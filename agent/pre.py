import json
import re
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

AGENT_DIR = Path(__file__).resolve().parent
CHUNKS_JSONL = AGENT_DIR.parent / "data" / "chunks.jsonl"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "day-14")
EMBEDDING_DIM = 3072  # text-embedding-3-large output dimension


def _load_env_file(env_path: Path) -> dict:
    env_map = {}
    if not env_path.exists():
        return env_map
    for line in env_path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        env_map[key.strip()] = value.strip().strip('"').strip("'")
    return env_map


def _to_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_text(text: str) -> str:
    cleaned = text.replace(" ", " ")
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _ocr_page_text(pdf_path: Path, page_number: int) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        return ""
    if not pdf_path.exists():
        return ""
    try:
        images = convert_from_path(str(pdf_path), first_page=page_number + 1, last_page=page_number + 1)
        if not images:
            return ""
        return normalize_text(pytesseract.image_to_string(images[0], lang="eng+vie"))
    except Exception:
        return ""


def load_pdf_documents(data_dir: Path):
    loader = DirectoryLoader(
        path=str(data_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def clean_documents(documents, enable_ocr_fallback: bool = True):
    cleaned_docs = []
    dropped_pages = 0
    ocr_fallback_used = 0
    for doc in documents:
        text = normalize_text(doc.page_content or "")
        if not text and enable_ocr_fallback:
            source = Path(str(doc.metadata.get("source", "")))
            page = int(doc.metadata.get("page", 0) or 0)
            ocr_text = _ocr_page_text(source, page)
            if ocr_text:
                text = ocr_text
                ocr_fallback_used += 1
                doc.metadata["extraction_method"] = "ocr_fallback"
        if not text:
            dropped_pages += 1
            continue
        doc.metadata.setdefault("extraction_method", "text_layer")
        doc.page_content = text
        cleaned_docs.append(doc)
    return cleaned_docs, {
        "total_pages": len(documents),
        "kept_pages": len(cleaned_docs),
        "dropped_pages": dropped_pages,
        "ocr_fallback_used": ocr_fallback_used,
    }


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True, strip_whitespace=True
    )
    return splitter.split_documents(documents)


def _get_or_create_index(pc: Pinecone) -> object:
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        import time
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc.Index(PINECONE_INDEX_NAME)


def ingest_documents(docs: list[Document], batch_size: int = 50):
    """Embed and upsert documents into Pinecone, using chunk_id as the vector ID."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = _get_or_create_index(pc)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Delete existing vectors before re-ingesting
    index.delete(delete_all=True)

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    ids = [doc.metadata.get("chunk_id", str(i)) for i, doc in enumerate(docs)]

    for i in tqdm(range(0, len(docs), batch_size), desc="Upserting to Pinecone", unit="batch"):
        batch_docs = docs[i: i + batch_size]
        batch_ids = ids[i: i + batch_size]
        vectorstore.add_documents(batch_docs, ids=batch_ids)

    print(f"Ingested {len(docs)} documents into Pinecone index '{PINECONE_INDEX_NAME}'.")


def main():
    if CHUNKS_JSONL.exists():
        print(f"Found {CHUNKS_JSONL} — ingesting from shared chunk registry.")
        docs = []
        with open(CHUNKS_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                docs.append(Document(
                    page_content=c["text"],
                    metadata={"chunk_id": c["id"], "source": c["source"], "header": c.get("header", "")},
                ))
        ingest_documents(docs)
        return

    # Fallback: ingest raw PDFs
    papers_dir = AGENT_DIR / "papers" / "ai_thucchien"
    env_values = _load_env_file(AGENT_DIR / ".env")
    ocr_enabled = _to_bool(
        os.getenv("OCR_FALLBACK_ENABLED", env_values.get("OCR_FALLBACK_ENABLED")), default=True
    )

    if not papers_dir.exists():
        raise FileNotFoundError(f"No chunks.jsonl and no papers dir: {papers_dir}")

    print("1) Loading PDFs...")
    docs = load_pdf_documents(papers_dir)
    if not docs:
        raise ValueError("No PDF files found.")

    print("2) Cleaning text...")
    cleaned_docs, stats = clean_documents(docs, enable_ocr_fallback=ocr_enabled)
    if not cleaned_docs:
        raise ValueError("All pages empty after cleaning.")
    print(f"   Kept {stats['kept_pages']}/{stats['total_pages']} pages.")

    print("3) Chunking...")
    chunks = chunk_documents(cleaned_docs)
    print(f"   Created {len(chunks)} chunks.")

    print("4) Ingesting into Pinecone...")
    ingest_documents(chunks)


if __name__ == "__main__":
    main()
