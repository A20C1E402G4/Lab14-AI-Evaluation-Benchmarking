import shutil
import re
import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def _load_env_file(env_path: Path) -> dict:
    """Load simple KEY=VALUE pairs from .env file."""
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


def load_pdf_documents(data_dir: Path):
    """Load all PDFs recursively from papers directory."""
    loader = DirectoryLoader(
        path=str(data_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def normalize_text(text: str) -> str:
    """Basic text cleaning for PDF extraction artifacts."""
    cleaned = text.replace("\u00a0", " ")
    # Merge line-break hyphenation: "learn-\ning" -> "learning"
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    # Replace hard line breaks with spaces to rebuild normal sentences.
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _ocr_page_text(pdf_path: Path, page_number: int) -> str:
    """Run OCR for a single page in a PDF. Return empty string on failure."""
    try:
        import pytesseract  # type: ignore[import-not-found]
        from pdf2image import convert_from_path  # type: ignore[import-not-found]
    except Exception:
        return ""

    if not pdf_path.exists():
        return ""

    try:
        images = convert_from_path(
            str(pdf_path),
            first_page=page_number + 1,
            last_page=page_number + 1,
        )
        if not images:
            return ""
        ocr_text = pytesseract.image_to_string(images[0], lang="eng+vie")
        return normalize_text(ocr_text)
    except Exception:
        return ""


def clean_documents(documents, enable_ocr_fallback: bool = True):
    """Clean extracted PDF text. OCR fallback only when extracted page text is empty."""
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

    stats = {
        "total_pages": len(documents),
        "kept_pages": len(cleaned_docs),
        "dropped_pages": dropped_pages,
        "ocr_fallback_used": ocr_fallback_used,
    }
    return cleaned_docs, stats


def chunk_documents(documents):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
    )
    return splitter.split_documents(documents)


def rebuild_chroma(chunks, persist_dir: Path, batch_size: int = 100):
    """Embed chunks with Ollama and persist vectors into ChromaDB."""
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=str(persist_dir))

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding", unit="batch"):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)


def main():
    project_root = Path(__file__).resolve().parent
    papers_dir = project_root / "papers" / "ai_thucchien"
    chroma_dir = project_root / "chroma_db"
    env_values = _load_env_file(project_root / ".env")
    ocr_enabled = _to_bool(
        os.getenv("OCR_FALLBACK_ENABLED", env_values.get("OCR_FALLBACK_ENABLED")),
        default=True,
    )

    if not papers_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc du lieu: {papers_dir}")

    print("1) Dang trich xuat text tu PDF trong papers/ai_thucchien...")
    docs = load_pdf_documents(papers_dir)
    if not docs:
        raise ValueError("Khong tim thay file PDF nao trong thu muc papers.")
    print(f"   Da tai {len(docs)} trang tai lieu.")

    print("2) Dang lam sach van ban...")
    print(f"   OCR fallback: {'BAT' if ocr_enabled else 'TAT'}")
    cleaned_docs, clean_stats = clean_documents(docs, enable_ocr_fallback=ocr_enabled)
    if not cleaned_docs:
        raise ValueError("Tat ca trang PDF rong sau buoc lam sach.")
    print(f"   Da giu {clean_stats['kept_pages']}/{clean_stats['total_pages']} trang hop le.")
    if ocr_enabled:
        print(f"   OCR fallback da dung cho {clean_stats['ocr_fallback_used']} trang rong.")
    if clean_stats["dropped_pages"] > 0:
        print(f"   Bo qua {clean_stats['dropped_pages']} trang khong the trich xuat/OCR.")

    print("3) Dang chunking...")
    chunks = chunk_documents(cleaned_docs)
    print(f"   Da tao {len(chunks)} chunks.")

    print("4) Dang embedding va luu vao ChromaDB...")
    rebuild_chroma(chunks, chroma_dir, batch_size=100)
    print(f"Hoan tat. Du lieu da duoc luu tai: {chroma_dir}")


if __name__ == "__main__":
    main()
