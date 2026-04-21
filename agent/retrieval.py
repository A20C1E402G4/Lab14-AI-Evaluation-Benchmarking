from dataclasses import dataclass
from typing import Dict, List, Tuple

from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings


DEFAULT_ANSWER_STYLE = (
    "Answer briefly in 1-3 short paragraphs and only include details needed for the question."
)

SUMMARY_ANSWER_STYLE = (
    "The user asks for a summary. Provide a comprehensive answer with enough detail. "
    "Use 2-4 short sections and include key points, important entities/numbers, and practical takeaways."
)


def build_answer_style_instruction(query: str) -> str:
    q = query.lower()
    summary_keywords = [
        "tom tat",
        "tóm tắt",
        "tong quan",
        "tổng quan",
        "khai quat",
        "khái quát",
        "summary",
        "overview",
    ]
    if any(keyword in q for keyword in summary_keywords):
        return SUMMARY_ANSWER_STYLE
    return DEFAULT_ANSWER_STYLE


@dataclass
class HybridRerankRetriever:
    vectorstore: Chroma
    bm25_retriever: BM25Retriever
    ranker: Ranker
    vector_k: int = 12
    bm25_k: int = 12
    final_k: int = 5
    score_threshold: float = 0.2

    def _rrf_fuse(
        self, vector_hits: List[Tuple[Document, float]], bm25_hits: List[Document]
    ) -> List[Document]:
        # Reciprocal rank fusion merges semantic and lexical rankings robustly.
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        rrf_k = 60

        for rank, (doc, score) in enumerate(vector_hits, start=1):
            if score < self.score_threshold:
                continue
            key = self._doc_key(doc)
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

        for rank, doc in enumerate(bm25_hits[: self.bm25_k], start=1):
            key = self._doc_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

        fused_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        return [doc_map[k] for k in fused_keys[: max(self.final_k * 4, self.final_k)]]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        src = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", ""))
        start_idx = str(doc.metadata.get("start_index", ""))
        text_head = doc.page_content[:120]
        return f"{src}|{page}|{start_idx}|{text_head}"

    def invoke(self, query: str) -> List[Document]:
        vector_hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.vector_k)
        bm25_hits = self.bm25_retriever.invoke(query)
        candidates = self._rrf_fuse(vector_hits, bm25_hits)

        if not candidates:
            return []

        rerank_passages = [
            {
                "id": str(i),
                "text": d.page_content,
                "meta": d.metadata,
            }
            for i, d in enumerate(candidates)
        ]

        try:
            rerank_request = RerankRequest(query=query, passages=rerank_passages)
            reranked = self.ranker.rerank(rerank_request)
            final_docs = []
            for item in reranked[: self.final_k]:
                doc_idx = int(item["id"])
                base_doc = candidates[doc_idx]
                metadata = dict(base_doc.metadata)
                metadata["rerank_score"] = float(item.get("score", 0.0))
                final_docs.append(Document(page_content=base_doc.page_content, metadata=metadata))
            return final_docs
        except Exception:
            # Fallback to fused ranking if reranker fails to initialize or run.
            return candidates[: self.final_k]


def _load_documents_from_chroma(vectorstore: Chroma) -> List[Document]:
    data = vectorstore.get(include=["documents", "metadatas"])
    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []

    out: List[Document] = []
    for i, text in enumerate(docs):
        if not text:
            continue
        metadata = metas[i] if i < len(metas) and metas[i] is not None else {}
        out.append(Document(page_content=text, metadata=metadata))
    return out


def get_rag_components():
    template = (
        "You are a strict, citation-focused assistant for a private knowledge base. \n"
        "RULES:\n"
        "1. Only use information from the provided context.\n"
        "2. If you don't know the answer, say so.\n"
        "3. Always include citations for any information you provide.\n"
        "4. Use the following format for citations: [source_name](source_url)\n"
        "5. Do not include any information that is not in the provided context.\n"
        "6. Be concise and to the point.\n"
        "7. Do not include any personal opinions or assumptions.\n"
        "8. Always prioritize accuracy and relevance over completeness.\n"
        "9. If the context contains conflicting information, present all sides and cite the sources.\n"
        "10. Follow this answer style instruction: {answer_style_instruction}\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
    )

    # 1. Khởi tạo Embeddings & VectorStore
    print("Đang khởi tạo Ollama Embeddings và ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # 2. Cấu hình Hybrid Retriever (Vector + BM25) + Rerank
    all_docs = _load_documents_from_chroma(vectorstore)
    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 12
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        retriever = HybridRerankRetriever(
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            ranker=ranker,
            vector_k=12,
            bm25_k=12,
            final_k=5,
            score_threshold=0.2,
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.2},
        )

    # 3. Thiết lập Prompt Template
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Hàm Format Docs
    def format_docs(docs):
        return "\n\n".join([f"Source: {d.metadata.get('source', 'N/A')}\nContent: {d.page_content}" for d in docs])

    # 5. Khởi tạo LLM (Qwen 2.5)
    llm = ChatOllama(
        model="qwen2.5", 
        temperature=0
    )

    return retriever, prompt, llm, format_docs


def get_rag_chain():
    retriever, prompt, llm, format_docs = get_rag_components()
    prompt_with_style = prompt.partial(answer_style_instruction=DEFAULT_ANSWER_STYLE)

    # 6. Tạo Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_with_style
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Xóa bỏ phần test bằng input() ở cuối file cũ đi để tránh bị chạy lúc import