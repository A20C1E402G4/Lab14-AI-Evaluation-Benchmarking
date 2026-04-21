import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List

from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

_CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.jsonl"
_MODEL = "gpt-4o-mini"
_SEMAPHORE = asyncio.Semaphore(10)

_chunks_cache: List[Dict] = []
_bm25_cache: BM25Okapi | None = None
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _ensure_loaded():
    global _chunks_cache, _bm25_cache
    if _bm25_cache is not None:
        return
    chunks = []
    with open(_CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    _chunks_cache = chunks
    tokenized = [c["text"].lower().split() for c in chunks]
    _bm25_cache = BM25Okapi(tokenized)


class MainAgent:
    """RAG agent using BM25 retrieval over chunks.jsonl and GPT-4o-mini for generation."""

    def __init__(self, top_k: int = 5, version: str = "v1"):
        self.top_k = top_k
        self.version = version
        self.name = f"RAGAgent-{version}"

    async def query(self, question: str) -> Dict:
        _ensure_loaded()

        # BM25 retrieval
        tokens = question.lower().split()
        scores = _bm25_cache.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]
        top_chunks = [_chunks_cache[i] for i in top_indices]

        retrieved_ids = [c["id"] for c in top_chunks]
        context_texts = [c["text"] for c in top_chunks]

        context = "\n\n---\n\n".join(
            f"[{c['id']}]\n{c['text'][:1500]}" for c in top_chunks
        )

        system_msg = (
            "You are a Machine Learning teaching assistant. "
            "Answer the question using ONLY the provided context. "
            "Be concise (2-4 sentences). "
            "If the context lacks relevant information, say 'I don't have information about this topic.'"
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"

        for attempt in range(5):
            try:
                async with _SEMAPHORE:
                    response = await _get_client().chat.completions.create(
                        model=_MODEL,
                        max_tokens=512,
                        temperature=0,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                    )
                break
            except Exception as e:
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt + 1)
                else:
                    raise

        answer = response.choices[0].message.content.strip()
        usage = response.usage
        tokens_used = (usage.prompt_tokens + usage.completion_tokens) if usage else 0

        return {
            "answer": answer,
            "contexts": context_texts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": _MODEL,
                "tokens_used": tokens_used,
                "version": self.version,
            },
        }


if __name__ == "__main__":
    agent = MainAgent()

    async def _test():
        resp = await agent.query("What is gradient descent?")
        print("Answer:", resp["answer"][:200])
        print("Retrieved IDs:", resp["retrieved_ids"])
        print("Tokens used:", resp["metadata"]["tokens_used"])

    asyncio.run(_test())
