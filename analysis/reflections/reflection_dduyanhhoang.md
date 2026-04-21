# Individual Reflection — dduyanhhoang

**Lab:** Day 14 — AI Evaluation Factory
**Role:** Agent & Integration
**Commit:** `0273b48` — `feat(agent): implement real RAG pipeline with BM25 retrieval and GPT-4o-mini generation`

---

## 1. What I Built

My responsibility was the RAG agent pipeline and the data ingestion layer — the two components that every other module depends on. Without a working agent that returns `retrieved_ids`, the retrieval evaluator has nothing to measure. Without a working ingestion script, the vector store is empty.

### `agent/main_agent.py` — The RAG Agent

The original file was a mock: it slept for 0.5 seconds and returned a hardcoded string. I replaced it with a real pipeline.

**Retrieval — BM25Okapi**

I chose BM25 (`rank-bm25`) over direct Pinecone queries for the benchmark agent. The reason is determinism and latency. Each Pinecone similarity search requires an OpenAI embedding API call first (one round-trip ~300–500ms). Across 140 benchmark queries (70 cases × 2 runs), that is 140 extra API calls adding roughly 50 seconds of wait time. BM25 runs in microseconds and produces the same results every run, making benchmarks reproducible.

The implementation loads all 20 chunks from `chunks.jsonl` into memory once (module-level singleton pattern), builds a `BM25Okapi` index from tokenized text, and on each query returns the top-k chunk IDs sorted by BM25 score:

```python
scores = _bm25_cache.get_scores(question.lower().split())
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
retrieved_ids = [_chunks_cache[i]["id"] for i in top_indices]
```

These IDs flow directly into `RetrievalEvaluator.calculate_hit_rate()` and `calculate_mrr()` for evaluation.

**Generation — GPT-4o-mini**

The agent sends the top-k chunks as formatted context to `gpt-4o-mini` with a strict system prompt requiring answers to come only from the provided context. I used OpenAI here instead of Claude — see Problem Solving section for why.

**V1 vs V2 design**

The two benchmark versions are the same class with different constructor parameters:
- `MainAgent(top_k=3)` = V1
- `MainAgent(top_k=5)` = V2

This isolates the single variable being tested (how much context the agent retrieves) while keeping everything else constant.

### `agent/pre.py` — Pinecone Ingestion Pipeline

This script was already partially written. It loaded PDFs from a local directory. I rewired it to read from `data/chunks.jsonl` instead — the shared chunk registry that `synthetic_gen.py` produces. Every chunk gets embedded with `text-embedding-3-large` (3072 dimensions) and upserted into Pinecone with `chunk_id` as the vector ID.

Using `chunk_id` as the Pinecone vector ID is critical for the evaluation system to work: it means a Pinecone similarity search returns the same IDs that `expected_retrieval_ids` in the golden dataset contains, making retrieval evaluation directly meaningful.

Current state: **20 vectors indexed in Pinecone index `ml-rag`** as of 2026-04-21.

---

## 2. Problems I Solved

### Problem 1 — Pinecone dimension mismatch (silent failure risk)

The account had two Pinecone indexes: `day-14` (dim=1024) and `ml-rag` (dim=3072). The code defaulted to `day-14` via `PINECONE_INDEX_NAME=day-14`. Attempting to upsert `text-embedding-3-large` vectors (3072-dim) into a 1024-dim index would throw an error at runtime.

**Fix:** I identified `ml-rag` as the correct index (3072-dim, already matching the embedding model), then added `PINECONE_INDEX_NAME=ml-rag` to `.env`. The `_get_or_create_index()` function already checked for existence before creating — so it would silently use the wrong-dimension index if it existed. Setting the env variable explicitly routes all upserts to the right index.

### Problem 2 — `delete(delete_all=True)` throws 404 on empty index

`ingest_documents()` called `index.delete(delete_all=True)` before upserting to avoid stale vectors. On a brand-new empty index, Pinecone returns a 404 because there is no default namespace to delete from.

**Fix:** Read `describe_index_stats()` first and skip the delete call when `total_vector_count == 0`:

```python
stats = index.describe_index_stats()
if stats.get("total_vector_count", 0) > 0:
    index.delete(delete_all=True)
```

### Problem 3 — Anthropic rate limit exhausted during benchmark

The first benchmark run hit Anthropic's 50 req/min rate limit at batch 8 of 14. The original plan was to use Claude Haiku for both agent generation and judging. With 70 cases × 2 runs × 3 Claude calls per case = 420 Claude API calls, the 50 req/min limit meant a minimum 8.4 minutes just in rate-limited wait time — and the concurrent batches made bursts worse.

**Fix:** Moved agent generation to `gpt-4o-mini` (OpenAI, much higher rate limit). This cut Claude calls from 3 to 1 per case (only for judging). Total Claude calls: 70 × 2 × 1 = 140, which completes without hitting the limit when combined with `asyncio.Semaphore(2)` for Anthropic calls and exponential backoff retry:

```python
for attempt in range(5):
    try:
        async with _SEMAPHORE:
            response = await _get_client().messages.create(...)
        break
    except Exception as e:
        if "rate_limit" in str(e).lower() and attempt < 4:
            await asyncio.sleep(2 ** attempt + 2)
        else:
            raise
```

### Problem 4 — `json.loads(line)` fails on multi-line golden_set.jsonl

The existing `golden_set.jsonl` file was pretty-printed (each JSON object spanning ~14 lines). The reader in `main.py` used `json.loads(line)` which expects each line to be a complete JSON object. This silently returned 0 valid records instead of 70.

**Fix:** Used Python's `json.JSONDecoder.raw_decode()` streaming parser to consume the file object-by-object regardless of line structure, then re-wrote the file as proper single-line JSONL. After this fix, all 70 records loaded correctly.

---

## 3. Technical Concepts

### Retrieval quality directly determines answer quality

The most important insight from running the benchmark is the correlation between retrieval hit rate and final judge score:

| Retrieval hit | Cases | Avg judge score |
|---------------|-------|----------------|
| Hit (correct chunk in top-3) | 52 | **4.38 / 5.0** |
| Miss (correct chunk not retrieved) | 18 | **3.17 / 5.0** |

A retrieval miss reduces the expected answer quality by 1.21 points (28% drop). This makes intuitive sense — if the agent doesn't retrieve the right chunk, it either hallucinates or says "I don't have information." Improving retrieval from 74.3% hit rate to 90% hit rate would mechanically improve the average judge score, even without touching the generation model or prompt.

This is why evaluating the retrieval stage separately (Hit Rate, MRR) is essential — aggregate answer quality scores hide where the system is actually failing.

### Hit Rate vs MRR

Both metrics evaluate retrieval quality but measure different things:

**Hit Rate** (binary): did the pipeline retrieve at least one correct chunk in the top-k?
- Top-3 hit rate of 74.3% means 26.7% of questions get wrong context regardless of generation quality.
- Does not penalize for rank — finding the right chunk at position 1 vs position 3 gives the same score.

**MRR** (positional): how high in the ranking is the first correct chunk?
```
MRR = 1/rank_of_first_relevant_result
      = 1.0 if found at rank 1
      = 0.5 if found at rank 2
      = 0.33 if found at rank 3
      = 0.0  if not found
```
Our MRR of 0.657 vs hit rate of 0.743 means that when the system does find the right chunk, it often finds it at rank 2 or 3 rather than rank 1. A better retriever would push the correct chunk to rank 1 more consistently (MRR approaching hit rate).

### BM25 and its limitations

BM25 (Best Matching 25) scores documents by term frequency (TF) weighted by inverse document frequency (IDF):

```
score(q, d) = Σ IDF(t) × (TF(t,d) × (k1+1)) / (TF(t,d) + k1 × (1 - b + b × |d|/avgdl))
```

Where `k1=1.5` and `b=0.75` are standard Okapi BM25 parameters. The key property is that it works on exact token matches. This is why 5 out of 12 failures were "vocabulary mismatch" — the user asked about "normal equations" but the chunk text says "closed-form solution." Same concept, different words. Zero BM25 score.

The fix is hybrid retrieval: BM25 for exact term matching + dense vector search (OpenAI embeddings) for semantic similarity. Our Pinecone index already stores the dense embeddings — the next version of the agent should query both and fuse results (Reciprocal Rank Fusion is already implemented in `agent/retrieval.py`).

### Cost vs quality trade-off

The full benchmark (70 cases × 2 runs) cost $0.024 total, or $0.00034 per case. Breakdown by stage:

- Agent generation (gpt-4o-mini, avg 892 tokens): ~$0.00013/case
- Judge calls (2 × Claude + GPT per case): ~$0.00021/case

To reduce cost by 30% without quality loss:
1. **Cache embeddings** — run BM25 retrieval without embedding API calls (already doing this)
2. **Batch judge calls** — OpenAI's Batch API offers 50% discount for async jobs; acceptable for non-realtime evaluation
3. **Skip judging for high-confidence cases** — if both judges agree instantly (diff=0) and score is 5, trust it; only run expensive tiebreaker logic for conflicts
4. **Use haiku for judge on easy questions** — classify question difficulty first, use GPT-3.5 for easy/medium, reserve GPT-4o for hard cases

---

## 4. What I Learned

**The contract between modules matters more than any individual module.** The hardest part of this lab was not writing BM25 retrieval or the GPT prompt — it was making sure `retrieved_ids` in the agent response matched the exact format that `RetrievalEvaluator` expected, which matched the exact format that `synthetic_gen.py` wrote into `expected_retrieval_ids`. One naming inconsistency anywhere in that chain and the retrieval metrics silently return 0.0 for everything.

**Rate limits are architecture constraints, not operational annoyances.** The decision to use OpenAI for agent generation was forced by Anthropic's 50 req/min limit. In production systems, API rate limits shape how you design concurrency, which models you use for which roles, and what your evaluation throughput can be. Designing around rate limits from the start (separate semaphores per provider, backoff retry, inter-batch pacing) is the right pattern.

**Evaluation systems need their own evaluation.** The 8 conflict cases (where Claude and GPT disagreed by 2+ points) were all on adversarial or ambiguous questions. This is expected — it reveals that our 1–5 scoring rubric is underspecified for edge cases. A production system would resolve this with a more detailed rubric, example anchors ("a score of 3 means X, here is an example"), and a calibration run before the main benchmark.
