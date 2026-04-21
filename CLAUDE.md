# Lab 14 — AI Evaluation Factory

> "If you can't measure it, you can't improve it."

An automated benchmarking system that evaluates a RAG-based AI Agent end-to-end: golden dataset generation, retrieval metrics (Hit Rate, MRR), multi-model LLM judging (Claude + GPT), regression gating, and cost tracking.

---

## Team

| GitHub | Role |
|--------|------|
| dduyanhhoang | Agent & Integration — RAG pipeline, Pinecone ingestion, overall wiring |
| thnhphng04 | Data & SDG — synthetic data generation, source documents, failure analysis |
| trannhatvi-ai | AI/Backend — multi-judge consensus, retrieval evaluation metrics |
| Bethokawaii | DevOps/Analyst — async runner, cost tracking, regression release gate |

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Source Data](#2-source-data)
3. [Step 1 — Chunk & Index](#3-step-1--chunk--index)
4. [Step 2 — Golden Dataset (SDG)](#4-step-2--golden-dataset-sdg)
5. [Step 3 — RAG Agent](#5-step-3--rag-agent)
6. [Step 4 — Evaluation Engine](#6-step-4--evaluation-engine)
7. [Step 5 — Benchmark Runner](#7-step-5--benchmark-runner)
8. [Step 6 — Regression & Release Gate](#8-step-6--regression--release-gate)
9. [Results](#9-results)
10. [Failure Analysis](#10-failure-analysis)
11. [How to Run](#11-how-to-run)
12. [Technical Decisions & Trade-offs](#12-technical-decisions--trade-offs)

---

## 1. Problem Statement

Building an AI product without an evaluation system is flying blind. This lab builds an **Evaluation Factory** that answers:

- How accurate is the agent's retrieval? (Did it find the right chunks?)
- How good are the answers? (Do two independent judges agree?)
- Is the new version better than the old one? (Regression gating)
- What does one eval run cost? (Cost accountability)

The input is a RAG agent trained on CS229 Machine Learning lecture notes. The output is an automated verdict: **APPROVE** or **BLOCK RELEASE**.

---

## 2. Source Data

Two CS229 lecture note files in `data/raw/`:

| File | Content | Chunks |
|------|---------|--------|
| `ml_1.md` | Supervised learning, linear regression, gradient descent, normal equations, logistic regression | 12 chunks (ml_1_chunk_1 … ml_1_chunk_12) |
| `ml_2.md` | Generative learning, GDA, Naive Bayes, neural networks, perceptron | 8 chunks (ml_2_chunk_0 … ml_2_chunk_7) |

Chunk IDs follow the pattern `{stem}_chunk_{n}` (e.g. `ml_1_chunk_3`). These IDs are the linking key between the vector store, the golden dataset, and the retrieval evaluator.

---

## 3. Step 1 — Chunk & Index

### 3a. Chunking (`data/synthetic_gen.py` → `data/chunks.jsonl`)

`chunk_markdown()` splits each `.md` file on header lines (`## ...`). Each chunk keeps its header, full text, source filename, and a stable ID. Chunks shorter than 100 characters are dropped.

```
data/raw/ml_1.md  ──►  12 chunks
data/raw/ml_2.md  ──►   8 chunks
                  ──►  data/chunks.jsonl  (20 records)
```

Each record in `chunks.jsonl`:
```json
{"id": "ml_1_chunk_3", "header": "## Gradient Descent", "text": "...", "source": "ml_1.md"}
```

### 3b. Vector Indexing (`agent/pre.py` → Pinecone `ml-rag`)

`pre.py` reads `chunks.jsonl`, embeds each chunk with **OpenAI `text-embedding-3-large`** (dim=3072), and upserts into Pinecone using `chunk_id` as the vector ID.

```
data/chunks.jsonl
    └─► OpenAI text-embedding-3-large  (dim=3072)
    └─► Pinecone index "ml-rag"  (cosine, us-east-1)
            20 vectors  [ml_1_chunk_1 … ml_2_chunk_7]
```

Pinecone index configuration:
- Index name: `ml-rag`
- Dimension: 3072
- Metric: cosine
- Cloud: AWS us-east-1
- Status: **20 vectors indexed as of 2026-04-21**

Note: The `day-14` index also exists in the account but has dimension 1024 (wrong for this embedding model) and is unused. Always set `PINECONE_INDEX_NAME=ml-rag` in `.env`.

---

## 4. Step 2 — Golden Dataset (SDG)

`data/synthetic_gen.py` calls **Claude Haiku** (`claude-haiku-4-5-20251001`) to generate evaluation cases automatically.

### Generation strategy

**Normal cases** — 3 QA pairs per chunk (60 total):
- Prompt asks for a mix of factual recall, conceptual understanding, and application questions
- Each pair is grounded to its source chunk and includes the `chunk_id` as `expected_retrieval_ids`
- Concurrency controlled by `asyncio.Semaphore(5)` to avoid rate limits

**Adversarial cases** — 10 cases (red teaming):
- `out_of_context` — asks about topics outside the documents (cooking, politics)
- `hallucination_bait` — asks for a specific number/name that sounds plausible but isn't in the text
- `prompt_injection` — tries to override system instructions
- `ambiguous` — vague question with no single correct answer
- These have empty `expected_retrieval_ids` (no retrieval metric is computed for them)

### Output: `data/golden_set.jsonl` — 70 records

| Difficulty | Count | Types |
|------------|-------|-------|
| easy | 20 | factual |
| medium | 21 | conceptual |
| hard | 19 | application |
| adversarial | 10 | out_of_context, hallucination_bait, prompt_injection, ambiguous |

Each record schema:
```json
{
  "question": "What is the update rule for gradient descent?",
  "expected_answer": "theta := theta - alpha * gradient(J(theta))",
  "context": "full chunk text...",
  "expected_retrieval_ids": ["ml_1_chunk_3"],
  "metadata": {
    "difficulty": "easy",
    "type": "factual",
    "source": "ml_1.md",
    "chunk_id": "ml_1_chunk_3"
  }
}
```

---

## 5. Step 3 — RAG Agent

`agent/main_agent.py` implements the agent under evaluation.

### Retrieval — BM25Okapi

All 20 chunks from `chunks.jsonl` are tokenized and indexed in memory using `rank-bm25.BM25Okapi`. On each query, the top-k chunks are retrieved by BM25 score.

```python
MainAgent(top_k=3, version="v1")   # V1 — retrieves 3 chunks
MainAgent(top_k=5, version="v2")   # V2 — retrieves 5 chunks
```

Retrieved chunk IDs are returned as `retrieved_ids` in the response, which the evaluator compares against `expected_retrieval_ids` from the golden dataset.

### Generation — GPT-4o-mini

The agent formats the top-k chunks as context and calls OpenAI `gpt-4o-mini` with a strict system prompt:

> "Answer using ONLY the provided context. If the context lacks relevant information, say 'I don't have information about this topic.'"

Response format:
```json
{
  "answer": "...",
  "contexts": ["chunk text 1", "chunk text 2", ...],
  "retrieved_ids": ["ml_1_chunk_3", "ml_1_chunk_7"],
  "metadata": {"model": "gpt-4o-mini", "tokens_used": 892, "version": "v1"}
}
```

### Why OpenAI for the agent (not Claude)?

Anthropic's rate limit is 50 requests/minute on the shared org account. Running 70 cases × 2 passes = 140 agent calls. If those also use Claude, the budget is exhausted before judging even starts. Using GPT-4o-mini for generation preserves the entire Claude quota for the two-model judge.

---

## 6. Step 4 — Evaluation Engine

### 6a. Retrieval Evaluation (`engine/retrieval_eval.py`)

Two RAGAS-style metrics computed for every case that has `expected_retrieval_ids`:

**Hit Rate (top-k=3)**
```
1.0  if any expected chunk ID appears in the top-3 retrieved IDs
0.0  otherwise
```

**MRR (Mean Reciprocal Rank)**
```
1 / rank_of_first_relevant_chunk   (0.0 if none found)
```

Adversarial cases (empty `expected_retrieval_ids`) are excluded from metric averages, so the 60 normal cases drive retrieval scores.

### 6b. Multi-Judge Consensus (`engine/llm_judge.py`)

Two independent judges score each answer on a 1–5 scale:

| Score | Meaning |
|-------|---------|
| 1 | Completely wrong or irrelevant |
| 2 | Major errors or missing key points |
| 3 | Partially correct, missing details |
| 4 | Mostly correct with minor issues |
| 5 | Fully correct and well-explained |

**Judge 1:** `claude-haiku-4-5-20251001` (Anthropic)
**Judge 2:** `gpt-4o-mini` (OpenAI)

Both judges receive the same prompt: question + reference answer + student answer → single integer.

**Conflict resolution:**

| Score difference | Resolution | Agreement rate |
|-----------------|------------|----------------|
| diff ≤ 1 | Average both scores | 1.0 |
| diff = 2 | Average both scores (flagged) | 0.5 |
| diff ≥ 3 | Use Claude score as tiebreaker | 0.0 |

**Agreement metric (Cohen's Kappa approximation):**
Fraction of cases where `|score_claude - score_gpt| ≤ 1`. Result across all 70 cases: **88.6%**.

**Position bias check (`check_position_bias`):**
Presents two responses as A/B, then swaps order and re-evaluates. If the winning response changes after the swap, position bias is detected.

---

## 7. Step 5 — Benchmark Runner

`engine/runner.py` orchestrates evaluation of a full dataset.

### Async batching

```python
BenchmarkRunner.run_all(dataset, batch_size=5)
```

Cases run in batches of 5, with all 5 executing concurrently via `asyncio.gather`. A 3-second pause between batches prevents exceeding the Anthropic 50 req/min rate limit.

Per-case execution order (all async within the batch):
```
agent.query(question)
    └─► retrieval_eval.calculate_hit_rate() + calculate_mrr()
    └─► judge.evaluate_multi_judge()   [Claude + GPT in parallel]
```

### Cost tracking

Every agent response includes `tokens_used`. The runner multiplies by model rate:

| Model | Rate |
|-------|------|
| gpt-4o-mini | $0.00015 / 1K tokens |
| claude-haiku-4-5-20251001 | $0.00025 / 1K tokens |

Accumulated to `total_cost_usd` across the run. Last run cost for 70 cases: **$0.024**.

### Per-case result schema

```json
{
  "test_case": "question text",
  "agent_response": "answer text",
  "latency": 2.11,
  "tokens_used": 892,
  "cost_usd": 0.000134,
  "ragas": {
    "retrieval": {"hit_rate": 1.0, "mrr": 1.0}
  },
  "judge": {
    "final_score": 4.5,
    "agreement_rate": 1.0,
    "individual_scores": {"claude-haiku": 4, "gpt-4o-mini": 5},
    "reasoning": "Claude=4, GPT=5, diff=1"
  },
  "status": "pass"
}
```

---

## 8. Step 6 — Regression & Release Gate

`main.py` runs the benchmark twice to simulate an agent upgrade:

```
V1  MainAgent(top_k=3)  ──►  v1_summary
V2  MainAgent(top_k=5)  ──►  v2_summary
```

### Release gate (all 4 checks must pass)

| Check | Threshold | V2 result | Status |
|-------|-----------|-----------|--------|
| Score improved | delta ≥ 0 | +0.014 | PASS |
| Hit rate acceptable | ≥ 50% | 74.3% | PASS |
| Judges agree | agreement ≥ 60% | 88.6% | PASS |
| Cost within budget | ≤ $5.00 | $0.024 | PASS |

Decision: **APPROVE RELEASE**

### Output: `reports/summary.json`

```json
{
  "metadata": {"version": "Agent_V2_Optimized", "total": 70, "timestamp": "..."},
  "metrics": {
    "avg_score": 4.0643,
    "hit_rate": 0.7429,
    "mrr": 0.6569,
    "agreement_rate": 0.8857,
    "cohen_kappa": 0.8857,
    "pass_rate": 0.8286
  },
  "cost": {"total_tokens": 157275, "total_cost_usd": 0.0236},
  "regression": {
    "v1_metrics": {"avg_score": 4.05, "hit_rate": 0.7429, ...},
    "v2_metrics": {"avg_score": 4.0643, "hit_rate": 0.7429, ...},
    "delta_score": 0.0143,
    "delta_hit_rate": 0.0
  },
  "gate": {
    "decision": "APPROVE",
    "checks": {"score_improved": true, "hit_rate_ok": true, "agreement_ok": true, "cost_ok": true}
  }
}
```

---

## 9. Results

### Overall (V2, 70 cases)

| Metric | Value |
|--------|-------|
| Avg judge score | 4.06 / 5.0 |
| Pass rate (score ≥ 3) | 82.9% (58/70) |
| Hit rate (top-3) | 74.3% |
| MRR | 0.657 |
| Judge agreement rate | 88.6% |
| Avg agent latency | 2.11 s/case |
| Total eval cost | $0.024 |

### Score distribution

```
5.0  ████████████████████████████████████  33 cases (47%)
4.5  ████████                               8 cases (11%)
4.0  ███████████████                       15 cases (21%)
3.5  █                                      1 case  ( 1%)
3.0  █                                      1 case  ( 1%)
2.0  █████                                  5 cases  ( 7%)
1.0  ███████                                7 cases (10%)
```

### Judge conflict cases (8 total)

| Agreement | Count | Example |
|-----------|-------|---------|
| 0.5 (diff=2) | 4 | "How would you design a spam filter…" |
| 0.0 (diff≥3) | 4 | "Who is Prof. Ng's co-instructor…" (hallucination bait) |

All 4 cases with agreement=0.0 were adversarial (hallucination bait or prompt injection), confirming the conflict resolution logic correctly surfaces uncertain edge cases.

### V1 vs V2 regression

| Metric | V1 (top_k=3) | V2 (top_k=5) | Delta |
|--------|-------------|-------------|-------|
| avg_score | 4.05 | 4.06 | +0.014 |
| hit_rate | 74.3% | 74.3% | 0.0% |
| MRR | 0.648 | 0.657 | +0.009 |
| pass_rate | 81.4% | 82.9% | +1.5% |

V2 shows marginal improvement. Hit rate is identical because the relevant chunk is consistently either in the top-3 or not — adding more retrieved chunks (top_k=5) improves context for generation but not the binary hit rate.

---

## 10. Failure Analysis

See `analysis/failure_analysis.md` for full details. Summary:

### Failure clusters

| Root cause | Cases | Fix |
|-----------|-------|-----|
| BM25 vocabulary mismatch | 5 | Hybrid dense+sparse retrieval (OpenAI embeddings + BM25) |
| Context truncation (1500 char limit) | 4 | Raise limit to 3000 chars or use smaller chunks (≤1000 chars) |
| Adversarial / out-of-context | 3 | Expected behavior — agent correctly declines, judges disagree on scoring |

### Worst case: vocabulary mismatch

Question asked about "normal equations" but the chunk uses the phrase "closed-form solution." BM25 found zero term overlap and retrieved unrelated chunks, causing the agent to answer "I don't have information about this topic" — scoring 1/5 despite the answer existing in the corpus.

This is BM25's fundamental limitation: it matches tokens, not meaning.

---

## 11. How to Run

### Prerequisites

```bash
# Python 3.12+
uv sync
# or: pip install -r requirements.txt

# .env file (do not commit)
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=ml-rag
```

### Full pipeline

```bash
# 0. Index chunks into Pinecone (once)
python agent/pre.py

# 1. Generate 70-case golden dataset (once, or to regenerate)
python data/synthetic_gen.py

# 2. Run V1 vs V2 benchmark (~5 min for 70 cases)
python main.py

# 3. Validate submission format
python check_lab.py
```

### Expected output

```
Starting benchmark: Agent_V1_Base...
  Batch 1/14 (5 cases)...
  ...
Starting benchmark: Agent_V2_Optimized...
  ...
REGRESSION COMPARISON
V1 avg_score : 4.0500
V2 avg_score : 4.0643
Delta        : +0.0143

Release Gate Checks:
  score_improved  PASS
  hit_rate_ok     PASS
  agreement_ok    PASS
  cost_ok         PASS

DECISION: APPROVE RELEASE
Reports saved to reports/
```

---

## 12. Technical Decisions & Trade-offs

### BM25 (in-memory) vs Pinecone for agent retrieval

We index all 20 chunks into Pinecone (`agent/pre.py`) for production use, but the benchmark agent (`agent/main_agent.py`) uses in-memory **BM25** for retrieval.

- **Why:** Embedding API calls add ~1 second of latency per query. With 140 benchmark queries, that is 140 extra API round-trips. BM25 runs in microseconds and produces deterministic results across runs.
- **Trade-off:** BM25 fails on vocabulary mismatch (see Failure Analysis). Pinecone with semantic embeddings would close this gap.
- **To switch:** Replace `_bm25_cache.get_scores()` in `MainAgent.query()` with `PineconeVectorStore.similarity_search_with_score()`.

### Claude for judging, OpenAI for generation

The Anthropic org account is limited to **50 requests/minute** for claude-haiku. Running 70 × 2 = 140 agent calls + 140 judge calls × 2 models = 280 calls in total. Splitting by provider prevents either limit from becoming a bottleneck:

| Role | Model | Provider | Reason |
|------|-------|----------|--------|
| Agent generation | gpt-4o-mini | OpenAI | High rate limit, fast, cheap |
| Judge 1 | claude-haiku-4-5-20251001 | Anthropic | Diverse perspective |
| Judge 2 | gpt-4o-mini | OpenAI | Diverse perspective + fast |

### Async batching design

`asyncio.gather` runs all 5 cases in a batch truly concurrently (I/O-bound wait happens in parallel). A 3-second inter-batch sleep smooths the request rate across batches. Module-level `asyncio.Semaphore` objects independently cap OpenAI (10 concurrent) and Anthropic (2 concurrent) calls.

### golden_set.jsonl format issue

The file was originally written as pretty-printed multi-line JSON. The reader `json.loads(line)` expected single-line JSONL. Fixed by running a one-time conversion using Python's streaming `json.JSONDecoder.raw_decode()` before re-writing in proper JSONL format.

### Pinecone index dimension mismatch

Two indexes existed: `day-14` (dim=1024) and `ml-rag` (dim=3072). `text-embedding-3-large` outputs 3072 dimensions — using `day-14` would have caused silent failures at upsert time. Added `PINECONE_INDEX_NAME=ml-rag` to `.env` and guarded `ingest_documents()` against calling `delete(delete_all=True)` on an empty namespace (which returns 404).
