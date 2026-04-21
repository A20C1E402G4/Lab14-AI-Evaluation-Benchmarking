# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a team lab project (Lab Day 14) to build an **AI Evaluation Factory** — an automated benchmarking system that evaluates a RAG-based AI Agent with metrics, multi-judge consensus, and regression gating.

---

## Commands

```bash
# Install dependencies (Python 3.12+, using uv)
uv sync
# or
pip install -r requirements.txt

# Step 0: Index chunks into Pinecone (run once, or after data changes)
python agent/pre.py

# Step 1: Generate golden dataset (run once, or to regenerate)
python data/synthetic_gen.py

# Step 2: Run full benchmark (V1 vs V2) and produce reports
python main.py

# Step 3: Validate submission format
python check_lab.py
```

---

## Architecture

```
data/raw/*.md
    └─► data/synthetic_gen.py
            ├─► data/chunks.jsonl       (20 chunks, shared IDs)
            └─► data/golden_set.jsonl   (70 QA test cases)

agent/pre.py
    └─► Embeds chunks.jsonl with OpenAI text-embedding-3-large
    └─► Upserts 20 vectors into Pinecone index "ml-rag" (dim=3072)

main.py  (orchestrates V1 vs V2 regression)
    └─► engine/runner.py  BenchmarkRunner.run_all()
            ├─► agent/main_agent.py  MainAgent.query()
            │       ├─ BM25Okapi retrieval over chunks.jsonl
            │       ├─ Returns retrieved_ids (chunk IDs)
            │       └─ GPT-4o-mini for answer generation
            │
            ├─► engine/retrieval_eval.py  RetrievalEvaluator
            │       ├─ calculate_hit_rate()  (top-k match)
            │       └─ calculate_mrr()       (Mean Reciprocal Rank)
            │
            └─► engine/llm_judge.py  LLMJudge
                    ├─ _judge_claude()    (claude-haiku-4-5-20251001)
                    ├─ _judge_openai()    (gpt-4o-mini)
                    ├─ conflict resolution (diff > 1 → Claude tiebreaker)
                    ├─ agreement_rate / Cohen's Kappa
                    └─ check_position_bias()  (swap A/B and re-judge)

reports/
    ├─ summary.json           (V2 metrics + V1 vs V2 regression + release gate)
    └─ benchmark_results.json (per-case details)
```

---

## Implementation Status

### Completed

#### `data/synthetic_gen.py`
- Reads `data/raw/*.md`, splits by markdown headers into chunks, writes `data/chunks.jsonl`.
- Calls `claude-haiku-4-5-20251001` (Anthropic API) to generate 3 QA pairs per chunk + 10 adversarial cases.
- Output: `data/golden_set.jsonl` — 70 records (easy: 20, medium: 21, hard: 19, adversarial: 10).
- Each record: `question`, `expected_answer`, `context`, `expected_retrieval_ids`, `metadata.difficulty/type/source/chunk_id`.

#### `agent/pre.py`
- Loads `data/chunks.jsonl` and embeds each chunk with `OpenAIEmbeddings(model="text-embedding-3-large")`.
- Upserts into Pinecone index `ml-rag` (dim=3072, cosine, us-east-1) using `chunk_id` as the vector ID.
- Skips `delete(delete_all=True)` if index is empty (fixes 404 on fresh index).
- **Status: 20 vectors indexed in `ml-rag` as of 2026-04-21.**

#### `agent/main_agent.py`
- Real RAG pipeline: BM25Okapi retrieval over `data/chunks.jsonl` + GPT-4o-mini generation.
- Constructor: `MainAgent(top_k=5, version="v1")` — V1 uses `top_k=3`, V2 uses `top_k=5`.
- Returns: `answer`, `contexts`, `retrieved_ids` (chunk IDs), `metadata.tokens_used`.
- Rate limiting: `asyncio.Semaphore(10)` + exponential backoff retry (5 attempts).
- Note: Agent uses OpenAI (not Claude) for generation to avoid Anthropic rate limits during benchmarking.

#### `engine/llm_judge.py`
- Two judge models: `claude-haiku-4-5-20251001` and `gpt-4o-mini`.
- Scoring rubric: 1–5 scale (1=wrong, 5=fully correct), single-integer response.
- Conflict resolution: diff ≤ 1 → average (agreement=1.0); diff=2 → average (agreement=0.5); diff ≥ 3 → Claude tiebreaker (agreement=0.0).
- `get_agreement_stats()`: returns Cohen's Kappa (fraction within 1 point) and exact agreement across all evaluated cases.
- `check_position_bias()`: swaps response A/B order and re-judges to detect positional preference.
- Rate limiting: `asyncio.Semaphore(2)` + retry on `rate_limit` errors.

#### `engine/retrieval_eval.py`
- `calculate_hit_rate(expected_ids, retrieved_ids, top_k=3)`: returns 1.0 if any expected ID in top-k retrieved.
- `calculate_mrr(expected_ids, retrieved_ids)`: 1/(position of first hit), 0 if none found.
- `evaluate_batch(dataset, agent_responses)`: computes avg hit_rate and avg MRR; skips adversarial cases (empty `expected_retrieval_ids`).

#### `engine/runner.py`
- `BenchmarkRunner(agent, evaluator, judge)` — evaluator is `RetrievalEvaluator`, judge is `LLMJudge`.
- `run_single_test()`: agent query → retrieval metrics → LLM judge (all async).
- Cost tracking: `tokens_used × rate_per_1k / 1000` per case, accumulated to `total_cost_usd`.
- Cost rates: gpt-4o-mini = $0.00015/1K, claude-haiku = $0.00025/1K.
- Inter-batch sleep: 3 seconds between batches to stay under Anthropic 50 req/min rate limit.

#### `main.py`
- Runs two benchmark passes: `Agent_V1_Base` (top_k=3) and `Agent_V2_Optimized` (top_k=5).
- Release gate checks (all must pass for APPROVE): `delta_score ≥ 0`, `hit_rate ≥ 0.50`, `agreement_rate ≥ 0.60`, `cost ≤ $5.00`.
- Writes `reports/summary.json` with full metrics, regression delta, and gate decision.
- Writes `reports/benchmark_results.json` with per-case details.

#### `analysis/failure_analysis.md`
- Benchmark numbers filled in from actual run (V1 vs V2 comparison table).
- 5-Whys analysis for 3 worst cases: BM25 vocabulary mismatch, context truncation, LaTeX/Unicode formula mismatch.
- Action plan: dense retrieval, smaller chunks, formula normalization, context budget increase.

---

## Benchmark Results (last run: 2026-04-21)

| Metric | V1 (top_k=3) | V2 (top_k=5) | Delta |
|--------|-------------|-------------|-------|
| avg_score | 4.05 / 5.0 | 4.06 / 5.0 | +0.014 |
| hit_rate | 74.3% | 74.3% | 0.0% |
| MRR | 0.648 | 0.657 | +0.009 |
| agreement_rate | 88.6% | 88.6% | — |
| pass_rate | 81.4% | 82.9% | +1.5% |
| total_cost | — | $0.024 | — |

Release gate: **APPROVED** (all 4 checks passed)

---

## Key Technical Decisions

- **BM25 over Pinecone for agent retrieval**: Pinecone is indexed (see `agent/pre.py`) but the benchmark agent uses in-memory BM25 to avoid latency from embedding API calls during the timed benchmark. To switch to Pinecone retrieval, update `MainAgent.query()` to call `PineconeVectorStore.similarity_search()`.
- **OpenAI for agent generation**: Using `gpt-4o-mini` instead of Claude avoids hitting the Anthropic 50 req/min org rate limit when running 70 × 2 benchmark passes (Claude is reserved for judging).
- **golden_set.jsonl format**: File was originally pretty-printed multi-line JSON; converted to single-line JSONL so `json.loads(line)` works correctly.
- **Pinecone index**: Use `ml-rag` (dim=3072, correct for `text-embedding-3-large`). The `day-14` index has dim=1024 and is unused. Set `PINECONE_INDEX_NAME=ml-rag` in `.env`.

---

## Grading Constraints

- Minimum 2 judge models — using only 1 caps score at 30/60.
- Minimum 50 test cases with `expected_retrieval_ids` for retrieval metrics (we have 60).
- Full pipeline should complete in < 2 minutes for 50 cases (async batching implemented; actual time ~5 min for 70 cases due to API rate limits).
- Individual reflection files: `analysis/reflections/reflection_[Name].md`.
- `.env` with API keys must **not** be committed.

---

## Environment Variables (`.env`)

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=ml-rag
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
```
