# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a team lab project (Lab Day 14) to build an **AI Evaluation Factory** — an automated benchmarking system that evaluates a RAG-based AI Agent with metrics, multi-judge consensus, and regression gating. The codebase is a template with placeholder implementations that students must replace with real logic.

## Commands

```bash
# Install dependencies (using uv, Python 3.12+)
uv sync
# or
pip install -r requirements.txt

# Step 1: Generate golden dataset (must run first)
python data/synthetic_gen.py

# Step 2: Run full benchmark and produce reports
python main.py

# Step 3: Validate submission format before handing in
python check_lab.py
```

## Architecture & Code Flow

The pipeline has four layers that execute in sequence during `main.py`:

```
data/raw/*.md  →  data/synthetic_gen.py  →  data/golden_set.jsonl  (70 cases)
                                         →  data/chunks.jsonl      (20 chunks, shared IDs for agent ingestion)
                                                  ↓
              main.py orchestrates two benchmark runs (V1 vs V2)
                                  ↓
           engine/runner.py  BenchmarkRunner.run_all()
             ├─ agent/main_agent.py  MainAgent.query()       ← RAG agent (mock)
             ├─ engine/retrieval_eval.py  RetrievalEvaluator ← RAGAS-style metrics
             └─ engine/llm_judge.py  LLMJudge                ← multi-model judge
                                  ↓
              reports/summary.json + reports/benchmark_results.json
                                  ↓
                          check_lab.py  validates output format
```

**`BenchmarkRunner.run_all()`** (`engine/runner.py`) batches test cases (default batch_size=5) and runs them concurrently with `asyncio.gather`. Each case goes through three async steps: agent query → RAGAS evaluation → multi-judge scoring.

**`main.py`** runs the benchmark twice (simulating Agent V1 vs V2), computes `delta` in `avg_score`, and writes `summary.json`. The release gate decision (`APPROVE` / `BLOCK RELEASE`) is simply `delta > 0` — students must extend this with quality/cost/performance thresholds.

## What Students Must Implement

Every file contains placeholder/mock logic marked with `TODO` comments or `# Giả lập` (simulated). The required implementations are:

### `data/synthetic_gen.py` — DONE
- Reads `data/raw/*.md`, splits by markdown headers into chunks, writes `data/chunks.jsonl` as shared chunk registry.
- Calls `claude-haiku-4-5-20251001` via `ANTHROPIC_API_KEY` to generate 3 QA pairs per chunk + 10 adversarial cases.
- **Output:** `data/golden_set.jsonl` — **70 records** (easy: 20, medium: 21, hard: 19, adversarial: 10) across types: factual/conceptual/application/out_of_context/hallucination_bait/prompt_injection/ambiguous.
- **20 chunks** across 2 source files (`ml_1.md`: chunks 1–12, `ml_2.md`: chunks 0–7), IDs like `ml_1_chunk_3`.
- Each record schema: `question`, `expected_answer`, `context`, `expected_retrieval_ids` (list of chunk IDs), `metadata.difficulty/type/source/chunk_id`.

### `agent/main_agent.py`
- Replace the mock `query()` with a real RAG pipeline (retrieval from a vector DB + LLM generation).
- Response must include `answer`, `contexts`, and retrieved document IDs (`retrieved_ids`) for retrieval eval.

### `engine/llm_judge.py` — `LLMJudge`
- Replace simulated scores with real calls to **at least 2 judge models** (e.g., GPT-4o + Claude).
- Implement conflict resolution when scores diverge by more than 1 point.
- Implement `check_position_bias()` by swapping response order and re-judging.
- Add Cohen's Kappa or Agreement Rate as a reliability metric.

### `engine/retrieval_eval.py` — `RetrievalEvaluator`
- `calculate_hit_rate()` and `calculate_mrr()` have correct logic already.
- `evaluate_batch()` must wire up to actual agent responses (use `retrieved_ids` from agent output vs `expected_retrieval_ids` from dataset).

### `engine/runner.py` — `BenchmarkRunner`
- Integrate `RetrievalEvaluator` into `run_single_test()` (currently it's wired to the placeholder `ExpertEvaluator` in `main.py`).
- Add cost tracking: accumulate `tokens_used` per case and compute total/per-case cost.

### `main.py`
- Replace `ExpertEvaluator` and `MultiModelJudge` stubs with real `RetrievalEvaluator` and `LLMJudge` instances.
- Extend the release gate beyond `delta > 0` to check hit_rate, agreement_rate, and cost thresholds.
- Persist regression comparison (`v1_summary` vs `v2_summary`) into `summary.json`.

### `analysis/failure_analysis.md`
- Fill in actual benchmark numbers and failure clusters after running.
- Complete the 5 Whys analysis for the 3 worst-performing cases.

## Required Output Schema

`reports/summary.json` must contain these fields for `check_lab.py` to pass:

```json
{
  "metadata": { "version": "...", "total": 50, "timestamp": "..." },
  "metrics": {
    "avg_score": 0.0,
    "hit_rate": 0.0,
    "agreement_rate": 0.0
  }
}
```

## Grading Constraints

- **Minimum 2 judge models** — using only 1 caps the group score at 30/60.
- **Minimum 50 test cases** with `expected_retrieval_ids` for retrieval metrics to count.
- Full pipeline must complete in **< 2 minutes** for 50 cases (requires async batching).
- Individual reflection files go in `analysis/reflections/reflection_[Name].md`.
- `.env` with API keys must **not** be committed.
