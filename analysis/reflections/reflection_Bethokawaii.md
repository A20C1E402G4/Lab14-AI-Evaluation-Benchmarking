# Individual Reflection — Bethokawaii

**Lab:** Day 14 — AI Evaluation Factory
**Role:** DevOps/Analyst — Async Runner, Cost Tracking & Regression Release Gate
**Commit:** `8c419cc` — `feat(pipeline): async benchmark runner with cost tracking and regression release gate`

---

## 1. What I Built

My responsibility was the orchestration layer: the runner that drives every test case through the pipeline, the cost accounting system, and the release gate logic that turns benchmark numbers into a binary deployment decision. These components touch every other module — they are the glue that makes the evaluation factory actually run end to end.

### `engine/runner.py` — BenchmarkRunner

The original `BenchmarkRunner` had a working async skeleton but delegated evaluation entirely to an external `ExpertEvaluator` stub that returned hardcoded numbers. I replaced the stub dependency with inline `RetrievalEvaluator` usage, added cost tracking, and completed the per-case result schema.

**RetrievalEvaluator integration**

Instead of calling `self.evaluator.score(test_case, response)` (which required a compatible external object), I instantiate a `RetrievalEvaluator` directly inside the runner:

```python
self.retrieval_evaluator = RetrievalEvaluator()
```

Per-case retrieval metrics are computed immediately after the agent response arrives, before the judge call, since they only require the response's `retrieved_ids` and the test case's `expected_retrieval_ids`:

```python
expected_ids = test_case.get("expected_retrieval_ids") or []
retrieved_ids = response.get("retrieved_ids") or []
hit_rate = self.retrieval_evaluator.calculate_hit_rate(expected_ids, retrieved_ids)
mrr = self.retrieval_evaluator.calculate_mrr(expected_ids, retrieved_ids)
```

The `or []` fallback handles both missing keys and explicit `None` values without crashing.

**Cost tracking**

Every agent response carries `metadata.tokens_used` and `metadata.model`. The runner looks up the model's cost rate and computes per-case cost:

```python
_COST_PER_1K = {
    "gpt-4o-mini":              0.00015,
    "claude-haiku-4-5-20251001": 0.00025,
    "gpt-4o":                    0.005,
    "default":                   0.00015,
}

rate = _COST_PER_1K.get(model, _COST_PER_1K["default"])
cost_usd = round(tokens_used * rate / 1000, 6)
self.total_tokens += tokens_used
self.total_cost_usd += cost_usd
```

`get_cost_summary()` returns the totals at the end of a run. For the full 70-case V2 benchmark: **$0.024 total**, **157,275 tokens**, **$0.00034 per case**.

**Async batching with inter-batch pacing**

```python
for i in range(0, total, batch_size):
    batch = dataset[i: i + batch_size]
    batch_results = await asyncio.gather(*[self.run_single_test(c) for c in batch])
    results.extend(batch_results)
    if batch_num < n_batches:
        await asyncio.sleep(3)
```

`asyncio.gather` runs all 5 cases in a batch truly in parallel (limited only by semaphores). The 3-second sleep between batches is a pacing mechanism — not waiting for anything to complete, but smoothing the request rate to stay under Anthropic's 50 req/min org limit. Without the sleep, 14 batches × 5 concurrent requests × 2 judge calls = up to 140 simultaneous Claude calls in bursts.

**Per-case result schema**

```python
return {
    "test_case": ...,
    "agent_response": ...,
    "latency": ...,       # perf_counter delta in seconds
    "tokens_used": ...,   # from agent metadata
    "cost_usd": ...,      # computed from tokens × rate
    "ragas": {
        "retrieval": {"hit_rate": ..., "mrr": ...}
    },
    "judge": {
        "final_score": ...,
        "agreement_rate": ...,
        "individual_scores": {...},
        "reasoning": ...
    },
    "status": "pass" | "fail"   # threshold: final_score >= 3
}
```

The `status` field binarizes the score for quick filtering. The threshold of 3/5 maps to "partially correct or better" — answers that are completely wrong (1) or have major errors (2) are flagged as failures for investigation.

### `main.py` — Regression Comparison & Release Gate

**V1 vs V2 setup**

The two benchmark passes test a specific hypothesis: does retrieving more context (top_k=5) produce better answers than retrieving less (top_k=3)?

```python
v1_agent = MainAgent(top_k=3, version="v1")
v2_agent = MainAgent(top_k=5, version="v2")
```

Everything else is identical: same golden dataset, same judge instance, same batch size. This isolation ensures any metric difference is caused by the top_k change, not by random variation in other parameters.

**Release gate with 4 independent checks**

```python
gate_checks = {
    "score_improved":  delta_score >= 0,
    "hit_rate_ok":     v2_summary["metrics"]["hit_rate"] >= _MIN_HIT_RATE,       # 0.50
    "agreement_ok":    v2_summary["metrics"]["agreement_rate"] >= _MIN_AGREEMENT_RATE,  # 0.60
    "cost_ok":         v2_summary["cost"]["total_cost_usd"] <= _MAX_COST_USD,    # $5.00
}
approved = all(gate_checks.values())
```

Using `all()` rather than a weighted sum is an intentional design choice — any single failed gate blocks the release. A new version that passes cost and quality thresholds but regresses on hit rate should not be released, even if the average score went up slightly. Each gate guards a different dimension of system health:

| Gate | What it protects |
|------|-----------------|
| `score_improved` | Answer quality does not regress |
| `hit_rate_ok` | Retrieval is functional (≥50% is the minimum bar for the system to be useful) |
| `agreement_ok` | Judge reliability is sufficient to trust the scores |
| `cost_ok` | Evaluation cost is within operational budget |

**Persisting the full regression record**

`summary.json` stores both the V1 and V2 metrics alongside the delta and gate decision, not just V2:

```json
"regression": {
    "v1_metrics": {...},
    "v2_metrics": {...},
    "delta_score": 0.0143,
    "delta_hit_rate": 0.0
},
"gate": {
    "decision": "APPROVE",
    "checks": {"score_improved": true, "hit_rate_ok": true, "agreement_ok": true, "cost_ok": true}
}
```

This makes the report self-contained — a reviewer reading `summary.json` weeks later can reconstruct the full regression comparison without re-running the benchmark.

---

## 2. Problems I Solved

### Problem 1 — Stub evaluator interface prevented real metric wiring

`BenchmarkRunner` accepted an `evaluator` parameter and called `self.evaluator.score(test_case, response)`. The `ExpertEvaluator` stub in `main.py` implemented this method by returning a hardcoded dict. If I had kept this architecture, wiring in `RetrievalEvaluator` would have required implementing `.score()` on it, which would have been a wrapper with a different interface than what the evaluator actually provides.

I took the simpler path: instantiate `RetrievalEvaluator` directly inside `BenchmarkRunner.__init__` and call its methods inline. The `evaluator` parameter in the constructor is preserved for interface compatibility but is no longer the primary evaluation mechanism.

### Problem 2 — Rate limit errors occurred mid-run, producing partial results

The first benchmark attempt crashed at batch 8/14 with a 429 rate limit error. All results up to that point were discarded because `run_all()` raised an exception before any results were written.

The fix was two-part:
1. The 3-second inter-batch sleep prevents the burst that triggered the limit
2. The retry logic in the judge and agent modules catches 429s and backs off, so a transient rate limit hit doesn't abort the entire run — it just delays that specific API call

Together these make the runner resilient: the sleep prevents hitting the limit in the first place, and the retry handles it if it does happen anyway.

### Problem 3 — `summary.json` schema didn't match `check_lab.py` requirements

`check_lab.py` validates the presence of `metrics.hit_rate` and `metrics.agreement_rate` at the top level of the `metrics` object. My initial version nested these under `metrics.retrieval.hit_rate` and `metrics.judge.agreement_rate` — a more organized structure but one that failed the validator.

Fix: flatten the metrics dict to put `hit_rate`, `agreement_rate`, `avg_score`, and `mrr` all at the same level, matching the expected schema:

```json
"metrics": {
    "avg_score": 4.0643,
    "hit_rate": 0.7429,
    "mrr": 0.6569,
    "agreement_rate": 0.8857
}
```

---

## 3. Technical Concepts

### asyncio.gather vs sequential execution

`asyncio.gather` schedules all coroutines to run concurrently in a single event loop. For I/O-bound work (API calls), this means the event loop sends all 5 requests in a batch nearly simultaneously and waits for them all to complete — rather than waiting for each one before sending the next.

Timing comparison for 5 test cases (avg 2.11s/case):
- Sequential: 5 × 2.11s = **10.55s**
- `asyncio.gather` (concurrent): max(2.11s) + overhead ≈ **2.2s**

For 70 cases in 14 batches: sequential would take ~148s just for agent calls alone. With concurrent batching, 70 cases complete in ~5 minutes total (including rate-limit sleeps, judge calls, and retry overhead).

This is not parallelism — there is still one thread. Python's event loop suspends a coroutine at every `await` and switches to another ready coroutine. For CPU-bound work (matrix math), this would not help. For API calls where the Python process is idle while waiting for a network response, it provides near-linear throughput improvement up to the concurrency limit.

### Release gate design principles

A release gate is a set of automated checks that must all pass before a new version is deployed. The design choices that matter:

**Absolute thresholds vs delta thresholds**

`score_improved` checks `delta >= 0` (delta threshold). `hit_rate_ok` checks `hit_rate >= 0.50` (absolute threshold). Both types are needed:

- Delta thresholds detect regression between versions. They don't catch a system that was already bad before.
- Absolute thresholds enforce minimum quality floors. A system that improves from 20% hit rate to 30% passes a delta check but is still broken.

Our gate uses both: `delta_score >= 0` (regression protection) + `hit_rate >= 0.50` + `agreement >= 0.60` (quality floors) + `cost <= $5.00` (operational constraint).

**Why all gates must pass, not a weighted average**

A weighted score of 90/100 sounds good. But if it is composed of 100/100 on cost and 80/100 on hit rate, the cost efficiency is masking a retrieval problem. In production, a 20% retrieval failure rate means 1 in 5 users gets a bad answer. Weighting cannot hide that. Using `all()` ensures each dimension is independently healthy.

**Thresholds should be calibrated from baselines, not arbitrary**

The thresholds I set (`hit_rate >= 0.50`, `agreement >= 0.60`) were chosen conservatively for this lab. In a real system, you would run the baseline version, record its metrics, and set the gate at `baseline - allowed_regression_margin`. A hit rate of 70% on V1 would set the V2 gate at `>= 67%` (allowing 3% regression), not a fixed 50%.

### Cost vs quality trade-off

The full two-run benchmark cost $0.048 (V1 + V2 combined). At $0.00034/case for the agent and approximately $0.00020/case for judging, the judging is actually cheaper than the agent call per case — because `gpt-4o-mini` at `max_tokens=512` for generation uses more tokens than `max_tokens=10` for judging.

To cut 30% evaluation cost without reducing accuracy:
1. **OpenAI Batch API** — submit all judge calls as a batch job (50% price reduction, ~24h turnaround for non-realtime eval)
2. **Skip judge for retrieval misses** — if `hit_rate=0.0`, the agent's answer is almost certainly wrong (our data: avg score 3.17 on misses vs 4.38 on hits). A heuristic score of 1.5 for retrieval misses would save 25% of judge calls with minimal accuracy loss
3. **One judge pass on easy cases** — cases where the expected answer is short and factual (easy difficulty) could be evaluated with a single judge; reserve two-judge consensus for medium/hard cases

---

## 4. What I Learned

**The runner is where all the assumptions meet.** Every interface contract between modules — what the agent returns, what the evaluator expects, what the judge consumes — is enforced (or violated) in `run_single_test()`. When a module returned data in an unexpected format or with an unexpected key name, the runner was the first place it broke. Writing the runner forced me to think about every integration point explicitly.

**Automation and reliability are different goals.** Making the benchmark run automatically (via `asyncio.gather` and batch scheduling) was straightforward. Making it run reliably — resuming correctly after a rate limit error, never silently producing wrong results, writing output atomically — required substantially more thought. The difference is the gap between a demo and a production system.
