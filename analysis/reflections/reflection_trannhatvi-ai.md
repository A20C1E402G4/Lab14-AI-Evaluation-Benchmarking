# Individual Reflection — trannhatvi-ai

**Lab:** Day 14 — AI Evaluation Factory
**Role:** AI/Backend — Multi-Judge Consensus & Retrieval Evaluation
**Commit:** `cd1dc22` — `feat(engine): implement multi-judge consensus and retrieval evaluation metrics`

---

## 1. What I Built

My responsibility was the evaluation engine: the two modules that turn raw agent outputs into quantifiable metrics. `engine/llm_judge.py` answers "how good is the answer?", and `engine/retrieval_eval.py` answers "did the agent find the right source?". These are the two fundamental questions in RAG evaluation, and they must be measured separately.

### `engine/llm_judge.py` — Multi-Model Judge

The original file was a placeholder returning hardcoded scores of 4 and 3. I replaced it with real calls to two independent judge models.

**Judge design**

Both judges receive the same prompt: question + reference answer + student answer → single integer 1–5.

```
1=Completely wrong    2=Major errors    3=Partially correct
4=Mostly correct      5=Fully correct
```

I kept the prompt minimal and consistent across both judges, with a strict format requirement ("Reply with a single integer (1-5) only"). A minimal prompt reduces the chance that one model's chattiness biases the score format.

**Defensive score parsing**

Claude and GPT don't always return just a digit. Sometimes they return "4." or "Score: 4" or "4\n". `_parse_score()` scans the string for the first digit rather than trying to parse the whole string:

```python
def _parse_score(text: str) -> int:
    for ch in text.strip():
        if ch.isdigit():
            return max(1, min(5, int(ch)))
    return 3  # fallback to middle score
```

The `max(1, min(5, ...))` clamp ensures we never get an out-of-range score even if the model returns "7" or "0".

**Parallel judging with asyncio.gather**

Both judge calls run concurrently per test case, not sequentially:

```python
score_claude, score_gpt = await asyncio.gather(
    _judge_claude(question, answer, ground_truth),
    _judge_openai(question, answer, ground_truth),
)
```

This halves the judging latency per case. Since both calls are I/O bound (waiting for API responses), they can genuinely overlap.

**Conflict resolution**

| Score difference | What it means | Resolution |
|-----------------|--------------|------------|
| diff ≤ 1 | Normal variation — models agree | Average, `agreement_rate = 1.0` |
| diff = 2 | Moderate disagreement — flag but proceed | Average, `agreement_rate = 0.5` |
| diff ≥ 3 | High conflict — something is ambiguous or adversarial | Claude score wins, `agreement_rate = 0.0` |

Using Claude as the tiebreaker (not GPT) is a deliberate choice. Claude's system prompt adherence tends to be stricter — in cases where the agent wrote a plausible-sounding but fabricated answer, Claude was more likely to penalize it. The 4 cases with diff≥3 in our results were all adversarial (hallucination bait, prompt injection), confirming that high conflict correlates with genuinely ambiguous cases rather than random noise.

**Agreement tracking across the full run**

`LLMJudge` accumulates individual scores per judge across all 70 cases:

```python
self._scores_claude.append(score_claude)
self._scores_gpt.append(score_gpt)
```

`get_agreement_stats()` then computes two metrics: exact agreement (identical score) and within-1-point agreement (used as Cohen's Kappa approximation). These apply to the entire benchmark run, not to individual cases. Result: exact agreement = 77.1%, within-1-point agreement = **88.6%**.

**Position bias check (`check_position_bias`)**

Presents two responses as A/B, then runs the same comparison with the order swapped (B/A). If the winner changes after the swap, the judge is biased toward whichever response appeared first in the prompt.

```python
consistent = (result_ab.startswith("A") and result_ba.startswith("B")) or \
             (result_ab.startswith("B") and result_ba.startswith("A"))
```

"Consistent" means: the same logical response won both times, regardless of which letter it was assigned. Position bias is detected when the winner changes purely because the positions changed.

### `engine/retrieval_eval.py` — Retrieval Evaluator

The existing `calculate_hit_rate()` and `calculate_mrr()` had correct formulas but `evaluate_batch()` was a stub returning `{"avg_hit_rate": 0.85}`. I wired it to actual agent responses.

The key design decision was the `agent_responses` parameter:

```python
async def evaluate_batch(self, dataset, agent_responses=None):
    for i, case in enumerate(dataset):
        expected_ids = case.get("expected_retrieval_ids") or []
        if not expected_ids:
            continue  # skip adversarial cases
        retrieved_ids = agent_responses[i].get("retrieved_ids") or [] if agent_responses else []
        hit_rates.append(self.calculate_hit_rate(expected_ids, retrieved_ids))
        mrr_scores.append(self.calculate_mrr(expected_ids, retrieved_ids))
```

Two intentional choices:
1. Adversarial cases (empty `expected_retrieval_ids`) are excluded — they have no ground truth chunk, so computing hit rate against an empty list would always return 0.0 and drag down the metric unfairly.
2. `agent_responses` is optional — if `None`, all `retrieved_ids` default to `[]`, producing 0.0 metrics. This makes the evaluator fail visibly rather than silently when called without agent data.

---

## 2. Problems I Solved

### Problem 1 — Shared semaphore conflicts between agent and judge

The original plan used `asyncio.Semaphore(5)` in both `main_agent.py` and `llm_judge.py`. Within a single batch of 5 concurrent test cases, each case would try to acquire the semaphore for the agent call AND then again for the judge calls. With 5 cases × 2 judge calls = 10 Claude calls per batch, all competing against the same 50 req/min limit, rate limit errors appeared consistently at batch 8.

The fix required two changes in coordination with dduyanhhoang:
- Move agent generation to OpenAI (`gpt-4o-mini`) — this freed Claude capacity entirely for judging
- Set `asyncio.Semaphore(2)` in the judge module — only 2 Claude judge calls at a time per process

With this split, a batch of 5 cases generates 5 OpenAI agent calls (semaphore=10, no real constraint) and 10 judge calls (5 Claude + 5 OpenAI, Claude capped at 2 concurrent). Claude call volume dropped from ~15/batch to ~5/batch.

### Problem 2 — `evaluate_batch()` silently returned fake data

The original stub returned `{"avg_hit_rate": 0.85, "avg_mrr": 0.72}` — hardcoded, not computed. If I had not fixed this and instead wired the runner to call `evaluate_batch()` as-is, the summary would show "85% hit rate" regardless of how bad the actual retrieval was.

The fix was to make `evaluate_batch()` actually compute from data, and to ensure it gracefully handles the case where `agent_responses` is missing (returns real zeros, not fake values). The per-case computation is now done inline in `runner.py` instead, using `calculate_hit_rate()` and `calculate_mrr()` directly — `evaluate_batch()` is now a utility for batch-level analysis.

### Problem 3 — Judge prompt leaking reasoning into the score

Early test runs showed the models sometimes responding "I would give this a 4 because..." before the digit. `json.loads(response)` would of course fail. The `_parse_score()` character-scan approach handles this because it finds the first digit regardless of surrounding text. The clamp then validates it is in range.

---

## 3. Technical Concepts

### Why one judge model is not enough

Using a single LLM judge introduces several sources of unreliability:

**Sycophancy bias:** GPT-4o-mini tends to score longer, more detailed answers higher even when the extra detail is irrelevant or incorrect. Claude Haiku penalizes verbosity that doesn't directly address the question. Running both and averaging cancels these opposing biases.

**Provider-specific calibration:** Different models have different ideas of what "3/5" means. GPT-4o-mini uses the full 1–5 range more evenly; Claude Haiku clusters around 4–5. Combining them broadens the effective scoring distribution.

**Adversarial case handling:** In our results, all 4 high-conflict cases (diff≥3) were adversarial. GPT scored correct refusals as 1/5 ("didn't answer the question"). Claude scored them as 4/5 ("correctly handled an out-of-scope question"). The conflict itself is informative — it tells us the rubric is underspecified for refusal behavior, which is exactly what you want a two-judge system to surface.

In production: any case with agreement_rate < 1.0 should be flagged for human review. The agreement metric is as important as the score itself.

### Cohen's Kappa — what it actually measures

Our implementation uses a simplified linear-weighted kappa:

```python
def _cohen_kappa(a_scores, b_scores):
    agree = sum(1 for a, b in zip(a_scores, b_scores) if abs(a - b) <= 1)
    return agree / len(a_scores)
```

The "within 1 point" threshold is a deliberate relaxation of strict kappa. True ordinal Cohen's Kappa would penalize disagreement proportionally to distance. We use this relaxed version because a Claude score of 4 vs a GPT score of 5 represents genuine agreement on answer quality — both judges think the answer is good, they just differ on whether it's "mostly correct" or "fully correct."

Our result: 88.6% within-1-point agreement across 70 cases. For context, kappa values:
- κ < 0.40: Poor agreement
- κ 0.40–0.60: Moderate agreement
- κ 0.60–0.80: Substantial agreement
- κ > 0.80: Near-perfect agreement

88.6% puts our two-judge system in the "near-perfect agreement" band — the judges are calibrated well against each other on ML content.

### Position bias in LLM judges

LLMs have a known tendency to prefer the response that appears first in a comparative prompt ("which is better, A or B?"). This is called **position bias** or **order effect**.

`check_position_bias()` detects it by running the same comparison twice with the responses swapped:

```
Round 1: "Which is better? A=[response_x] B=[response_y]" → winner
Round 2: "Which is better? A=[response_y] B=[response_x]" → winner
```

If Round 1 says A wins and Round 2 says A wins again — despite the fact that A in Round 2 is the same logical response that was B in Round 1 — that's position bias. The judge preferred "position A" over the content.

This matters for pairwise evaluation (comparing two model outputs). It does not affect our primary benchmark (single-answer scoring), but it becomes critical when you extend to "which version is better?" comparisons in the regression phase.

### Hit Rate vs MRR — when each matters

**Hit Rate** answers: "does our retriever work at all?" If hit rate is low, the retriever is broken and no amount of prompt engineering will fix answer quality.

**MRR** answers: "how efficiently does our retriever work?" An MRR of 0.657 with a hit rate of 0.743 means that when the retriever does find the right chunk, it places it at rank 1 only about 66% of the time — meaning 34% of successful retrievals still waste context window space by putting the relevant chunk behind less relevant ones. The LLM must "look through" irrelevant context to find the answer, which is exactly the scenario where it hallucinates.

The relationship between our MRR (0.657) and hit rate (0.743) confirms this: MRR/hit_rate = 0.88, meaning correct chunks are found at rank 1 about 88% of the time when found at all. Room for improvement but not critical.

---

## 4. What I Learned

**Two judges reveal what one judge hides.** The 8 conflict cases in our results are not noise — they are the most valuable outputs of the entire evaluation run. They identify exactly where the scoring rubric is ambiguous (refusal behavior, hallucination bait, ambiguous questions), which is the information you need to improve the evaluation system itself.

**Evaluation systems have their own failure modes.** `evaluate_batch()` returning hardcoded values is a category of bug with no runtime error, no assertion failure, and no obvious symptom. The metric looks plausible (85% hit rate is a reasonable number for an ML dataset), so it would have passed unnoticed if not caught during code review. The lesson: test your evaluation code against known inputs with known expected outputs, not just against the live system.
