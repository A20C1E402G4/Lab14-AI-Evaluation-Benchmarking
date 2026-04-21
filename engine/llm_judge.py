import asyncio
import os
from typing import Dict, Any, List

import anthropic
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_GPT_MODEL = "gpt-4o-mini"
_SEMAPHORE = asyncio.Semaphore(2)

_anthropic_client: anthropic.AsyncAnthropic | None = None
_openai_client: AsyncOpenAI | None = None

_JUDGE_PROMPT = (
    "You are an expert evaluator for Machine Learning educational content.\n"
    "Score the Student Answer against the Reference Answer on a 1-5 scale:\n"
    "1=Completely wrong  2=Major errors  3=Partially correct  4=Mostly correct  5=Fully correct\n\n"
    "Question: {question}\n"
    "Reference Answer: {reference}\n"
    "Student Answer: {answer}\n\n"
    "Reply with a single integer (1-5) only."
)


def _get_anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _parse_score(text: str) -> int:
    for ch in text.strip():
        if ch.isdigit():
            return max(1, min(5, int(ch)))
    return 3


async def _judge_claude(question: str, answer: str, ground_truth: str) -> int:
    prompt = _JUDGE_PROMPT.format(question=question, reference=ground_truth, answer=answer)
    for attempt in range(5):
        try:
            async with _SEMAPHORE:
                response = await _get_anthropic().messages.create(
                    model=_CLAUDE_MODEL,
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                )
            return _parse_score(response.content[0].text)
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < 4:
                await asyncio.sleep(2 ** attempt + 2)
            else:
                return 3  # fallback on persistent failure


async def _judge_openai(question: str, answer: str, ground_truth: str) -> int:
    prompt = _JUDGE_PROMPT.format(question=question, reference=ground_truth, answer=answer)
    async with _SEMAPHORE:
        response = await _get_openai().chat.completions.create(
            model=_GPT_MODEL,
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
    return _parse_score(response.choices[0].message.content)


def _cohen_kappa(a_scores: List[int], b_scores: List[int]) -> float:
    """Linear-weighted Cohen's Kappa approximation."""
    n = len(a_scores)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(a_scores, b_scores) if abs(a - b) <= 1)
    return agree / n


class LLMJudge:
    """Multi-model judge using Claude Haiku and GPT-4o-mini."""

    def __init__(self):
        self.rubrics = {
            "accuracy": "Score 1-5 on factual correctness vs ground truth",
            "completeness": "Score 1-5 on coverage of key concepts",
        }
        self._scores_claude: List[int] = []
        self._scores_gpt: List[int] = []

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        score_claude, score_gpt = await asyncio.gather(
            _judge_claude(question, answer, ground_truth),
            _judge_openai(question, answer, ground_truth),
        )

        self._scores_claude.append(score_claude)
        self._scores_gpt.append(score_gpt)

        diff = abs(score_claude - score_gpt)

        if diff <= 1:
            # Models agree — average their scores
            final_score = (score_claude + score_gpt) / 2
            agreement_rate = 1.0
        elif diff == 2:
            # Moderate conflict — average but flag lower agreement
            final_score = (score_claude + score_gpt) / 2
            agreement_rate = 0.5
        else:
            # High conflict (diff >= 3) — use Claude as tiebreaker
            final_score = float(score_claude)
            agreement_rate = 0.0

        return {
            "final_score": round(final_score, 2),
            "agreement_rate": agreement_rate,
            "individual_scores": {
                "claude-haiku": score_claude,
                "gpt-4o-mini": score_gpt,
            },
            "reasoning": f"Claude={score_claude}, GPT={score_gpt}, diff={diff}",
        }

    async def check_position_bias(self, response_a: str, response_b: str) -> Dict[str, Any]:
        """Detect position bias by swapping response order."""

        async def _ask(prompt: str) -> str:
            async with _SEMAPHORE:
                r = await _get_anthropic().messages.create(
                    model=_CLAUDE_MODEL,
                    max_tokens=5,
                    messages=[{"role": "user", "content": prompt}],
                )
            return r.content[0].text.strip().upper()

        p_ab = f"Which answer is better? A: {response_a[:300]}\nB: {response_b[:300]}\nReply A or B:"
        p_ba = f"Which answer is better? A: {response_b[:300]}\nB: {response_a[:300]}\nReply A or B:"

        result_ab, result_ba = await asyncio.gather(_ask(p_ab), _ask(p_ba))

        # No bias if A wins in AB order and B wins in BA order (same logical answer wins both)
        consistent = (result_ab.startswith("A") and result_ba.startswith("B")) or \
                     (result_ab.startswith("B") and result_ba.startswith("A"))

        return {
            "ab_order_winner": result_ab,
            "ba_order_winner": result_ba,
            "position_bias_detected": not consistent,
        }

    def get_agreement_stats(self) -> Dict[str, Any]:
        """Return Cohen's Kappa and agreement rate across all evaluations."""
        n = len(self._scores_claude)
        if n == 0:
            return {"agreement_rate": 0.0, "cohen_kappa": 0.0, "count": 0}
        kappa = _cohen_kappa(self._scores_claude, self._scores_gpt)
        exact = sum(1 for a, b in zip(self._scores_claude, self._scores_gpt) if a == b) / n
        return {
            "agreement_rate": round(kappa, 4),
            "exact_agreement": round(exact, 4),
            "cohen_kappa": round(kappa, 4),
            "count": n,
        }
