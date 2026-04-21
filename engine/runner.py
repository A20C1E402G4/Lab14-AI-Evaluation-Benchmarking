import asyncio
import time
from typing import List, Dict

from engine.retrieval_eval import RetrievalEvaluator

# Cost constants (USD per 1K tokens)
_COST_PER_1K = {
    "claude-haiku-4-5-20251001": 0.00025,
    "gpt-4o-mini": 0.00015,
    "gpt-4o": 0.005,
    "default": 0.00015,
}


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator  # LLMJudge
        self.judge = judge           # LLMJudge (same instance in practice)
        self.retrieval_evaluator = RetrievalEvaluator()
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        # 1. Agent query (RAG)
        response = await self.agent.query(test_case["question"])
        latency = round(time.perf_counter() - start_time, 3)

        # 2. Cost tracking
        meta = response.get("metadata", {})
        tokens_used = int(meta.get("tokens_used", 0))
        model = meta.get("model", "default")
        rate = _COST_PER_1K.get(model, _COST_PER_1K["default"])
        cost_usd = round(tokens_used * rate / 1000, 6)
        self.total_tokens += tokens_used
        self.total_cost_usd += cost_usd

        # 3. Retrieval metrics
        expected_ids = test_case.get("expected_retrieval_ids") or []
        retrieved_ids = response.get("retrieved_ids") or []
        hit_rate = self.retrieval_evaluator.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.retrieval_evaluator.calculate_mrr(expected_ids, retrieved_ids)

        # 4. Multi-model LLM judge
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
        )

        # Faithfulness proxy: 1.0 if relevant context was retrieved, 0.5 otherwise
        # Relevancy proxy: MRR (higher rank of relevant doc = more relevant context)
        faithfulness = round(0.5 + 0.5 * hit_rate, 2)
        relevancy = round(mrr, 2) if mrr > 0 else 0.0

        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "ragas": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "faithfulness": faithfulness,
                "relevancy": relevancy,
            },
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Run benchmark in async batches to respect rate limits."""
        results: List[Dict] = []
        total = len(dataset)
        n_batches = (total + batch_size - 1) // batch_size
        for i in range(0, total, batch_size):
            batch = dataset[i: i + batch_size]
            batch_num = i // batch_size + 1
            print(f"  Batch {batch_num}/{n_batches} ({len(batch)} cases)...")
            batch_results = await asyncio.gather(*[self.run_single_test(c) for c in batch])
            results.extend(batch_results)
            # Pause between batches to respect API rate limits (each case = 3 Claude calls)
            if batch_num < n_batches:
                await asyncio.sleep(3)
        return results

    def get_cost_summary(self) -> Dict:
        n = self.total_tokens
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_tokens_per_case": round(n / max(1, 1), 1),
        }
