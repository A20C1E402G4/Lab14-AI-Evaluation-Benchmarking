from typing import List, Dict


class RetrievalEvaluator:
    """RAGAS-style retrieval metrics: Hit Rate and MRR."""

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """Return 1.0 if at least one expected_id appears in the top_k retrieved docs."""
        top = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """Mean Reciprocal Rank — 1/(position of first relevant doc), 0 if none found."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(
        self,
        dataset: List[Dict],
        agent_responses: List[Dict] | None = None,
    ) -> Dict:
        """
        Compute avg hit_rate and avg MRR across the dataset.

        Args:
            dataset: list of test cases with 'expected_retrieval_ids'.
            agent_responses: list of agent outputs with 'retrieved_ids', aligned with dataset.
                             If None or misaligned, treats all retrieved_ids as empty.
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "evaluated": 0}

        hit_rates: List[float] = []
        mrr_scores: List[float] = []

        for i, case in enumerate(dataset):
            expected_ids = case.get("expected_retrieval_ids") or []
            if not expected_ids:
                # Adversarial / out-of-context cases — skip retrieval metric
                continue

            retrieved_ids: List[str] = []
            if agent_responses and i < len(agent_responses):
                retrieved_ids = agent_responses[i].get("retrieved_ids") or []

            hit_rates.append(self.calculate_hit_rate(expected_ids, retrieved_ids))
            mrr_scores.append(self.calculate_mrr(expected_ids, retrieved_ids))

        n = len(hit_rates)
        return {
            "avg_hit_rate": round(sum(hit_rates) / n, 4) if n else 0.0,
            "avg_mrr": round(sum(mrr_scores) / n, 4) if n else 0.0,
            "evaluated": n,
        }
