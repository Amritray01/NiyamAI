from __future__ import annotations

from typing import Any, Dict, List


class RetrievalEvaluator:
    @staticmethod
    def calculate_hit_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
        if not expected_ids:
            return 1.0
        top_k = set(retrieved_ids[:k])
        return 1.0 if any(eid in top_k for eid in expected_ids) else 0.0

    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
        if not expected_ids:
            return 1.0
        top_k = set(retrieved_ids[:k])
        hits = sum(1 for eid in expected_ids if eid in top_k)
        return hits / len(expected_ids)

    @staticmethod
    def calculate_mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
        if not expected_ids:
            return 1.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in expected_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_fact_coverage(retrieved_texts: List[str], required_facts: List[str]) -> float:
        if not required_facts:
            return 1.0
        combined = " ".join(retrieved_texts).lower()
        matched = sum(1 for fact in required_facts if fact.lower() in combined)
        return matched / len(required_facts)

    def evaluate_query(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        expected_chunk_ids: List[str],
        required_facts: List[str],
        k_values: List[int] = [1, 3, 5],
    ) -> Dict[str, float]:
        retrieved_ids = [c.get("chunk_id", "") for c in retrieved_chunks]
        retrieved_texts = [c.get("text", "") for c in retrieved_chunks]

        metrics: Dict[str, float] = {}
        for k in k_values:
            metrics[f"hit@{k}"] = self.calculate_hit_at_k(retrieved_ids, expected_chunk_ids, k)
            metrics[f"recall@{k}"] = self.calculate_recall_at_k(retrieved_ids, expected_chunk_ids, k)

        metrics["mrr"] = self.calculate_mrr(retrieved_ids, expected_chunk_ids)
        metrics["fact_coverage"] = self.calculate_fact_coverage(retrieved_texts, required_facts)
        return metrics
