from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.rag.retriever import FlashRankFSSAIReranker, FSSAIRetriever, RRFCombiner
from app.rag.vector_store import BM25FSSAIIndex
from evaluation.retrieval_eval import RetrievalEvaluator


class StagedRAGEvaluator:
    def __init__(
        self,
        retriever: FSSAIRetriever,
        vector_store: Any,
        bm25_index: BM25FSSAIIndex,
    ) -> None:
        self.retriever = retriever
        self.vs = vector_store
        self.bm25 = bm25_index
        self.evaluator = RetrievalEvaluator()
        self.rrf = RRFCombiner(k=60)
        self.reranker = FlashRankFSSAIReranker()

    def inspect_query_steps(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        candidate_k = top_k * 3

        dense = self.vs.query(query, top_k=candidate_k)
        sparse = self.bm25.query(query, top_k=candidate_k)
        fused = self.rrf.fuse(dense, sparse, query=query)
        reranked = self.reranker.rerank(query, fused[:15], top_k=top_k)

        return {
            "query": query,
            "step_1_dense": [{"chunk_id": c.chunk_id, "dense_dist": round(d, 4), "rank": i + 1} for i, (c, d) in enumerate(dense[:5])],
            "step_2_sparse_bm25": [{"chunk_id": c.chunk_id, "bm25_score": round(s, 4), "rank": i + 1} for i, (c, s) in enumerate(sparse[:5])],
            "step_3_rrf_fusion": [{"chunk_id": r.chunk.chunk_id, "rrf_score": round(r.rrf_score, 4)} for r in fused[:5]],
            "step_4_rerank": [{"chunk_id": r.chunk.chunk_id, "rerank_score": round(r.rerank_score or 0.0, 4)} for r in reranked],
        }

    def evaluate_stages(
        self,
        query: str,
        expected_chunk_ids: List[str],
        required_facts: List[str],
        category_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        where_filter = {"category": {"$eq": category_filter}} if category_filter and category_filter != "All" else None

        stage_queries = [
            ("stage_1_bm25_only", [{"chunk_id": c.chunk_id, "text": c.text} for c, _ in self.bm25.query(query, top_k=top_k)]),
            ("stage_2_semantic_only", [{"chunk_id": c.chunk_id, "text": c.text} for c, _ in self.vs.query(query, top_k=top_k)]),
            ("stage_3_after_filter", [{"chunk_id": c.chunk_id, "text": c.text} for c, _ in self.vs.query(query, top_k=top_k, where=where_filter)]),
            ("stage_4_hybrid_rerank", [
                {"chunk_id": r.chunk.chunk_id, "text": r.chunk.text}
                for r in self.retriever.retrieve(query=query, top_k=top_k, category=category_filter if category_filter != "All" else None, active_only=True, rerank=True)
            ]),
        ]

        return {
            stage_name: self.evaluator.evaluate_query(chunks, expected_chunk_ids, required_facts, k_values=[1, 3, 5])
            for stage_name, chunks in stage_queries
        }
