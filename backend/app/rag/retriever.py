"""
Hybrid Retrieval & Fusion Engine (RRF, Dense, & BM25)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.rag.models import ChunkType, FSSAIChunk, HybridSearchResult
from app.rag.vector_store import BM25FSSAIIndex, ChromaFSSAIStore


class RRFCombiner:
    """Reciprocal Rank Fusion (Cormack et al., 2009)."""

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        dense: List[Tuple[FSSAIChunk, float]],
        sparse: List[Tuple[FSSAIChunk, float]],
        query: str = "",
    ) -> List[HybridSearchResult]:
        scores: Dict[str, Dict[str, Any]] = {}
        
        # Extract domain-specific query entities (excluding generic regulatory words)
        GENERIC_TERMS = {
            "what", "are", "the", "in", "of", "for", "and", "to", "is", "a", "an", "various", "any", "does", "which",
            "maximum", "minimum", "permitted", "limit", "limits", "tolerance", "prescribed", "standard", "standards",
            "food", "foods", "product", "products", "article", "articles", "specified", "regulation", "regulations",
            "clause", "section", "requirement", "requirements", "shall", "under", "per", "level", "levels", "much", "how"
        }
        q_lower = query.lower()
        all_words = re.findall(r"[a-z0-9]+", q_lower)
        specific_terms = [w for w in all_words if w not in GENERIC_TERMS and len(w) > 2]

        for rank, (chunk, dist) in enumerate(dense, start=1):
            cid = chunk.chunk_id
            scores.setdefault(cid, {
                "chunk": chunk, "dense_rank": None,
                "sparse_rank": None, "dense_dist": None,
            })
            scores[cid]["dense_rank"] = rank
            scores[cid]["dense_dist"] = dist

        for rank, (chunk, _) in enumerate(sparse, start=1):
            cid = chunk.chunk_id
            scores.setdefault(cid, {
                "chunk": chunk, "dense_rank": None,
                "sparse_rank": None, "dense_dist": None,
            })
            scores[cid]["sparse_rank"] = rank

        def rrf_score(e: Dict[str, Any]) -> float:
            s = 0.0
            if e["dense_rank"] is not None:
                s += 1.0 / (self.k + e["dense_rank"])
            if e["sparse_rank"] is not None:
                s += 1.0 / (self.k + e["sparse_rank"])
            
            # Domain entity relevance bonus: prioritize chunks matching target chemical/food/substance
            if specific_terms:
                c_text = (e["chunk"].text + " " + (e["chunk"].embedding_text or "")).lower()
                matches = sum(1 for kw in specific_terms if kw in c_text)
                if matches > 0:
                    s += 0.020 * matches
            return s

        ranked = sorted(scores.values(), key=rrf_score, reverse=True)
        return [
            HybridSearchResult(
                chunk=e["chunk"],
                dense_rank=e["dense_rank"],
                sparse_rank=e["sparse_rank"],
                rrf_score=rrf_score(e),
                dense_distance=e["dense_dist"],
            )
            for e in ranked
        ]


class FSSAIRetriever:
    """Main hybrid retrieval engine for NiyamAI."""

    def __init__(
        self,
        vector_store: ChromaFSSAIStore,
        bm25_index: BM25FSSAIIndex,
        rrf_k: int = 60,
    ) -> None:
        self._vs = vector_store
        self._bm25 = bm25_index
        self._rrf = RRFCombiner(k=rrf_k)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        regulation_id: Optional[str] = None,
        category: Optional[str] = None,
        food_category: Optional[str] = None,
        active_only: bool = True,
        chunk_type: Optional[ChunkType] = None,
        is_binding: Optional[bool] = None,
    ) -> List[HybridSearchResult]:
        where = self._build_where(
            regulation_id, category, food_category,
            active_only, chunk_type, is_binding,
        )

        candidate_k = top_k * 3
        dense = self._vs.query(query, top_k=candidate_k, where=where or None)

        sparse_raw = self._bm25.query(query, top_k=candidate_k)
        sparse = [
            (c, s) for c, s in sparse_raw
            if self._passes_filter(
                c, regulation_id, category, food_category,
                active_only, chunk_type, is_binding,
            )
        ]

        fused = self._rrf.fuse(dense, sparse, query=query)
        return fused[:top_k]

    @staticmethod
    def _build_where(
        regulation_id: Optional[str],
        category: Optional[str],
        food_category: Optional[str],
        active_only: bool,
        chunk_type: Optional[ChunkType],
        is_binding: Optional[bool],
    ) -> Optional[Dict[str, Any]]:
        conditions: List[Dict[str, Any]] = []
        if regulation_id:
            conditions.append({"regulation_id": {"$eq": regulation_id}})
        if category:
            conditions.append({"category": {"$eq": category}})
        if food_category:
            conditions.append({"food_category": {"$eq": food_category}})
        if active_only:
            conditions.append({"is_active": {"$eq": True}})
        if chunk_type:
            conditions.append({"chunk_type": {"$eq": chunk_type.value}})
        if is_binding is not None:
            conditions.append({"is_binding": {"$eq": is_binding}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _passes_filter(
        chunk: FSSAIChunk,
        regulation_id: Optional[str],
        category: Optional[str],
        food_category: Optional[str],
        active_only: bool,
        chunk_type: Optional[ChunkType],
        is_binding: Optional[bool],
    ) -> bool:
        m = chunk.metadata
        if regulation_id and m.get("regulation_id") != regulation_id:
            return False
        if category and m.get("category") != category:
            return False
        if food_category and m.get("food_category") != food_category:
            return False
        if active_only and not m.get("is_active", True):
            return False
        if chunk_type and m.get("chunk_type") != chunk_type.value:
            return False
        if is_binding is not None and m.get("is_binding") != is_binding:
            return False
        return True


class BaselineRetriever:
    """Dense-only baseline retriever for evaluation baseline comparison."""

    def __init__(self, vector_store: ChromaFSSAIStore) -> None:
        self._vs = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[HybridSearchResult]:
        dense = self._vs.query(query, top_k=top_k)
        return [
            HybridSearchResult(
                chunk=c,
                dense_rank=idx + 1,
                sparse_rank=None,
                rrf_score=1.0 / (60 + idx + 1),
                dense_distance=dist,
            )
            for idx, (c, dist) in enumerate(dense)
        ]
