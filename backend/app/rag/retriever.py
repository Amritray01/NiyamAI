from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from app.rag.models import ChunkType, FSSAIChunk, HybridSearchResult
from app.rag.vector_store import BM25FSSAIIndex, ChromaFSSAIStore, QdrantFSSAIStore


class FlashRankFSSAIReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._ranker = None
        try:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=model_name)
        except Exception:
            self._ranker = None

    def rerank(self, query: str, candidates: List[HybridSearchResult], top_k: int = 5) -> List[HybridSearchResult]:
        if not candidates:
            return []

        if self._ranker is not None:
            try:
                from flashrank import RerankRequest
                passages = [
                    {"id": i, "text": f"{c.chunk.text} {c.chunk.embedding_text or ''}"}
                    for i, c in enumerate(candidates)
                ]
                results = self._ranker.rerank(RerankRequest(query=query, passages=passages))
                return [
                    HybridSearchResult(
                        chunk=candidates[r["id"]].chunk,
                        dense_rank=candidates[r["id"]].dense_rank,
                        sparse_rank=candidates[r["id"]].sparse_rank,
                        rrf_score=candidates[r["id"]].rrf_score,
                        rerank_score=float(r.get("score", 0.0)),
                        dense_distance=candidates[r["id"]].dense_distance,
                    )
                    for r in results[:top_k]
                ]
            except Exception:
                pass

        return candidates[:top_k]


class RRFCombiner:
    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        dense: List[Tuple[FSSAIChunk, float]],
        sparse: List[Tuple[FSSAIChunk, float]],
        query: str = "",
    ) -> List[HybridSearchResult]:
        scores: Dict[str, Dict[str, Any]] = {}
        all_words = re.findall(r"[a-z0-9]+", query.lower())
        specific_terms = [w for w in all_words if w not in BM25FSSAIIndex.STOPWORDS and len(w) > 2]

        for rank, (chunk, dist) in enumerate(dense, start=1):
            cid = chunk.chunk_id
            scores.setdefault(cid, {"chunk": chunk, "dense_rank": None, "sparse_rank": None, "dense_dist": None})
            scores[cid]["dense_rank"] = rank
            scores[cid]["dense_dist"] = dist

        for rank, (chunk, _) in enumerate(sparse, start=1):
            cid = chunk.chunk_id
            scores.setdefault(cid, {"chunk": chunk, "dense_rank": None, "sparse_rank": None, "dense_dist": None})
            scores[cid]["sparse_rank"] = rank

        def rrf_score(e: Dict[str, Any]) -> float:
            s = 0.0
            if e["dense_rank"] is not None:
                s += 1.0 / (self.k + e["dense_rank"])
            if e["sparse_rank"] is not None:
                s += 1.0 / (self.k + e["sparse_rank"])
            if specific_terms:
                c_text = f"{e['chunk'].text} {e['chunk'].embedding_text or ''}".lower()
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
    def __init__(
        self,
        vector_store: Union[QdrantFSSAIStore, ChromaFSSAIStore],
        bm25_index: BM25FSSAIIndex,
        rrf_k: int = 60,
    ) -> None:
        self._vs = vector_store
        self._bm25 = bm25_index
        self._rrf = RRFCombiner(k=rrf_k)
        self._reranker = FlashRankFSSAIReranker()

    def retrieve(
        self,
        query: str,
        top_k: int = 7,
        *,
        regulation_id: Optional[str] = None,
        category: Optional[str] = None,
        food_category: Optional[str] = None,
        active_only: bool = True,
        chunk_type: Optional[ChunkType] = None,
        is_binding: Optional[bool] = None,
        rerank: bool = True,
    ) -> List[HybridSearchResult]:
        where = self._build_where(
            regulation_id, category, food_category,
            active_only, chunk_type, is_binding,
        )

        # Top 20 from both BM25 and Semantic (Dense) rank
        candidate_k = 20
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
        # Top 20 combined from RRF fusion passed to FlashRank Cross-Encoder for Top 7 final selection
        top_20_fused = fused[:20]
        if rerank and len(top_20_fused) > 1:
            return self._reranker.rerank(query, top_20_fused, top_k=top_k)
        return top_20_fused[:top_k]

    @classmethod
    def _normalize_category(cls, cat: Optional[str]) -> Optional[str]:
        if not cat or cat.strip().lower() in ("all", "general", ""):
            return None
        c = cat.strip().lower()
        if any(w in c for w in ["advertis", "claim", "packag", "label"]):
            return "Advertising & Claims"
        if any(w in c for w in ["contaminan", "metal", "lead", "pesticid"]):
            return "Contaminants"
        if any(w in c for w in ["prohibit", "adulteran"]):
            return "Prohibitions"
        if any(w in c for w in ["sweet", "honey", "sugar"]):
            return "Sweetening Agents & Standards"
        if any(w in c for w in ["additiv", "preservat"]):
            return "Additives"
        if "microbio" in c:
            return "Microbiological"
        return cat.strip()

    @classmethod
    def _build_where(
        cls,
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
        norm_cat = cls._normalize_category(category)
        if norm_cat:
            conditions.append({"category": {"$eq": norm_cat}})
        elif category and category.lower() not in ("all", "general", ""):
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
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    @classmethod
    def _passes_filter(
        cls,
        chunk: FSSAIChunk,
        regulation_id: Optional[str],
        category: Optional[str],
        food_category: Optional[str],
        active_only: bool,
        chunk_type: Optional[ChunkType],
        is_binding: Optional[bool],
    ) -> bool:
        m = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        if regulation_id and m.get("regulation_id") != regulation_id:
            return False
        if category:
            norm_target = cls._normalize_category(category)
            chunk_cat = m.get("category", "")
            norm_chunk = cls._normalize_category(chunk_cat)
            if norm_target and norm_chunk:
                if norm_target != norm_chunk:
                    return False
            elif category.lower() not in ("all", "general", "") and chunk_cat.lower() != category.lower():
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
