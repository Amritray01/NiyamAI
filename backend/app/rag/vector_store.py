from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.rag.models import ChunkType, FSSAIChunk

DEFAULT_CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "niyamai_chroma"))
DEFAULT_QDRANT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "niyamai_qdrant"))


def get_default_embedding_fn() -> Callable[[List[str]], List[List[float]]]:
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).tolist()
    except Exception:
        import hashlib
        def fallback_embed(texts: List[str]) -> List[List[float]]:
            res = []
            for t in texts:
                h = hashlib.sha256(t.encode("utf-8")).digest()
                res.append([float(b) / 255.0 for b in h[:384]])
            return res
        return fallback_embed


class QdrantFSSAIStore:
    def __init__(
        self,
        collection_name: str = "fssai_regulations",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        local_path: Optional[str] = DEFAULT_QDRANT_DIR,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        vector_size: int = 384,
    ) -> None:
        self.collection_name = collection_name
        self._embed = embedding_fn or get_default_embedding_fn()
        self.vector_size = vector_size
        self._client = None

        target_url = url or getattr(settings, "qdrant_url", "")
        target_key = api_key or getattr(settings, "qdrant_api_key", "")

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels

            if target_url and "cloud.qdrant.io" in target_url:
                try:
                    self._client = QdrantClient(url=target_url, api_key=target_key, timeout=2, check_compatibility=False)
                    self._client.get_collections()
                except Exception:
                    self._client = None

            if self._client is None:
                os.makedirs(local_path, exist_ok=True)
                self._client = QdrantClient(path=local_path, check_compatibility=False)

            existing_colls = [c.name for c in self._client.get_collections().collections]
            if collection_name not in existing_colls:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception as e:
            print(f"[Qdrant Store Warning] Client init error: {e}")

    def upsert(self, chunks: List[FSSAIChunk]) -> None:
        if not chunks or self._client is None:
            return
        from qdrant_client.http import models as qmodels

        unique_chunks = list({c.chunk_id: c for c in chunks}.values())
        embeddings = self._embed([c.embedding_text for c in unique_chunks])

        points = [
            qmodels.PointStruct(
                id=abs(hash(c.chunk_id)) % (2**63 - 1),
                vector=emb,
                payload={
                    "chunk_id": c.chunk_id,
                    "chunk_type": c.chunk_type.value if hasattr(c.chunk_type, "value") else str(c.chunk_type),
                    "source_doc": c.source_doc,
                    "page": c.page_numbers[0] if c.page_numbers else 1,
                    "text": c.text,
                    "display_text": c.display_text or c.text,
                    "embedding_text": c.embedding_text,
                    **c.metadata,
                },
            )
            for c, emb in zip(unique_chunks, embeddings)
        ]

        self._client.upsert(collection_name=self.collection_name, points=points)

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[FSSAIChunk, float]]:
        if self._client is None:
            return []
        from qdrant_client.http import models as qmodels

        qe = self._embed([query_text])[0]
        q_filter = None

        if where:
            must_conditions = []

            def _extract_conditions(w: Any):
                if not isinstance(w, dict):
                    return
                if "$and" in w and isinstance(w["$and"], list):
                    for sub in w["$and"]:
                        _extract_conditions(sub)
                else:
                    for k, v in w.items():
                        if k.startswith("$"):
                            continue
                        val = v["$eq"] if isinstance(v, dict) and "$eq" in v else v
                        if val is not None and isinstance(val, (str, int, float, bool)):
                            must_conditions.append(
                                qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=val))
                            )

            _extract_conditions(where)
            if must_conditions:
                q_filter = qmodels.Filter(must=must_conditions)

        search_result = self._client.query_points(
            collection_name=self.collection_name,
            query=qe,
            query_filter=q_filter,
            limit=top_k,
        ).points

        results: List[Tuple[FSSAIChunk, float]] = []
        for hit in search_result:
            p = hit.payload or {}
            raw_ctype = p.get("chunk_type", "clause")
            try:
                ctype = ChunkType(raw_ctype)
            except ValueError:
                ctype = ChunkType.CLAUSE

            p_chunk_id = str(p.get("chunk_id") or hit.id)
            p_source_doc = str(p.get("source_doc") or "FSSAI Regulation")
            p_text = str(p.get("text") or "")
            p_display = str(p.get("display_text") or p_text)
            p_embed = str(p.get("embedding_text") or p_text)
            raw_page = p.get("page", 1)
            try:
                page_num = int(raw_page) if raw_page is not None else 1
            except (ValueError, TypeError):
                page_num = 1

            chunk = FSSAIChunk(
                chunk_id=p_chunk_id,
                chunk_type=ctype,
                source_doc=p_source_doc,
                page_numbers=[page_num],
                text=p_text,
                display_text=p_display,
                embedding_text=p_embed,
                metadata=p,
            )
            results.append((chunk, max(0.0, 1.0 - hit.score)))

        return results


class ChromaFSSAIStore:
    def __init__(
        self,
        collection_name: str = "fssai_regulations",
        persist_dir: str = DEFAULT_CHROMA_DIR,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed = embedding_fn or get_default_embedding_fn()

    def upsert(self, chunks: List[FSSAIChunk]) -> None:
        if not chunks:
            return
        deduped = list({c.chunk_id: c for c in chunks}.values())
        existing_ids = set(self._collection.get(include=[])["ids"]) if self._collection.count() > 0 else set()
        to_upsert = [c for c in deduped if c.chunk_id not in existing_ids]

        if to_upsert:
            embeddings = self._embed([c.embedding_text for c in to_upsert])
            self._collection.upsert(
                ids=[c.chunk_id for c in to_upsert],
                embeddings=embeddings,
                documents=[c.text for c in to_upsert],
                metadatas=[c.metadata for c in to_upsert],
            )

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[FSSAIChunk, float]]:
        qe = self._embed([query_text])[0]
        kwargs: Dict[str, Any] = {
            "query_embeddings": [qe],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        res = self._collection.query(**kwargs)
        chunks_with_scores: List[Tuple[FSSAIChunk, float]] = []

        if res["documents"] and res["documents"][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            )):
                raw_ctype = meta.get("chunk_type", "clause")
                try:
                    ctype = ChunkType(raw_ctype)
                except ValueError:
                    ctype = ChunkType.CLAUSE

                meta_dict = meta if isinstance(meta, dict) else {}
                p_chunk_id = str(res["ids"][0][i] or "")
                p_source_doc = str(meta_dict.get("source_doc") or "FSSAI Regulation")
                p_text = str(doc or "")
                p_display = str(meta_dict.get("display_text") or p_text)
                p_embed = str(meta_dict.get("embedding_text") or p_text)
                raw_page = meta_dict.get("page", 1)
                try:
                    page_num = int(raw_page) if raw_page is not None else 1
                except (ValueError, TypeError):
                    page_num = 1

                chunk = FSSAIChunk(
                    chunk_id=p_chunk_id,
                    chunk_type=ctype,
                    source_doc=p_source_doc,
                    page_numbers=[page_num],
                    text=p_text,
                    display_text=p_display,
                    embedding_text=p_embed,
                    metadata=meta_dict,
                )
                chunks_with_scores.append((chunk, float(dist)))

        return chunks_with_scores


class BM25FSSAIIndex:
    STOPWORDS = {
        "what", "are", "the", "for", "in", "is", "of", "and", "a", "an", "to", "on", "as",
        "by", "various", "any", "which", "how", "much", "does", "do", "under", "with",
        "from", "shall", "be", "all", "such", "said", "or", "per", "at", "where",
    }

    def __init__(self) -> None:
        self._chunks: List[FSSAIChunk] = []
        self._index: Optional[BM25Okapi] = None

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        raw_tokens = re.findall(r"[\w./-]+", text.lower())
        tokens = [t for t in raw_tokens if t not in cls.STOPWORDS and len(t) > 1]
        return tokens or raw_tokens

    def build(self, chunks: List[FSSAIChunk]) -> None:
        self._chunks = chunks
        corpus = [self._tokenize(f"{c.text} {c.embedding_text or ''}") for c in chunks]
        self._index = BM25Okapi(corpus) if corpus else None

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[FSSAIChunk, float]]:
        if not self._index:
            return []
        tokens = self._tokenize(query_text)
        scores = self._index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            (self._chunks[i], float(score))
            for i, score in ranked[:top_k]
            if score > 0.0
        ]
