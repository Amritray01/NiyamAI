"""
Vector & Keyword Storage Module (ChromaDB + rank-bm25)
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from app.rag.models import ChunkType, FSSAIChunk


DEFAULT_CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "niyamai_chroma"))


class ChromaFSSAIStore:
    """
    Persistent local vector store backed by ChromaDB with BAAI/bge-small-en-v1.5 embeddings.
    """

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
        self._embed = embedding_fn or self._default_embed()

    @staticmethod
    def _default_embed() -> Callable[[List[str]], List[List[float]]]:
        """Lazy load local sentence-transformer model on CPU."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).tolist()

    def upsert(self, chunks: List[FSSAIChunk]) -> None:
        """Batch upsert chunks into ChromaDB."""
        if not chunks:
            return
        # Deduplicate chunks by chunk_id to prevent ChromaDB DuplicateIDError
        unique_map: Dict[str, FSSAIChunk] = {}
        for c in chunks:
            unique_map[c.chunk_id] = c
        deduped_chunks = list(unique_map.values())

        # Check existing IDs in collection to avoid re-embedding if already stored
        existing_ids = set(self._collection.get(include=[])["ids"]) if self._collection.count() > 0 else set()
        to_upsert = [c for c in deduped_chunks if c.chunk_id not in existing_ids]

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
        """Dense cosine vector search with optional metadata where clause."""
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

                chunk = FSSAIChunk(
                    chunk_id=res["ids"][0][i],
                    chunk_type=ctype,
                    source_doc=meta.get("source_doc", ""),
                    page_numbers=[meta.get("page", 1)],
                    text=doc,
                    embedding_text=doc,
                    metadata=meta,
                )
                chunks_with_scores.append((chunk, float(dist)))

        return chunks_with_scores


class BM25FSSAIIndex:
    """In-process sparse BM25 index using rank-bm25 (BM25Okapi)."""
    STOPWORDS = {
        "what", "are", "the", "for", "in", "is", "of", "and", "a", "an", "to", "on", "as", 
        "by", "various", "any", "which", "how", "much", "does", "do", "under", "with", 
        "from", "shall", "be", "all", "such", "said", "or", "per", "at", "where"
    }

    def __init__(self) -> None:
        self._chunks: List[FSSAIChunk] = []
        self._index: Optional[BM25Okapi] = None

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Tokenize preserving compound terms and filtering generic stopwords."""
        raw_tokens = re.findall(r"[\w./-]+", text.lower())
        tokens = [t for t in raw_tokens if t not in cls.STOPWORDS and len(t) > 1]
        return tokens or raw_tokens

    def build(self, chunks: List[FSSAIChunk]) -> None:
        """Build BM25 index over the provided chunks with contextual headers."""
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
