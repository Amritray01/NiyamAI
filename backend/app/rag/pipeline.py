"""
FSSAI Document Ingestion Pipeline Orchestrators
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

from app.rag.chunkers import Baseline1PageChunker, FSSAIClauseChunker, FSSAITableChunker
from app.rag.models import ChunkType, FSSAIChunk
from app.rag.parsers import DoclingFSSAIParser
from app.rag.retriever import BaselineRetriever, FSSAIRetriever
from app.rag.vector_store import BM25FSSAIIndex, ChromaFSSAIStore


DEFAULT_CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "niyamai_chroma"))


class FSSAIIngestionPipeline:
    """Production-ready FSSAI PDF ingestion orchestrator."""

    def __init__(
        self,
        chroma_persist: str = DEFAULT_CHROMA_DIR,
        collection_name: str = "fssai_regulations",
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        self._parser = DoclingFSSAIParser()
        self._vector_store = ChromaFSSAIStore(
            collection_name=collection_name,
            persist_dir=chroma_persist,
            embedding_fn=embedding_fn,
        )
        self._bm25 = BM25FSSAIIndex()
        self._all_chunks: List[FSSAIChunk] = []

    def ingest_batch(self, pdf_paths: List[str]) -> Dict[str, int]:
        """Ingest multiple PDFs and rebuild BM25 once at the end."""
        totals = {"table_chunks": 0, "clause_chunks": 0, "total_chunks": 0}
        batch_new: List[FSSAIChunk] = []

        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"[Pipeline Warning] File not found: {pdf_path}")
                continue
            chunks = self._parse_and_chunk(pdf_path)
            batch_new.extend(chunks)
            self._vector_store.upsert(chunks)

            tbl_count = sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE_ROW)
            cls_count = sum(1 for c in chunks if c.chunk_type == ChunkType.CLAUSE)
            totals["table_chunks"] += tbl_count
            totals["clause_chunks"] += cls_count
            totals["total_chunks"] += len(chunks)

        self._all_chunks.extend(batch_new)
        print(f"[Pipeline] Rebuilding BM25 index over {len(self._all_chunks)} chunks...")
        self._bm25.build(self._all_chunks)
        print(f"[Pipeline] Ingestion complete: {totals}")
        return totals

    def get_retriever(self) -> FSSAIRetriever:
        return FSSAIRetriever(self._vector_store, self._bm25)

    def _parse_and_chunk(self, pdf_path: str) -> List[FSSAIChunk]:
        source_doc = os.path.basename(pdf_path)
        regulation_id = self._infer_regulation_id(source_doc)
        category = self._infer_category(source_doc)

        print(f"[Pipeline] Parsing document: {source_doc}")
        parsed = self._parser.parse(pdf_path)

        table_chunker = FSSAITableChunker(
            regulation_id=regulation_id,
            category=category,
            source_doc=source_doc,
            footnote_registry=parsed["footnote_registry"],
        )
        clause_chunker = FSSAIClauseChunker(
            source_doc=source_doc,
            regulation_id=regulation_id,
            document_title=source_doc.replace(".pdf", ""),
            default_category=category,
        )

        table_chunks: List[FSSAIChunk] = []
        for el in parsed["tables"]:
            table_chunks.extend(table_chunker.chunk(el["html"], el["page"], el["table_idx"]))

        clause_chunks: List[FSSAIChunk] = []
        for el in parsed["clauses"]:
            clause_chunks.extend(clause_chunker.chunk(el))

        # Attach highlights by page number
        page_to_highlights: Dict[int, List[str]] = {}
        for h in parsed["highlights"]:
            page_to_highlights.setdefault(h["page"], []).append(h["text"])

        for chunk in table_chunks + clause_chunks:
            pg = chunk.page_numbers[0] if chunk.page_numbers else 1
            chunk.highlight_annotations = page_to_highlights.get(pg, [])

        return table_chunks + clause_chunks

    @staticmethod
    def _infer_regulation_id(filename: str) -> str:
        f = filename.lower()
        if "contaminant" in f:
            return "FSS_Contaminants_2011"
        if "prohibition" in f:
            return "FSS_Prohibition_2011"
        if "sweetening" in f or "honey" in f:
            return "FSS_Standards_Sweeteners_2023"
        if "advertising" in f or "claim" in f:
            return "FSS_Advertising_Claims_2018"
        return "FSS_General_Regulation"

    @staticmethod
    def _infer_category(filename: str) -> str:
        f = filename.lower()
        if "contaminant" in f:
            return "Contaminants"
        if "prohibition" in f:
            return "Prohibitions"
        if "sweetening" in f or "honey" in f:
            return "Standards"
        if "advertising" in f or "claim" in f:
            return "Labelling & Claims"
        return "General"


class BaselineIngestionPipeline:
    """Naive 1-page ingestion pipeline for evaluation baseline comparison."""

    def __init__(self, chroma_persist: str = "./baseline_chroma") -> None:
        self._vector_store = ChromaFSSAIStore(
            collection_name="baseline_1page",
            persist_dir=chroma_persist,
        )

    def ingest_batch(self, pdf_paths: List[str]) -> None:
        import pymupdf as fitz
        for pdf_path in pdf_paths:

            if not os.path.exists(pdf_path):
                continue
            source_doc = os.path.basename(pdf_path)
            doc = fitz.open(pdf_path)
            chunker = Baseline1PageChunker(source_doc=source_doc, regulation_id="BASELINE")
            chunks = []
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                text = page.get_text("text")
                if text.strip():
                    chunks.append(chunker.chunk(page_idx + 1, text))
            doc.close()
            self._vector_store.upsert(chunks)

    def get_retriever(self) -> BaselineRetriever:
        return BaselineRetriever(self._vector_store)
