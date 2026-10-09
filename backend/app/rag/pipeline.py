from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Union

from app.rag.chunkers import Baseline1PageChunker, FSSAIClauseChunker, FSSAITableChunker
from app.rag.models import ChunkType, FSSAIChunk
from app.rag.parsers import DoclingFSSAIParser
from app.rag.retriever import FSSAIRetriever
from app.rag.vector_store import BM25FSSAIIndex, ChromaFSSAIStore, DEFAULT_CHROMA_DIR, DEFAULT_QDRANT_DIR, QdrantFSSAIStore


class FSSAIIngestionPipeline:
    def __init__(
        self,
        collection_name: str = "fssai_regulations",
        use_chroma: bool = False,
        mode: str = "advanced",
        chroma_persist: str = DEFAULT_CHROMA_DIR,
        qdrant_persist: str = DEFAULT_QDRANT_DIR,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        self._parser = DoclingFSSAIParser()
        self.use_chroma = use_chroma
        self.mode = mode

        if use_chroma:
            self._vector_store: Union[QdrantFSSAIStore, ChromaFSSAIStore] = ChromaFSSAIStore(
                collection_name=collection_name,
                persist_dir=chroma_persist,
                embedding_fn=embedding_fn,
            )
        else:
            self._vector_store = QdrantFSSAIStore(
                collection_name=collection_name,
                local_path=qdrant_persist,
                embedding_fn=embedding_fn,
            )

        self._bm25 = BM25FSSAIIndex()
        self._all_chunks: List[FSSAIChunk] = []

    def ingest_batch(self, pdf_paths: List[str]) -> Dict[str, int]:
        totals = {"table_chunks": 0, "clause_chunks": 0, "total_chunks": 0}
        batch_new: List[FSSAIChunk] = []

        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                continue
            chunks = self._parse_and_chunk(pdf_path)
            batch_new.extend(chunks)
            self._vector_store.upsert(chunks)

            totals["table_chunks"] += sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE_ROW)
            totals["clause_chunks"] += sum(1 for c in chunks if c.chunk_type == ChunkType.CLAUSE)
            totals["total_chunks"] += len(chunks)

        self._all_chunks.extend(batch_new)
        self._bm25.build(self._all_chunks)
        return totals

    def get_retriever(self) -> FSSAIRetriever:
        return FSSAIRetriever(self._vector_store, self._bm25)

    def _parse_and_chunk(self, pdf_path: str) -> List[FSSAIChunk]:
        source_doc = os.path.basename(pdf_path)
        regulation_id = self._infer_regulation_id(source_doc)
        category = self._infer_category(source_doc)

        if self.mode == "baseline":
            import pymupdf as fitz
            doc = fitz.open(pdf_path)
            chunker = Baseline1PageChunker(source_doc=source_doc, regulation_id=regulation_id)
            chunks: List[FSSAIChunk] = []
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                text = page.get_text("text")
                if text.strip():
                    chunks.append(chunker.chunk(page_idx + 1, text))
            doc.close()
            return chunks

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
            return "Sweetening Agents & Standards"
        if "advertising" in f or "claim" in f:
            return "Advertising & Claims"
        return "General"
