"""
Chunking Strategies for FSSAI Documents (Table Row, Clause, & Baseline 1-Page)
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

from app.rag.models import ChunkType, FSSAIChunk, ClauseMetadata
from app.rag.parsers import FootnoteRegistry


def generate_deterministic_chunk_id(
    source_doc: str,
    chunk_type: str,
    identifier: str,
    text_snippet: str
) -> str:
    """
    Generates a deterministic 16-character SHA-256 chunk ID.
    Guarantees stable chunk IDs across evaluation & ingestion runs.
    """
    raw_str = f"{source_doc}:{chunk_type}:{identifier}:{text_snippet}"
    return "chunk_" + hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:12]


class FSSAITableChunker:
    """Row-level chunker for FSSAI tables."""
    _FOOD_COL_CANDIDATES = ["food category", "food product", "product", "food item", "article of food", "name of food"]
    _PARAM_COL_CANDIDATES = ["parameter", "microorganism", "contaminant", "additive", "substance", "test", "limit"]

    def __init__(
        self,
        regulation_id: str,
        category: str,
        source_doc: str,
        footnote_registry: FootnoteRegistry,
    ) -> None:
        self.regulation_id = regulation_id
        self.category = category
        self.source_doc = source_doc
        self.footnote_registry = footnote_registry

    def chunk(self, table_html: str, page: int, table_idx: int = 0) -> List[FSSAIChunk]:
        soup = BeautifulSoup(table_html, "html.parser")
        headers = self._extract_headers(soup)
        chunks: List[FSSAIChunk] = []

        data_rows = soup.select("tbody tr") or soup.select("tr")[1:]
        for row_idx, row in enumerate(data_rows):
            cells = [td.get_text(separator=" ", strip=True) for td in row.find_all(["td", "th"])]
            if not any(cells):
                continue

            row_data = (
                dict(zip(headers, cells)) if headers
                else {f"col_{i}": v for i, v in enumerate(cells)}
            )
            row_html = str(row)
            footnotes = self.footnote_registry.resolve_all_in_html(row_html)
            food_cat = self._find_value(row_data, self._FOOD_COL_CANDIDATES) or "General"
            param = self._find_value(row_data, self._PARAM_COL_CANDIDATES) or "Standard"
            raw_text = " | ".join(f"{k}: {v}" for k, v in row_data.items() if v)
            embed_text = self._to_embedding_text(row_data, footnotes)

            chunk_id = generate_deterministic_chunk_id(
                self.source_doc, ChunkType.TABLE_ROW.value, f"pg_{page}_tbl_{table_idx}_row_{row_idx}", raw_text
            )


            metadata: Dict[str, Any] = {
                "regulation_id": self.regulation_id,
                "category": self.category,
                "food_category": food_cat,
                "food_product": food_cat,
                "parameter": param,
                "source_doc": self.source_doc,
                "page": page,
                "is_active": True,
                "is_binding": True,
                "has_footnotes": len(footnotes) > 0,
                "chunk_type": ChunkType.TABLE_ROW.value,
            }

            chunks.append(FSSAIChunk(
                chunk_id=chunk_id,
                chunk_type=ChunkType.TABLE_ROW,
                source_doc=self.source_doc,
                page_numbers=[page],
                text=raw_text,
                embedding_text=embed_text,
                metadata=metadata,
                raw_html=row_html,
                footnote_ids=footnotes,
            ))
        return chunks

    @staticmethod
    def _extract_headers(soup: BeautifulSoup) -> List[str]:
        thead = soup.find("thead")
        if thead:
            return [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]
        rows = soup.find_all("tr")
        return [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])] if rows else []

    @staticmethod
    def _find_value(row: Dict[str, str], candidates: List[str]) -> Optional[str]:
        for key in row:
            for cand in candidates:
                if cand.lower() in key.lower():
                    return row[key] or None
        return None

    @staticmethod
    def _to_embedding_text(row_data: Dict[str, str], footnotes: List[str]) -> str:
        meaningful = [f"{k} is {v}" for k, v in row_data.items() if v and v not in ("--", "N/A", "Nil", "")]
        text = "In this FSSAI regulation table, " + ", ".join(meaningful) + "."
        if footnotes:
            text += " Additional notes: " + "; ".join(footnotes) + "."
        return text


class FSSAIClauseChunker:
    """Sliding window chunker for regulation clauses."""
    MAX_CHARS = 800
    OVERLAP_CHARS = 120

    def __init__(
        self,
        source_doc: str,
        regulation_id: str,
        document_title: str = "",
        default_category: str = "General",
    ) -> None:
        self.source_doc = source_doc
        self.regulation_id = regulation_id
        self.document_title = document_title
        self.default_category = default_category

    def chunk(self, clause: Dict[str, Any]) -> List[FSSAIChunk]:
        text = clause["text"]
        page = clause.get("page", 1)
        clause_id = clause.get("clause_id", "unknown")
        title = clause.get("title", "")
        category = self._infer_category(text) or self.default_category
        food_cat = self._infer_food_category(text)
        segments = self._sliding_window(text, self.MAX_CHARS, self.OVERLAP_CHARS)

        chunks: List[FSSAIChunk] = []
        for seg_idx, segment in enumerate(segments):
            checksum = ClauseMetadata.compute_checksum(segment)
            clause_path = f"Clause_{clause_id}_seg_{seg_idx}" if not clause_id.startswith("Page_") else f"{clause_id}_seg_{seg_idx}"

            chunk_id = generate_deterministic_chunk_id(
                self.source_doc, ChunkType.CLAUSE.value, clause_path, checksum
            )

            # Build enriched embedding text
            prefix = ""
            if not segment.startswith("["):
                header_parts = [self.document_title]
                if title:
                    header_parts.append(title)
                elif clause_id and clause_id != "General":
                    header_parts.append(f"Regulation {clause_id}")
                prefix = f"[{' | '.join(header_parts)}]\n"
            
            embed_text = prefix + segment

            metadata: Dict[str, Any] = {
                "regulation_id": self.regulation_id,
                "document_title": self.document_title,
                "clause_id": clause_id,
                "clause_title": title,
                "clause_path": clause_path,
                "category": category,
                "food_category": food_cat,
                "food_product": food_cat,
                "source_doc": self.source_doc,
                "page": page,
                "is_active": True,
                "is_binding": True,
                "source_checksum": checksum,
                "chunk_type": ChunkType.CLAUSE.value,
            }

            chunks.append(FSSAIChunk(
                chunk_id=chunk_id,
                chunk_type=ChunkType.CLAUSE,
                source_doc=self.source_doc,
                page_numbers=[page],
                text=segment,
                embedding_text=embed_text,
                metadata=metadata,
            ))
        return chunks

    @staticmethod
    def _sliding_window(text: str, max_chars: int, overlap: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        segments, start = [], 0
        while start < len(text):
            segments.append(text[start : start + max_chars])
            start += max_chars - overlap
        return segments

    @staticmethod
    def _infer_category(text: str) -> Optional[str]:
        t = text.lower()
        if any(w in t for w in ["additive", "preservative", "colour", "flavor", "flavour"]):
            return "Additives"
        if any(w in t for w in ["contaminant", "pesticide", "residue", "heavy metal", "mycotoxin", "lead"]):
            return "Contaminants"
        if any(w in t for w in ["coliform", "salmonella", "yeast", "mould", "microbiolog"]):
            return "Microbiological"
        if any(w in t for w in ["label", "claim", "marking", "display", "advertising"]):
            return "Labelling"
        if any(w in t for w in ["honey", "sweetener", "sugar", "jaggery"]):
            return "Sweetening agents"
        return None

    @staticmethod
    def _infer_food_category(text: str) -> str:
        keywords: Dict[str, List[str]] = {
            "Honey & Sweeteners": ["honey", "jaggery", "sugar", "dextrose", "sucrose", "sweetener"],
            "Dairy Products": ["milk", "butter", "cheese", "cream", "paneer", "ghee"],
            "Beverages": ["juice", "water", "beverage", "drink", "squash"],
            "Contaminants & Residues": ["metal", "lead", "arsenic", "aflatoxin", "insecticide"],
            "Advertising & Claims": ["claim", "advertisement", "misleading", "nutritional"],
        }
        t = text.lower()
        for cat, kws in keywords.items():
            if any(kw in t for kw in kws):
                return cat
        return "General"


class Baseline1PageChunker:
    """Naive 1-page text chunker used strictly as an evaluation baseline."""

    def __init__(self, source_doc: str, regulation_id: str) -> None:
        self.source_doc = source_doc
        self.regulation_id = regulation_id

    def chunk(self, page_num: int, page_text: str) -> FSSAIChunk:
        chunk_id = generate_deterministic_chunk_id(
            self.source_doc, "baseline_1page", f"page_{page_num}", page_text[:100]
        )
        metadata = {
            "regulation_id": self.regulation_id,
            "source_doc": self.source_doc,
            "page": page_num,
            "is_active": True,
            "chunk_type": "baseline_page",
        }
        return FSSAIChunk(
            chunk_id=chunk_id,
            chunk_type=ChunkType.CLAUSE,
            source_doc=self.source_doc,
            page_numbers=[page_num],
            text=page_text,
            embedding_text=page_text,
            metadata=metadata,
        )
