from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class ChunkType(str, Enum):
    TABLE_ROW = "table_row"
    CLAUSE = "clause"
    FOOTNOTE = "footnote"
    HEADER = "header"
    AMENDMENT = "amendment"
    BASELINE_PAGE = "baseline_page"


class ICMSFSamplingPlan(BaseModel):
    n: int
    c: int
    m: str
    M: Optional[str] = None
    plan_class: str = "2-class"

    @model_validator(mode="before")
    @classmethod
    def determine_plan_class(cls, values: dict) -> dict:
        m_val = values.get("M")
        values["plan_class"] = (
            "3-class" if m_val and str(m_val).strip() not in ("--", "nil", "", "None") else "2-class"
        )
        return values

    def to_text(self) -> str:
        base = f"ICMSF {self.plan_class} sampling plan with sample size n={self.n}, c={self.c} marginal units permitted, acceptable limit m={self.m}"
        if self.plan_class == "3-class" and self.M:
            base += f", unacceptable limit M={self.M}"
        return base


class ClauseMetadata(BaseModel):
    regulation_id: str
    document_title: str
    clause_path: str
    category: str
    food_category: str = "General"
    food_product: str = ""
    is_binding: bool = True
    language: str = "en"
    source_checksum: str = ""

    @staticmethod
    def compute_checksum(text: str) -> str:
        return f"sha256_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


class FSSAIChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_type: ChunkType = ChunkType.CLAUSE
    source_doc: str = Field(default="FSSAI Regulation")
    page_numbers: List[int] = Field(default_factory=lambda: [1])
    text: str = Field(default="")
    display_text: Optional[str] = None
    embedding_text: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_html: Optional[str] = None
    footnote_ids: List[str] = Field(default_factory=list)
    highlight_annotations: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def sanitize_chunk_fields(cls, values: dict) -> dict:
        if isinstance(values, dict):
            if values.get("source_doc") is None:
                values["source_doc"] = "FSSAI Regulation"
            if values.get("text") is None:
                values["text"] = ""
            if values.get("embedding_text") is None:
                values["embedding_text"] = values.get("text", "")
            if not values.get("display_text"):
                values["display_text"] = values.get("text", "")
            if not values.get("chunk_id"):
                values["chunk_id"] = str(uuid4())
            raw_pages = values.get("page_numbers")
            if not raw_pages or not isinstance(raw_pages, list):
                page = values.get("page", 1)
                try:
                    values["page_numbers"] = [int(page)]
                except (ValueError, TypeError):
                    values["page_numbers"] = [1]
            else:
                cleaned_pages = []
                for p in raw_pages:
                    try:
                        cleaned_pages.append(int(p))
                    except (ValueError, TypeError):
                        pass
                values["page_numbers"] = cleaned_pages or [1]
        return values


class HybridSearchResult(BaseModel):
    chunk: FSSAIChunk
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    dense_distance: Optional[float] = None
