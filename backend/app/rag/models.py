"""
Domain Data Models & Schemas for NiyamAI FSSAI RAG Pipeline
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class ChunkType(str, Enum):
    """Every chunk entering the index is tagged with one of these."""
    TABLE_ROW     = "table_row"     # Single data row from a microbiological / additive table
    CLAUSE        = "clause"        # Numbered clause from regulation prose
    FOOTNOTE      = "footnote"      # Footnote resolved from superscript reference
    HEADER        = "header"        # Chapter / section heading
    AMENDMENT     = "amendment"     # Amendment notice / legal update
    BASELINE_PAGE = "baseline_page" # Naive 1-page evaluation baseline chunk



class ICMSFSamplingPlan(BaseModel):
    """
    Standardized ICMSF sampling plan notation used across FSSAI microbiological tables.
    Auto-classifies 2-class vs 3-class.
    """
    n: int = Field(..., description="Sample units drawn from the lot.")
    c: int = Field(..., description="Max units allowed above m (marginal acceptability).")
    m: str = Field(..., description="Lower microbiological limit.")
    M: Optional[str] = Field(default=None, description="Upper limit for 3-class plans.")
    plan_class: str = Field(default="2-class", description="Auto-set: '2-class' or '3-class'.")

    @model_validator(mode="before")
    @classmethod
    def determine_plan_class(cls, values: dict) -> dict:
        M_val = values.get("M")
        values["plan_class"] = (
            "3-class" if M_val and str(M_val).strip() not in ("--", "nil", "", "None") else "2-class"
        )
        return values

    def to_text(self) -> str:
        """Prose representation for embedding models."""
        base = (
            f"ICMSF {self.plan_class} sampling plan with "
            f"sample size n={self.n}, c={self.c} marginal units permitted, "
            f"acceptable limit m={self.m}"
        )
        if self.plan_class == "3-class" and self.M:
            base += f", unacceptable limit M={self.M}"
        return base


class FSSAILimitRule(BaseModel):
    """Microbiological or chemical parameter limit for a food category."""
    regulation_id: str = Field(default="FSS_General_2011")
    food_category: str = Field(..., description="Macro category, e.g. 'Dairy Products'")
    food_product: str = Field(default="", description="Specific product, e.g. 'Pasteurised Butter'")
    parameter: str = Field(..., description="e.g. 'Coliform Count'")
    sampling_plan: Optional[ICMSFSamplingPlan] = None
    limit_value: Optional[str] = None
    footnotes: List[str] = Field(default_factory=list)

    def text_for_embedding(self) -> str:
        prod_str = f"for {self.food_product}" if self.food_product else f"under category {self.food_category}"
        parts = [f"Under {self.regulation_id}, {prod_str}, the {self.parameter} limit"]
        if self.sampling_plan:
            parts.append("follows an " + self.sampling_plan.to_text() + ".")
        elif self.limit_value:
            parts.append(f"is set at {self.limit_value}.")
        else:
            parts.append("is regulated strictly.")
        if self.footnotes:
            parts.append("Notes: " + "; ".join(self.footnotes) + ".")
        return " ".join(parts)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        return {
            "regulation_id": self.regulation_id,
            "food_category": self.food_category,
            "food_product": self.food_product,
            "parameter": self.parameter,
            "sampling_n": self.sampling_plan.n if self.sampling_plan else 0,
            "sampling_c": self.sampling_plan.c if self.sampling_plan else 0,
            "sampling_m": str(self.sampling_plan.m) if self.sampling_plan else "",
            "sampling_M": str(self.sampling_plan.M) if self.sampling_plan and self.sampling_plan.M else "",
            "limit_value": self.limit_value or "",
            "has_footnotes": len(self.footnotes) > 0,
            "is_active": True,
            "chunk_type": ChunkType.TABLE_ROW.value,
        }


class AmendedLimitRule(BaseModel):
    """FSSAI limit rule with full amendment & supersession tracking."""
    regulation_id: str
    parameter: str
    food_category: str
    food_product: str = ""
    limit_value: str
    is_active: bool = True
    superseded_by: Optional[List[str]] = None
    amended_by: Optional[List[str]] = None
    effective_date: str = "2011-08-05"

    def text_for_embedding(self) -> str:
        status = (
            "currently active"
            if self.is_active
            else f"superseded by {', '.join(self.superseded_by or ['subsequent amendment'])}"
        )
        prod = f"in {self.food_product}" if self.food_product else f"under {self.food_category}"
        return (
            f"Under {self.regulation_id}, {self.parameter} {prod} "
            f"has a limit of {self.limit_value}. Status: {status}. "
            f"Effective date: {self.effective_date}."
        )

    def to_chroma_metadata(self) -> Dict[str, Any]:
        return {
            "regulation_id": self.regulation_id,
            "parameter": self.parameter,
            "food_category": self.food_category,
            "food_product": self.food_product,
            "limit_value": str(self.limit_value),
            "is_active": self.is_active,
            "superseded_by": json.dumps(self.superseded_by or []),
            "amended_by": json.dumps(self.amended_by or []),
            "effective_date": self.effective_date,
            "chunk_type": ChunkType.AMENDMENT.value,
        }


class ClauseMetadata(BaseModel):
    """Metadata schema for clause-level chunks."""
    regulation_id: str
    document_title: str
    clause_path: str
    category: str
    food_category: str = "General"
    food_product: str = ""
    is_binding: bool = True
    language: str = "en"
    source_checksum: str = ""

    @classmethod
    def compute_checksum(cls, text: str) -> str:
        return "sha256_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def to_chroma_metadata(self) -> Dict[str, Any]:
        return {
            "regulation_id": self.regulation_id,
            "document_title": self.document_title,
            "clause_path": self.clause_path,
            "category": self.category,
            "food_category": self.food_category,
            "food_product": self.food_product,
            "is_active": True,
            "is_binding": self.is_binding,
            "language": self.language,
            "source_checksum": self.source_checksum,
            "chunk_type": ChunkType.CLAUSE.value,
        }


class DocumentMetadata(BaseModel):
    """PDF Document Model."""
    filename: str
    title: str
    regulation_id: str
    category: str
    total_pages: int
    sha256_checksum: str


class FSSAIChunk(BaseModel):
    """Canonical retrieval unit."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_type: ChunkType
    source_doc: str
    page_numbers: List[int]
    text: str
    embedding_text: str
    metadata: Dict[str, Any]
    raw_html: Optional[str] = None
    footnote_ids: List[str] = Field(default_factory=list)
    highlight_annotations: List[str] = Field(default_factory=list)


class HybridSearchResult(BaseModel):
    """Result container returned by FSSAIRetriever."""
    chunk: FSSAIChunk
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    dense_distance: Optional[float] = None
