from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ==========================================
# 1. Search & Citation Schemas
# ==========================================

class SearchRequest(BaseModel):
    """Request payload for FSSAI regulatory search."""
    user_id: str = Field(default="user_demo", description="Unique identifier for the querying user")
    question: str = Field(..., min_length=2, description="Regulatory question or food safety query")
    category: Optional[str] = Field(default="All", description="Filter by category (e.g. Additives, Labeling, Licensing)")
    conversation_id: Optional[str] = Field(default=None, description="Optional ID to bind query to user conversation memory")


class Citation(BaseModel):
    """Schema representing exact legal clause citation."""
    document_name: str = Field(..., description="Name of the official document, e.g. FSS (Food Additives) Regs 2011")
    section: str = Field(..., description="Section / Clause number, e.g. Section 3.1.2")
    page_number: Optional[int] = Field(default=1, description="Page number in original PDF")
    verbatim_text: str = Field(..., description="Exact quoted text from regulation")


class SearchResponse(BaseModel):
    """Structured explainable answer returned to the frontend."""
    summary: str = Field(..., description="Executive plain-English summary")
    detailed_analysis: str = Field(..., description="Step-by-step regulatory analysis and compliance rationale")
    citations: List[Citation] = Field(default_factory=list, description="Verbatim source citations")
    confidence_label: str = Field(default="High", description="High / Medium / Low confidence rating")
    confidence_score: float = Field(default=0.92, description="Numeric confidence score (0.0 to 1.0)")
    execution_time_ms: float = Field(default=0.0, description="Time taken to process query in milliseconds")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated LLM API cost in USD")


# ==========================================
# 2. Global Comparison Schemas
# ==========================================

class GlobalCompareRequest(BaseModel):
    """Request payload to compare food regulations across regions."""
    parameter: str = Field(..., description="Food parameter, additive, or labeling rule (e.g. MSG in noodles, Trans fat limits)")


class RegionalStandard(BaseModel):
    """Regulatory standard details for a specific region."""
    region: str = Field(..., description="Region name: India (FSSAI), USA (FDA), Europe (EFSA)")
    authority: str = Field(..., description="Regulatory body name")
    status: str = Field(..., description="Permitted, Banned, Restricted, or Regulated")
    permitted_limit: str = Field(..., description="Allowed concentration or rule summary")
    citation: str = Field(..., description="Reference code / clause")


class GlobalCompareResponse(BaseModel):
    """Comparison matrix response."""
    parameter: str
    key_takeaway: str
    standards: List[RegionalStandard]


# ==========================================
# 3. Admin & Telemetry Schemas
# ==========================================

class TelemetryLog(BaseModel):
    """Telemetry record for tracking request metrics, execution time, and API cost."""
    request_id: str
    user_id: str
    endpoint: str
    execution_time_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    timestamp: str


class AdminMetricsResponse(BaseModel):
    """Aggregated stats for Admin Telemetry Dashboard."""
    total_requests: int
    total_cost_usd: float
    avg_response_time_ms: float
    active_users_count: int
    logs: List[TelemetryLog]
