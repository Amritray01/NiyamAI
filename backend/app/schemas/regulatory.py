from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class SearchRequest(BaseModel):
    user_id: str = Field(default="user_demo")
    question: str = Field(..., min_length=2)
    category: Optional[str] = Field(default="All")


class Citation(BaseModel):
    document_name: str = Field(default="FSSAI Regulation")
    section: str = Field(default="Regulation Clause")
    page_number: Optional[int] = Field(default=1)
    verbatim_text: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def sanitize_citation(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if values.get("document_name") is None:
                values["document_name"] = "FSSAI Regulation"
            if values.get("section") is None:
                values["section"] = "Regulation Clause"
            if values.get("verbatim_text") is None:
                values["verbatim_text"] = ""
            raw_page = values.get("page_number")
            if raw_page is None:
                values["page_number"] = 1
            else:
                try:
                    values["page_number"] = int(raw_page)
                except (ValueError, TypeError):
                    values["page_number"] = 1
        return values


class SearchResponse(BaseModel):
    summary: str = Field(default="")
    detailed_analysis: str = Field(default="")
    citations: List[Citation] = Field(default_factory=list)
    confidence_label: str = Field(default="High")
    confidence_score: float = Field(default=0.92)
    execution_time_ms: float = Field(default=0.0)
    estimated_cost_usd: float = Field(default=0.0)


class GlobalCompareRequest(BaseModel):
    parameter: str


class RegionalStandard(BaseModel):
    region: str
    authority: str
    status: str
    permitted_limit: str
    citation: str


class GlobalCompareResponse(BaseModel):
    parameter: str
    key_takeaway: str
    standards: List[RegionalStandard]


class TelemetryLog(BaseModel):
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
    total_requests: int
    total_cost_usd: float
    avg_response_time_ms: float
    active_users_count: int
    logs: List[TelemetryLog]
