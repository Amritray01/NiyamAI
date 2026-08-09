import time
from fastapi import APIRouter, Depends
from app.schemas.regulatory import SearchRequest, SearchResponse
from app.rag.chains import run_fssai_rag_search
from app.core.telemetry import record_request_telemetry
from app.core.auth import get_current_user

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search_fssai(request: SearchRequest, current_user: dict = Depends(get_current_user)) -> SearchResponse:
    """
    FSSAI Regulatory Search Endpoint.
    Executes search, measures latency, records telemetry, and returns citations & analysis.
    """
    start_time = time.time()
    
    # Execute starter search
    response = run_fssai_rag_search(request.question, category=request.category or "All")
    
    execution_time_ms = (time.time() - start_time) * 1000
    response.execution_time_ms = round(execution_time_ms, 2)
    
    # Record telemetry for logging / admin cost engine
    record_request_telemetry(
        user_id=current_user.get("user_id", "user_demo"),
        endpoint="/api/search",
        execution_time_ms=execution_time_ms,
        prompt_tokens=120,
        completion_tokens=210
    )
    
    return response
