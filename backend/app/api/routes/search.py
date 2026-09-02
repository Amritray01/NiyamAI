import time
from fastapi import APIRouter
from app.schemas.regulatory import SearchRequest, SearchResponse
from app.rag.chains import run_fssai_rag_search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search_fssai(request: SearchRequest) -> SearchResponse:
    """
    FSSAI Regulatory Search Endpoint.
    Executes RAG search and returns explainable analysis with verbatim citations.
    """
    start_time = time.time()
    
    # Execute RAG search
    response = run_fssai_rag_search(request.question, category=request.category or "All")
    
    execution_time_ms = (time.time() - start_time) * 1000
    response.execution_time_ms = round(execution_time_ms, 2)
    
    return response
