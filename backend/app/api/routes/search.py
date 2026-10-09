import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.rag.chains import run_fssai_rag_search, stream_fssai_rag_search
from app.schemas.regulatory import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search_fssai(request: SearchRequest) -> SearchResponse:
    start_time = time.time()
    response = run_fssai_rag_search(request.question, category=request.category or "All")
    response.execution_time_ms = round((time.time() - start_time) * 1000, 2)
    return response


@router.post("/search/stream")
def search_fssai_stream(request: SearchRequest):
    return StreamingResponse(
        stream_fssai_rag_search(request.question, category=request.category or "All"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
