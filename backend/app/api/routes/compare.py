from fastapi import APIRouter

from app.rag.chains import run_global_comparison
from app.schemas.regulatory import GlobalCompareRequest, GlobalCompareResponse

router = APIRouter(tags=["compare"])


@router.post("/compare", response_model=GlobalCompareResponse)
def compare_standards(request: GlobalCompareRequest) -> GlobalCompareResponse:
    return run_global_comparison(request.parameter)
