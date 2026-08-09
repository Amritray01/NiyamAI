from fastapi import APIRouter
from app.schemas.regulatory import GlobalCompareRequest, GlobalCompareResponse
from app.rag.chains import run_global_comparison

router = APIRouter(tags=["compare"])


@router.post("/compare", response_model=GlobalCompareResponse)
def compare_standards(request: GlobalCompareRequest) -> GlobalCompareResponse:
    """
    Global Regulatory Comparison Endpoint (FSSAI India vs. US FDA vs. EU EFSA).
    """
    return run_global_comparison(request.parameter)
