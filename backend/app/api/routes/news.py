from typing import Any, Dict
from fastapi import APIRouter

from app.rag.agents.news_agent import run_fssai_news_agent

router = APIRouter(tags=["news"])


@router.get("/news", response_model=Dict[str, Any])
def get_regulatory_news() -> Dict[str, Any]:
    return run_fssai_news_agent()


@router.post("/news/refresh", response_model=Dict[str, Any])
def refresh_regulatory_news() -> Dict[str, Any]:
    return run_fssai_news_agent()
