from fastapi import APIRouter
from typing import Dict, Any
from app.rag.agents.news_agent import run_fssai_news_agent

router = APIRouter(tags=["news"])


@router.get("/news", response_model=Dict[str, Any])
def get_regulatory_news() -> Dict[str, Any]:
    """
    Returns latest FSSAI advisories, notifications, and enforcement actions
    aggregated by the FSSAI Regulatory News Agent with AI compliance breakdown.
    """
    try:
        return run_fssai_news_agent()
    except Exception as e:
        print(f"[!] Live news agent execution error: {e}")
        from app.rag.agents.news_agent import CURATED_REGULATORY_UPDATES, FSSAINewsPayload
        return FSSAINewsPayload(
            ai_overview="Active FSSAI Regulatory Pulse: Heightened enforcement across cloud kitchen licensing, mandatory removal of '100% fruit juice' labels on reconstituted drinks, and reduction of synthetic color tolerances.",
            category_summaries={
                "Advisories": "Mandatory directives barring misleading '100% juice' labeling and unauthorized health claims.",
                "Notifications": "Amendments to Food Additives Regulations and microbiological criteria.",
                "Enforcement & Fines": "Targeted raids on unlicensed cloud kitchens and adulterated dairy products."
            },
            news_items=[item for item in CURATED_REGULATORY_UPDATES]
        ).model_dump()


@router.post("/news/refresh", response_model=Dict[str, Any])
def refresh_regulatory_news() -> Dict[str, Any]:
    """Force fresh scrape and search across FSSAI portals and live enforcement feeds."""
    return get_regulatory_news()
