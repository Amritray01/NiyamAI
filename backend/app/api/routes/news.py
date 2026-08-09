from fastapi import APIRouter
from typing import List, Dict, Any

router = APIRouter(tags=["news"])


@router.get("/news", response_model=List[Dict[str, Any]])
def get_regulatory_news() -> List[Dict[str, Any]]:
    """
    Returns latest FSSAI advisories, enforcement news, and penalty circulars.
    
    Implementation Idea for User:
    Build a simple agent or background task using RSS/BeautifulSoup to fetch live alerts from fssai.gov.in.
    """
    return [
        {
            "id": "news_1",
            "title": "FSSAI Advisory on Misleading '100% Fruit Juice' Claims",
            "category": "Advisory Circular",
            "date": "2026-07-28",
            "summary": "FSSAI directs all Food Business Operators (FBOs) to remove claims of '100% Fruit Juice' from reconstituted juices with added water or sweeteners.",
            "source_url": "https://www.fssai.gov.in/cms/advisories.php"
        },
        {
            "id": "news_2",
            "title": "Penalty Imposed for Unlicensed Cloud Kitchen Operations in Mumbai",
            "category": "Enforcement & Fine",
            "date": "2026-07-15",
            "summary": "FDA Maharashtra penalized 14 cloud kitchen units operating without mandatory FSSAI state licenses under Section 63 of FSS Act.",
            "source_url": "https://www.fssai.gov.in/cms/press-releases.php"
        },
        {
            "id": "news_3",
            "title": "Draft Amendment: Revision of Permissible Limits for Synthetic Colors in Sweets",
            "category": "Standard Amendment",
            "date": "2026-06-30",
            "summary": "Proposed reduction of Tartrazine and Sunset Yellow maximum permissible limit from 100 ppm to 50 ppm in traditional confectionery.",
            "source_url": "https://www.fssai.gov.in/cms/gazette-notifications.php"
        }
    ]
