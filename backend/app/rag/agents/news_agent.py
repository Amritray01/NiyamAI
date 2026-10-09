from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
import html
import re
import time
from typing import Any, Dict, List, Tuple, TypedDict
import urllib.request
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import httpx
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


class FSSAINewsItem(BaseModel):
    id: str
    title: str
    category: str
    summary: str
    compliance_action: str = ""
    affected_sectors: List[str] = Field(default_factory=list)
    date: str
    source_url: str
    source_name: str = Field(default="News Article")
    severity: str = "Medium"


class FSSAINewsPayload(BaseModel):
    ai_overview: str
    last_updated: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    category_summaries: Dict[str, str] = Field(default_factory=dict)
    news_items: List[FSSAINewsItem] = Field(default_factory=list)


_CACHE: Dict[str, Any] = {"payload": None, "timestamp": 0.0}
CACHE_TTL_SECONDS = 600


def _clean_text(text: str) -> str:
    cleaned = html.unescape(text)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _infer_affected_sectors(text: str) -> List[str]:
    t = text.lower()
    sectors = []
    mapping = {
        "Dairy & Milk": ["dairy", "milk", "paneer", "ghee", "butter", "cheese", "curd"],
        "Beverages & Juices": ["beverage", "juice", "drink", "water", "tea", "coffee", "nectar", "energy drink", "red bull"],
        "Spices & Condiments": ["spice", "chili", "turmeric", "condiment", "masala", "curry"],
        "Cloud Kitchens & QSR": ["cloud kitchen", "restaurant", "swiggy", "zomato", "delivery", "hotel", "knife", "eatery"],
        "Supplements & Nutraceuticals": ["supplement", "nutraceutical", "ashwagandha", "protein", "ayurvedic", "vitamin"],
        "Bakery & Confectionery": ["bakery", "bread", "sweet", "confectionery", "chocolate", "cake", "biscuit"],
        "Edible Oils & Fats": ["oil", "mustard", "palm", "vegetable oil", "fat"],
        "Packaged & Processed Foods": ["packaged", "fbo", "label", "claim", "brand", "snack", "noodle"],
    }
    for sector, keywords in mapping.items():
        if any(kw in t for kw in keywords):
            sectors.append(sector)
    return sectors[:3] or ["Food Business Operators", "Packaged Foods"]


def _infer_severity(title: str, category: str) -> str:
    t = title.lower()
    if any(w in t for w in ["ban", "raid", "seize", "penalty", "quash", "stay", "court", "cancel", "licence", "license", "order"]):
        return "High"
    if category == "Enforcement & Fines":
        return "High"
    if any(w in t for w in ["notice", "propose", "warn", "strict", "advisory", "amendment"]):
        return "Medium"
    return "Low"


def _generate_compliance_action(title: str, category: str, sectors: List[str]) -> str:
    t = title.lower()
    sec_str = ", ".join(sectors) if sectors else "affected food categories"
    if "label" in t or "claim" in t or "100%" in t:
        return f"Audit packaging declarations for {sec_str} against FSSAI Advertising & Claims Regulations 2018."
    if "raid" in t or "seize" in t or "license" in t or "licence" in t or "unlicensed" in t:
        return f"Verify active FSSAI Central/State premise registration and display FSSAI license numbers across all storefronts."
    if "contaminat" in t or "residue" in t or "knife" in t or "hygiene" in t or "equipment" in t:
        return f"Implement standard food contact equipment sanitation protocols and batch-level quality audits for {sec_str}."
    if "sweet" in t or "colour" in t or "color" in t or "sugar" in t or "fat" in t:
        return f"Review product formulations to ensure additives and nutritional thresholds adhere strictly to statutory limits."
    if category == "Notifications":
        return f"Review gazette amendment timelines and update internal standard operating procedures for {sec_str}."
    return f"Review official regulatory directives and consult legal compliance team for operational impacts on {sec_str}."


class NewsAgentState(TypedDict):
    raw_items: List[Dict[str, Any]]
    news_items: List[FSSAINewsItem]
    ai_overview: str
    category_summaries: Dict[str, str]


def node_scrape_feeds(state: NewsAgentState) -> Dict[str, Any]:
    feeds = [
        ("FSSAI+advisory+order+guidelines+India", "Advisories"),
        ("FSSAI+gazette+notification+standards+amendment+India", "Notifications"),
        ("FSSAI+raid+seized+penalty+FDA+India", "Enforcement & Fines"),
        ("FSSAI+standards+testing+MRL+food+safety+India", "Standards & Guidance"),
    ]
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for query, default_cat in feeds:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = resp.read()
                root = ET.fromstring(data)
                for entry in root.findall(".//item")[:5]:
                    raw_title = entry.findtext("title") or ""
                    link = entry.findtext("link") or ""
                    pub_date = entry.findtext("pubDate") or ""
                    desc = entry.findtext("description") or ""

                    source_tag = entry.find("source")
                    source_name = source_tag.text.strip() if (source_tag is not None and source_tag.text) else ""
                    if not source_name and " - " in raw_title:
                        source_name = raw_title.split(" - ")[-1].strip()
                    if not source_name:
                        source_name = "News Media"

                    clean_title = _clean_text(re.sub(r"\s*-\s*[^-]+$", "", raw_title))
                    if not clean_title or len(clean_title) < 15 or clean_title.lower() in seen:
                        continue
                    seen.add(clean_title.lower())

                    try:
                        date_str = parsedate_to_datetime(pub_date).strftime("%Y-%m-%d")
                    except Exception:
                        date_str = datetime.now().strftime("%Y-%m-%d")

                    clean_desc = _clean_text(desc)
                    summary = clean_desc if (clean_desc and len(clean_desc) > 30 and clean_desc != clean_title) else f"News report: {clean_title}."

                    items.append({
                        "id": f"rss_{abs(hash(clean_title)) % 1000000}",
                        "title": clean_title,
                        "category": default_cat,
                        "summary": summary,
                        "date": date_str,
                        "source_url": link,
                        "source_name": source_name,
                    })
        except Exception:
            continue

    try:
        gov_resp = httpx.get("https://www.fssai.gov.in/food-law/advisories", headers={"User-Agent": "Mozilla/5.0"}, timeout=3.0, follow_redirects=True)
        if gov_resp.status_code == 200:
            soup = BeautifulSoup(gov_resp.text, "html.parser")
            for a in soup.find_all("a")[:6]:
                txt = _clean_text(a.get_text(strip=True))
                href = a.get("href", "")
                if txt and len(txt) > 25 and not any(k in txt.lower() for k in ["home", "contact", "feedback", "sitemap"]):
                    full_url = href if href.startswith("http") else f"https://www.fssai.gov.in{href}"
                    items.append({
                        "id": f"gov_{abs(hash(txt)) % 1000000}",
                        "title": txt,
                        "category": "Advisories",
                        "summary": f"Official FSSAI gazette directive regarding: '{txt}'.",
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "source_url": full_url,
                        "source_name": "FSSAI Official Gazette",
                    })
    except Exception:
        pass

    return {"raw_items": items}


def node_analyze_items(state: NewsAgentState) -> Dict[str, Any]:
    raw = state.get("raw_items", [])
    seen = set()
    deduped: List[FSSAINewsItem] = []

    for item in raw:
        normalized = re.sub(r"[^\w\s]", "", item["title"].lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            sectors = _infer_affected_sectors(item["title"] + " " + item["summary"])
            action = _generate_compliance_action(item["title"], item["category"], sectors)
            severity = _infer_severity(item["title"], item["category"])

            deduped.append(FSSAINewsItem(
                id=item["id"],
                title=item["title"],
                category=item["category"],
                summary=item["summary"],
                compliance_action=action,
                affected_sectors=sectors,
                date=item["date"],
                source_url=item["source_url"],
                source_name=item.get("source_name", "News Article"),
                severity=severity,
            ))

    return {"news_items": deduped[:12]}


def node_synthesize_pulse(state: NewsAgentState) -> Dict[str, Any]:
    news_items = state.get("news_items", [])
    titles_by_cat: Dict[str, List[str]] = {}
    for item in news_items:
        titles_by_cat.setdefault(item.category, []).append(item.title)

    category_summaries: Dict[str, str] = {}
    for cat, titles in titles_by_cat.items():
        sample_topics = "; ".join([t[:60] + "..." if len(t) > 60 else t for t in titles[:2]])
        category_summaries[cat] = f"Active focus on {len(titles)} key developments: {sample_topics}."

    for fallback_cat, default_desc in [
        ("Advisories", "Active directives governing non-standard health claims and food labeling compliance."),
        ("Notifications", "Statutory amendments to FSSAI product standards and sampling criteria."),
        ("Enforcement & Fines", "Targeted state FDA surveillance, hygiene inspections, and licensing crackdowns."),
        ("Standards & Guidance", "Updated testing protocols for chemical residues, contaminants, and microbiological safety."),
    ]:
        if fallback_cat not in category_summaries:
            category_summaries[fallback_cat] = default_desc

    top_headlines = " | ".join([item.title for item in news_items[:3]])
    ai_overview = (
        f"Live FSSAI Regulatory Pulse ({datetime.now().strftime('%B %Y')}): "
        f"Real-time monitoring reflects key regulatory activities including {top_headlines}. "
        f"Food Business Operators are advised to audit label declarations and maintain rigorous batch-level quality verification."
    )

    return {"ai_overview": ai_overview, "category_summaries": category_summaries}


def create_news_graph():
    graph = StateGraph(NewsAgentState)
    graph.add_node("scrape_feeds", node_scrape_feeds)
    graph.add_node("analyze_items", node_analyze_items)
    graph.add_node("synthesize_pulse", node_synthesize_pulse)

    graph.set_entry_point("scrape_feeds")
    graph.add_edge("scrape_feeds", "analyze_items")
    graph.add_edge("analyze_items", "synthesize_pulse")
    graph.add_edge("synthesize_pulse", END)

    return graph.compile()


news_agent_app = create_news_graph()


def run_fssai_news_agent(force_refresh: bool = False) -> Dict[str, Any]:
    current_time = time.time()
    if not force_refresh and _CACHE["payload"] and (current_time - _CACHE["timestamp"] < CACHE_TTL_SECONDS):
        return _CACHE["payload"]

    initial_state: NewsAgentState = {
        "raw_items": [],
        "news_items": [],
        "ai_overview": "",
        "category_summaries": {},
    }

    final_state = news_agent_app.invoke(initial_state)

    payload = FSSAINewsPayload(
        ai_overview=final_state["ai_overview"],
        category_summaries=final_state["category_summaries"],
        news_items=final_state["news_items"],
    ).model_dump()

    _CACHE["payload"] = payload
    _CACHE["timestamp"] = current_time
    return payload
