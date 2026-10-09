from __future__ import annotations

import json
import os
import time
from typing import Generator, Optional

from openai import OpenAI

from app.core.config import settings
from app.rag.agents.regulatory_graph import run_agentic_fssai_rag, _verify_and_guard_numeric_precision
from app.rag.pipeline import FSSAIIngestionPipeline
from app.schemas.regulatory import GlobalCompareResponse, RegionalStandard, SearchResponse

_PIPELINE: Optional[FSSAIIngestionPipeline] = None


def get_retriever(use_chroma: bool = True):
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = FSSAIIngestionPipeline(use_chroma=use_chroma)
        pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "pdfs"))
        if os.path.exists(pdf_dir):
            pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
            _PIPELINE.ingest_batch(pdf_files)
    return _PIPELINE.get_retriever()


def run_fssai_rag_search(question: str, category: str = "All") -> SearchResponse:
    retriever = get_retriever()
    return run_agentic_fssai_rag(question=question, retriever=retriever, category=category)


from app.rag.synsets import FSSAISynsetRegistry


def stream_fssai_rag_search(question: str, category: str = "All") -> Generator[str, None, None]:
    start_time = time.time()
    retriever = get_retriever()
    filter_cat = category if category and category != "All" else None

    # Step 1: Synset Query Expansion & Hybrid Retrieval
    expanded_queries = FSSAISynsetRegistry.expand_query(question)
    all_results: dict = {}

    for q_text in expanded_queries[:3]:
        sub_results = retriever.retrieve(query=q_text, top_k=20, category=filter_cat, active_only=True, rerank=False)
        for r in sub_results:
            cid = r.chunk.chunk_id
            if cid not in all_results or r.rrf_score > all_results[cid].rrf_score:
                all_results[cid] = r

    # Step 1B: Fallback to Overall Agent Search across all categories if filtered results < 2
    if len(all_results) < 2 and filter_cat is not None:
        for q_text in expanded_queries[:3]:
            fallback_sub = retriever.retrieve(query=q_text, top_k=20, category=None, active_only=True, rerank=False)
            for r in fallback_sub:
                cid = r.chunk.chunk_id
                if cid not in all_results or r.rrf_score > all_results[cid].rrf_score:
                    all_results[cid] = r

    candidates = sorted(all_results.values(), key=lambda x: x.rrf_score, reverse=True)[:20]
    if len(candidates) > 1:
        results = retriever._reranker.rerank(question, candidates, top_k=7)
    else:
        results = candidates[:7]

    chunks = [r.chunk for r in results]

    citations = [
        {
            "document_name": str(r.chunk.source_doc or "FSSAI Regulation"),
            "section": str((r.chunk.metadata or {}).get("clause_path") or (r.chunk.metadata or {}).get("parameter") or "FSSAI Regulation Clause"),
            "page_number": int(r.chunk.page_numbers[0]) if (r.chunk.page_numbers and r.chunk.page_numbers[0] is not None) else 1,
            "verbatim_text": str(r.chunk.display_text or r.chunk.text or "")[:500],
        }
        for r in results
    ]

    # Grounded confidence calculation
    all_text = " ".join([f"{c.text} {c.embedding_text or ''}" for c in chunks]).lower()
    q_words = [w for w in question.lower().split() if len(w) > 3]
    term_coverage = (sum(1 for w in q_words if w in all_text) / len(q_words)) if q_words else 0.5
    top_r = results[0] if results else None
    dense_dist = top_r.dense_distance if (top_r and top_r.dense_distance is not None) else 0.40
    dense_sim = max(0.0, min(1.0, 1.0 - dense_dist))

    confidence_score = round(min(0.98, max(0.20, (dense_sim * 0.4) + (term_coverage * 0.45) + (min(5, len(results)) * 0.03))), 2)
    confidence_label = "High" if confidence_score >= 0.75 else "Medium" if confidence_score >= 0.50 else "Low"

    # Step 2: Yield initial metadata with citations immediately
    yield f"data: {json.dumps({'type': 'init', 'citations': citations, 'confidence_score': confidence_score, 'confidence_label': confidence_label})}\n\n"

    if not chunks:
        no_res_summary = f"No direct FSSAI statutory regulation found for '{question}' under category '{category}'."
        yield f"data: {json.dumps({'type': 'token', 'token': no_res_summary})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'summary': no_res_summary, 'detailed_analysis': 'Please refine your search query or check across All FSSAI categories.', 'execution_time_ms': round((time.time() - start_time) * 1000, 2)})}\n\n"
        return

    # Step 3: Stream LLM Generation
    context_str = "\n\n---\n\n".join([
        f"Document: {c.source_doc} (Page {c.page_numbers[0] if c.page_numbers else 1})\nContext: {c.text}"
        for c in chunks[:7]
    ])

    prompt = f"""You are NiyamAI, the premier FSSAI Statutory Compliance & Regulatory Intelligence AI for India.
Provide an exhaustive, legally authoritative, and technically comprehensive Explainable AI (xAI) compliance advisory tailored for Food Business Operators (FBOs), Quality Assurance Directors, and Regulatory Affairs Officers.

Answer strictly based on the retrieved statutory regulation contexts below.
Always preserve exact numeric limits (e.g. 0.02 mg/kg, 15 ppb, 20 percent, 5 g/100g), units, sampling plans (n, c, m, M), and statutory clause numbers without rounding.

User Question: {question}

Retrieved Statutory Regulations Context:
{context_str}

Structure your response into the following exact Markdown format:

## Short Summary
[1-2 authoritative, concise sentences providing the definitive statutory verdict, core threshold, or claim legality]

## Explainable AI (xAI) Statutory & Legal Document

### 1. Statutory Data & Limits Matrix
[Tabular or structured breakdown of exact parameters, contaminants, limits, test methods, or statutory clauses cited from the official Gazette]

### 2. Comprehensive Ingredient Classification (Allowed vs Prohibited)
[Detailed evaluation of ingredients. For sugar/claim queries, explicitly categorize:
- Prohibited Disqualifiers: Sucrose, Fructose, Glucose, Dextrose, Maltose, Invert Sugar, Honey, Jaggery, Syrups, Malt Extract, Sugar-containing Fruit Juice Concentrates.
- Sweeteners & Polyols: Sugar Alcohols (Sorbitol, Maltitol, Xylitol, Erythritol) added as sweetening agents.
- Permitted Processing Ingredients: Non-sweetening processing aids, unadulterated base foods, permitted natural flavors.]

### 3. Food Business Operator (FBO) Legal Liabilities & Compliance Risks
[Detailed legal risk analysis:
- Statutory liability under Section 24 (Misleading Advertisements) and Section 53 (Penalties up to ₹10 Lakhs) of the Food Safety and Standards Act, 2006.
- Misbranding, license suspension, and product recall liabilities under FSS Regulations.
- Mandatory disclaimer wording (e.g., 'CONTAINS NATURALLY OCCURRING SUGARS'), font height ratios, and front-of-pack display rules.]

### 4. Medical, Clinical & Public Health Rationale
[Clinical and nutritional science backing:
- Difference between intrinsic/cellular sugars and added/free extrinsic sugars.
- Glycemic response, insulin sensitivity, metabolic syndrome, and lifestyle disease prevention.]"""

    full_generated = ""
    groq_key = os.getenv("GROQ_API_KEY") or getattr(settings, "groq_api_key", "")
    google_key = os.getenv("GOOGLE_API_KEY") or getattr(settings, "google_api_key", "")

    if groq_key:
        for model_name in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
            try:
                client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1", timeout=30.0)
                stream_resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2200,
                    stream=True,
                )
                for chunk_obj in stream_resp:
                    delta = chunk_obj.choices[0].delta if chunk_obj.choices else None
                    token_content = delta.content if delta and delta.content else None
                    if token_content:
                        full_generated += token_content
                        yield f"data: {json.dumps({'type': 'token', 'token': token_content})}\n\n"
                if full_generated:
                    break
            except Exception:
                pass

    if not full_generated and google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_key, temperature=0.1, max_retries=1, timeout=10.0)
            for chunk_obj in llm.stream(prompt):
                if chunk_obj.content:
                    full_generated += chunk_obj.content
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk_obj.content})}\n\n"
        except Exception:
            pass

    if not full_generated:
        fallback_summary = f"Statutory FSSAI compliance provisions retrieved for '{question}'."
        analysis_bullets = [
            f"• [{c.source_doc} (Page {c.page_numbers[0] if c.page_numbers else 1})] {c.text[:220].replace(chr(10), ' ')}..."
            for c in chunks[:3]
        ]
        fallback_analysis = "\n\n".join(analysis_bullets)
        full_generated = f"## Short Summary\n{fallback_summary}\n\n## Explainable AI (xAI) Statutory & Legal Document\n{fallback_analysis}"
        for word in full_generated.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'token': word + ' '})}\n\n"

    # Step 4: Split into Short Summary vs Explainable AI (xAI) Document
    summary = ""
    detailed_analysis = ""

    split_markers = [
        "## Explainable AI (xAI) Statutory & Legal Document",
        "## Explainable AI",
        "## Detailed Analysis",
    ]
    split_done = False
    for marker in split_markers:
        if marker in full_generated:
            parts = full_generated.split(marker)
            summary_raw = parts[0].replace("## Short Summary", "").replace("## Executive Summary", "").strip()
            analysis_raw = parts[1].strip() if len(parts) > 1 else ""
            summary = summary_raw
            detailed_analysis = analysis_raw
            split_done = True
            break

    if not split_done:
        first_p = full_generated.split("\n\n")[0].strip()
        summary = first_p.replace("## Short Summary", "").replace("## Executive Summary", "").replace("##", "").strip()
        detailed_analysis = full_generated.replace(first_p, "").strip()

    summary, detailed_analysis, _ = _verify_and_guard_numeric_precision(summary, detailed_analysis, chunks)
    exec_time = round((time.time() - start_time) * 1000, 2)

    yield f"data: {json.dumps({'type': 'done', 'summary': summary, 'detailed_analysis': detailed_analysis, 'execution_time_ms': exec_time})}\n\n"


def run_global_comparison(parameter: str) -> GlobalCompareResponse:
    standards = [
        RegionalStandard(
            region="India",
            authority="FSSAI",
            status="Restricted",
            permitted_limit="Regulated under FSSAI Food Product Standards / Contaminants Regulations",
            citation="FSSAI Standards 2011/2023",
        ),
        RegionalStandard(
            region="United States",
            authority="US FDA",
            status="Permitted (GRAS)",
            permitted_limit="Allowed within FDA CFR Title 21 limits",
            citation="21 CFR 172",
        ),
        RegionalStandard(
            region="Europe",
            authority="EU EFSA",
            status="Strictly Banned",
            permitted_limit="Prohibited or restricted under EU Food Regulations",
            citation="EU Regulation 2022/617",
        ),
    ]

    return GlobalCompareResponse(
        parameter=parameter,
        key_takeaway=f"For '{parameter}', compliance requirements vary across India (FSSAI), US (FDA), and Europe (EFSA).",
        standards=standards,
    )
