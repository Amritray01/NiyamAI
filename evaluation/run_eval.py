"""
NiyamAI Benchmark Evaluation Runner (Baseline vs Advanced FSSAI Pipeline)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List


# Add root project and backend directories to Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.rag.pipeline import FSSAIIngestionPipeline, BaselineIngestionPipeline
from app.rag.chains import run_fssai_rag_search
from evaluation.retrieval_eval import RetrievalEvaluator
from evaluation.deepeval_suite import run_deepeval_test_case
from evaluation.ragas_eval import GroqRagasJudge





def load_golden_dataset(path: str = "./evaluation/golden_dataset.jsonl") -> List[Dict[str, Any]]:
    dataset = []
    if not os.path.exists(path):
        print(f"[Eval Error] Dataset not found at: {path}")
        return dataset
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def run_benchmark():
    pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "data", "raw", "pdfs"))
    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.endswith(".pdf")
    ] if os.path.exists(pdf_dir) else []

    print(f"\n=======================================================")
    print(f"  NiyamAI RAG Evaluation & Benchmark Suite")
    print(f"=======================================================")
    print(f"Found {len(pdf_files)} PDFs in target folder: {pdf_dir}\n")

    # 1. Ingest into Advanced Pipeline
    print("[1/4] Ingesting PDFs into Advanced FSSAI Pipeline (Row Tables + Clauses + RRF)...")
    adv_pipeline = FSSAIIngestionPipeline(chroma_persist="./niyamai_chroma")
    adv_pipeline.ingest_batch(pdf_files)
    adv_retriever = adv_pipeline.get_retriever()

    # 2. Ingest into Baseline Pipeline
    print("\n[2/4] Ingesting PDFs into Baseline Pipeline (1-Page Chunker)...")
    base_pipeline = BaselineIngestionPipeline(chroma_persist="./baseline_chroma")
    base_pipeline.ingest_batch(pdf_files)
    base_retriever = base_pipeline.get_retriever()

    # 3. Load Golden Dataset
    print("\n[3/4] Loading Golden Dataset (25 Canonical Cases)...")
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} evaluation test cases.")

    evaluator = RetrievalEvaluator()
    groq_judge = GroqRagasJudge()
    results_rows = []

    cat_metrics_adv: Dict[str, Dict[str, float]] = {}
    cat_metrics_base: Dict[str, Dict[str, float]] = {}
    cat_counts: Dict[str, int] = {}

    print("\n[4/4] Executing Comparative Retrieval Evaluation + Groq LLM Judge...\n")

    for item in golden_dataset:
        qid = item["id"]
        qtext = item["question"]
        qtype = item["question_type"]
        ref_ans = item.get("reference_answer", "")
        expected_ids = item.get("expected_chunk_ids", [])
        required_facts = item.get("required_facts", [])

        # Query Advanced Hybrid Retriever & RAG Search Chain
        adv_res = adv_retriever.retrieve(qtext, top_k=5)
        adv_chunks = [{"chunk_id": r.chunk.chunk_id, "text": r.chunk.text} for r in adv_res]
        adv_metrics = evaluator.evaluate_query(adv_chunks, expected_ids, required_facts)

        # Generate full RAG Search Response
        rag_response = run_fssai_rag_search(qtext, category=item.get("metadata_filters", {}).get("category", "All"))
        generated_answer = f"{rag_response.summary}\n\n{rag_response.detailed_analysis}"

        # Query Baseline Retriever
        base_res = base_retriever.retrieve(qtext, top_k=5)
        base_chunks = [{"chunk_id": r.chunk.chunk_id, "text": r.chunk.text} for r in base_res]
        base_metrics = evaluator.evaluate_query(base_chunks, expected_ids, required_facts)

        # DeepEval metric execution
        adv_contexts = [r.chunk.text for r in adv_res]
        deepeval_res = run_deepeval_test_case(
            query=qtext,
            actual_output=generated_answer,
            retrieved_contexts=adv_contexts,
            expected_output=ref_ans,
        )

        # Groq LLM-as-a-Judge evaluation on full generated response
        judge_res = groq_judge.evaluate_response(
            query=qtext,
            retrieved_contexts=adv_contexts,
            generated_answer=generated_answer,
            reference_answer=ref_ans,
        )

        # Record CSV row
        row = {
            "id": qid,
            "question_type": qtype,
            "question": qtext,
            "adv_hit@5": adv_metrics["hit@5"],
            "adv_recall@5": adv_metrics["recall@5"],
            "adv_mrr": adv_metrics["mrr"],
            "adv_fact_coverage": adv_metrics["fact_coverage"],
            "base_hit@5": base_metrics["hit@5"],
            "base_fact_coverage": base_metrics["fact_coverage"],
            "deepeval_relevancy": deepeval_res.get("answer_relevancy", 0.0),
            "deepeval_faithfulness": deepeval_res.get("faithfulness", 0.0),
            "groq_faithfulness": judge_res.get("faithfulness", 0.0),
            "groq_relevance": judge_res.get("answer_relevance", 0.0),
            "groq_precision": judge_res.get("context_precision", 0.0),
            "groq_judge_score": judge_res.get("overall_score", 0.0),
            "groq_model": judge_res.get("model_used", "none"),
        }
        results_rows.append(row)
        time.sleep(2.5)  # Rate limiting throttle to prevent Groq API 429




        # Aggregate per-category
        cat_counts[qtype] = cat_counts.get(qtype, 0) + 1
        if qtype not in cat_metrics_adv:
            cat_metrics_adv[qtype] = {"hit@5": 0.0, "mrr": 0.0, "fact_coverage": 0.0}
            cat_metrics_base[qtype] = {"hit@5": 0.0, "mrr": 0.0, "fact_coverage": 0.0}

        cat_metrics_adv[qtype]["hit@5"] += adv_metrics["hit@5"]
        cat_metrics_adv[qtype]["mrr"] += adv_metrics["mrr"]
        cat_metrics_adv[qtype]["fact_coverage"] += adv_metrics["fact_coverage"]

        cat_metrics_base[qtype]["hit@5"] += base_metrics["hit@5"]
        cat_metrics_base[qtype]["mrr"] += base_metrics["mrr"]
        cat_metrics_base[qtype]["fact_coverage"] += base_metrics["fact_coverage"]

    # Calculate overall averages
    tot = len(golden_dataset) if golden_dataset else 1
    avg_adv_fact_coverage = sum(r["adv_fact_coverage"] for r in results_rows) / tot
    avg_base_fact_coverage = sum(r["base_fact_coverage"] for r in results_rows) / tot
    avg_adv_hit5 = sum(r["adv_hit@5"] for r in results_rows) / tot
    avg_base_hit5 = sum(r["base_hit@5"] for r in results_rows) / tot
    avg_adv_mrr = sum(r["adv_mrr"] for r in results_rows) / tot

    avg_groq_faithfulness = sum(r["groq_faithfulness"] for r in results_rows) / tot
    avg_groq_relevance = sum(r["groq_relevance"] for r in results_rows) / tot
    avg_groq_precision = sum(r["groq_precision"] for r in results_rows) / tot
    avg_groq_score = sum(r["groq_judge_score"] for r in results_rows) / tot

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "baseline_vs_advanced.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_rows[0].keys() if results_rows else [])
        writer.writeheader()
        writer.writerows(results_rows)

    category_breakdown = {}
    for cat in cat_counts:
        n = cat_counts[cat]
        category_breakdown[cat] = {
            "adv_hit@5": round(cat_metrics_adv[cat]["hit@5"] / n, 4),
            "adv_mrr": round(cat_metrics_adv[cat]["mrr"] / n, 4),
            "adv_fact_coverage": round(cat_metrics_adv[cat]["fact_coverage"] / n, 4),
            "base_hit@5": round(cat_metrics_base[cat]["hit@5"] / n, 4),
            "base_fact_coverage": round(cat_metrics_base[cat]["fact_coverage"] / n, 4),
        }

    summary = {
        "dataset_size": tot,
        "overall_metrics": {
            "advanced_pipeline": {
                "hit@5": round(avg_adv_hit5, 4),
                "mrr": round(avg_adv_mrr, 4),
                "required_fact_coverage": round(avg_adv_fact_coverage, 4),
            },
            "baseline_1page_pipeline": {
                "hit@5": round(avg_base_hit5, 4),
                "required_fact_coverage": round(avg_base_fact_coverage, 4),
            },
            "groq_llm_as_a_judge": {
                "active": True,
                "model_used": results_rows[0].get("groq_model", "openai/gpt-oss-20b") if results_rows else "none",
                "avg_faithfulness": round(avg_groq_faithfulness, 4),
                "avg_answer_relevance": round(avg_groq_relevance, 4),
                "avg_context_precision": round(avg_groq_precision, 4),
                "avg_overall_score": round(avg_groq_score, 4),
            },
            "fact_coverage_improvement_pct": round((avg_adv_fact_coverage - avg_base_fact_coverage) * 100, 2),
        },
        "category_breakdown": category_breakdown,
    }

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("  FINAL EVALUATION BENCHMARK RESULTS")
    print("=======================================================")
    print(f"Advanced FSSAI Pipeline Fact Coverage: {avg_adv_fact_coverage * 100:.2f}%")
    print(f"Baseline 1-Page Pipeline Fact Coverage: {avg_base_fact_coverage * 100:.2f}%")
    print(f"Advanced Pipeline MRR                : {avg_adv_mrr:.4f}")
    print(f"Advanced Pipeline Hit@5              : {avg_adv_hit5 * 100:.2f}%")
    print(f"Groq LLM Judge Faithfulness          : {avg_groq_faithfulness * 100:.2f}%")
    print(f"Groq LLM Judge Answer Relevance      : {avg_groq_relevance * 100:.2f}%")
    print(f"Groq LLM Judge Overall Score         : {avg_groq_score * 100:.2f}%\n")


    print(f"Results written to:")
    print(f"  - CSV : {csv_path}")
    print(f"  - JSON: {json_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    run_benchmark()
