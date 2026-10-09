from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.rag.pipeline import FSSAIIngestionPipeline
from evaluation.staged_evaluator import StagedRAGEvaluator


def load_golden_dataset(path: str = "./evaluation/golden_dataset.jsonl") -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        alt_path = os.path.join(root_dir, "evaluation", "golden_dataset.jsonl")
        path = alt_path if os.path.exists(alt_path) else path

    if not os.path.exists(path):
        print(f"[Eval Error] Dataset not found at: {path}", flush=True)
        return dataset

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def run_staged_benchmark(
    inspect_sample_steps: bool = True,
    use_chroma: bool = True,
) -> Dict[str, Any]:
    pdf_dir = os.path.join(backend_dir, "app", "data", "raw", "pdfs")
    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.endswith(".pdf")
    ] if os.path.exists(pdf_dir) else []

    print("\n=======================================================", flush=True)
    print("   NiyamAI Modular Stage & Step Evaluation Benchmark   ", flush=True)
    print("=======================================================", flush=True)
    print(f"Target Documents : {len(pdf_files)} PDFs in {pdf_dir}", flush=True)
    print(f"Backend Store    : {'ChromaDB (Test Mode)' if use_chroma else 'Qdrant (Production Mode)'}", flush=True)

    print("\n[Step 1/3] Initializing Ingestion & Hybrid Indices...", flush=True)
    pipeline = FSSAIIngestionPipeline(use_chroma=use_chroma)
    pipeline.ingest_batch(pdf_files)
    retriever = pipeline.get_retriever()

    staged_evaluator = StagedRAGEvaluator(
        retriever=retriever,
        vector_store=pipeline._vector_store,
        bm25_index=pipeline._bm25,
    )

    golden_dataset = load_golden_dataset()
    print(f"[Step 2/3] Loaded {len(golden_dataset)} golden evaluation queries.", flush=True)

    if inspect_sample_steps and golden_dataset:
        print("\n--- Step-by-Step Lifecycle Inspection (Sample Query 1) ---", flush=True)
        sample_q = golden_dataset[0]["question"]
        steps = staged_evaluator.inspect_query_steps(sample_q, top_k=3)
        print(f"Query: '{sample_q}'", flush=True)
        print("  • Step 1 (Dense Retrieval)     :", [f"{c['chunk_id']} (dist: {c['dense_dist']})" for c in steps["step_1_dense"][:3]], flush=True)
        print("  • Step 2 (Sparse BM25)         :", [f"{c['chunk_id']} (score: {c['bm25_score']})" for c in steps["step_2_sparse_bm25"][:3]], flush=True)
        print("  • Step 3 (RRF Fusion)          :", [f"{c['chunk_id']} (rrf: {c['rrf_score']})" for c in steps["step_3_rrf_fusion"][:3]], flush=True)
        print("  • Step 4 (FlashRank Rerank)    :", [f"{c['chunk_id']} (score: {c['rerank_score']})" for c in steps["step_4_rerank"][:3]], flush=True)
        print("----------------------------------------------------------------\n", flush=True)

    print("[Step 3/3] Executing 4-Stage Benchmark across Golden Dataset...", flush=True)

    stages = [
        "stage_1_bm25_only",
        "stage_2_semantic_only",
        "stage_3_after_filter",
        "stage_4_hybrid_rerank",
    ]

    stage_totals: Dict[str, Dict[str, float]] = {s: {} for s in stages}
    per_query_rows: List[Dict[str, Any]] = []

    for item in golden_dataset:
        qid = item["id"]
        qtext = item["question"]
        qtype = item["question_type"]
        expected_ids = item.get("expected_chunk_ids", [])
        required_facts = item.get("required_facts", [])
        cat_filter = item.get("metadata_filters", {}).get("category")

        stage_results = staged_evaluator.evaluate_stages(
            query=qtext,
            expected_chunk_ids=expected_ids,
            required_facts=required_facts,
            category_filter=cat_filter,
            top_k=5,
        )

        row: Dict[str, Any] = {"id": qid, "question_type": qtype, "question": qtext}

        for s in stages:
            m = stage_results[s]
            for metric_name, val in m.items():
                stage_totals[s][metric_name] = stage_totals[s].get(metric_name, 0.0) + val
                row[f"{s}_{metric_name}"] = round(val, 4)

        per_query_rows.append(row)

    tot_queries = len(golden_dataset) if golden_dataset else 1
    stage_averages: Dict[str, Dict[str, float]] = {}
    for s in stages:
        stage_averages[s] = {
            metric_name: round(val / tot_queries, 4)
            for metric_name, val in stage_totals[s].items()
        }

    out_dir = os.path.join(root_dir, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "staged_evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if per_query_rows:
            writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_query_rows)

    json_path = os.path.join(out_dir, "staged_summary.json")
    summary = {
        "dataset_size": tot_queries,
        "metrics_description": "Evaluated Hit@1, Hit@3, Hit@5, MRR, Recall@5, Fact Coverage (No nDCG)",
        "stage_progression": stage_averages,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=========================================================================================", flush=True)
    print("                    PROGRESSION ACROSS ARCHITECTURAL STAGES                              ", flush=True)
    print("=========================================================================================", flush=True)
    print(f"{'Stage Name':<26} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'Fact Cover':<10}", flush=True)
    print("-" * 89, flush=True)
    for s in stages:
        avg = stage_averages[s]
        stage_label = s.replace("stage_", "Stage ").replace("_", " ").title()
        print(
            f"{stage_label:<26} | {avg['hit@1']*100:>6.1f}% | {avg['hit@3']*100:>6.1f}% | "
            f"{avg['hit@5']*100:>6.1f}% | {avg['mrr']:>8.4f} | {avg['fact_coverage']*100:>8.1f}%",
            flush=True,
        )
    print("=========================================================================================\n", flush=True)
    print(f"Detailed CSV Report : {csv_path}", flush=True)
    print(f"Summary JSON Report : {json_path}\n", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NiyamAI Modular Stage & Step RAG Evaluation")
    parser.add_argument("--use-qdrant", action="store_true", help="Use Qdrant production store instead of Chroma")
    parser.add_argument("--no-steps", action="store_true", help="Disable step lifecycle preview")
    args = parser.parse_args()

    run_staged_benchmark(
        inspect_sample_steps=not args.no_steps,
        use_chroma=not args.use_qdrant,
    )
