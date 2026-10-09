from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
load_dotenv(os.path.join(root_dir, ".env"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.config import settings


class GroqRagasJudge:
    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY") or getattr(settings, "groq_api_key", "")
        self.api_key = key
        self.client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1") if key else None
        self.candidate_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

    def evaluate_response(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        reference_answer: str = "",
    ) -> Dict[str, Any]:
        # Step 1: Calculate factual grounded confidence from retrieved documents
        all_context = " ".join(retrieved_contexts).lower()
        q_words = [w for w in query.lower().split() if len(w) > 3]
        term_matches = sum(1 for w in q_words if w in all_context)
        fact_density = (term_matches / len(q_words)) if q_words else 0.5
        grounded_confidence = round(min(0.98, max(0.20, (fact_density * 0.70) + (min(5, len(retrieved_contexts)) * 0.06))), 2)

        if not self.client:
            return {
                "groq_eval_active": False,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "ingredient_classification_accuracy": 0.0,
                "grounded_confidence": grounded_confidence,
                "overall_judge_score": 0.0,
                "overall_score": 0.0,
                "reasoning": "GROQ_API_KEY not configured or client unavailable. Zeroed score without synthetic fallback.",
            }

        context_str = "\n---\n".join(retrieved_contexts[:4])
        prompt = f"""You are an expert FSSAI legal compliance auditor evaluating an AI assistant's Long-Format Explainable AI (xAI) Regulatory Document against retrieved statutory regulations.

User Query: {query}

Retrieved Statutory Regulations Context:
{context_str}

Generated Long-Format Statutory Document:
{generated_answer}

Reference Answer (Golden Ground Truth):
{reference_answer or "N/A"}

Evaluate the generated response strictly against the retrieved context on a scale of 0.0 to 1.0 for:
1. "faithfulness": Does the response state ONLY facts, limits, ingredient classifications, and clauses supported by the retrieved context? (1.0 = zero hallucination, 0.0 = completely hallucinated)
2. "answer_relevance": Does the response exhaustively address the user's compliance question, FBO legal angle, and ingredient requirements? (1.0 = perfectly relevant & comprehensive)
3. "context_precision": How well do the retrieved contexts contain the required numerical thresholds and statutory provisions? (1.0 = highly precise)
4. "ingredient_classification_accuracy": Are prohibited vs permitted ingredients correctly distinguished based on statutory rules? (1.0 = flawless classification)

Return ONLY a raw JSON object with keys: "faithfulness", "answer_relevance", "context_precision", "ingredient_classification_accuracy", "overall_score", and "reasoning".
JSON:"""

        import time
        max_retries = 3

        for model_name in self.candidate_models:
            for attempt in range(max_retries):
                try:
                    res = self.client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=2048,
                    )
                    raw_text = res.choices[0].message.content.strip()
                    json_str = raw_text[raw_text.find("{") : raw_text.rfind("}") + 1]
                    data = json.loads(json_str)
                    data["groq_eval_active"] = True
                    data["model_used"] = model_name
                    data["grounded_confidence"] = grounded_confidence
                    return data
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1.0 * (2 ** attempt))
                    continue

        return {
            "groq_eval_active": False,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "ingredient_classification_accuracy": 0.0,
            "grounded_confidence": grounded_confidence,
            "overall_score": 0.0,
            "overall_judge_score": 0.0,
            "reasoning": "LLM evaluation failed after all exponential backoff retries. Strict 0.0 applied without random values.",
        }


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


def run_ragas_benchmark(
    max_samples: Optional[int] = None,
    use_chroma: bool = True,
) -> Dict[str, Any]:
    import csv
    import time
    from app.rag.chains import get_retriever, run_fssai_rag_search

    print("\n=======================================================", flush=True)
    print("   NiyamAI RAGAS & Generation Quality Audit Benchmark  ", flush=True)
    print("=======================================================", flush=True)

    judge = GroqRagasJudge()
    golden_dataset = load_golden_dataset()
    if max_samples:
        golden_dataset = golden_dataset[:max_samples]

    print(f"Active Judge Models : {judge.candidate_models}", flush=True)
    print(f"Total Test Cases    : {len(golden_dataset)} queries from golden dataset", flush=True)
    print(f"Vector Store Engine : {'ChromaDB (Local Test Store)' if use_chroma else 'Qdrant (Production Store)'}\n", flush=True)

    # Pre-warm retriever
    print("[1/2] Pre-warming pipeline & retriever...", flush=True)
    retriever = get_retriever(use_chroma=use_chroma)
    print("Retriever ready.\n", flush=True)

    print("[2/2] Running RAG Generation & LLM-as-a-Judge Evaluation...", flush=True)
    print("-" * 92, flush=True)
    print(f"{'ID':<6} | {'Faithful':<9} | {'Relevance':<10} | {'Precision':<10} | {'Score':<8} | {'Query Snippet':<40}", flush=True)
    print("-" * 92, flush=True)

    results_rows: List[Dict[str, Any]] = []
    cat_stats: Dict[str, Dict[str, float]] = {}
    cat_counts: Dict[str, int] = {}

    for idx, item in enumerate(golden_dataset, start=1):
        qid = item.get("id", f"Q{idx:03d}")
        qtext = item["question"]
        qtype = item.get("question_type", "general")
        ref_ans = item.get("reference_answer", "")
        cat_filter = item.get("metadata_filters", {}).get("category", "All")

        try:
            rag_res = run_fssai_rag_search(qtext, category=cat_filter)
            generated_answer = f"{rag_res.summary}\n\n{rag_res.detailed_analysis}"
            contexts = [c.verbatim_text for c in rag_res.citations if getattr(c, "verbatim_text", None)]
            if not contexts:
                retrieved_chunks = retriever.retrieve(qtext, top_k=4, category=cat_filter if cat_filter != "All" else None)
                contexts = [r.chunk.text for r in retrieved_chunks]
        except Exception as e:
            generated_answer = f"Error during generation: {str(e)}"
            contexts = []

        judge_eval = judge.evaluate_response(
            query=qtext,
            retrieved_contexts=contexts,
            generated_answer=generated_answer,
            reference_answer=ref_ans,
        )

        faith = float(judge_eval.get("faithfulness", 0.0))
        rel = float(judge_eval.get("answer_relevance", 0.0))
        prec = float(judge_eval.get("context_precision", 0.0))
        ingr = float(judge_eval.get("ingredient_classification_accuracy", 0.0))
        overall = float(judge_eval.get("overall_score", 0.0))
        model_used = judge_eval.get("model_used", "none")

        row = {
            "id": qid,
            "question_type": qtype,
            "question": qtext,
            "faithfulness": round(faith, 4),
            "answer_relevance": round(rel, 4),
            "context_precision": round(prec, 4),
            "ingredient_classification_accuracy": round(ingr, 4),
            "overall_score": round(overall, 4),
            "grounded_confidence": round(judge_eval.get("grounded_confidence", 0.0), 4),
            "model_used": model_used,
            "reasoning": judge_eval.get("reasoning", ""),
        }
        results_rows.append(row)

        cat_counts[qtype] = cat_counts.get(qtype, 0) + 1
        if qtype not in cat_stats:
            cat_stats[qtype] = {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0, "overall_score": 0.0}
        cat_stats[qtype]["faithfulness"] += faith
        cat_stats[qtype]["answer_relevance"] += rel
        cat_stats[qtype]["context_precision"] += prec
        cat_stats[qtype]["overall_score"] += overall

        snip = (qtext[:37] + "...") if len(qtext) > 40 else qtext
        print(
            f"{qid:<6} | {faith*100:>7.1f}% | {rel*100:>8.1f}% | {prec*100:>8.1f}% | {overall*100:>6.1f}% | {snip:<40}",
            flush=True,
        )
        time.sleep(1.2)  # Throttle to stay within Groq rate limits

    total_n = len(results_rows) if results_rows else 1
    avg_faith = sum(r["faithfulness"] for r in results_rows) / total_n
    avg_rel = sum(r["answer_relevance"] for r in results_rows) / total_n
    avg_prec = sum(r["context_precision"] for r in results_rows) / total_n
    avg_ingr = sum(r["ingredient_classification_accuracy"] for r in results_rows) / total_n
    avg_score = sum(r["overall_score"] for r in results_rows) / total_n

    cat_breakdown = {}
    for ct, cnt in cat_counts.items():
        cat_breakdown[ct] = {
            "avg_faithfulness": round(cat_stats[ct]["faithfulness"] / cnt, 4),
            "avg_answer_relevance": round(cat_stats[ct]["answer_relevance"] / cnt, 4),
            "avg_context_precision": round(cat_stats[ct]["context_precision"] / cnt, 4),
            "avg_overall_score": round(cat_stats[ct]["overall_score"] / cnt, 4),
            "sample_count": cnt,
        }

    out_dir = os.path.join(root_dir, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ragas_evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if results_rows:
            writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
            writer.writeheader()
            writer.writerows(results_rows)

    summary = {
        "dataset_size": total_n,
        "overall_metrics": {
            "groq_ragas_judge": {
                "active": True,
                "model_used": results_rows[0].get("model_used", "openai/gpt-oss-120b") if results_rows else "none",
                "avg_faithfulness": round(avg_faith, 4),
                "avg_answer_relevance": round(avg_rel, 4),
                "avg_context_precision": round(avg_prec, 4),
                "avg_ingredient_accuracy": round(avg_ingr, 4),
                "avg_overall_score": round(avg_score, 4),
            }
        },
        "category_breakdown": cat_breakdown,
    }

    json_path = os.path.join(out_dir, "ragas_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Sync into global summary.json as well
    global_summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(global_summary_path):
        try:
            with open(global_summary_path, "r", encoding="utf-8") as gf:
                gdata = json.load(gf)
            if "overall_metrics" in gdata:
                gdata["overall_metrics"]["groq_llm_as_a_judge"] = {
                    "active": True,
                    "model_used": summary["overall_metrics"]["groq_ragas_judge"]["model_used"],
                    "avg_faithfulness": round(avg_faith, 4),
                    "avg_answer_relevance": round(avg_rel, 4),
                    "avg_context_precision": round(avg_prec, 4),
                    "avg_overall_score": round(avg_score, 4),
                }
            with open(global_summary_path, "w", encoding="utf-8") as gf:
                json.dump(gdata, gf, indent=2)
        except Exception:
            pass

    print("-" * 92, flush=True)
    print("\n=======================================================", flush=True)
    print("           FINAL RAGAS EVALUATION METRICS              ", flush=True)
    print("=======================================================", flush=True)
    print(f"Faithfulness (Zero-Hallucination) : {avg_faith * 100:.2f}%", flush=True)
    print(f"Answer Relevance                  : {avg_rel * 100:.2f}%", flush=True)
    print(f"Context Precision                 : {avg_prec * 100:.2f}%", flush=True)
    print(f"Ingredient Classification Accuracy: {avg_ingr * 100:.2f}%", flush=True)
    print(f"Overall RAGAS Judge Score         : {avg_score * 100:.2f}%\n", flush=True)
    print(f"Detailed CSV Report : {csv_path}", flush=True)
    print(f"Summary JSON Report : {json_path}\n", flush=True)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NiyamAI RAGAS LLM-as-a-Judge Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of golden dataset queries to test")
    parser.add_argument("--use-qdrant", action="store_true", help="Use Qdrant store instead of Chroma")
    args = parser.parse_args()

    run_ragas_benchmark(max_samples=args.limit, use_chroma=not args.use_qdrant)
