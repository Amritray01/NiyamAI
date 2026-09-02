"""
FSSAI Document Batch Ingestion CLI Script
"""

import os
import sys
import argparse

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.pipeline import FSSAIIngestionPipeline, BaselineIngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest FSSAI PDFs into ChromaDB and BM25 index.")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="./app/data/raw/pdfs",
        help="Directory containing official FSSAI PDFs"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["advanced", "baseline", "both"],
        default="both",
        help="Pipeline mode to execute (advanced, baseline, or both)"
    )

    args = parser.parse_args()

    pdf_dir = os.path.abspath(args.pdf_dir)
    if not os.path.exists(pdf_dir):
        print(f"[!] Directory '{pdf_dir}' does not exist.")
        return

    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.endswith(".pdf")
    ]

    print(f"[*] Found {len(pdf_files)} PDF files in {pdf_dir}:")
    for f in pdf_files:
        print(f"  - {os.path.basename(f)}")

    if args.mode in ("advanced", "both"):
        print("\n--- Running Advanced FSSAI Ingestion Pipeline ---")
        adv_pipeline = FSSAIIngestionPipeline(chroma_persist="./niyamai_chroma")
        adv_stats = adv_pipeline.ingest_batch(pdf_files)
        print(f"[+] Advanced Ingestion Summary: {adv_stats}")

    if args.mode in ("baseline", "both"):
        print("\n--- Running Baseline 1-Page Ingestion Pipeline ---")
        base_pipeline = BaselineIngestionPipeline(chroma_persist="./baseline_chroma")
        base_pipeline.ingest_batch(pdf_files)
        print(f"[+] Baseline 1-Page Ingestion Complete.")


if __name__ == "__main__":
    main()
