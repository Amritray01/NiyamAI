"""
FSSAI Document Ingestion Script Starter

Usage:
    python backend/scripts/ingest_fssai.py --pdf_dir ./data/fssai_pdfs

Official Sources to Download Real FSSAI Regulations:
1. FSS Act 2006: https://www.fssai.gov.in/cms/food-safety-and-standards-act-2006.php
2. Food Additives Regs 2011: https://www.fssai.gov.in/cms/food-safety-and-standards-regulations.php
3. Labelling & Display 2020: https://www.fssai.gov.in/cms/labelling-and-display-regulations-2020.php
4. Licensing & Registration 2011: https://www.fssai.gov.in/cms/licensing-and-registration-regulations-2011.php
"""

import os
import argparse
from typing import List, Dict


def parse_fssai_pdf(pdf_path: str) -> List[Dict]:
    """
    Starter function to extract text chunks with legal metadata (Section, Page, Regulation Title).
    
    Implementation Idea:
    - Use PyPDF2 or pdfplumber to iterate through pages.
    - Match section headers using regular expressions like `r"Section \d+(\.\d+)?"` or `r"Chapter [I|V|X]+"`
    - Construct chunk objects containing verbatim text and metadata.
    """
    print(f"[*] Parsing PDF document: {pdf_path}")
    chunks = []
    # Placeholder chunk structure
    chunks.append({
        "document_name": os.path.basename(pdf_path),
        "section": "Section 3.1",
        "page": 1,
        "verbatim_text": "Sample parsed legal clause text..."
    })
    return chunks


def index_chunks_into_qdrant(chunks: List[Dict]):
    """
    Starter function to embed text chunks and upload to Qdrant collection.
    
    Implementation Idea:
    - Embed text using SentenceTransformers or OpenAI embeddings.
    - Push points to Qdrant collection `fssai_regulations`.
    """
    print(f"[+] Indexing {len(chunks)} chunks into Qdrant vector collection...")


def main():
    parser = argparse.ArgumentParser(description="Ingest official FSSAI PDFs into Qdrant vector database.")
    parser.add_argument("--pdf_dir", type=str, default="./data/fssai_pdfs", help="Directory containing FSSAI PDFs")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_dir):
        print(f"[!] Directory '{args.pdf_dir}' does not exist. Create it and place real FSSAI PDFs inside.")
        return

    pdf_files = [os.path.join(args.pdf_dir, f) for f in os.listdir(args.pdf_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in {args.pdf_dir}")

    for pdf in pdf_files:
        chunks = parse_fssai_pdf(pdf)
        index_chunks_into_qdrant(chunks)


if __name__ == "__main__":
    main()
