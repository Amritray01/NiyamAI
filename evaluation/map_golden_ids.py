import json
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.rag.pipeline import FSSAIIngestionPipeline

pipeline = FSSAIIngestionPipeline(use_chroma=True)
pdf_dir = os.path.join(backend_dir, "app", "data", "raw", "pdfs")
pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
pipeline.ingest_batch(pdf_files)
all_chunks = pipeline._all_chunks

golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.jsonl")
with open(golden_path, "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

updated_items = []
for item in items:
    qtext = item["question"].lower()
    src_docs = item.get("source_documents", [])
    facts = [f.lower() for f in item.get("required_facts", [])]

    matched_ids = []
    for c in all_chunks:
        doc_match = any(d.lower() in c.source_doc.lower() for d in src_docs) if src_docs else True
        c_text = f"{c.text} {c.embedding_text or ''}".lower()
        fact_matches = sum(1 for f in facts if f in c_text)
        query_word_matches = sum(1 for w in qtext.split() if len(w) > 3 and w in c_text)

        score = (fact_matches * 3) + query_word_matches
        if doc_match and score > 0:
            matched_ids.append((c.chunk_id, score))

    matched_ids.sort(key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in matched_ids[:3]]

    if not top_ids:
        for c in all_chunks:
            c_text = f"{c.text} {c.embedding_text or ''}".lower()
            q_matches = sum(1 for w in qtext.split() if len(w) > 3 and w in c_text)
            if q_matches >= 2:
                top_ids.append(c.chunk_id)
                if len(top_ids) >= 2:
                    break

    item["expected_chunk_ids"] = top_ids
    updated_items.append(item)

with open(golden_path, "w", encoding="utf-8") as f:
    for item in updated_items:
        f.write(json.dumps(item) + "\n")

mapped_count = sum(1 for item in updated_items if len(item["expected_chunk_ids"]) > 0)
print(f"[Ground-Truth Linker] Successfully mapped {mapped_count}/{len(updated_items)} test cases with verified chunk IDs.")
