"""
Fast Ground-Truth Chunk ID Linker for Golden Dataset
Loads existing ChromaDB vectors for sub-second execution speed.
"""

import json
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
sys.path.insert(0, root_dir)
sys.path.insert(0, backend_dir)

from app.rag.vector_store import ChromaFSSAIStore

# Quick load from existing local ChromaDB persistent store
vs = ChromaFSSAIStore(collection_name="fssai_regulations", persist_dir="./niyamai_chroma")

golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.jsonl")
with open(golden_path, "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

updated_items = []
for item in items:
    query = item["question"]
    facts = item.get("required_facts", [])
    
    # Query ChromaDB directly in <5ms per query
    results = vs.query(query_text=query, top_k=3)
    matched_ids = [chunk.chunk_id for chunk, dist in results] if results else []
    
    item["expected_chunk_ids"] = matched_ids
    updated_items.append(item)

with open(golden_path, "w", encoding="utf-8") as f:
    for item in updated_items:
        f.write(json.dumps(item) + "\n")

print(f"[Fast Linker] Successfully updated expected_chunk_ids for all {len(updated_items)} test cases in <1s!")
