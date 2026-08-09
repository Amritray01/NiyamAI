"""
Vector Store Adapter (Qdrant Cloud)

Provides connection helper for Qdrant Cloud and simple search starter function stubs.
"""

from typing import List, Dict, Any
from app.core.config import settings


def get_qdrant_client():
    """
    Initialize and return QdrantClient instance for Qdrant Cloud.
    
    Implementation Idea:
    from qdrant_client import QdrantClient
    
    if settings.qdrant_api_key:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantClient(url=settings.qdrant_url)
    """
    return None


def search_fssai_vector_store(query: str, category: str = "All", top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Basic vector search starter stub supporting category filtering and top_k parameters.
    
    Implementation Idea for Qdrant Cloud:
    1. Generate embedding vector for `query`.
    2. Build optional Qdrant Filter for `category`.
    3. Query collection `settings.qdrant_collection` on Qdrant Cloud with limit `top_k`.
    """
    # Basic fallback starter structure until Qdrant Cloud collection is populated
    return [
        {
            "document_name": "FSS (Food Additives) Regulations 2011",
            "section": "Section 3.1.2 - Preservatives in Beverages",
            "page": 14,
            "verbatim_text": "Sodium Benzoate limit in synthetic syrups and fruit squashes shall not exceed 600 ppm."
        },
        {
            "document_name": "FSS (Labelling & Display) Regulations 2020",
            "section": "Clause 5(2) - Nutritional Information",
            "page": 8,
            "verbatim_text": "Every prepackaged food shall display energy value, protein, carbohydrate, total sugars, and added sugars per 100g or per serve."
        }
    ]
