"""
Basic Telemetry & Cost Calculation Module

This module provides starter functions for tracking API cost, token counts, and execution latency.
You can replace or extend these functions to save logs directly into PostgreSQL.
"""

from typing import List
from datetime import datetime
import uuid

# Sample mock in-memory store for user requests (Replace with PostgreSQL query)
_MOCK_TELEMETRY_LOGS = []

# Model pricing rates per 1,000 tokens (USD)
PRICING_RATES = {
    "default": {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000},
    "gemini-flash": {"prompt": 0.00015 / 1000, "completion": 0.0006 / 1000},
    "gpt-4o-mini": {"prompt": 0.00015 / 1000, "completion": 0.0006 / 1000},
}


def calculate_api_cost(prompt_tokens: int, completion_tokens: int, model_name: str = "default") -> float:
    """
    Calculate estimated API cost in USD based on input and output token counts.
    
    Implementation Idea:
    Pass the token counts returned from your LangChain LLM response callback or API header
    to compute exact cost per user request.
    """
    rates = PRICING_RATES.get(model_name, PRICING_RATES["default"])
    cost = (prompt_tokens * rates["prompt"]) + (completion_tokens * rates["completion"])
    return round(cost, 6)


def record_request_telemetry(
    user_id: str,
    endpoint: str,
    execution_time_ms: float,
    prompt_tokens: int = 150,
    completion_tokens: int = 250,
    model_name: str = "default"
) -> dict:
    """
    Record a single request's telemetry log.
    
    Implementation Idea:
    Insert this payload into your PostgreSQL `api_request_logs` table.
    """
    total_tokens = prompt_tokens + completion_tokens
    cost = calculate_api_cost(prompt_tokens, completion_tokens, model_name)
    
    log_entry = {
        "request_id": str(uuid.uuid4()),
        "user_id": user_id,
        "endpoint": endpoint,
        "execution_time_ms": round(execution_time_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    _MOCK_TELEMETRY_LOGS.append(log_entry)
    return log_entry


def get_all_telemetry_logs() -> List[dict]:
    """Retrieve all recorded telemetry logs for the Admin dashboard."""
    return _MOCK_TELEMETRY_LOGS
