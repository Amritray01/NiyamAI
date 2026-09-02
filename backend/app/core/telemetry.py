"""
Telemetry & Cost Calculation Module

Provides utility functions for tracking API cost, token counts, and execution latency.
"""

from typing import List, Dict, Any
from datetime import datetime
import uuid

# In-memory session telemetry records
_TELEMETRY_LOGS: List[Dict[str, Any]] = []

# Model pricing rates per 1,000 tokens (USD)
PRICING_RATES = {
    "default": {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000},
    "gemini-flash": {"prompt": 0.00015 / 1000, "completion": 0.0006 / 1000},
    "gpt-4o-mini": {"prompt": 0.00015 / 1000, "completion": 0.0006 / 1000},
}


def calculate_api_cost(prompt_tokens: int, completion_tokens: int, model_name: str = "default") -> float:
    """Calculate estimated API cost in USD based on token counts."""
    rates = PRICING_RATES.get(model_name, PRICING_RATES["default"])
    cost = (prompt_tokens * rates["prompt"]) + (completion_tokens * rates["completion"])
    return round(cost, 6)


def record_request_telemetry(
    user_id: str,
    endpoint: str,
    execution_time_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model_name: str = "default"
) -> Dict[str, Any]:
    """Record a single request's telemetry metrics."""
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
    
    _TELEMETRY_LOGS.append(log_entry)
    return log_entry


def get_all_telemetry_logs() -> List[Dict[str, Any]]:
    """Retrieve recorded telemetry logs."""
    return _TELEMETRY_LOGS

