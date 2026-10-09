from datetime import datetime
from typing import Any, Dict, List
import uuid

_TELEMETRY_LOGS: List[Dict[str, Any]] = []

PRICING_RATES: Dict[str, Dict[str, float]] = {
    "default": {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000},
    "gemini-flash": {"prompt": 0.00015 / 1000, "completion": 0.0006 / 1000},
    "gpt-4o-mini": {"prompt": 0.00015 / 1000, "completion": 0.0006 / 1000},
}


def calculate_api_cost(prompt_tokens: int, completion_tokens: int, model_name: str = "default") -> float:
    rates = PRICING_RATES.get(model_name, PRICING_RATES["default"])
    cost = (prompt_tokens * rates["prompt"]) + (completion_tokens * rates["completion"])
    return round(cost, 6)


def record_request_telemetry(
    user_id: str,
    endpoint: str,
    execution_time_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model_name: str = "default",
) -> Dict[str, Any]:
    total_tokens = prompt_tokens + completion_tokens
    cost = calculate_api_cost(prompt_tokens, completion_tokens, model_name)

    log_entry: Dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "user_id": user_id,
        "endpoint": endpoint,
        "execution_time_ms": round(execution_time_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
        "timestamp": datetime.utcnow().isoformat(),
    }
    _TELEMETRY_LOGS.append(log_entry)
    return log_entry


def get_all_telemetry_logs() -> List[Dict[str, Any]]:
    return _TELEMETRY_LOGS
