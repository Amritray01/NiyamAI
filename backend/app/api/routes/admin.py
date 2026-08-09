from fastapi import APIRouter, Depends
from app.schemas.regulatory import AdminMetricsResponse, TelemetryLog
from app.core.telemetry import get_all_telemetry_logs
from app.core.auth import get_current_user

router = APIRouter(tags=["admin"])


@router.get("/admin/metrics", response_model=AdminMetricsResponse)
def get_admin_metrics(current_user: dict = Depends(get_current_user)) -> AdminMetricsResponse:
    """
    Admin Telemetry Dashboard Endpoint.
    Returns request count, aggregated cost in USD, response latency, and logs.
    """
    raw_logs = get_all_telemetry_logs()
    
    # Format logs into Pydantic models
    logs = [TelemetryLog(**log) for log in raw_logs]
    
    total_requests = len(logs)
    total_cost = sum(log.estimated_cost_usd for log in logs)
    avg_latency = (sum(log.execution_time_ms for log in logs) / total_requests) if total_requests > 0 else 0.0
    
    return AdminMetricsResponse(
        total_requests=total_requests,
        total_cost_usd=round(total_cost, 6),
        avg_response_time_ms=round(avg_latency, 2),
        active_users_count=len(set(log.user_id for log in logs)),
        logs=logs
    )
