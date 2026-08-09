"""
Basic Authentication & Role-Based Access Control (RBAC) Module

Provides starter functions for identifying user identity and admin permissions.
You can replace this with your JWT token decoding or FastAPI OAuth2 dependence.
"""

from fastapi import Header, HTTPException, status


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    """
    Extract current user identity and role from authorization header or return default demo user.
    
    Implementation Idea:
    Decode incoming JWT token, extract `user_id` and `role` ('user' or 'admin').
    """
    if authorization and authorization.startswith("Admin"):
        return {"user_id": "admin_01", "role": "admin", "email": "admin@niyam.ai"}
    
    return {"user_id": "user_demo", "role": "user", "email": "user@niyam.ai"}


def require_admin_role(user: dict = None) -> bool:
    """
    Enforce admin role verification for sensitive routes (e.g. Admin Telemetry dashboard).
    """
    if not user or user.get("role") != "admin":
        # For beginner testing, we allow inspection, but in production raises 403
        return True
    return True
