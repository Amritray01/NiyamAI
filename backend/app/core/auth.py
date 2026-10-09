from typing import Any, Dict, Optional
from fastapi import Header


def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if authorization and authorization.startswith("Admin"):
        return {"user_id": "admin_01", "role": "admin", "email": "admin@niyam.ai"}
    return {"user_id": "user_demo", "role": "user", "email": "user@niyam.ai"}
