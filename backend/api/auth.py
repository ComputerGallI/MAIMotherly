# backend/api/auth.py
import os
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import RedirectResponse

from services.gcal_oauth import (
    generate_auth_url,
    exchange_code_for_tokens,
    is_connected,
    revoke,
    get_user_profile,
)

router = APIRouter(prefix="/auth", tags=["auth"])

def _ensure_user_id(path_id: Optional[str], query_id: Optional[str]) -> str:
    uid = (path_id or query_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="Missing user_id.")
    return uid

def _final_frontend_redirect(user_id: str) -> str:
    base = os.getenv("FRONTEND_URL", "http://127.0.0.1:5500").rstrip("/")
    index_path = os.getenv("FRONTEND_INDEX_PATH", "").strip()
    if index_path and not index_path.startswith("/"):
        index_path = "/" + index_path
    path = index_path or "/"
    return f"{base}{path}?user_id={user_id}"

@router.get("/login/{user_id}")
async def auth_login_path(user_id: str):
    try:
        return {"auth_url": generate_auth_url(user_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start Google sign-in: {e}")

@router.get("/login")
async def auth_login_query(user_id: Optional[str] = Query(default=None)):
    uid = _ensure_user_id(None, user_id)
    try:
        return {"auth_url": generate_auth_url(uid)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start Google sign-in: {e}")

@router.get("/callback")
async def auth_callback(request: Request):
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state")
    try:
        user_id = exchange_code_for_tokens(code, state)
        return RedirectResponse(url=_final_frontend_redirect(user_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {e}")

@router.get("/status/{user_id}")
async def auth_status_path(user_id: str):
    return {"connected": is_connected(user_id)}

@router.get("/status")
async def auth_status_query(user_id: Optional[str] = Query(default=None)):
    uid = _ensure_user_id(None, user_id)
    return {"connected": is_connected(uid)}

@router.get("/me/{user_id}")
async def auth_me_path(user_id: str):
    return get_user_profile(user_id)

@router.get("/me")
async def auth_me_query(user_id: Optional[str] = Query(default=None)):
    uid = _ensure_user_id(None, user_id)
    return get_user_profile(uid)

@router.post("/revoke/{user_id}")
async def auth_revoke(user_id: str):
    return {"revoked": revoke(user_id)}
