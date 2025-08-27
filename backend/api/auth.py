# backend/api/auth.py
import os
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse
from services.gcal_oauth import (
    generate_auth_url,
    exchange_code_for_tokens,
    is_connected,
    revoke,
)

router = APIRouter(prefix="/auth", tags=["auth"])

def _ensure_user_id(path_id: Optional[str], query_id: Optional[str]) -> str:
    uid = (path_id or query_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="Missing user_id.")
    return uid

def _final_frontend_redirect(user_id: str) -> str:
    """
    Builds a robust redirect target that works whether your app is served as:
      - http://127.0.0.1:5500/                  (index at "/")
      - http://127.0.0.1:5500/index.html        (explicit index)
      - http://127.0.0.1:5500/frontend/index.html (served from repo root)
    Configure via:
      FRONTEND_URL         -> base origin + optional subpath (e.g., http://127.0.0.1:5500 or http://127.0.0.1:5500/frontend)
      FRONTEND_INDEX_PATH  -> "" (default, means "/"), or "/index.html"
                              or "/frontend/index.html" if needed.
    """
    base = os.getenv("FRONTEND_URL", "http://127.0.0.1:5500").rstrip("/")
    index_path = os.getenv("FRONTEND_INDEX_PATH", "").strip()  # "", "/index.html", "/frontend/index.html", etc.

    # Normalize index path
    if index_path and not index_path.startswith("/"):
        index_path = "/" + index_path

    # If no index path provided, redirect to "/" (most static servers map this to index.html)
    path = index_path or "/"
    return f"{base}{path}?user_id={user_id}"

# --- Login: path or query ---
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

# --- Callback: store tokens, then redirect to your frontend ---
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

# --- Status: path or query ---
@router.get("/status/{user_id}")
async def auth_status_path(user_id: str):
    return {"connected": is_connected(user_id)}

@router.get("/status")
async def auth_status_query(user_id: Optional[str] = Query(default=None)):
    uid = _ensure_user_id(None, user_id)
    return {"connected": is_connected(uid)}

@router.post("/revoke/{user_id}")
async def auth_revoke(user_id: str):
    return {"revoked": revoke(user_id)}
