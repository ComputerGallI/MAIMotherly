# backend/services/gcal_oauth.py
import os
from typing import Dict, Optional, List

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request

# ===== In-memory token store (demo). Replace with DB/Redis in production. =====
_TOKEN_STORE: Dict[str, Dict] = {}  # user_id -> token dict

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"

def _env(key: str, default: str = "") -> str:
    """Read an env var at call time, stripping quotes and whitespace."""
    v = os.getenv(key, default)
    if v is None:
        v = default
    return v.strip().strip('"').strip("'")

def _scopes() -> List[str]:
    raw = _env(
        "GOOGLE_CALENDAR_SCOPES",
        "https://www.googleapis.com/auth/calendar.events",
    )
    return [s.strip() for part in raw.split(",") for s in part.split() if s.strip()]

def _client_config() -> dict:
    """Google OAuth client config structure for server-side web flow."""
    client_id = _env("GOOGLE_CLIENT_ID")
    client_secret = _env("GOOGLE_CLIENT_SECRET")
    redirect_uri = _env("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/callback")

    if not client_id or not client_secret:
        raise RuntimeError("Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET")

    return {
        "web": {
            "client_id": client_id,
            "project_id": "mai-backend",
            "auth_uri": AUTH_URI,
            "token_uri": TOKEN_URI,
            "client_secret": client_secret,
            "redirect_uris": [redirect_uri],
        }
    }

def create_flow(state: str) -> Flow:
    """Create OAuth flow bound to the redirect URI and state (use user_id as state)."""
    flow = Flow.from_client_config(_client_config(), scopes=_scopes())
    flow.redirect_uri = _env("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/callback")
    return flow

def generate_auth_url(user_id: str) -> str:
    """
    Build the Google OAuth consent URL.
    Use user_id as 'state' to correlate the callback.
    """
    flow = create_flow(state=user_id)
    auth_url, _state = flow.authorization_url(
        access_type="offline",              # request refresh_token
        include_granted_scopes="true",
        prompt="consent",                   # force refresh_token on repeated consents
        state=user_id
    )
    return auth_url

def exchange_code_for_tokens(code: str, state: str) -> str:
    """
    Exchange 'code' for tokens and persist to the in-memory store.
    Returns the user_id (same as state).
    """
    user_id = state
    flow = create_flow(state=user_id)
    flow.fetch_token(code=code)
    creds: Credentials = flow.credentials

    # Store token payload
    _TOKEN_STORE[user_id] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
        "expiry": creds.expiry.timestamp() if creds.expiry else None,
    }
    return user_id

def _to_credentials(token_payload: Dict) -> Credentials:
    """Create Credentials object from stored payload."""
    return Credentials(
        token=token_payload.get("token"),
        refresh_token=token_payload.get("refresh_token"),
        token_uri=token_payload.get("token_uri", TOKEN_URI),
        client_id=token_payload.get("client_id", _env("GOOGLE_CLIENT_ID")),
        client_secret=token_payload.get("client_secret", _env("GOOGLE_CLIENT_SECRET")),
        scopes=token_payload.get("scopes", _scopes()),
    )

def get_user_credentials(user_id: str) -> Optional[Credentials]:
    """
    Return valid (refreshed if needed) credentials for user_id,
    or None if not authorized yet.
    """
    payload = _TOKEN_STORE.get(user_id)
    if not payload:
        return None

    creds = _to_credentials(payload)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                payload["token"] = creds.token
                payload["expiry"] = creds.expiry.timestamp() if creds.expiry else None
                _TOKEN_STORE[user_id] = payload
            except Exception as e:
                print(f"[gcal_oauth] Token refresh failed for {user_id}: {e}")
                return None
        else:
            return None
    return creds

def is_connected(user_id: str) -> bool:
    """Quick status check."""
    creds = get_user_credentials(user_id)
    return bool(creds and creds.valid)

def revoke(user_id: str) -> bool:
    """Delete tokens (demo)."""
    return _TOKEN_STORE.pop(user_id, None) is not None
