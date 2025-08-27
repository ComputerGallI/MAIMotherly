# backend/services/google_auth.py
import os
import time
from typing import Dict, Optional, Tuple, List

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request

# ===== In-memory token store (demo). Replace with DB/Redis in production. =====
_TOKEN_STORE: Dict[str, Dict] = {}  # user_id -> token dict

# ===== Config from environment =====
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")
SCOPES_RAW = os.getenv(
    "GOOGLE_CALENDAR_SCOPES",
    "https://www.googleapis.com/auth/calendar.events"
)

# Split scopes by comma or whitespace
SCOPES: List[str] = [s.strip() for part in SCOPES_RAW.split(",") for s in part.split() if s.strip()]

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"


def _client_config() -> dict:
    """Google OAuth client config structure for server-side web flow."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise RuntimeError("Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET")
    return {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "project_id": "mai-backend",
            "auth_uri": AUTH_URI,
            "token_uri": TOKEN_URI,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uris": [GOOGLE_REDIRECT_URI],
        }
    }


def create_flow(state: str) -> Flow:
    """Create OAuth flow bound to the redirect URI and state (use user_id as state)."""
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES)
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    # Note: we return flow; caller will call authorization_url(...) or fetch_token(...)
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
    flow.fetch_token(code=code)  # validates & contacts TOKEN_URI
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
        client_id=token_payload.get("client_id", GOOGLE_CLIENT_ID),
        client_secret=token_payload.get("client_secret", GOOGLE_CLIENT_SECRET),
        scopes=token_payload.get("scopes", SCOPES),
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
    # Refresh if expired/invalid
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Update store with new access token and expiry
                payload["token"] = creds.token
                payload["expiry"] = creds.expiry.timestamp() if creds.expiry else None
                _TOKEN_STORE[user_id] = payload
            except Exception as e:
                print(f"[google_auth] Token refresh failed for {user_id}: {e}")
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
