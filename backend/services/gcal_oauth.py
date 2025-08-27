# backend/services/gcal_oauth.py
import os
from typing import Dict, Optional, List, Any
import requests

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request

# ===== In-memory token/profile store (demo). Replace with DB/Redis in production. =====
_TOKEN_STORE: Dict[str, Dict[str, Any]] = {}

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USERINFO_URI = "https://openidconnect.googleapis.com/v1/userinfo"

# --- helpers ------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    v = os.getenv(key, default)
    if v is None:
        v = default
    # strip stray quotes/spaces
    return v.strip().strip('"').strip("'")

def _normalize_scopes(raw: str) -> List[str]:
    """
    Normalize scope strings so 'email/profile' == 'userinfo.email/userinfo.profile'.
    Accept comma or space separated values. Return a stable, deduped, sorted list.
    """
    # default set (canonical)
    default = (
        "openid "
        "https://www.googleapis.com/auth/userinfo.email "
        "https://www.googleapis.com/auth/userinfo.profile "
        "https://www.googleapis.com/auth/calendar.events"
    )
    text = raw or default

    # split on commas, then spaces
    parts: List[str] = []
    for chunk in text.split(","):
        parts += chunk.split()

    # mapping of synonyms -> canonical
    canon = {
        "email": "https://www.googleapis.com/auth/userinfo.email",
        "profile": "https://www.googleapis.com/auth/userinfo.profile",
        "userinfo.email": "https://www.googleapis.com/auth/userinfo.email",
        "userinfo.profile": "https://www.googleapis.com/auth/userinfo.profile",
        # keep these canonical as-is:
        "openid": "openid",
        "https://www.googleapis.com/auth/userinfo.email": "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile": "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/calendar.events": "https://www.googleapis.com/auth/calendar.events",
    }

    norm: List[str] = []
    for p in (s.strip() for s in parts if s.strip()):
        # collapse known synonyms
        norm.append(canon.get(p, p))

    # de-dup + stable order (sort for deterministic comparison inside google-auth)
    # Also, ensure required scopes are present if user omitted them in .env
    required = {
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/calendar.events",
    }
    uniq = set(norm) | required
    return sorted(uniq)

def _scopes() -> List[str]:
    return _normalize_scopes(_env("GOOGLE_SCOPES", ""))

def _client_config() -> dict:
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
    flow = Flow.from_client_config(_client_config(), scopes=_scopes())
    flow.redirect_uri = _env("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/callback")
    return flow

def generate_auth_url(user_id: str) -> str:
    flow = create_flow(state=user_id)
    # NOTE: keep prompt=consent so we can add/refresh scopes cleanly during dev
    auth_url, _state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=user_id,
    )
    return auth_url

def _fetch_userinfo(access_token: str) -> Dict[str, Any]:
    try:
        r = requests.get(
            USERINFO_URI,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if r.ok:
            return r.json()
        else:
            print(f"[gcal_oauth] userinfo error {r.status_code}: {r.text}")
            return {}
    except Exception as e:
        print(f"[gcal_oauth] userinfo request failed: {e}")
        return {}

def exchange_code_for_tokens(code: str, state: str) -> str:
    """
    Exchange 'code' for tokens, fetch profile, persist. The Flow is re-created
    with the same normalized scopes so scope comparison during fetch_token doesn't fail.
    """
    user_id = state
    flow = create_flow(state=user_id)
    flow.fetch_token(code=code)
    creds: Credentials = flow.credentials

    profile = _fetch_userinfo(creds.token)
    name = profile.get("name") or profile.get("given_name") or ""
    email = profile.get("email") or ""
    picture = profile.get("picture") or ""

    _TOKEN_STORE[user_id] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
        "expiry": creds.expiry.timestamp() if creds.expiry else None,
        "profile": {"name": name, "email": email, "picture": picture},
    }
    return user_id

def _to_credentials(token_payload: Dict) -> Credentials:
    return Credentials(
        token=token_payload.get("token"),
        refresh_token=token_payload.get("refresh_token"),
        token_uri=token_payload.get("token_uri", TOKEN_URI),
        client_id=token_payload.get("client_id", _env("GOOGLE_CLIENT_ID")),
        client_secret=token_payload.get("client_secret", _env("GOOGLE_CLIENT_SECRET")),
        scopes=token_payload.get("scopes", _scopes()),
    )

def get_user_credentials(user_id: str) -> Optional[Credentials]:
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
    creds = get_user_credentials(user_id)
    return bool(creds and creds.valid)

def get_user_profile(user_id: str) -> Dict[str, Any]:
    data = _TOKEN_STORE.get(user_id) or {}
    prof = data.get("profile") or {}
    return {
        "user_id": user_id,
        "name": prof.get("name"),
        "email": prof.get("email"),
        "picture": prof.get("picture"),
        "connected": is_connected(user_id),
    }

def revoke(user_id: str) -> bool:
    return _TOKEN_STORE.pop(user_id, None) is not None
