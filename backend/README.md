# MAI Backend (FastAPI)

Summary of Backend

## Overview
FastAPI backend for **MAI** with:
- Empathetic chat generation (graceful fallbacks if models/indexes missing)
- Google OAuth login (profile/email) with **scope normalization**
- Google Calendar event creation with **custom reminder minutes**
- Open CORS for local HTML/JS/CSS frontends

## Project Layout
```
backend/
├─ api/
│  ├─ auth.py               # Google OAuth endpoints (gcal_oauth)
│  ├─ calendar.py           # Calendar endpoints
│  └─ chat.py               # Chat endpoints: /health, /generate, /debug/*
├─ artifacts/
│  ├─ knowledge_corpus.pkl  # REQUIRED (chat works with fallback if missing)
│  ├─ faiss_index.bin       # OPTIONAL (semantic search)
│  ├─ config.json           # OPTIONAL
│  ├─ retriever_model/      # OPTIONAL (SentenceTransformer)
│  ├─ generator_model/      # OPTIONAL (seq2seq model)
│  └─ generator_tokenizer/  # OPTIONAL
├─ modelsAI/
│  ├─ model_loader.py
│  └─ search_engine.py
├─ pydantic/
│  └─ schema.py
├─ services/
│  ├─ gcal_oauth.py
│  └─ google_calendar.py
├─ main.py
└─ requirements.txt
```

## Prerequisites
- Python 3.11+
- (Recommended) virtualenv

## Setup
### 1) Install deps
```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Artifacts
Place your corpus at:
```
backend/artifacts/knowledge_corpus.pkl
```
Optional accelerators auto-detected if present.

### 3) Environment (`backend/.env`)
Start from `.env.template` and fill in:
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI` → `http://127.0.0.1:8000/auth/callback`
- (Optional) `FRONTEND_URL` + `FRONTEND_INDEX_PATH` for redirect after OAuth
- (Optional) `GEMINI_API_KEY`

**Default scopes** (normalized internally): `openid userinfo.email userinfo.profile calendar.events`

### 4) Google Cloud Console
- OAuth 2.0 Web Client → **Authorized redirect URI**: `http://127.0.0.1:8000/auth/callback`
- Enable **Google Calendar API**
- OAuth consent screen: **Testing**, add your user as **Test user**

### 5) Run
```bash
#Clear Cache (Prevent so many bugs)
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
# Windows
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
# macOS/Linux
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
### 6) **Run the tests**
Make sure that the server is running before in another terminal.\
  On windows PowerShell:
  ```bash
  python -m pytest -q
  ```
  Other
  ```bash
  pytest -q
  ```

## Endpoints

### Chat
- `GET /health`
- `POST /generate`
```json
{
  "user_input": "I'm anxious about a presentation",
  "quiz_summary": "",
  "subscription_tier": "free"
}
```
Response:
```json
{
  "response": "I hear you—presentations can be stressful...",
  "suggestions": ["Try deep breathing", "Visualize the first 60s going well"]
}
```

### Auth
- `GET /auth/login?user_id=YOU` → returns `auth_url`
- `GET /auth/callback?...` → exchanges code and
  - redirects to your frontend (`FRONTEND_URL + FRONTEND_INDEX_PATH?user_id=...`) **if** FRONTEND_URL is set
  - otherwise returns JSON `{"ok":true,"user_id":"..."}`
- `GET /auth/status?user_id=YOU` → `{"connected": true|false}`
- `GET /auth/me?user_id=YOU` → `{"user_id","name","email","picture","connected"}`
- `POST /auth/revoke/YOU` → clears in-memory token

### Calendar
- `POST /calendar/events?user_id=YOU`
```json
{
  "summary": "MAI wellness check-in",
  "start_iso": "2025-09-01T10:00:00",
  "end_iso": "2025-09-01T10:30:00",
  "timezone": "America/Chicago",
  "description": "Breathing + 2 wins",
  "location": "",
  "attendees": ["friend@example.com"],
  "remind_minutes": 60
}
```
- `GET /calendar/upcoming?user_id=YOU&max_results=10`



## Frontend Integration (Local)

- Open `auth_url` from `/auth/login?user_id=YOU` in a new tab.
- After consent:
  - If `FRONTEND_URL` is set, you’ll be redirected back to your frontend with `?user_id=...`
  - Else the callback returns JSON; store the `user_id` in `localStorage`
- Show profile with `/auth/me` and create calendar events with `/calendar/events`.
- Chat via `/generate`.


## Troubleshooting

- **redirect_uri_mismatch (400):** Make sure the OAuth client has `http://127.0.0.1:8000/auth/callback`.
- **access_denied (403):** Add your Google account as a **Test user**.
- **Calendar API not enabled:** Enable it in **APIs & Services**.
- **Scope mismatch:** Restart backend, `POST /auth/revoke/YOU`, re-consent.
- **Chat 404:** Ensure `main.py` includes the chat router.

---
