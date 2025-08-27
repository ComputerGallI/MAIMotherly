# backend/services/google_calendar.py
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Make google api client optional so this module always imports cleanly ---
try:
    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.errors import HttpError  # type: ignore
    _GOOGLE_API_OK = True
except Exception as _e:
    build = None  # type: ignore
    HttpError = Exception  # fallback
    _GOOGLE_API_OK = False
    _GOOGLE_API_ERR = _e

from .gcal_oauth import get_user_credentials  # OAuth layer (we created earlier)


def _calendar_service(user_id: str):
    """
    Return a Google Calendar service or None if not authorized or google libs missing.
    """
    if not _GOOGLE_API_OK:
        # Library not available; return None so callers can get a friendly error
        return None

    creds = get_user_credentials(user_id)
    if not creds:
        return None

    # cache_discovery=False avoids disk writes on some envs
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def create_event(
    user_id: str,
    summary: str,
    start_iso: str,
    end_iso: str,
    timezone: str = "UTC",
    description: str = "",
    location: str = "",
    attendees: Optional[List[str]] = None,
    remind_minutes: int = 1440  # 1 day
) -> Dict[str, Any]:
    """
    Create a calendar event with a default 1-day reminder (email + popup).
    start_iso / end_iso: ISO 8601 timestamps (e.g., '2025-08-27T10:00:00')
    timezone: IANA tz (e.g., 'America/Chicago')
    """
    if not _GOOGLE_API_OK:
        return {"error": f"google-api-python-client not available: {_GOOGLE_API_ERR}"}

    service = _calendar_service(user_id)
    if not service:
        return {"error": "Not authorized or Google API unavailable. Please connect Google first."}

    event_body: Dict[str, Any] = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start_iso, "timeZone": timezone},
        "end": {"dateTime": end_iso, "timeZone": timezone},
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": remind_minutes},
                {"method": "popup", "minutes": remind_minutes},
            ],
        },
    }

    if attendees:
        event_body["attendees"] = [{"email": a} for a in attendees if a]

    try:
        created = service.events().insert(calendarId="primary", body=event_body).execute()
        return {"ok": True, "event": created}
    except HttpError as e:  # type: ignore
        return {"error": f"Google API error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


def list_upcoming_events(user_id: str, max_results: int = 10) -> Dict[str, Any]:
    if not _GOOGLE_API_OK:
        return {"error": f"google-api-python-client not available: {_GOOGLE_API_ERR}"}

    service = _calendar_service(user_id)
    if not service:
        return {"error": "Not authorized or Google API unavailable. Please connect Google first."}

    now = datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time
    try:
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])
        return {"ok": True, "events": events}
    except HttpError as e:  # type: ignore
        return {"error": f"Google API error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}
