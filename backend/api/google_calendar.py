# backend/api/google_calendar.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Use the service wrapper, not a relative import
from services.google_calendar import create_event, list_upcoming_events

router = APIRouter(prefix="/calendar", tags=["calendar"])

class EventIn(BaseModel):
    user_id: str
    summary: str
    start_iso: str   # e.g., "2025-08-27T10:00:00"
    end_iso: str     # e.g., "2025-08-27T11:00:00"
    timezone: str = "UTC"
    description: str = ""
    location: str = ""
    attendees: Optional[List[str]] = None
    remind_minutes: int = 1440

@router.post("/create")
async def create_event_route(body: EventIn):
    res = create_event(
        user_id=body.user_id,
        summary=body.summary,
        start_iso=body.start_iso,
        end_iso=body.end_iso,
        timezone=body.timezone,
        description=body.description,
        location=body.location,
        attendees=body.attendees,
        remind_minutes=body.remind_minutes,
    )
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@router.get("/list/{user_id}")
async def list_events_route(user_id: str, max_results: int = 10):
    res = list_upcoming_events(user_id, max_results=max_results)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res
