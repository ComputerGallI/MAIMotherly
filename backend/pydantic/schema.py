from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    user_input: str
    quiz_summary: Optional[str] = ""
    subscription_tier: Optional[str] = "free"

class ChatResponse(BaseModel):
    response: str
    suggestions: Optional[List[str]] = []
