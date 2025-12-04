from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Message(BaseModel):
    role: str      # "user" hoặc "ai"
    content: str
    createdDate: Optional[Any] = None

class ChatRequest(BaseModel):
    text: str
    history: List[Message] = []

class ChatResponse(BaseModel):
    answer: str
    debug_info: Optional[Dict[str, Any]] = None