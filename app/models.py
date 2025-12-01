from pydantic import BaseModel
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    answer: str
    debug_info: Optional[Dict[str, Any]] = None