"""
Chat artifacts for streaming responses and scoring.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class StreamChunk(BaseModel):
    """Single chunk of streaming response."""
    type: Literal["stream_chunk"] = "stream_chunk"
    content: str
    done: bool = False


class ChatMessage(BaseModel):
    """Complete chat message."""
    type: Literal["chat_message"] = "chat_message"
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    
    # Assistant-specific
    trace_id: Optional[str] = None
    tokens_used: Optional[int] = None
    confidence: Optional[float] = None
    
    # Retrieval metadata (for assistant messages)
    retrieval_summary: Optional[dict] = None  # {vector: 5, episodic: 3, ...}


class ScoreRequest(BaseModel):
    """User scoring a response."""
    type: Literal["score_request"] = "score_request"
    trace_id: str
    accuracy: int = Field(..., ge=1, le=10)
    temporal: int = Field(..., ge=1, le=10)
    tone: int = Field(..., ge=1, le=10)
    feedback: Optional[str] = None


class ScoreAck(BaseModel):
    """Acknowledgment of score submission."""
    type: Literal["score_ack"] = "score_ack"
    trace_id: str
    overall: float = Field(..., ge=0, le=1)  # Computed weighted average
    saved: bool = True