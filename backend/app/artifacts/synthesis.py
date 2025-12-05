"""
Synthesis artifacts for multi-memory combination.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class SynthesisRequest(BaseModel):
    """Request to synthesize multiple memories."""
    type: Literal["synthesis_request"] = "synthesis_request"
    memory_ids: list[str]
    prompt: Optional[str] = None  # Optional user guidance


class SynthesisResult(BaseModel):
    """Result of memory synthesis."""
    type: Literal["synthesis_result"] = "synthesis_result"
    id: str
    source_ids: list[str]  # Memory IDs that were synthesized
    title: str
    content: str
    insight: Optional[str] = None  # Key insight extracted
    created_at: datetime
    
    # Metadata
    confidence: float = Field(..., ge=0, le=1)
    token_count: Optional[int] = None


class InsightCard(BaseModel):
    """Proactive insight surfaced by the system."""
    type: Literal["insight_card"] = "insight_card"
    id: str
    title: str
    content: str
    relevance: float = Field(..., ge=0, le=1)
    source_memory_ids: list[str]
    created_at: datetime
    
    # Actions
    dismissable: bool = True
    expandable: bool = True