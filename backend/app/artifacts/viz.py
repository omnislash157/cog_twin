"""
Visualization artifacts for drift bar and trending topics.
"""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


class DriftData(BaseModel):
    """Drift bar visualization data."""
    type: Literal["drift_data"] = "drift_data"
    
    # Current position on spectrum
    position: float = Field(..., ge=0, le=1)  # 0 = left topic, 1 = right topic
    
    # Topic anchors
    left_topic: str
    right_topic: str
    
    # Historical drift points
    history: list[dict] = []  # [{timestamp, position}, ...]


class TrendingTopic(BaseModel):
    """Single trending topic."""
    type: Literal["trending_topic"] = "trending_topic"
    term: str
    count: int
    trend: Literal["rising", "stable", "falling"]
    recent_count: int  # Last 7 days
    
    # Visual
    bar_percent: float = Field(..., ge=0, le=1)


class TrendingUpdate(BaseModel):
    """Pushed trending topics update."""
    type: Literal["trending_update"] = "trending_update"
    topics: list[TrendingTopic]
    updated_at: datetime


class CognitiveStateUpdate(BaseModel):
    """Real-time cognitive state for dashboard."""
    type: Literal["cognitive_state"] = "cognitive_state"
    phase: str  # idle, focused, exploring, consolidating, drifting, crisis
    temperature: float = Field(..., ge=0, le=1)
    drift_detected: bool = False
    gap_count: int = 0
    session_message_count: int = 0