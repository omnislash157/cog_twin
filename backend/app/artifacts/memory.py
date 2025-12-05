"""
Memory artifacts for 3D visualization and sidebar cards.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class MemoryBubble(BaseModel):
    """3D memory node for Threlte visualization."""
    type: Literal["memory_bubble"] = "memory_bubble"
    id: str
    content_preview: str = Field(..., max_length=150)
    timestamp: datetime
    age_label: str  # "TODAY", "3 days ago", etc.
    
    # Visual properties
    source: Literal["vector", "episodic", "grep", "live"]
    relevance: float = Field(..., ge=0, le=1)
    size: float = Field(default=1.0, description="Bubble size based on relevance")
    
    # Position hints for 3D layout
    cluster_id: Optional[int] = None
    position_hint: Optional[dict] = None  # {x, y, z} suggestion
    
    # Interaction state
    selected: bool = False
    highlighted: bool = False


class MemoryCard(BaseModel):
    """Expanded memory artifact for sidebar."""
    type: Literal["memory_card"] = "memory_card"
    id: str
    title: str
    human_content: str
    assistant_content: str
    timestamp: datetime
    age_label: str
    source: Literal["vector", "episodic", "grep", "live"]
    relevance: float
    
    # Metadata
    tags: list[str] = []
    cluster_label: Optional[str] = None
    
    # Actions
    editable: bool = True
    synthesizable: bool = True


class RetrievalUpdate(BaseModel):
    """Pushed when retrieval completes."""
    type: Literal["retrieval_update"] = "retrieval_update"
    vector_count: int
    episodic_count: int
    grep_count: int
    live_count: int
    bubbles: list[MemoryBubble]
    retrieval_ms: Optional[float] = None  # Performance tracking