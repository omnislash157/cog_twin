"""Artifact action schemas - the contract between LLM and frontend."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from enum import Enum


class ArtifactType(str, Enum):
    """Available artifact types in the bank."""
    MEMORY_CARD = "memory_card"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    CODE = "code"
    SYNTHESIS = "synthesis"
    LIST = "list"


class ArtifactAction(BaseModel):
    """Parsed from LLM output: [ARTIFACT type="..." ...]"""
    type: ArtifactType
    title: Optional[str] = None

    # For memory-based artifacts
    ids: Optional[List[str]] = Field(default=None, description="Memory IDs to display")
    query: Optional[str] = Field(default=None, description="Query to retrieve memories")

    # For timeline
    range: Optional[str] = Field(default=None, description="Time range like '30d', '6m'")

    # For code blocks
    lang: Optional[str] = Field(default="python", description="Language for syntax highlighting")
    code: Optional[str] = Field(default=None, description="The code content")

    # For lists
    items: Optional[List[str]] = Field(default=None, description="List items")


class ArtifactEmit(BaseModel):
    """WebSocket payload sent to frontend."""
    type: Literal["artifact_emit"] = "artifact_emit"
    artifact: ArtifactAction
    suggested: bool = Field(default=False, description="True if LLM added this unprompted")
