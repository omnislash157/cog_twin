"""
Artifact schemas for CogTwin UI.

Pydantic models that define the contract between backend and frontend.
Each artifact has a `type` field for frontend routing.
"""

from app.artifacts.memory import MemoryBubble, MemoryCard, RetrievalUpdate
from app.artifacts.chat import StreamChunk, ChatMessage, ScoreRequest, ScoreAck
from app.artifacts.viz import DriftData, TrendingTopic, TrendingUpdate, CognitiveStateUpdate
from app.artifacts.synthesis import SynthesisRequest, SynthesisResult, InsightCard
from .actions import ArtifactType, ArtifactAction, ArtifactEmit
from .parser import extract_artifacts, parse_artifact_tag

__all__ = [
    # Memory
    "MemoryBubble",
    "MemoryCard",
    "RetrievalUpdate",
    # Chat
    "StreamChunk",
    "ChatMessage",
    "ScoreRequest",
    "ScoreAck",
    # Visualization
    "DriftData",
    "TrendingTopic",
    "TrendingUpdate",
    "CognitiveStateUpdate",
    # Synthesis
    "SynthesisRequest",
    "SynthesisResult",
    "InsightCard",
    # Actions
    "ArtifactType",
    "ArtifactAction",
    "ArtifactEmit",
    "extract_artifacts",
    "parse_artifact_tag",
]