"""
Memory Schemas - Dual Pipeline Data Models

Two distinct memory types:
1. MemoryNode: 1:1 Q/A pairs, clustered, process-focused (what/how)
2. EpisodicMemory: Full conversations, preserved whole (why/when)

The Venom principle: "We" not "I" - these memories merge with the user's
cognition to form a unified external brain.

Version: 1.0.0 (cog_twin)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class Source(str, Enum):
    """Chat export providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROK = "grok"
    GEMINI = "gemini"


class Role(str, Enum):
    """Message author role."""
    HUMAN = "human"
    ASSISTANT = "assistant"


class IntentType(str, Enum):
    """Detected intent from heuristics."""
    QUESTION = "question"
    REQUEST = "request"
    STATEMENT = "statement"
    COMPLAINT = "complaint"
    CELEBRATION = "celebration"


class Complexity(str, Enum):
    """Content complexity level."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ConversationMode(str, Enum):
    """Detected conversation mode."""
    CHAT = "chat"
    DEBUG = "debug"
    BRAINSTORM = "brainstorm"
    REVIEW = "review"


class Urgency(str, Enum):
    """Urgency level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EmotionalValence(str, Enum):
    """Emotional tone."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY NODE (Process Memory - What/How)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryNode:
    """
    Single Q/A exchange - the atomic unit of process memory.

    These get clustered by HDBSCAN to form process patterns.
    50% noise removal is a FEATURE - we want signal, not chatter.

    Retrieval: NumPy cosine similarity (pure geometry, fast)
    """
    # ─── Identity ─────────────────────────────────────────────────────
    id: str                          # UUID or hash
    conversation_id: str             # Parent conversation
    sequence_index: int              # Position in conversation

    # ─── Content ──────────────────────────────────────────────────────
    human_content: str               # The question/request
    assistant_content: str           # The response

    # ─── Source & Time ────────────────────────────────────────────────
    source: Source                   # Provider
    created_at: datetime             # When this exchange happened

    # ─── Embedding ────────────────────────────────────────────────────
    embedding: Optional[List[float]] = None  # BGE-M3 1024-dim vector

    # ─── Heuristic Signals (computed at ingest, zero LLM) ─────────────
    intent_type: IntentType = IntentType.STATEMENT
    complexity: Complexity = Complexity.SIMPLE
    technical_depth: int = 0         # 0-10
    emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL
    urgency: Urgency = Urgency.LOW
    conversation_mode: ConversationMode = ConversationMode.CHAT
    action_required: bool = False
    has_code: bool = False
    has_error: bool = False

    # ─── Dynamic Tags (emergent from data) ────────────────────────────
    tags: Dict[str, List[str]] = field(default_factory=lambda: {
        "domains": [],       # Cluster-derived domains
        "topics": [],        # Extracted/clustered topics
        "entities": [],      # NER-extracted entities
        "processes": [],     # Action patterns (debugging, designing, etc)
    })

    # ─── Cluster Metadata (assigned by HDBSCAN) ───────────────────────
    cluster_id: Optional[int] = None          # -1 = noise (filtered out)
    cluster_label: Optional[str] = None       # Human-readable label
    cluster_confidence: float = 0.0           # Membership probability

    # ─── Retrieval Metadata ───────────────────────────────────────────
    access_count: int = 0            # How often retrieved
    last_accessed: Optional[datetime] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            content_hash = hashlib.sha256(
                f"{self.conversation_id}:{self.sequence_index}:{self.human_content[:100]}".encode()
            ).hexdigest()[:16]
            self.id = f"mem_{content_hash}"

    @property
    def combined_content(self) -> str:
        """Full content for embedding."""
        return f"Human: {self.human_content}\n\nAssistant: {self.assistant_content}"

    @property
    def is_noise(self) -> bool:
        """Check if marked as noise by HDBSCAN."""
        return self.cluster_id == -1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "sequence_index": self.sequence_index,
            "human_content": self.human_content,
            "assistant_content": self.assistant_content,
            "source": self.source.value,
            "created_at": self.created_at.isoformat(),
            "intent_type": self.intent_type.value,
            "complexity": self.complexity.value,
            "technical_depth": self.technical_depth,
            "emotional_valence": self.emotional_valence.value,
            "urgency": self.urgency.value,
            "conversation_mode": self.conversation_mode.value,
            "action_required": self.action_required,
            "has_code": self.has_code,
            "has_error": self.has_error,
            "tags": self.tags,
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "cluster_confidence": self.cluster_confidence,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        """Deserialize from storage."""
        data["source"] = Source(data["source"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["intent_type"] = IntentType(data["intent_type"])
        data["complexity"] = Complexity(data["complexity"])
        data["emotional_valence"] = EmotionalValence(data["emotional_valence"])
        data["urgency"] = Urgency(data["urgency"])
        data["conversation_mode"] = ConversationMode(data["conversation_mode"])
        if data.get("last_accessed"):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        # Don't load embedding from dict - loaded separately from numpy
        data.pop("embedding", None)
        return cls(**data)


# ═══════════════════════════════════════════════════════════════════════════
# EPISODIC MEMORY (Context Memory - Why/When)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodicMemory:
    """
    Full conversation preserved - the narrative memory.

    No clustering, no noise removal - the WHOLE context matters.
    "We were working on X but had to stop because Y"

    Retrieval: FAISS + heuristic pre-filter + LLM final pass
    """
    # ─── Identity ─────────────────────────────────────────────────────
    id: str                          # Original conversation ID
    title: str                       # Conversation title

    # ─── Content ──────────────────────────────────────────────────────
    messages: List[Dict[str, Any]]   # Full message array from parser
    message_count: int               # Quick access

    # ─── Source & Time ────────────────────────────────────────────────
    source: Source
    created_at: datetime             # First message
    updated_at: datetime             # Last message
    duration_minutes: float          # Conversation length

    # ─── Embedding ────────────────────────────────────────────────────
    embedding: Optional[List[float]] = None  # Summary embedding

    # ─── LLM-Generated Tags (cheap: ~$0.30 per 2200 convos) ───────────
    llm_tags: Dict[str, Any] = field(default_factory=lambda: {
        "summary": "",               # One-line summary
        "primary_intent": "",        # What were we trying to do
        "outcome": "",               # resolved/abandoned/ongoing/paused
        "interruption_reason": "",   # Why we stopped (if applicable)
        "key_entities": [],          # People, projects, concepts
        "emotional_arc": "",         # start→end emotional journey
        "domains": [],               # What domains touched
        "time_context": "",          # "late night session", "quick check"
    })

    # ─── Heuristic Aggregates (computed from nodes) ───────────────────
    dominant_intent: IntentType = IntentType.STATEMENT
    avg_complexity: float = 0.0
    avg_technical_depth: float = 0.0
    emotional_arc: str = "neutral→neutral"  # start→end
    has_code: bool = False
    has_errors: bool = False
    urgency_max: Urgency = Urgency.LOW

    # ─── Cross-References ─────────────────────────────────────────────
    memory_node_ids: List[str] = field(default_factory=list)  # Child nodes
    related_episode_ids: List[str] = field(default_factory=list)  # Linked convos

    # ─── Retrieval Metadata ───────────────────────────────────────────
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def __post_init__(self):
        """Compute derived fields."""
        if not self.id:
            self.id = f"ep_{hashlib.sha256(self.title.encode()).hexdigest()[:16]}"

        if self.messages and not self.duration_minutes:
            # Estimate duration from message timestamps if available
            pass

    @property
    def full_text(self) -> str:
        """Concatenated conversation for embedding."""
        parts = []
        for msg in self.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(parts)

    @property
    def summary_text(self) -> str:
        """Short text for quick display."""
        summary = self.llm_tags.get("summary", "")
        if summary:
            return summary
        # Fallback to title + first message
        if self.messages:
            first = self.messages[0].get("content", "")[:200]
            return f"{self.title}: {first}..."
        return self.title

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "title": self.title,
            "messages": self.messages,
            "message_count": self.message_count,
            "source": self.source.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "duration_minutes": self.duration_minutes,
            "llm_tags": self.llm_tags,
            "dominant_intent": self.dominant_intent.value,
            "avg_complexity": self.avg_complexity,
            "avg_technical_depth": self.avg_technical_depth,
            "emotional_arc": self.emotional_arc,
            "has_code": self.has_code,
            "has_errors": self.has_errors,
            "urgency_max": self.urgency_max.value,
            "memory_node_ids": self.memory_node_ids,
            "related_episode_ids": self.related_episode_ids,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        """Deserialize from storage."""
        data["source"] = Source(data["source"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["dominant_intent"] = IntentType(data["dominant_intent"])
        data["urgency_max"] = Urgency(data["urgency_max"])
        if data.get("last_accessed"):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data.pop("embedding", None)
        return cls(**data)


# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVAL RESULT (Unified output from both pipelines)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """
    Combined result from dual-pipeline retrieval.

    Process memories tell us WHAT and HOW.
    Episodic memories tell us WHY and WHEN.
    Together: "We were working on X using Y approach because Z context."
    """
    query: str

    # Process memories (what/how)
    process_memories: List[MemoryNode] = field(default_factory=list)
    process_scores: List[float] = field(default_factory=list)

    # Episodic memories (why/when)
    episodic_memories: List[EpisodicMemory] = field(default_factory=list)
    episodic_scores: List[float] = field(default_factory=list)

    # Merged context for answer builder
    merged_context: str = ""

    # Metadata
    retrieval_time_ms: float = 0.0
    process_candidates_scanned: int = 0
    episodic_candidates_scanned: int = 0

    def build_venom_context(self) -> str:
        """
        Build context string in Venom voice (we/us/our).

        This is what gets passed to the answer builder LLM.
        """
        parts = []

        # Process memories - the HOW
        if self.process_memories:
            parts.append("=== WHAT WE KNOW (Process Memory) ===")
            for mem, score in zip(self.process_memories[:5], self.process_scores[:5]):
                cluster_info = f"[{mem.cluster_label}]" if mem.cluster_label else ""
                parts.append(f"\n{cluster_info} (relevance: {score:.2f})")
                parts.append(f"Q: {mem.human_content[:200]}...")
                parts.append(f"A: {mem.assistant_content[:300]}...")

        # Episodic memories - the WHY/WHEN
        if self.episodic_memories:
            parts.append("\n\n=== WHY WE WERE DOING THIS (Episodic Memory) ===")
            for ep, score in zip(self.episodic_memories[:3], self.episodic_scores[:3]):
                parts.append(f"\n[{ep.title}] (relevance: {score:.2f})")
                parts.append(f"Summary: {ep.llm_tags.get('summary', 'No summary')}")
                if ep.llm_tags.get('interruption_reason'):
                    parts.append(f"Interrupted because: {ep.llm_tags['interruption_reason']}")
                parts.append(f"Outcome: {ep.llm_tags.get('outcome', 'unknown')}")

        self.merged_context = "\n".join(parts)
        return self.merged_context


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTER INFO (for labeling and navigation)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterInfo:
    """
    Metadata about a process memory cluster.

    Generated after HDBSCAN runs, labeled by LLM once per cluster.
    """
    cluster_id: int
    label: str                       # Human-readable name
    description: str                 # What this cluster represents

    # Statistics
    member_count: int
    avg_technical_depth: float
    dominant_intent: IntentType
    top_domains: List[str]
    top_entities: List[str]

    # Centroid for similarity
    centroid_embedding: Optional[List[float]] = None

    # Time range
    earliest_memory: Optional[datetime] = None
    latest_memory: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "description": self.description,
            "member_count": self.member_count,
            "avg_technical_depth": self.avg_technical_depth,
            "dominant_intent": self.dominant_intent.value,
            "top_domains": self.top_domains,
            "top_entities": self.top_entities,
            "earliest_memory": self.earliest_memory.isoformat() if self.earliest_memory else None,
            "latest_memory": self.latest_memory.isoformat() if self.latest_memory else None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def conversation_to_nodes(
    conversation: Dict[str, Any],
    source: Source
) -> List[MemoryNode]:
    """
    Split a parsed conversation into 1:1 Q/A memory nodes.

    Args:
        conversation: Normalized conversation from ChatParserFactory
        source: Provider source

    Returns:
        List of MemoryNode objects (one per Q/A exchange)
    """
    nodes = []
    messages = conversation.get("messages", [])
    conv_id = conversation.get("id", "")

    # Pair up human/assistant messages
    i = 0
    seq = 0
    while i < len(messages):
        msg = messages[i]

        if msg.get("role") == "human":
            human_content = msg.get("content", "")

            # Look for assistant response
            assistant_content = ""
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                assistant_content = messages[i + 1].get("content", "")
                i += 1  # Skip assistant in next iteration

            # Create node
            created_at = msg.get("created_at", "")
            if isinstance(created_at, str) and created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    created_dt = datetime.now()
            else:
                created_dt = datetime.now()

            node = MemoryNode(
                id="",  # Will be generated in __post_init__
                conversation_id=conv_id,
                sequence_index=seq,
                human_content=human_content,
                assistant_content=assistant_content,
                source=source,
                created_at=created_dt,
            )
            nodes.append(node)
            seq += 1

        i += 1

    return nodes


def conversation_to_episode(
    conversation: Dict[str, Any],
    source: Source
) -> EpisodicMemory:
    """
    Convert a parsed conversation to an EpisodicMemory.

    Args:
        conversation: Normalized conversation from ChatParserFactory
        source: Provider source

    Returns:
        EpisodicMemory object with full conversation preserved
    """
    messages = conversation.get("messages", [])

    # Parse timestamps
    created_at_str = conversation.get("created_at", "")
    updated_at_str = conversation.get("updated_at", "")

    def parse_ts(ts_str: str) -> datetime:
        if isinstance(ts_str, str) and ts_str:
            try:
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now()

    created_at = parse_ts(created_at_str)
    updated_at = parse_ts(updated_at_str) if updated_at_str else created_at

    # Calculate duration
    duration = (updated_at - created_at).total_seconds() / 60.0

    return EpisodicMemory(
        id=conversation.get("id", ""),
        title=conversation.get("title", "Untitled"),
        messages=messages,
        message_count=len(messages),
        source=source,
        created_at=created_at,
        updated_at=updated_at,
        duration_minutes=max(0, duration),
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Memory Schemas - Dual Pipeline Models")
    print("=" * 60)

    # Test node creation
    test_conv = {
        "id": "test-conv-123",
        "title": "Debugging async handlers",
        "created_at": "2024-01-15T10:30:00",
        "updated_at": "2024-01-15T11:45:00",
        "messages": [
            {"role": "human", "content": "I'm getting a weird error with async", "created_at": "2024-01-15T10:30:00"},
            {"role": "assistant", "content": "That error usually means...", "created_at": "2024-01-15T10:31:00"},
            {"role": "human", "content": "How do I fix it?", "created_at": "2024-01-15T10:32:00"},
            {"role": "assistant", "content": "Try wrapping it in...", "created_at": "2024-01-15T10:33:00"},
        ],
        "metadata": {"source": "anthropic"}
    }

    # Create memory nodes
    nodes = conversation_to_nodes(test_conv, Source.ANTHROPIC)
    print(f"\nCreated {len(nodes)} memory nodes:")
    for node in nodes:
        print(f"  - {node.id}: {node.human_content[:50]}...")

    # Create episodic memory
    episode = conversation_to_episode(test_conv, Source.ANTHROPIC)
    print(f"\nCreated episodic memory:")
    print(f"  - {episode.id}: {episode.title}")
    print(f"  - Duration: {episode.duration_minutes:.1f} minutes")
    print(f"  - Messages: {episode.message_count}")

    # Test retrieval result
    result = RetrievalResult(
        query="How do we handle async errors?",
        process_memories=nodes,
        process_scores=[0.92, 0.85],
        episodic_memories=[episode],
        episodic_scores=[0.78],
    )

    print("\n" + "=" * 60)
    print("VENOM CONTEXT (for answer builder)")
    print("=" * 60)
    print(result.build_venom_context())
