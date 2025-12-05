"""
reasoning_trace.py - Record and search model reasoning chains.
Stores: what memories were touched, what was cited, what worked.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import hashlib
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StepType(Enum):
    RETRIEVE = "retrieve"
    REFLECT = "reflect"
    SYNTHESIZE = "synthesize"
    GAP_DETECT = "gap_detect"
    GREP = "grep"
    EXPLORE = "explore"
    CITE = "cite"


@dataclass
class ReasoningStep:
    """Single step in a reasoning chain."""
    step_type: StepType
    content: str
    memories_touched: List[str] = field(default_factory=list)
    clusters_touched: List[int] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "step_type": self.step_type.value,
            "content": self.content,
            "memories_touched": self.memories_touched,
            "clusters_touched": self.clusters_touched,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReasoningStep":
        return cls(
            step_type=StepType(data["step_type"]),
            content=data["content"],
            memories_touched=data.get("memories_touched", []),
            clusters_touched=data.get("clusters_touched", []),
            duration_ms=data.get("duration_ms", 0.0),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class ReasoningTrace:
    """Complete reasoning chain for a single query."""
    id: str
    query: str
    query_embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # What was retrieved
    memories_retrieved: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    memories_cited: List[str] = field(default_factory=list)
    clusters_traversed: List[int] = field(default_factory=list)

    # The chain
    steps: List[ReasoningStep] = field(default_factory=list)

    # Response
    response: str = ""
    response_confidence: float = 0.0
    cognitive_phase: str = ""
    response_mode: str = ""

    # Touch counts (salience signals)
    memory_touch_counts: Dict[str, int] = field(default_factory=dict)
    cluster_touch_counts: Dict[str, int] = field(default_factory=dict)

    # Scoring (filled by Phase 3)
    score: Optional[Dict[str, float]] = None
    feedback_notes: Dict[str, str] = field(default_factory=dict)

    # Performance
    total_duration_ms: float = 0.0
    tokens_used: int = 0

    def add_step(self, step: ReasoningStep):
        """Add a step and update touch counts."""
        self.steps.append(step)

        for mem_id in step.memories_touched:
            self.memory_touch_counts[mem_id] = self.memory_touch_counts.get(mem_id, 0) + 1

        for cluster_id in step.clusters_touched:
            key = str(cluster_id)
            self.cluster_touch_counts[key] = self.cluster_touch_counts.get(key, 0) + 1

    def record_citation(self, memory_id: str):
        """Record that a memory was actually cited in response."""
        if memory_id not in self.memories_cited:
            self.memories_cited.append(memory_id)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "query": self.query,
            "query_embedding": self.query_embedding,
            "timestamp": self.timestamp.isoformat(),
            "memories_retrieved": self.memories_retrieved,
            "retrieval_scores": self.retrieval_scores,
            "memories_cited": self.memories_cited,
            "clusters_traversed": self.clusters_traversed,
            "steps": [s.to_dict() for s in self.steps],
            "response": self.response,
            "response_confidence": self.response_confidence,
            "cognitive_phase": self.cognitive_phase,
            "response_mode": self.response_mode,
            "memory_touch_counts": self.memory_touch_counts,
            "cluster_touch_counts": self.cluster_touch_counts,
            "score": self.score,
            "feedback_notes": self.feedback_notes,
            "total_duration_ms": self.total_duration_ms,
            "tokens_used": self.tokens_used,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReasoningTrace":
        trace = cls(
            id=data["id"],
            query=data["query"],
            query_embedding=data.get("query_embedding"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            memories_retrieved=data.get("memories_retrieved", []),
            retrieval_scores=data.get("retrieval_scores", []),
            memories_cited=data.get("memories_cited", []),
            clusters_traversed=data.get("clusters_traversed", []),
            response=data.get("response", ""),
            response_confidence=data.get("response_confidence", 0.0),
            cognitive_phase=data.get("cognitive_phase", ""),
            response_mode=data.get("response_mode", ""),
            memory_touch_counts=data.get("memory_touch_counts", {}),
            cluster_touch_counts=data.get("cluster_touch_counts", {}),
            score=data.get("score"),
            feedback_notes=data.get("feedback_notes", {}),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            tokens_used=data.get("tokens_used", 0),
        )
        trace.steps = [ReasoningStep.from_dict(s) for s in data.get("steps", [])]
        return trace

    def format_for_context(self) -> str:
        """Format trace for injection into LLM context."""
        lines = [
            f"PAST REASONING TRACE [{self.id[:8]}]",
            f"Query: {self.query[:100]}...",
            f"Phase: {self.cognitive_phase} | Mode: {self.response_mode}",
            f"Retrieved: {len(self.memories_retrieved)} | Cited: {len(self.memories_cited)}",
        ]

        if self.score:
            score_str = ", ".join(f"{k}={v:.2f}" for k, v in self.score.items())
            lines.append(f"Score: {score_str}")

        if self.feedback_notes:
            for field_name, note in self.feedback_notes.items():
                lines.append(f"Feedback ({field_name}): {note}")

        lines.append(f"Response preview: {self.response[:200]}...")

        return "\n".join(lines)


class CognitiveTracer:
    """
    Records reasoning chains during query processing.

    Phase 5: Now streams traces to memory_pipeline on finalize,
    making them immediately grepable and retrievable.

    Usage:
        tracer = CognitiveTracer(data_dir, memory_pipeline=pipeline)
        tracer.start_trace(query, retrieved_ids)
        tracer.record_step(StepType.REFLECT, "thinking about...", touched_ids)
        tracer.record_citation(memory_id)
        trace = await tracer.finalize_trace(response, confidence, tokens)
    """

    def __init__(self, data_dir: Path, memory_pipeline=None):
        self.data_dir = Path(data_dir)
        self.traces_dir = self.data_dir / "reasoning_traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        # Phase 5: Reference to live memory pipeline for streaming
        self.memory_pipeline = memory_pipeline

        self.current_trace: Optional[ReasoningTrace] = None
        self.start_time: float = 0.0

        # In-memory index for search
        self.traces: List[ReasoningTrace] = []
        self.trace_map: Dict[str, ReasoningTrace] = {}

        self._load_traces()

    def _load_traces(self):
        """Load existing traces from disk."""
        for trace_file in self.traces_dir.glob("trace_*.json"):
            try:
                with open(trace_file) as f:
                    data = json.load(f)
                trace = ReasoningTrace.from_dict(data)
                self.traces.append(trace)
                self.trace_map[trace.id] = trace
            except Exception as e:
                logger.warning(f"Failed to load trace {trace_file}: {e}")

        # Sort by timestamp
        self.traces.sort(key=lambda t: t.timestamp, reverse=True)
        logger.info(f"CognitiveTracer: loaded {len(self.traces)} existing traces")

    def start_trace(
        self,
        query: str,
        retrieved_memory_ids: List[str],
        retrieval_scores: Optional[List[float]] = None,
        query_embedding: Optional[List[float]] = None,
        cognitive_phase: str = "",
        response_mode: str = "",
    ) -> str:
        """Start recording a new trace. Returns trace ID."""
        trace_id = hashlib.sha256(
            f"{time.time()}_{query[:50]}".encode()
        ).hexdigest()[:16]

        self.current_trace = ReasoningTrace(
            id=trace_id,
            query=query,
            query_embedding=None,  # Don't store - saves ~8KB per trace
            memories_retrieved=retrieved_memory_ids,
            retrieval_scores=retrieval_scores or [],
            cognitive_phase=cognitive_phase,
            response_mode=response_mode,
        )

        self.start_time = time.time()

        # Record initial retrieval step
        self.record_step(
            StepType.RETRIEVE,
            f"Retrieved {len(retrieved_memory_ids)} memories",
            memories_touched=retrieved_memory_ids,
        )

        return trace_id

    def record_step(
        self,
        step_type: StepType,
        content: str,
        memories_touched: Optional[List[str]] = None,
        clusters_touched: Optional[List[int]] = None,
        **metadata
    ):
        """Record a reasoning step."""
        if not self.current_trace:
            return

        step = ReasoningStep(
            step_type=step_type,
            content=content,
            memories_touched=memories_touched or [],
            clusters_touched=clusters_touched or [],
            metadata=metadata,
        )

        self.current_trace.add_step(step)

    def record_citation(self, memory_id: str):
        """Record that a memory was cited in the response."""
        if self.current_trace:
            self.current_trace.record_citation(memory_id)

    def record_grep(self, term: str, occurrences: int, memories_found: int):
        """Record a grep tool invocation."""
        self.record_step(
            StepType.GREP,
            f"Grep '{term}': {occurrences} occurrences across {memories_found} memories",
            metadata={"term": term, "occurrences": occurrences, "memories_found": memories_found}
        )

    async def finalize_trace(
        self,
        response: str,
        confidence: float,
        tokens_used: int,
    ) -> ReasoningTrace:
        """
        Finalize, save, and stream trace to memory pipeline.

        Phase 5: Now async - streams trace to memory_pipeline
        for immediate retrieval and grep access.
        """
        if not self.current_trace:
            raise ValueError("No active trace to finalize")

        self.current_trace.response = response
        self.current_trace.response_confidence = confidence
        self.current_trace.tokens_used = tokens_used
        self.current_trace.total_duration_ms = (time.time() - self.start_time) * 1000

        # Save to disk
        trace_file = self.traces_dir / f"trace_{self.current_trace.id}.json"
        with open(trace_file, "w") as f:
            json.dump(self.current_trace.to_dict(), f, indent=2)

        # Add to in-memory index
        self.traces.insert(0, self.current_trace)
        self.trace_map[self.current_trace.id] = self.current_trace

        # Phase 5: Stream to memory pipeline for immediate retrieval
        if self.memory_pipeline:
            await self._stream_to_memory(self.current_trace)

        trace = self.current_trace
        self.current_trace = None

        logger.info(f"Trace {trace.id[:8]} finalized: {len(trace.steps)} steps, {trace.tokens_used} tokens")

        return trace

    async def _stream_to_memory(self, trace: ReasoningTrace):
        """
        Stream trace to memory pipeline as CognitiveOutput.

        This is the Phase 5 hook that closes the loop:
        - Trace becomes immediately grepable
        - Trace becomes immediately retrievable
        - Model can see its own reasoning history in real-time
        """
        # Import here to avoid circular dependency
        from memory_pipeline import CognitiveOutput, ThoughtType

        # Format trace as searchable content
        content = f"""REASONING TRACE [{trace.id[:8]}]
Query: {trace.query}
Phase: {trace.cognitive_phase}
Memories retrieved: {len(trace.memories_retrieved)}
Memories cited: {len(trace.memories_cited)}

Response preview:
{trace.response[:500]}{'...' if len(trace.response) > 500 else ''}
"""

        # Include score info if available
        reasoning = None
        if trace.score:
            score_parts = [f"{k}={v:.2f}" for k, v in trace.score.items()]
            reasoning = f"Scores: {', '.join(score_parts)}"
            if trace.feedback_notes:
                notes_str = ", ".join(f"{k}: {v}" for k, v in trace.feedback_notes.items())
                reasoning += f"\nFeedback: {notes_str}"

        output = CognitiveOutput(
            id=f"trace_{trace.id}",
            timestamp=trace.timestamp,
            thought_type=ThoughtType.REFLECT,  # Traces are reflective outputs
            content=content,
            reasoning=reasoning,
            source_memory_ids=trace.memories_retrieved[:10],  # Cap for efficiency
            cognitive_phase=trace.cognitive_phase,
            confidence=trace.response_confidence,
        )

        await self.memory_pipeline.ingest(output)
        logger.info(f"Trace {trace.id[:8]} streamed to memory pipeline")

    def search_traces(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[ReasoningTrace]:
        """
        Search past traces by query similarity.

        For Phase 2: Simple keyword matching.
        Future: Vector similarity on query_embedding.
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored = []
        for trace in self.traces:
            trace_terms = set(trace.query.lower().split())
            overlap = len(query_terms & trace_terms)
            if overlap > 0:
                score = overlap / max(len(query_terms), len(trace_terms))
                if min_score is None or trace.score is None or (trace.score.get("overall", 0) >= min_score):
                    scored.append((trace, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:top_k]]

    def get_high_scored_traces(
        self,
        min_overall: float = 0.7,
        top_k: int = 10
    ) -> List[ReasoningTrace]:
        """Get traces with high user scores."""
        scored = [
            t for t in self.traces
            if t.score and t.score.get("overall", 0) >= min_overall
        ]
        scored.sort(key=lambda t: t.score.get("overall", 0), reverse=True)
        return scored[:top_k]

    def get_traces_for_memory(self, memory_id: str) -> List[ReasoningTrace]:
        """Get all traces that touched a specific memory."""
        return [
            t for t in self.traces
            if memory_id in t.memories_retrieved or memory_id in t.memories_cited
        ]

    def get_recent_traces(self, n: int = 10) -> List[ReasoningTrace]:
        """Get N most recent traces."""
        return self.traces[:n]

    def add_score_to_trace(
        self,
        trace_id: str,
        score: Dict[str, float],
        feedback_notes: Optional[Dict[str, str]] = None
    ):
        """Add user score to an existing trace."""
        if trace_id not in self.trace_map:
            raise ValueError(f"Trace {trace_id} not found")

        trace = self.trace_map[trace_id]
        trace.score = score
        if feedback_notes:
            trace.feedback_notes = feedback_notes

        # Save updated trace
        trace_file = self.traces_dir / f"trace_{trace_id}.json"
        with open(trace_file, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)

        logger.info(f"Score added to trace {trace_id[:8]}: overall={score.get('overall', 0):.2f}")

    def load_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """Load a trace by ID from disk."""
        trace_file = self.traces_dir / f"trace_{trace_id}.json"
        if not trace_file.exists():
            matches = list(self.traces_dir.glob(f"trace_{trace_id[:8]}*.json"))
            if matches:
                trace_file = matches[0]
            else:
                return None
        try:
            with open(trace_file) as f:
                return ReasoningTrace.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load trace {trace_id}: {e}")
            return None

    def load_recent_traces(self, n: int = 100) -> List[ReasoningTrace]:
        """Load N most recent traces."""
        trace_files = sorted(
            self.traces_dir.glob("trace_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )[:n]
        traces = []
        for tf in trace_files:
            try:
                with open(tf) as f:
                    traces.append(ReasoningTrace.from_dict(json.load(f)))
            except Exception as e:
                logger.warning(f"Failed to load {tf}: {e}")
        return traces
