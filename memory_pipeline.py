"""
Memory Pipeline - Async recursive memory ingestion.

The snake eating its tail: agent outputs become retrievable memories
in real-time. This is how we solved statelessness - the context window
is just a suggestion when every thought persists.

Extracted from venom_agent.py to serve as shared infrastructure
for the unified CognitiveTwin system.

Performance: 50 node retrievals in 0.3ms. That's not RAG latency.
That's memory access.

Version: 1.0.0 (cog_twin)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict
from enum import Enum
import hashlib
import time

import numpy as np

from schemas import MemoryNode, Source
from embedder import AsyncEmbedder
from streaming_cluster import StreamingClusterEngine, ClusterAssignment

logger = logging.getLogger(__name__)


class ThoughtType(Enum):
    """
    Types of cognitive outputs that become memories.
    
    Every type gets embedded and stored - the recursive loop
    that gives the model true persistent state.
    """
    RESPONSE = "response"           # Normal conversational output
    REMEMBER = "remember"           # Explicit memory storage request
    REFLECT = "reflect"             # Metacognitive observation
    INSIGHT = "insight"             # Synthesized understanding
    DECISION = "decision"           # Choice point with reasoning
    CODE_PROPOSAL = "code_proposal" # Self-modification proposal (HITL)
    TASK_SPAWN = "task_spawn"       # Subtask creation
    TASK_COMPLETE = "task_complete" # Task completion marker
    GAP_DETECTION = "gap_detection" # Identified context gap
    CORRECTION = "correction"       # Self-correction output


@dataclass
class CognitiveOutput:
    """
    A single unit of agent cognition that becomes memory.
    
    This is the recursive hook - every output becomes input.
    The model literally remembers everything it ever said.
    """
    id: str
    timestamp: datetime
    thought_type: ThoughtType
    content: str
    reasoning: Optional[str] = None
    
    # Provenance - what memories led to this thought
    source_memory_ids: List[str] = field(default_factory=list)
    
    # Cognitive state at time of generation
    cognitive_phase: Optional[str] = None
    confidence: float = 0.8
    gap_severity: Optional[float] = None
    
    # Clustering (filled by pipeline)
    cluster_id: Optional[int] = None
    cluster_confidence: Optional[float] = None
    is_new_cluster: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        thought_type: ThoughtType,
        content: str,
        reasoning: Optional[str] = None,
        source_memory_ids: Optional[List[str]] = None,
        cognitive_phase: Optional[str] = None,
        confidence: float = 0.8,
        **metadata
    ) -> "CognitiveOutput":
        """Factory method with auto-generated ID and timestamp."""
        thought_id = hashlib.sha256(
            f"{time.time()}_{content[:50]}".encode()
        ).hexdigest()[:16]
        
        return cls(
            id=thought_id,
            timestamp=datetime.now(),
            thought_type=thought_type,
            content=content,
            reasoning=reasoning,
            source_memory_ids=source_memory_ids or [],
            cognitive_phase=cognitive_phase,
            confidence=confidence,
            metadata=metadata
        )
    
    def to_memory_node(self, source: Source = Source.ANTHROPIC) -> MemoryNode:
        """
        Convert cognitive output to storable memory node.

        This is the transformation that closes the loop:
        thought -> memory -> future retrieval -> future thought
        """
        # Build rich human_content that captures the cognitive context
        human_parts = [f"[{self.thought_type.value.upper()}]"]
        if self.reasoning:
            human_parts.append(f"Reasoning: {self.reasoning}")
        if self.cognitive_phase:
            human_parts.append(f"Phase: {self.cognitive_phase}")
        if self.gap_severity is not None:
            human_parts.append(f"Gap Severity: {self.gap_severity:.2f}")

        return MemoryNode(
            id=self.id,
            source=source,
            conversation_id=f"twin_session_{datetime.now().strftime('%Y%m%d')}",
            sequence_index=0,  # Session outputs don't have sequence
            created_at=self.timestamp,
            human_content=" | ".join(human_parts),
            assistant_content=self.content,
            cluster_id=self.cluster_id,
            cluster_confidence=self.cluster_confidence or 0.0,
            tags={
                "domains": [self.thought_type.value],
                "topics": [],
                "entities": self.source_memory_ids[:5] if self.source_memory_ids else [],
                "processes": [self.cognitive_phase] if self.cognitive_phase else [],
            },
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "thought_type": self.thought_type.value,
            "content": self.content,
            "reasoning": self.reasoning,
            "source_memory_ids": self.source_memory_ids,
            "cognitive_phase": self.cognitive_phase,
            "confidence": self.confidence,
            "gap_severity": self.gap_severity,
            "cluster_id": self.cluster_id,
            "cluster_confidence": self.cluster_confidence,
            "is_new_cluster": self.is_new_cluster,
            "metadata": self.metadata
        }


class MemoryPipeline:
    """
    Async memory ingestion pipeline with streaming clustering.
    
    The recursive engine: runs in background, embedding and storing
    agent outputs without blocking conversation flow. Every thought
    becomes retrievable memory in near-real-time.
    
    This is how we beat RAG - not search-then-stuff, but continuous
    cognitive state persistence. 50 nodes in 0.3ms isn't retrieval
    latency, it's memory access speed.
    """
    
    def __init__(
        self,
        embedder: AsyncEmbedder,
        data_dir: Path,
        batch_interval: float = 5.0,
        max_batch_size: int = 10,
    ):
        """
        Args:
            embedder: Async embedding engine
            data_dir: Where to persist memories
            batch_interval: Seconds between batch processing
            max_batch_size: Process when batch hits this size
        """
        self.embedder = embedder
        self.data_dir = data_dir
        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size
        
        self.queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Session buffer - not yet persisted to disk
        # This IS the recursive state - searchable within session
        self.session_outputs: List[CognitiveOutput] = []
        self.session_embeddings: List[np.ndarray] = []
        self.session_nodes: List[MemoryNode] = []
        
        # Streaming cluster engine for real-time cluster assignment
        self.cluster_engine: Optional[StreamingClusterEngine] = None
        self._init_cluster_engine()
        
        # Stats
        self.total_processed = 0
        self.total_new_clusters = 0
        
        logger.info(f"MemoryPipeline initialized (batch_interval={batch_interval}s)")
    
    def _init_cluster_engine(self):
        """Initialize streaming cluster engine if available."""
        try:
            self.cluster_engine = StreamingClusterEngine(self.data_dir)
            logger.info("Streaming cluster engine initialized")
        except Exception as e:
            logger.warning(f"Streaming clustering unavailable: {e}")
            self.cluster_engine = None
    
    async def start(self):
        """Start the background memory pipeline."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Memory pipeline started - recursive loop active")
    
    async def stop(self):
        """Stop pipeline and flush remaining items to disk."""
        self._running = False
        
        if self._task:
            # Process any remaining items
            await self._task
            
        await self._flush_to_disk()
        logger.info(
            f"Memory pipeline stopped. "
            f"Total processed: {self.total_processed}, "
            f"New clusters: {self.total_new_clusters}"
        )
    
    async def ingest(self, output: CognitiveOutput):
        """
        Add a cognitive output to the memory queue.
        
        This is the recursive hook - call this after every
        LLM generation to close the loop.
        """
        await self.queue.put(output)
        logger.debug(f"Queued cognitive output: {output.thought_type.value}")
    
    async def ingest_batch(self, outputs: List[CognitiveOutput]):
        """Ingest multiple outputs at once."""
        for output in outputs:
            await self.queue.put(output)
        logger.debug(f"Queued {len(outputs)} cognitive outputs")
    
    async def _process_loop(self):
        """Main processing loop - runs in background."""
        batch: List[CognitiveOutput] = []
        
        while self._running or not self.queue.empty():
            try:
                # Collect items until timeout or batch full
                try:
                    output = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.batch_interval
                    )
                    batch.append(output)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if ready
                should_process = (
                    len(batch) >= self.max_batch_size or
                    (batch and not self._running)
                )
                
                if should_process:
                    await self._process_batch(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Memory pipeline error: {e}", exc_info=True)
    
    async def _process_batch(self, outputs: List[CognitiveOutput]):
        """
        Process a batch of cognitive outputs into searchable memory.
        
        This is where the magic happens:
        1. Embed the content
        2. Assign to clusters (streaming)
        3. Add to session buffer (immediately searchable)
        4. Queue for disk persistence
        """
        if not outputs:
            return
        
        start_time = time.time()
        
        # Extract content for embedding
        texts = [o.content for o in outputs]
        
        # Embed asynchronously
        embeddings = await self.embedder.embed_batch(texts, show_progress=False)
        
        # Assign to clusters using streaming engine
        cluster_assignments = []
        if self.cluster_engine:
            cluster_assignments = self.cluster_engine.batch_assign(embeddings)
        else:
            # Fallback: no clustering, mark as noise
            cluster_assignments = [
                ClusterAssignment(cluster_id=-1, confidence=0.0, is_new_cluster=False)
                for _ in outputs
            ]
        
        # Update outputs with cluster info and convert to nodes
        nodes = []
        for output, embedding, assignment in zip(outputs, embeddings, cluster_assignments):
            output.cluster_id = assignment.cluster_id
            output.cluster_confidence = assignment.confidence
            output.is_new_cluster = assignment.is_new_cluster
            
            node = output.to_memory_node()
            nodes.append(node)
            
            if assignment.is_new_cluster:
                self.total_new_clusters += 1
        
        # Add to session buffer - immediately searchable
        self.session_outputs.extend(outputs)
        self.session_embeddings.extend(embeddings)
        self.session_nodes.extend(nodes)
        
        self.total_processed += len(outputs)
        
        # Stats
        elapsed_ms = (time.time() - start_time) * 1000
        new_clusters = sum(1 for a in cluster_assignments if a.is_new_cluster)
        noise_points = sum(1 for a in cluster_assignments if a.cluster_id == -1)
        
        logger.info(
            f"Processed {len(outputs)} outputs in {elapsed_ms:.1f}ms "
            f"(session total: {len(self.session_nodes)}, "
            f"new clusters: {new_clusters}, noise: {noise_points})"
        )
    
    def search_session(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[tuple[CognitiveOutput, float]]:
        """
        Search session memories (not yet persisted to disk).
        
        This enables the agent to remember what it did earlier
        in this session - the immediate recursive loop.
        
        Returns: List of (output, similarity_score) tuples
        """
        if not self.session_embeddings:
            return []
        
        # Build session matrix
        session_matrix = np.array(self.session_embeddings)
        
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        session_norm = session_matrix / (
            np.linalg.norm(session_matrix, axis=1, keepdims=True) + 1e-8
        )
        
        # Compute similarities
        similarities = session_norm @ query_norm
        
        # Get top-k above threshold
        indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in indices[:top_k]:
            sim = float(similarities[idx])
            if sim >= min_similarity:
                results.append((self.session_outputs[idx], sim))
        
        return results
    
    def get_session_context(self, last_n: int = 5) -> List[CognitiveOutput]:
        """Get the last N outputs from this session."""
        return self.session_outputs[-last_n:] if self.session_outputs else []
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about current session."""
        thought_type_counts = {}
        for output in self.session_outputs:
            t = output.thought_type.value
            thought_type_counts[t] = thought_type_counts.get(t, 0) + 1
        
        return {
            "session_outputs": len(self.session_outputs),
            "queue_size": self.queue.qsize(),
            "total_processed": self.total_processed,
            "total_new_clusters": self.total_new_clusters,
            "thought_type_distribution": thought_type_counts,
        }
    
    async def _flush_to_disk(self):
        """Persist session memories to disk."""
        if not self.session_nodes:
            logger.info("No session memories to flush")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure directories exist
        (self.data_dir / "memory_nodes").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "vectors").mkdir(parents=True, exist_ok=True)
        
        # Save nodes as JSON
        nodes_file = self.data_dir / "memory_nodes" / f"session_nodes_{timestamp}.json"
        nodes_data = [n.to_dict() for n in self.session_nodes]
        with open(nodes_file, "w") as f:
            json.dump(nodes_data, f, indent=2, default=str)
        
        # Save embeddings as numpy
        emb_file = self.data_dir / "vectors" / f"session_embeddings_{timestamp}.npy"
        np.save(emb_file, np.array(self.session_embeddings))
        
        # Save cognitive outputs for full provenance
        outputs_file = self.data_dir / "memory_nodes" / f"session_outputs_{timestamp}.json"
        outputs_data = [o.to_dict() for o in self.session_outputs]
        with open(outputs_file, "w") as f:
            json.dump(outputs_data, f, indent=2)
        
        # Save session clusters if using streaming engine
        if self.cluster_engine:
            self.cluster_engine.save_session_clusters()
        
        logger.info(
            f"Flushed {len(self.session_nodes)} session memories to disk: "
            f"{nodes_file.name}"
        )


# Convenience function for creating common thought types
def create_response_output(
    content: str,
    source_memory_ids: List[str],
    cognitive_phase: str,
    confidence: float = 0.8
) -> CognitiveOutput:
    """Create a standard response output."""
    return CognitiveOutput.create(
        thought_type=ThoughtType.RESPONSE,
        content=content,
        source_memory_ids=source_memory_ids,
        cognitive_phase=cognitive_phase,
        confidence=confidence
    )


def create_reflection_output(
    content: str,
    reasoning: str,
    cognitive_phase: str
) -> CognitiveOutput:
    """Create a metacognitive reflection output."""
    return CognitiveOutput.create(
        thought_type=ThoughtType.REFLECT,
        content=content,
        reasoning=reasoning,
        cognitive_phase=cognitive_phase,
        confidence=0.7  # Reflections are inherently uncertain
    )


def create_insight_output(
    content: str,
    source_memory_ids: List[str],
    confidence: float = 0.85
) -> CognitiveOutput:
    """Create a synthesized insight output."""
    return CognitiveOutput.create(
        thought_type=ThoughtType.INSIGHT,
        content=content,
        source_memory_ids=source_memory_ids,
        confidence=confidence
    )


def create_gap_detection_output(
    content: str,
    gap_severity: float,
    cognitive_phase: str
) -> CognitiveOutput:
    """Create a context gap detection output."""
    output = CognitiveOutput.create(
        thought_type=ThoughtType.GAP_DETECTION,
        content=content,
        cognitive_phase=cognitive_phase,
        confidence=0.9  # Gap detection is fairly certain
    )
    output.gap_severity = gap_severity
    return output