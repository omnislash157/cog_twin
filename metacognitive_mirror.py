"""
Metacognitive Mirror - The system watching itself think.

Analyzes patterns in HOW the memory system is being used - not just what's
stored, but how it's accessed, what patterns emerge, and where the cognitive
architecture itself could evolve.

Components:
  - QueryArchaeologist: Reconstruct cognitive intent from query patterns
  - MemoryThermodynamics: Track entropy and energy in memory access
  - CognitiveSeismograph: Detect shifts in mental state
  - PredictivePrefetcher: Anticipate future memory needs
  - ArchitecturalIntrospector: Self-optimization recommendations

Version: 1.0.0 (cog_twin)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque, Counter
import numpy as np
from numpy.typing import NDArray
import json
from pathlib import Path
from enum import Enum
import heapq


logger = logging.getLogger(__name__)


class CognitivePhase(Enum):
    """
    Mental state classifications based on access patterns.

    Why: Not all thinking is the same. Exploration looks different from
    exploitation, learning different from retrieval. By classifying phases,
    we can optimize behavior for the current cognitive mode.
    """
    EXPLORATION = "exploration"  # Wide-ranging queries, low repetition
    EXPLOITATION = "exploitation"  # Focused queries, high repetition
    LEARNING = "learning"  # Building new memory structures
    CONSOLIDATION = "consolidation"  # Reviewing and connecting existing memories
    IDLE = "idle"  # Low activity, background processing
    CRISIS = "crisis"  # Rapid, unfocused access - lost or confused


class DriftSignal(Enum):
    """
    Types of cognitive drift we can detect.

    Why: Change is inevitable, but different types of change require
    different responses. Topic drift is natural, but semantic collapse
    might indicate a problem.
    """
    TOPIC_SHIFT = "topic_shift"  # Natural evolution to new subjects
    SEMANTIC_EXPANSION = "semantic_expansion"  # Vocabulary growing
    SEMANTIC_COLLAPSE = "semantic_collapse"  # Vocabulary narrowing (concern)
    TEMPORAL_DRIFT = "temporal_drift"  # Time-based pattern changes
    STRUCTURAL_DRIFT = "structural_drift"  # Access pattern changes
    ANOMALOUS_SPIKE = "anomalous_spike"  # Sudden unexplained changes


@dataclass
class QueryEvent:
    """
    A single query captured for analysis.

    Why: The atomic unit of metacognitive analysis. Every question asked
    reveals something about cognitive state, and the sequence of questions
    tells a story about thought progression.
    """
    timestamp: datetime
    query_text: str
    query_embedding: NDArray[np.float32]
    retrieved_memory_ids: List[str]
    retrieval_scores: List[float]
    execution_time_ms: float
    result_count: int
    semantic_gate_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage (embedding as list)."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "query_text": self.query_text,
            "query_embedding": self.query_embedding.tolist(),
            "retrieved_memory_ids": self.retrieved_memory_ids,
            "retrieval_scores": self.retrieval_scores,
            "execution_time_ms": self.execution_time_ms,
            "result_count": self.result_count,
            "semantic_gate_passed": self.semantic_gate_passed,
        }


@dataclass
class MemoryAccessEvent:
    """
    A memory being accessed/retrieved.

    Why: Memories have temperatures. Some are hot (frequently accessed),
    some are cold (rarely touched), some are burning (suddenly popular).
    Tracking this reveals what matters to the cognitive system.
    """
    timestamp: datetime
    memory_id: str
    access_type: str  # 'retrieval', 'update', 'creation'
    query_context: Optional[str]
    access_score: float  # How strongly was this memory selected
    co_accessed_memories: List[str]  # What else was accessed in same query


@dataclass
class CognitiveSnapshot:
    """
    A moment-in-time capture of cognitive state.

    Why: The system's "mental state" at a given moment, characterized by
    query patterns, access frequencies, and semantic centroids. Snapshots
    allow temporal drift analysis and state comparison.
    """
    timestamp: datetime
    query_centroid: NDArray[np.float32]  # Average query embedding
    query_variance: float  # How scattered are queries
    access_entropy: float  # How evenly distributed are memory accesses
    dominant_topics: List[Tuple[str, float]]  # Top query clusters
    phase: CognitivePhase
    temperature: float  # Overall system activity level (0-1)
    focus_score: float  # How concentrated vs diffuse is attention


@dataclass
class PredictionEvent:
    """
    A prediction about future memory needs.

    Why: If we can predict what memories will be needed next, we can
    preload them, reduce latency, and improve user experience. This
    tracks prediction accuracy for continuous improvement.
    """
    timestamp: datetime
    predicted_memory_ids: List[str]
    confidence_scores: List[float]
    actual_accessed: Optional[List[str]] = None  # Filled in after validation
    prediction_accuracy: Optional[float] = None


@dataclass
class MetacognitiveInsight:
    """
    A system-generated recommendation about itself.

    Why: The ultimate goal - the system understanding its own weaknesses
    and suggesting improvements. Self-optimization through introspection.
    """
    timestamp: datetime
    insight_type: str
    severity: str  # 'info', 'warning', 'critical'
    description: str
    metrics: Dict[str, float]
    suggested_action: str
    estimated_impact: str


class QueryArchaeologist:
    """
    Reconstructs cognitive patterns from query history.

    Why: Individual queries are noisy, but sequences reveal intent.
    This class finds the patterns, clusters, and evolution in how
    questions are being asked over time.
    """

    def __init__(
        self,
        window_size: int = 100,
        cluster_epsilon: float = 0.3,
        min_pattern_length: int = 3
    ):
        self.window_size = window_size
        self.cluster_epsilon = cluster_epsilon
        self.min_pattern_length = min_pattern_length
        self.query_history: deque[QueryEvent] = deque(maxlen=window_size)
        self.query_clusters: Dict[int, List[QueryEvent]] = defaultdict(list)

    def record_query(self, event: QueryEvent) -> None:
        """Add query to history and update clustering."""
        self.query_history.append(event)
        self._update_clusters()

    def _update_clusters(self) -> None:
        """
        Incrementally cluster queries by semantic similarity.

        Why: Simple online clustering using embedding distance. Queries
        within epsilon distance join same cluster. Reveals recurring
        question themes without expensive recomputation.
        """
        if not self.query_history:
            return

        latest = self.query_history[-1]

        # Find closest existing cluster
        min_dist = float('inf')
        closest_cluster = None

        for cluster_id, cluster_queries in self.query_clusters.items():
            if not cluster_queries:
                continue
            # Use centroid of cluster
            centroid = np.mean(
                [q.query_embedding for q in cluster_queries],
                axis=0
            )
            dist = np.linalg.norm(latest.query_embedding - centroid)

            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster_id

        # Assign to cluster or create new one
        if min_dist < self.cluster_epsilon:
            self.query_clusters[closest_cluster].append(latest)
        else:
            new_id = len(self.query_clusters)
            self.query_clusters[new_id] = [latest]

    def detect_recurring_patterns(self) -> List[Tuple[str, int, float]]:
        """
        Find query patterns that repeat over time.

        Returns: List of (pattern_description, frequency, recency_score)

        Why: Repeated query patterns indicate stable cognitive needs.
        These are candidates for optimization, caching, or automation.
        """
        patterns = []

        for cluster_id, queries in self.query_clusters.items():
            if len(queries) < 2:
                continue

            # Calculate frequency
            frequency = len(queries)

            # Calculate recency score (more recent = higher score)
            if queries:
                latest_timestamp = max(q.timestamp for q in queries)
                hours_ago = (datetime.now() - latest_timestamp).total_seconds() / 3600
                recency_score = np.exp(-hours_ago / 24)  # Exponential decay
            else:
                recency_score = 0.0

            # Generate pattern description from cluster centroid
            centroid = np.mean([q.query_embedding for q in queries], axis=0)
            representative = queries[-1].query_text  # Use most recent as label

            patterns.append((representative, frequency, recency_score))

        # Sort by combined score (frequency * recency)
        patterns.sort(key=lambda x: x[1] * x[2], reverse=True)
        return patterns

    def calculate_query_entropy(self) -> float:
        """
        Measure the diversity/focus of recent queries.

        Returns: Entropy value (higher = more diverse)

        Why: High entropy = exploration, low entropy = exploitation.
        This single metric reveals cognitive phase and can trigger
        different optimization strategies.
        """
        if len(self.query_clusters) <= 1:
            return 0.0

        # Calculate entropy based on cluster distribution
        total_queries = len(self.query_history)
        if total_queries == 0:
            return 0.0

        probabilities = [
            len(queries) / total_queries
            for queries in self.query_clusters.values()
        ]

        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy

    def detect_semantic_drift(
        self,
        lookback_hours: int = 24
    ) -> Tuple[float, DriftSignal]:
        """
        Detect if query semantics are shifting over time.

        Returns: (drift_magnitude, drift_type)

        Why: Cognitive drift is natural but monitoring it helps detect
        topic shifts, confusion, or system misalignment. Large sudden
        drifts might indicate the user is lost or frustrated.
        """
        if len(self.query_history) < 10:
            return 0.0, DriftSignal.TOPIC_SHIFT

        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent = [q for q in self.query_history if q.timestamp > cutoff]
        old = [q for q in self.query_history if q.timestamp <= cutoff]

        if not recent or not old:
            return 0.0, DriftSignal.TOPIC_SHIFT

        # Calculate centroid shift
        recent_centroid = np.mean([q.query_embedding for q in recent], axis=0)
        old_centroid = np.mean([q.query_embedding for q in old], axis=0)

        drift_magnitude = np.linalg.norm(recent_centroid - old_centroid)

        # Calculate variance to detect expansion vs collapse
        recent_variance = np.var([q.query_embedding for q in recent])
        old_variance = np.var([q.query_embedding for q in old])

        variance_ratio = recent_variance / (old_variance + 1e-6)

        # Classify drift type
        if drift_magnitude > 0.5:
            signal = DriftSignal.TOPIC_SHIFT
        elif variance_ratio > 1.5:
            signal = DriftSignal.SEMANTIC_EXPANSION
        elif variance_ratio < 0.5:
            signal = DriftSignal.SEMANTIC_COLLAPSE
        elif drift_magnitude > 0.3:
            signal = DriftSignal.TEMPORAL_DRIFT
        else:
            signal = DriftSignal.TOPIC_SHIFT

        return float(drift_magnitude), signal


class MemoryThermodynamics:
    """
    Tracks the "temperature" and access patterns of memories.

    Why: Not all memories are equal. Some are accessed constantly (hot),
    some rarely (cold), some show bursts of activity (plasma state).
    Understanding memory thermodynamics enables intelligent caching,
    prefetching, and pruning decisions.
    """

    def __init__(
        self,
        decay_half_life_hours: float = 48.0,
        burst_threshold: float = 3.0
    ):
        self.decay_half_life = decay_half_life_hours
        self.burst_threshold = burst_threshold

        # Memory access tracking
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, datetime] = {}
        self.access_history: Dict[str, List[datetime]] = defaultdict(list)

        # Co-access graph (memories accessed together)
        self.co_access_graph: Dict[str, Counter] = defaultdict(Counter)

        # Temperature cache
        self._temperature_cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    def record_access(self, event: MemoryAccessEvent) -> None:
        """Record a memory access event."""
        mid = event.memory_id
        now = event.timestamp

        self.access_counts[mid] += 1
        self.last_access[mid] = now
        self.access_history[mid].append(now)

        # Update co-access graph
        for co_mid in event.co_accessed_memories:
            if co_mid != mid:
                self.co_access_graph[mid][co_mid] += 1

        # Invalidate cache for this memory
        self._temperature_cache.pop(mid, None)

    def calculate_temperature(self, memory_id: str) -> float:
        """
        Calculate memory "temperature" (0-1, higher = hotter).

        Why: Combines access frequency and recency with exponential decay.
        Hot memories should be cached in fast storage, cold ones can be
        archived. Temperature guides resource allocation.
        """
        # Check cache
        if memory_id in self._temperature_cache:
            temp, cached_at = self._temperature_cache[memory_id]
            if datetime.now() - cached_at < self._cache_ttl:
                return temp

        if memory_id not in self.last_access:
            return 0.0

        # Calculate recency decay
        hours_since_access = (
            datetime.now() - self.last_access[memory_id]
        ).total_seconds() / 3600

        decay_factor = 0.5 ** (hours_since_access / self.decay_half_life)

        # Calculate frequency score (normalized by total access history)
        total_accesses = sum(self.access_counts.values())
        if total_accesses == 0:
            frequency_score = 0.0
        else:
            frequency_score = self.access_counts[memory_id] / total_accesses

        # Combined temperature (weighted average)
        temperature = 0.6 * decay_factor + 0.4 * frequency_score

        # Cache result
        self._temperature_cache[memory_id] = (temperature, datetime.now())

        return float(temperature)

    def detect_hotspots(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the hottest memories (most frequently/recently accessed).

        Why: Hotspots are optimization targets. Cache them, preload them,
        optimize their retrieval paths. This is where the system spends
        most of its energy.
        """
        temperatures = [
            (mid, self.calculate_temperature(mid))
            for mid in self.access_counts.keys()
        ]

        # Use heapq for efficient top-k
        hottest = heapq.nlargest(top_k, temperatures, key=lambda x: x[1])
        return hottest

    def detect_bursts(
        self,
        time_window_hours: float = 1.0
    ) -> List[Tuple[str, float]]:
        """
        Detect memories experiencing sudden burst of activity.

        Returns: List of (memory_id, burst_intensity)

        Why: Bursts indicate emergent importance. A memory that suddenly
        becomes hot might signal a topic becoming critical. These deserve
        immediate optimization attention.
        """
        bursts = []
        cutoff = datetime.now() - timedelta(hours=time_window_hours)

        for mid, history in self.access_history.items():
            recent_accesses = sum(1 for ts in history if ts > cutoff)

            # Compare to baseline rate
            total_hours = (
                (datetime.now() - history[0]).total_seconds() / 3600
                if history else 1.0
            )
            baseline_rate = len(history) / max(total_hours, 1.0)
            current_rate = recent_accesses / time_window_hours

            if baseline_rate > 0:
                burst_ratio = current_rate / baseline_rate

                if burst_ratio > self.burst_threshold:
                    bursts.append((mid, float(burst_ratio)))

        bursts.sort(key=lambda x: x[1], reverse=True)
        return bursts

    def find_memory_communities(
        self,
        min_co_access: int = 3
    ) -> List[Set[str]]:
        """
        Find groups of memories that are frequently accessed together.

        Why: Co-accessed memories form conceptual communities. If you
        access memory A, you'll likely need B and C too. Communities
        enable intelligent prefetching and reveal conceptual structure.
        """
        communities = []
        visited = set()

        def dfs_community(start_mid: str, community: Set[str]) -> None:
            """Depth-first search to build community."""
            if start_mid in visited:
                return

            visited.add(start_mid)
            community.add(start_mid)

            # Add strongly connected neighbors
            for neighbor, count in self.co_access_graph[start_mid].items():
                if count >= min_co_access and neighbor not in visited:
                    dfs_community(neighbor, community)

        # Build communities
        for mid in self.co_access_graph.keys():
            if mid not in visited:
                community = set()
                dfs_community(mid, community)
                if len(community) > 1:  # Only keep multi-member communities
                    communities.append(community)

        return communities

    def calculate_access_entropy(self) -> float:
        """
        Measure how evenly distributed memory accesses are.

        Returns: Entropy value (higher = more evenly distributed)

        Why: Low entropy = focus on few memories (exploitation).
        High entropy = broad exploration. This reveals cognitive mode
        and can guide system behavior.
        """
        if not self.access_counts:
            return 0.0

        total = sum(self.access_counts.values())
        probabilities = [count / total for count in self.access_counts.values()]

        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy


class CognitiveSeismograph:
    """
    Detects shifts in mental state and cognitive phases.

    Why: The system has different modes - exploring vs exploiting,
    learning vs retrieving, focused vs scattered. Detecting these
    phases enables adaptive behavior optimization.
    """

    def __init__(self, snapshot_interval_minutes: int = 15):
        self.snapshot_interval = timedelta(minutes=snapshot_interval_minutes)
        self.snapshots: List[CognitiveSnapshot] = []
        self.current_phase = CognitivePhase.IDLE
        self.phase_history: List[Tuple[datetime, CognitivePhase]] = []

    def capture_snapshot(
        self,
        archaeologist: QueryArchaeologist,
        thermodynamics: MemoryThermodynamics
    ) -> CognitiveSnapshot:
        """
        Capture current cognitive state snapshot.

        Why: A snapshot freezes the current moment for later comparison.
        Analyzing snapshot sequences reveals how mental state evolves.
        """
        if not archaeologist.query_history:
            # Empty state
            snapshot = CognitiveSnapshot(
                timestamp=datetime.now(),
                query_centroid=np.zeros(384),  # Assuming SBERT dimension
                query_variance=0.0,
                access_entropy=0.0,
                dominant_topics=[],
                phase=CognitivePhase.IDLE,
                temperature=0.0,
                focus_score=0.0
            )
        else:
            # Calculate query centroid and variance
            embeddings = [q.query_embedding for q in archaeologist.query_history]
            query_centroid = np.mean(embeddings, axis=0)
            query_variance = float(np.var(embeddings))

            # Calculate access entropy
            access_entropy = thermodynamics.calculate_access_entropy()

            # Detect dominant topics
            patterns = archaeologist.detect_recurring_patterns()
            dominant_topics = [(p[0], p[1] * p[2]) for p in patterns[:5]]

            # Detect phase
            phase = self._classify_phase(
                archaeologist,
                thermodynamics,
                query_variance,
                access_entropy
            )

            # Calculate temperature (activity level)
            recent_queries = sum(
                1 for q in archaeologist.query_history
                if (datetime.now() - q.timestamp).total_seconds() < 3600
            )
            temperature = min(1.0, recent_queries / 20.0)  # Normalize to 0-1

            # Calculate focus score (inverse of entropy)
            max_entropy = np.log2(len(archaeologist.query_clusters) or 1)
            focus_score = 1.0 - (access_entropy / max(max_entropy, 1.0))

            snapshot = CognitiveSnapshot(
                timestamp=datetime.now(),
                query_centroid=query_centroid,
                query_variance=query_variance,
                access_entropy=access_entropy,
                dominant_topics=dominant_topics,
                phase=phase,
                temperature=float(temperature),
                focus_score=float(focus_score)
            )

        self.snapshots.append(snapshot)
        self.current_phase = snapshot.phase
        self.phase_history.append((snapshot.timestamp, snapshot.phase))

        return snapshot

    def _classify_phase(
        self,
        archaeologist: QueryArchaeologist,
        thermodynamics: MemoryThermodynamics,
        query_variance: float,
        access_entropy: float
    ) -> CognitivePhase:
        """
        Classify current cognitive phase based on metrics.

        Why: Different phases need different optimizations. Exploration
        needs broad search, exploitation needs fast caching, learning
        needs persistence, crisis needs simplified responses.
        """
        query_entropy = archaeologist.calculate_query_entropy()

        # Get recent query count
        recent_count = sum(
            1 for q in archaeologist.query_history
            if (datetime.now() - q.timestamp).total_seconds() < 600
        )

        # Classification logic
        if recent_count == 0:
            return CognitivePhase.IDLE

        # Crisis: High activity + high variance + rapid queries
        if recent_count > 15 and query_variance > 0.5 and query_entropy > 2.5:
            return CognitivePhase.CRISIS

        # Exploration: High entropy, high variance
        if query_entropy > 2.0 and query_variance > 0.3:
            return CognitivePhase.EXPLORATION

        # Exploitation: Low entropy, low variance, recurring patterns
        if query_entropy < 1.0 and query_variance < 0.2:
            return CognitivePhase.EXPLOITATION

        # Learning: Moderate activity, expanding access patterns
        patterns = archaeologist.detect_recurring_patterns()
        if len(patterns) > 0 and access_entropy > 2.0:
            return CognitivePhase.LEARNING

        # Consolidation: Moderate entropy, reviewing past memories
        if 1.0 <= query_entropy <= 2.0:
            return CognitivePhase.CONSOLIDATION

        return CognitivePhase.IDLE

    def detect_phase_transitions(self) -> List[Tuple[datetime, CognitivePhase, CognitivePhase]]:
        """
        Find moments where cognitive phase changed.

        Returns: List of (timestamp, old_phase, new_phase)

        Why: Phase transitions are inflection points. They indicate
        significant cognitive shifts and are often where optimization
        opportunities or problems emerge.
        """
        transitions = []

        for i in range(1, len(self.phase_history)):
            prev_ts, prev_phase = self.phase_history[i - 1]
            curr_ts, curr_phase = self.phase_history[i]

            if prev_phase != curr_phase:
                transitions.append((curr_ts, prev_phase, curr_phase))

        return transitions

    def calculate_cognitive_stability(self, lookback_hours: int = 24) -> float:
        """
        Measure how stable the cognitive state has been.

        Returns: Stability score (0-1, higher = more stable)

        Why: Rapid phase changes might indicate confusion or system
        problems. Stable phases suggest effective cognitive flow.
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent_snapshots = [s for s in self.snapshots if s.timestamp > cutoff]

        if len(recent_snapshots) < 2:
            return 1.0

        # Count phase changes
        phase_changes = 0
        for i in range(1, len(recent_snapshots)):
            if recent_snapshots[i].phase != recent_snapshots[i - 1].phase:
                phase_changes += 1

        # Calculate stability (fewer changes = more stable)
        max_possible_changes = len(recent_snapshots) - 1
        stability = 1.0 - (phase_changes / max_possible_changes)

        return stability


class PredictivePrefetcher:
    """
    Predicts future memory needs and generates preloading suggestions.

    Why: If we can predict what the user will ask next, we can preload
    those memories, reducing latency and improving responsiveness. This
    turns reactive retrieval into proactive anticipation.
    """

    def __init__(
        self,
        markov_order: int = 2,
        confidence_threshold: float = 0.3
    ):
        self.markov_order = markov_order
        self.confidence_threshold = confidence_threshold

        # Transition matrix: (memory_sequence) -> {next_memory: count}
        self.transition_matrix: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

        # Query -> memory mappings
        self.query_memory_pairs: List[Tuple[str, List[str]]] = []

        # Prediction tracking
        self.predictions: List[PredictionEvent] = []

    def record_access_sequence(self, memory_sequence: List[str]) -> None:
        """
        Record a sequence of memory accesses for Markov chain learning.

        Why: Memory access patterns have structure. Given memories A and B
        were accessed, what's likely next? Build transition probabilities
        from observed sequences.
        """
        if len(memory_sequence) < self.markov_order + 1:
            return

        for i in range(len(memory_sequence) - self.markov_order):
            # Get state (past N memories)
            state = tuple(memory_sequence[i:i + self.markov_order])
            # Get next memory
            next_memory = memory_sequence[i + self.markov_order]

            self.transition_matrix[state][next_memory] += 1

    def predict_next_memories(
        self,
        current_sequence: List[str],
        top_k: int = 5
    ) -> PredictionEvent:
        """
        Predict next memories based on current access sequence.

        Returns: PredictionEvent with predicted memory IDs and confidences

        Why: Given the current context (recently accessed memories),
        what memories are likely to be needed next? Use learned transition
        probabilities to make informed predictions.
        """
        if len(current_sequence) < self.markov_order:
            # Not enough context, return empty prediction
            return PredictionEvent(
                timestamp=datetime.now(),
                predicted_memory_ids=[],
                confidence_scores=[]
            )

        # Get current state
        state = tuple(current_sequence[-self.markov_order:])

        # Get transition probabilities
        if state not in self.transition_matrix:
            return PredictionEvent(
                timestamp=datetime.now(),
                predicted_memory_ids=[],
                confidence_scores=[]
            )

        next_counts = self.transition_matrix[state]
        total_transitions = sum(next_counts.values())

        # Calculate probabilities and get top-k
        predictions = [
            (mid, count / total_transitions)
            for mid, count in next_counts.items()
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        predictions = predictions[:top_k]

        # Filter by confidence threshold
        predictions = [
            (mid, conf) for mid, conf in predictions
            if conf >= self.confidence_threshold
        ]

        event = PredictionEvent(
            timestamp=datetime.now(),
            predicted_memory_ids=[mid for mid, _ in predictions],
            confidence_scores=[conf for _, conf in predictions]
        )

        self.predictions.append(event)
        return event

    def validate_prediction(
        self,
        prediction_event: PredictionEvent,
        actual_accessed: List[str]
    ) -> float:
        """
        Validate a previous prediction against actual access.

        Returns: Accuracy score (0-1)

        Why: Continuous evaluation of prediction quality. If predictions
        are poor, adjust the model. Track accuracy over time to measure
        improvement and guide parameter tuning.
        """
        prediction_event.actual_accessed = actual_accessed

        if not prediction_event.predicted_memory_ids or not actual_accessed:
            prediction_event.prediction_accuracy = 0.0
            return 0.0

        # Calculate hits (predicted memories that were actually accessed)
        predicted_set = set(prediction_event.predicted_memory_ids)
        actual_set = set(actual_accessed)

        hits = len(predicted_set & actual_set)
        total_predictions = len(predicted_set)

        accuracy = hits / total_predictions if total_predictions > 0 else 0.0
        prediction_event.prediction_accuracy = accuracy

        return accuracy

    def calculate_prediction_performance(
        self,
        lookback_hours: int = 24
    ) -> Dict[str, float]:
        """
        Calculate aggregate prediction performance metrics.

        Why: Track how well the prediction system is working. Are
        predictions improving? What's the hit rate? This guides whether
        to trust and act on predictions.
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent_predictions = [
            p for p in self.predictions
            if p.timestamp > cutoff and p.prediction_accuracy is not None
        ]

        if not recent_predictions:
            return {
                "accuracy_mean": 0.0,
                "accuracy_std": 0.0,
                "total_predictions": 0,
                "validated_predictions": 0
            }

        accuracies = [p.prediction_accuracy for p in recent_predictions]

        return {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "total_predictions": len(self.predictions),
            "validated_predictions": len(recent_predictions)
        }


class ArchitecturalIntrospector:
    """
    Analyzes system performance and suggests optimization improvements.

    Why: The ultimate meta-cognitive capability - the system examining
    itself and recommending its own improvements. Self-optimization
    through introspection.
    """

    def __init__(
        self,
        performance_window_hours: int = 24,
        anomaly_threshold_std: float = 3.0
    ):
        self.performance_window = timedelta(hours=performance_window_hours)
        self.anomaly_threshold = anomaly_threshold_std
        self.insights: List[MetacognitiveInsight] = []

    def analyze_system_health(
        self,
        archaeologist: QueryArchaeologist,
        thermodynamics: MemoryThermodynamics,
        seismograph: CognitiveSeismograph,
        prefetcher: PredictivePrefetcher
    ) -> List[MetacognitiveInsight]:
        """
        Perform comprehensive system health analysis.

        Returns: List of insights and recommendations

        Why: Periodic health checks detect degradation, inefficiencies,
        and optimization opportunities before they become problems.
        """
        insights = []

        # Check query performance
        if archaeologist.query_history:
            exec_times = [q.execution_time_ms for q in archaeologist.query_history]
            mean_time = np.mean(exec_times)
            std_time = np.std(exec_times)

            # Detect slow queries
            slow_queries = [
                t for t in exec_times
                if t > mean_time + (2 * std_time)
            ]

            if len(slow_queries) > len(exec_times) * 0.1:  # >10% slow
                insights.append(MetacognitiveInsight(
                    timestamp=datetime.now(),
                    insight_type="performance_degradation",
                    severity="warning",
                    description=f"Detected {len(slow_queries)} slow queries (>2σ). Mean execution time: {mean_time:.2f}ms",
                    metrics={
                        "mean_exec_time_ms": float(mean_time),
                        "std_exec_time_ms": float(std_time),
                        "slow_query_count": len(slow_queries)
                    },
                    suggested_action="Consider optimizing retrieval pipeline, adding caching, or investigating hot memories",
                    estimated_impact="Could reduce mean query time by 20-40%"
                ))

        # Check memory access patterns
        hotspots = thermodynamics.detect_hotspots(top_k=10)
        if hotspots:
            total_temp = sum(temp for _, temp in hotspots)
            concentration = total_temp / len(thermodynamics.access_counts) if thermodynamics.access_counts else 0

            if concentration > 0.5:  # Top 10 memories are >50% of all access
                insights.append(MetacognitiveInsight(
                    timestamp=datetime.now(),
                    insight_type="memory_concentration",
                    severity="info",
                    description=f"Access highly concentrated on {len(hotspots)} memories ({concentration*100:.1f}% of activity)",
                    metrics={
                        "concentration_ratio": float(concentration),
                        "hotspot_count": len(hotspots),
                        "total_memories": len(thermodynamics.access_counts)
                    },
                    suggested_action="Implement aggressive caching for hotspot memories",
                    estimated_impact="Could reduce 50%+ of retrieval operations"
                ))

        # Check cognitive stability
        stability = seismograph.calculate_cognitive_stability()
        if stability < 0.5:
            insights.append(MetacognitiveInsight(
                timestamp=datetime.now(),
                insight_type="cognitive_instability",
                severity="warning",
                description=f"Low cognitive stability detected ({stability:.2f}). Frequent phase transitions suggest confusion or inefficient workflow",
                metrics={
                    "stability_score": float(stability),
                    "current_phase": seismograph.current_phase.value,
                    "transition_count": len(seismograph.detect_phase_transitions())
                },
                suggested_action="Analyze phase transition patterns. Consider providing user guidance or simplifying interface during crisis phases",
                estimated_impact="Could improve user experience and task completion rate"
            ))

        # Check prediction performance
        pred_perf = prefetcher.calculate_prediction_performance()
        if pred_perf["validated_predictions"] > 10:
            if pred_perf["accuracy_mean"] < 0.3:
                insights.append(MetacognitiveInsight(
                    timestamp=datetime.now(),
                    insight_type="poor_prediction_accuracy",
                    severity="info",
                    description=f"Prefetcher accuracy is low ({pred_perf['accuracy_mean']:.2%}). Predictions may not be useful",
                    metrics=pred_perf,
                    suggested_action="Consider increasing Markov order, or disable prefetching to save resources",
                    estimated_impact="Minor - prefetching currently providing little value"
                ))
            elif pred_perf["accuracy_mean"] > 0.6:
                insights.append(MetacognitiveInsight(
                    timestamp=datetime.now(),
                    insight_type="high_prediction_accuracy",
                    severity="info",
                    description=f"Prefetcher performing well ({pred_perf['accuracy_mean']:.2%}). Consider aggressive preloading",
                    metrics=pred_perf,
                    suggested_action="Enable automatic prefetching and increase cache size for predicted memories",
                    estimated_impact="Could reduce query latency by 30-50%"
                ))

        # Check for semantic drift issues
        drift_mag, drift_signal = archaeologist.detect_semantic_drift()
        if drift_signal == DriftSignal.SEMANTIC_COLLAPSE:
            insights.append(MetacognitiveInsight(
                timestamp=datetime.now(),
                insight_type="semantic_collapse",
                severity="critical",
                description=f"Semantic collapse detected (drift={drift_mag:.3f}). Query vocabulary is narrowing - possible fixation or confusion",
                metrics={
                    "drift_magnitude": float(drift_mag),
                    "drift_type": drift_signal.value,
                    "query_entropy": archaeologist.calculate_query_entropy()
                },
                suggested_action="URGENT: Investigate user workflow. May indicate frustration or being stuck in loop",
                estimated_impact="High - user may be blocked or experiencing poor system behavior"
            ))

        # Store insights
        self.insights.extend(insights)
        return insights

    def suggest_architectural_improvements(
        self,
        thermodynamics: MemoryThermodynamics
    ) -> List[MetacognitiveInsight]:
        """
        Generate long-term architectural optimization suggestions.

        Why: Beyond immediate performance issues, look for structural
        improvements. Different usage patterns might benefit from
        different architectures.
        """
        suggestions = []

        # Analyze memory community structure
        communities = thermodynamics.find_memory_communities()

        if len(communities) > 5:
            avg_community_size = np.mean([len(c) for c in communities])

            suggestions.append(MetacognitiveInsight(
                timestamp=datetime.now(),
                insight_type="community_structure_detected",
                severity="info",
                description=f"Detected {len(communities)} memory communities (avg size {avg_community_size:.1f}). Strong conceptual clustering present",
                metrics={
                    "community_count": len(communities),
                    "avg_community_size": float(avg_community_size),
                    "total_memories": len(thermodynamics.access_counts)
                },
                suggested_action="Consider implementing community-aware retrieval or hierarchical memory organization",
                estimated_impact="Could improve retrieval relevance and enable community-based caching"
            ))

        # Analyze access entropy
        entropy = thermodynamics.calculate_access_entropy()
        max_entropy = np.log2(len(thermodynamics.access_counts) or 1)
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0

        if entropy_ratio < 0.3:
            suggestions.append(MetacognitiveInsight(
                timestamp=datetime.now(),
                insight_type="low_access_diversity",
                severity="info",
                description=f"Access entropy is low ({entropy:.2f}/{max_entropy:.2f}). System heavily exploits small memory subset",
                metrics={
                    "access_entropy": float(entropy),
                    "max_entropy": float(max_entropy),
                    "entropy_ratio": float(entropy_ratio)
                },
                suggested_action="Consider aggressive memory pruning. Most memories are rarely accessed and could be archived",
                estimated_impact="Could reduce memory footprint by 60-80% with minimal impact"
            ))

        self.insights.extend(suggestions)
        return suggestions

    def export_diagnostics(self, output_path: Path) -> None:
        """
        Export comprehensive diagnostic report.

        Why: Enables offline analysis and long-term tracking. Export
        everything the system knows about itself for human review.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_insights": len(self.insights),
            "insights_by_severity": {
                "info": len([i for i in self.insights if i.severity == "info"]),
                "warning": len([i for i in self.insights if i.severity == "warning"]),
                "critical": len([i for i in self.insights if i.severity == "critical"])
            },
            "insights": [
                {
                    "timestamp": i.timestamp.isoformat(),
                    "type": i.insight_type,
                    "severity": i.severity,
                    "description": i.description,
                    "metrics": i.metrics,
                    "suggested_action": i.suggested_action,
                    "estimated_impact": i.estimated_impact
                }
                for i in self.insights
            ]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported diagnostic report to {output_path}")


class MetacognitiveMirror:
    """
    Main orchestrator for metacognitive analysis.

    Why: Coordinates all meta-analysis components to provide unified
    view of system self-awareness. The mirror reflects back to the
    system what it's doing and how it could improve.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize metacognitive mirror.

        Args:
            config: Configuration dict with component parameters

        Why: Config-driven architecture allows experimentation with
        different parameters and easy tuning based on system characteristics.
        """
        config = config or {}

        # Initialize components
        self.archaeologist = QueryArchaeologist(
            window_size=config.get("query_window_size", 100),
            cluster_epsilon=config.get("query_cluster_epsilon", 0.3),
            min_pattern_length=config.get("min_pattern_length", 3)
        )

        self.thermodynamics = MemoryThermodynamics(
            decay_half_life_hours=config.get("temperature_decay_hours", 48.0),
            burst_threshold=config.get("burst_threshold", 3.0)
        )

        self.seismograph = CognitiveSeismograph(
            snapshot_interval_minutes=config.get("snapshot_interval_minutes", 15)
        )

        self.prefetcher = PredictivePrefetcher(
            markov_order=config.get("markov_order", 2),
            confidence_threshold=config.get("prediction_confidence", 0.3)
        )

        self.introspector = ArchitecturalIntrospector(
            performance_window_hours=config.get("performance_window_hours", 24),
            anomaly_threshold_std=config.get("anomaly_threshold", 3.0)
        )

        logger.info("MetacognitiveMirror initialized - system self-awareness active")

    def record_query(self, event: QueryEvent) -> None:
        """Record a query event for analysis."""
        self.archaeologist.record_query(event)

        # Record memory accesses
        for i, memory_id in enumerate(event.retrieved_memory_ids):
            access_event = MemoryAccessEvent(
                timestamp=event.timestamp,
                memory_id=memory_id,
                access_type="retrieval",
                query_context=event.query_text,
                access_score=event.retrieval_scores[i] if i < len(event.retrieval_scores) else 0.0,
                co_accessed_memories=event.retrieved_memory_ids
            )
            self.thermodynamics.record_access(access_event)

        # Update sequence for prediction
        if event.retrieved_memory_ids:
            self.prefetcher.record_access_sequence(event.retrieved_memory_ids)

    def get_real_time_insights(self) -> Dict[str, Any]:
        """
        Get real-time metacognitive insights.

        Returns: Dict with current system state and immediate insights

        Why: Provides instantaneous feedback about system state for
        real-time adaptation and user feedback.
        """
        # Capture current snapshot
        snapshot = self.seismograph.capture_snapshot(
            self.archaeologist,
            self.thermodynamics
        )

        # Get hotspots
        hotspots = self.thermodynamics.detect_hotspots(top_k=5)

        # Get bursts
        bursts = self.thermodynamics.detect_bursts()

        # Detect drift
        drift_mag, drift_signal = self.archaeologist.detect_semantic_drift()

        # Get recurring patterns
        patterns = self.archaeologist.detect_recurring_patterns()

        return {
            "timestamp": datetime.now().isoformat(),
            "cognitive_phase": snapshot.phase.value,
            "temperature": snapshot.temperature,
            "focus_score": snapshot.focus_score,
            "access_entropy": snapshot.access_entropy,
            "query_entropy": self.archaeologist.calculate_query_entropy(),
            "drift_magnitude": float(drift_mag),
            "drift_signal": drift_signal.value,
            "hotspot_memories": [
                {"memory_id": mid, "temperature": temp}
                for mid, temp in hotspots
            ],
            "burst_memories": [
                {"memory_id": mid, "burst_intensity": intensity}
                for mid, intensity in bursts[:5]
            ],
            "recurring_patterns": [
                {"pattern": p[0], "frequency": p[1], "recency": p[2]}
                for p in patterns[:5]
            ],
            "dominant_topics": [
                {"topic": topic, "strength": strength}
                for topic, strength in snapshot.dominant_topics
            ]
        }

    def predict_next_access(
        self,
        current_memories: List[str],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Predict what memories will be accessed next.

        Returns: Dict with predictions and confidence scores
        """
        prediction = self.prefetcher.predict_next_memories(
            current_memories,
            top_k=top_k
        )

        return {
            "timestamp": prediction.timestamp.isoformat(),
            "predictions": [
                {
                    "memory_id": mid,
                    "confidence": conf
                }
                for mid, conf in zip(
                    prediction.predicted_memory_ids,
                    prediction.confidence_scores
                )
            ]
        }

    def run_health_check(self) -> List[Dict[str, Any]]:
        """
        Run comprehensive system health analysis.

        Returns: List of insights and recommendations
        """
        insights = self.introspector.analyze_system_health(
            self.archaeologist,
            self.thermodynamics,
            self.seismograph,
            self.prefetcher
        )

        return [
            {
                "timestamp": i.timestamp.isoformat(),
                "type": i.insight_type,
                "severity": i.severity,
                "description": i.description,
                "metrics": i.metrics,
                "suggested_action": i.suggested_action,
                "estimated_impact": i.estimated_impact
            }
            for i in insights
        ]

    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """
        Generate architectural optimization suggestions.

        Returns: List of optimization recommendations
        """
        suggestions = self.introspector.suggest_architectural_improvements(
            self.thermodynamics
        )

        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "type": s.insight_type,
                "description": s.description,
                "metrics": s.metrics,
                "suggested_action": s.suggested_action,
                "estimated_impact": s.estimated_impact
            }
            for s in suggestions
        ]

    def export_full_report(self, output_dir: Path) -> None:
        """
        Export comprehensive analysis report.

        Args:
            output_dir: Directory to write report files

        Why: Enables detailed offline analysis. Export everything the
        system has learned about itself.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export diagnostic insights
        self.introspector.export_diagnostics(
            output_dir / f"diagnostics_{timestamp}.json"
        )

        # Export query history
        query_export = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self.archaeologist.query_history),
            "query_clusters": len(self.archaeologist.query_clusters),
            "queries": [
                q.to_dict() for q in self.archaeologist.query_history
            ]
        }
        with open(output_dir / f"queries_{timestamp}.json", 'w') as f:
            json.dump(query_export, f, indent=2)

        # Export memory thermodynamics
        thermo_export = {
            "timestamp": datetime.now().isoformat(),
            "total_memories_tracked": len(self.thermodynamics.access_counts),
            "access_entropy": self.thermodynamics.calculate_access_entropy(),
            "hotspots": [
                {"memory_id": mid, "temperature": temp}
                for mid, temp in self.thermodynamics.detect_hotspots(top_k=20)
            ],
            "communities": [
                {"members": list(comm), "size": len(comm)}
                for comm in self.thermodynamics.find_memory_communities()
            ]
        }
        with open(output_dir / f"thermodynamics_{timestamp}.json", 'w') as f:
            json.dump(thermo_export, f, indent=2)

        # Export cognitive snapshots
        snapshot_export = {
            "timestamp": datetime.now().isoformat(),
            "total_snapshots": len(self.seismograph.snapshots),
            "current_phase": self.seismograph.current_phase.value,
            "cognitive_stability": self.seismograph.calculate_cognitive_stability(),
            "phase_transitions": [
                {
                    "timestamp": ts.isoformat(),
                    "from_phase": old.value,
                    "to_phase": new.value
                }
                for ts, old, new in self.seismograph.detect_phase_transitions()
            ]
        }
        with open(output_dir / f"cognitive_state_{timestamp}.json", 'w') as f:
            json.dump(snapshot_export, f, indent=2)

        # Export prediction performance
        pred_perf = self.prefetcher.calculate_prediction_performance()
        pred_export = {
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": pred_perf,
            "total_predictions": len(self.prefetcher.predictions)
        }
        with open(output_dir / f"predictions_{timestamp}.json", 'w') as f:
            json.dump(pred_export, f, indent=2)

        logger.info(f"Exported full metacognitive report to {output_dir}")


# Example usage and testing (if run directly)
if __name__ == "__main__":
    """
    Demonstration of metacognitive mirror capabilities.

    Why: Shows how to use the system and provides sanity check that
    all components work together correctly.
    """
    print("MetacognitiveMirror - Experimental Prototype")
    print("=" * 60)

    # Initialize mirror
    config = {
        "query_window_size": 50,
        "query_cluster_epsilon": 0.25,
        "markov_order": 2,
        "prediction_confidence": 0.3
    }
    mirror = MetacognitiveMirror(config)

    # Simulate some query events
    print("\nSimulating query events...")

    # Create synthetic embeddings (in real use, these come from SBERT)
    np.random.seed(42)

    for i in range(20):
        # Generate synthetic query
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        event = QueryEvent(
            timestamp=datetime.now(),
            query_text=f"Example query {i}",
            query_embedding=embedding,
            retrieved_memory_ids=[f"mem_{j}" for j in range(i % 5 + 1)],
            retrieval_scores=[0.8 - j * 0.1 for j in range(i % 5 + 1)],
            execution_time_ms=50.0 + np.random.randn() * 10,
            result_count=i % 5 + 1,
            semantic_gate_passed=True
        )

        mirror.record_query(event)

    # Get real-time insights
    print("\n--- Real-Time Insights ---")
    insights = mirror.get_real_time_insights()
    print(f"Cognitive Phase: {insights['cognitive_phase']}")
    print(f"Temperature: {insights['temperature']:.3f}")
    print(f"Focus Score: {insights['focus_score']:.3f}")
    print(f"Query Entropy: {insights['query_entropy']:.3f}")
    print(f"Drift Signal: {insights['drift_signal']} (magnitude: {insights['drift_magnitude']:.3f})")

    # Get predictions
    print("\n--- Predictions ---")
    predictions = mirror.predict_next_access(["mem_0", "mem_1"], top_k=3)
    print(f"Predicted next memories:")
    for pred in predictions['predictions']:
        print(f"  {pred['memory_id']}: {pred['confidence']:.2%}")

    # Run health check
    print("\n--- Health Check ---")
    health = mirror.run_health_check()
    print(f"Generated {len(health)} insights:")
    for insight in health:
        print(f"  [{insight['severity'].upper()}] {insight['type']}: {insight['description']}")

    # Get optimization suggestions
    print("\n--- Optimization Suggestions ---")
    opts = mirror.suggest_optimizations()
    print(f"Generated {len(opts)} suggestions:")
    for opt in opts:
        print(f"  {opt['type']}: {opt['suggested_action']}")

    print("\n" + "=" * 60)
    print("Prototype demonstration complete!")
    print("\nThis is a standalone experimental module.")
    print("To integrate with AGI Engine v13:")
    print("  1. Connect to retrieval_api.py for query events")
    print("  2. Add to config.yaml for parameter tuning")
    print("  3. Expose via API endpoints for real-time monitoring")
    print("  4. Set up background job for periodic health checks")
