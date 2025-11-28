"""
Cognitive Agent - Autonomous conversation driver.

This agent doesn't just answer questions - it THINKS about what it doesn't
know, detects context gaps, chains through memory communities, and drives
conversations toward insight based on cognitive state.

Components:
  - ContextGapDetector: Analyzes what's MISSING from context
  - StrategicFrameworkEngine: Injects decision frameworks when needed
  - ConversationDriver: Manages multi-turn proactive exploration

Cognitive Modes:
  - crisis: Grounding, simplified, probing for stability
  - exploration: Expansive, connecting dots, following threads
  - exploitation: Focused, direct, efficient answers
  - learning: Teaching mode, building structure
  - consolidation: Synthesis, connecting past context

Version: 1.0.0 (cog_twin)
"""

import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from numpy.typing import NDArray
import anthropic

# Import from local modules (no agi_engine dependency)
from metacognitive_mirror import (
    MetacognitiveMirror,
    QueryEvent,
    CognitivePhase,
    DriftSignal
)

logger = logging.getLogger(__name__)


class ResponseMode(Enum):
    """
    How the agent should respond.

    Why: Not all questions need the same treatment. Quick factual queries
    get direct answers. Complex decisions need exploration, framework
    injection, and synthesis.
    """
    DIRECT_ANSWER = "direct_answer"  # Straight retrieval, no exploration
    SHALLOW_EXPLORE = "shallow_explore"  # One hop into context
    DEEP_EXPLORE = "deep_explore"  # Multi-hop community traversal
    FRAMEWORK_INJECTION = "framework_injection"  # Add strategic lenses
    CRISIS_INTERVENTION = "crisis_intervention"  # Stabilize, simplify
    PATTERN_INTERRUPT = "pattern_interrupt"  # Break semantic collapse


class ContextGapType(Enum):
    """
    Types of missing context we can detect.

    Why: Different gaps require different exploration strategies.
    Missing entities need entity lookup, missing values need elicitation,
    missing relationships need graph traversal.
    """
    MISSING_ENTITY = "missing_entity"  # Query mentions entity not in results
    MISSING_EMOTIONAL_CONTEXT = "missing_emotional_context"  # Tone without stakes
    MISSING_VALUES = "missing_values"  # Decision without constraints
    MISSING_RELATIONSHIPS = "missing_relationships"  # People without dynamics
    MISSING_HISTORY = "missing_history"  # Reference without precedent
    MISSING_CONSTRAINTS = "missing_constraints"  # Problem without boundaries
    MISSING_STAKES = "missing_stakes"  # Decision without consequences
    TEMPORAL_GAP = "temporal_gap"  # Time discontinuity in context


class StrategicFramework(Enum):
    """
    Lens through which to analyze decisions.

    Why: Every decision can be viewed through multiple frameworks.
    Machiavellian prioritizes effectiveness, Boy Scout prioritizes
    integrity, Pragmatic finds the calibrated middle. Showing multiple
    perspectives reveals blind spots.
    """
    MACHIAVELLIAN = "machiavellian"  # Power, leverage, effectiveness > ethics
    BOY_SCOUT = "boy_scout"  # Integrity, trust, long-term relationships
    PRAGMATIC_HYBRID = "pragmatic"  # Strategic honesty, calibrated approach
    UTILITARIAN = "utilitarian"  # Greatest good for greatest number
    DEONTOLOGICAL = "deontological"  # Rules, duties, principles
    VIRTUE_ETHICS = "virtue_ethics"  # Character, excellence, flourishing


@dataclass
class ContextGap:
    """
    A detected gap in retrieved context.

    Why: Explicit representation of what's missing enables targeted
    exploration. Each gap type suggests specific retrieval strategies
    and probing questions.
    """
    gap_type: ContextGapType
    description: str
    severity: float  # 0-1, how critical is this gap
    suggested_exploration: List[str]  # Memory IDs or query terms to explore
    probing_questions: List[str]  # Questions to ask user/system


@dataclass
class MemoryChain:
    """
    A chain of related memories traversed during exploration.

    Why: Context isn't isolated - it chains through communities.
    Tracking the chain shows how we got from A to B and reveals
    the conceptual path through memory space.
    """
    root_memory_id: str
    chain: List[str]  # Memory IDs in traversal order
    relevance_scores: List[float]
    chain_coherence: float  # How well does this chain hang together
    termination_reason: str  # Why we stopped exploring


@dataclass
class StrategicAnalysis:
    """
    Multi-framework analysis of a decision.

    Why: Single-perspective analysis creates blind spots. Showing
    the same decision through multiple lenses reveals trade-offs,
    hidden assumptions, and strategic choices.
    """
    query: str
    frameworks: Dict[StrategicFramework, str]  # Framework → Analysis
    tensions: List[Tuple[StrategicFramework, StrategicFramework, str]]  # Conflicts
    synthesis: str  # Integrated recommendation
    confidence: float


@dataclass
class ProactiveResponse:
    """
    The agent's autonomous, context-aware response.

    Why: Goes beyond simple answers. Includes exploration chains,
    detected gaps, strategic analysis, and proactive suggestions.
    This is the full cognitive output.
    """
    query: str
    response_mode: ResponseMode
    cognitive_phase: CognitivePhase

    # Retrieved context
    primary_memories: List[Dict[str, Any]]
    explored_chains: List[MemoryChain]

    # Gap analysis
    detected_gaps: List[ContextGap]
    gap_severity: float  # Overall severity score

    # Strategic analysis (if applicable)
    strategic_analysis: Optional[StrategicAnalysis]

    # LLM synthesis
    synthesized_response: str
    probing_questions: List[str]  # Questions to drive conversation
    suggested_next_topics: List[str]  # Where conversation could go

    # Metadata
    timestamp: datetime
    exploration_depth: int  # How many hops we explored
    confidence: float
    processing_time_ms: float


class ContextGapDetector:
    """
    Analyzes retrieved context to find what's MISSING.

    Why: The most important information is often what you don't have.
    By explicitly detecting gaps, we can proactively fill them before
    generating a response. This prevents incomplete answers.
    """

    def __init__(
        self,
        gap_threshold: float = 0.3,
        entity_extraction_confidence: float = 0.5
    ):
        """
        Args:
            gap_threshold: Minimum severity to report a gap
            entity_extraction_confidence: Min confidence for entity detection

        Why: Configurable thresholds allow tuning sensitivity. Too sensitive
        creates noise, too relaxed misses critical gaps.
        """
        self.gap_threshold = gap_threshold
        self.entity_confidence = entity_extraction_confidence

    async def detect_gaps(
        self,
        query: str,
        query_embedding: NDArray[np.float32],
        retrieved_memories: List[Dict[str, Any]],
        cognitive_phase: CognitivePhase
    ) -> List[ContextGap]:
        """
        Detect context gaps in retrieved results.

        Why: Analyzes the query against retrieved context to find
        mismatches, missing entities, and conceptual holes that need
        to be filled before answering.
        """
        gaps = []

        # Detect missing entities
        entity_gaps = await self._detect_missing_entities(
            query,
            retrieved_memories
        )
        gaps.extend(entity_gaps)

        # Detect missing emotional context
        if self._has_emotional_markers(query):
            emotional_gap = await self._detect_missing_emotional_context(
                query,
                retrieved_memories
            )
            if emotional_gap:
                gaps.append(emotional_gap)

        # Detect missing values for decision queries
        if self._is_decision_query(query):
            values_gap = await self._detect_missing_values(
                query,
                retrieved_memories
            )
            if values_gap:
                gaps.append(values_gap)

            constraints_gap = await self._detect_missing_constraints(
                query,
                retrieved_memories
            )
            if constraints_gap:
                gaps.append(constraints_gap)

        # Detect missing relationship context
        relationship_gaps = await self._detect_missing_relationships(
            query,
            retrieved_memories
        )
        gaps.extend(relationship_gaps)

        # Detect temporal gaps
        temporal_gap = await self._detect_temporal_gaps(retrieved_memories)
        if temporal_gap:
            gaps.append(temporal_gap)

        # Filter by threshold
        significant_gaps = [
            gap for gap in gaps
            if gap.severity >= self.gap_threshold
        ]

        logger.info(
            f"Detected {len(significant_gaps)} significant context gaps "
            f"(filtered from {len(gaps)} total)"
        )

        return significant_gaps

    async def _detect_missing_entities(
        self,
        query: str,
        memories: List[Dict[str, Any]]
    ) -> List[ContextGap]:
        """
        Find entities mentioned in query but missing from retrieved context.

        Why: If you ask about your "boss" but no boss context is retrieved,
        that's a critical gap. We need to find those memories before answering.
        """
        gaps = []

        # Extract entities from query (simple heuristic - could use NER)
        query_lower = query.lower()

        # Common entity indicators
        entity_indicators = {
            "boss": ["manager", "supervisor", "leadership"],
            "team": ["colleagues", "coworkers", "group"],
            "project": ["initiative", "work", "task"],
            "client": ["customer", "stakeholder"],
            "deadline": ["timeline", "due date"],
            "meeting": ["discussion", "call", "sync"]
        }

        # Check each indicator
        for entity, synonyms in entity_indicators.items():
            if entity in query_lower:
                # Check if entity context exists in memories
                has_context = any(
                    entity in str(mem.get("content", "")).lower() or
                    any(syn in str(mem.get("content", "")).lower() for syn in synonyms)
                    for mem in memories
                )

                if not has_context:
                    gaps.append(ContextGap(
                        gap_type=ContextGapType.MISSING_ENTITY,
                        description=f"Query mentions '{entity}' but no {entity} context retrieved",
                        severity=0.8,  # High severity - direct entity reference
                        suggested_exploration=[entity] + synonyms,
                        probing_questions=[
                            f"Can you tell me more about the {entity} context?",
                            f"What's your relationship with this {entity}?",
                            f"Any previous history with this {entity}?"
                        ]
                    ))

        return gaps

    async def _detect_missing_emotional_context(
        self,
        query: str,
        memories: List[Dict[str, Any]]
    ) -> Optional[ContextGap]:
        """
        Detect urgent/emotional tone without corresponding stakes context.

        Why: "I need help NOW" without understanding what's at stake leads
        to misaligned advice. Emotional queries need emotional context.
        """
        if not self._has_emotional_markers(query):
            return None

        # Check if memories contain stakes/consequences
        has_stakes = any(
            any(marker in str(mem.get("content", "")).lower()
                for marker in ["consequence", "risk", "impact", "stake", "critical"])
            for mem in memories
        )

        if not has_stakes:
            return ContextGap(
                gap_type=ContextGapType.MISSING_EMOTIONAL_CONTEXT,
                description="Query has urgent/emotional tone but no stakes context retrieved",
                severity=0.9,  # Very high - emotional mismatch is dangerous
                suggested_exploration=["stakes", "consequences", "risks", "impact"],
                probing_questions=[
                    "What's at stake here?",
                    "What happens if this doesn't go well?",
                    "Why is this urgent right now?"
                ]
            )

        return None

    async def _detect_missing_values(
        self,
        query: str,
        memories: List[Dict[str, Any]]
    ) -> Optional[ContextGap]:
        """
        Decision queries without value/priority context.

        Why: "Should I do X?" can't be answered without knowing what you
        value. Integrity? Speed? Relationships? Values guide decisions.
        """
        # Check if memories contain values/priorities
        has_values = any(
            any(marker in str(mem.get("content", "")).lower()
                for marker in ["value", "priority", "important", "care about", "principle"])
            for mem in memories
        )

        if not has_values:
            return ContextGap(
                gap_type=ContextGapType.MISSING_VALUES,
                description="Decision query without value/priority context",
                severity=0.85,
                suggested_exploration=["values", "priorities", "principles", "what matters"],
                probing_questions=[
                    "What's most important to you in this situation?",
                    "What values are you trying to honor?",
                    "What would success look like?"
                ]
            )

        return None

    async def _detect_missing_constraints(
        self,
        query: str,
        memories: List[Dict[str, Any]]
    ) -> Optional[ContextGap]:
        """
        Decision queries without constraint context.

        Why: Constraints shape feasible solutions. Without knowing
        boundaries, we might suggest impossible actions.
        """
        has_constraints = any(
            any(marker in str(mem.get("content", "")).lower()
                for marker in ["constraint", "limit", "can't", "must", "requirement"])
            for mem in memories
        )

        if not has_constraints:
            return ContextGap(
                gap_type=ContextGapType.MISSING_CONSTRAINTS,
                description="Decision query without constraint context",
                severity=0.7,
                suggested_exploration=["constraints", "limitations", "requirements"],
                probing_questions=[
                    "What constraints are you working within?",
                    "What's off the table?",
                    "Any requirements that must be met?"
                ]
            )

        return None

    async def _detect_missing_relationships(
        self,
        query: str,
        memories: List[Dict[str, Any]]
    ) -> List[ContextGap]:
        """
        People mentioned without relationship/power dynamics.

        Why: Advice about dealing with "John" is very different if John
        is your peer vs your CEO. Power dynamics matter.
        """
        gaps = []

        # Simple heuristic: capital words might be names
        words = query.split()
        potential_names = [
            w for w in words
            if w[0].isupper() and w.lower() not in ["i", "the", "a", "an"]
        ]

        for name in potential_names:
            # Check if relationship context exists
            has_relationship = any(
                name in str(mem.get("content", "")) and
                any(marker in str(mem.get("content", "")).lower()
                    for marker in ["reports to", "manager", "peer", "relationship"])
                for mem in memories
            )

            if not has_relationship:
                gaps.append(ContextGap(
                    gap_type=ContextGapType.MISSING_RELATIONSHIPS,
                    description=f"Person '{name}' mentioned without relationship context",
                    severity=0.75,
                    suggested_exploration=[f"{name} relationship", f"{name} role"],
                    probing_questions=[
                        f"What's your relationship with {name}?",
                        f"What role does {name} play?",
                        f"How much influence does {name} have?"
                    ]
                ))

        return gaps

    async def _detect_temporal_gaps(
        self,
        memories: List[Dict[str, Any]]
    ) -> Optional[ContextGap]:
        """
        Detect discontinuities in temporal context.

        Why: If retrieved memories are from 6 months ago and now,
        with nothing in between, we're missing evolution context.
        """
        if len(memories) < 2:
            return None

        # Extract timestamps (if available)
        timestamps = []
        for mem in memories:
            if "timestamp" in mem:
                try:
                    ts = datetime.fromisoformat(mem["timestamp"])
                    timestamps.append(ts)
                except (ValueError, TypeError) as e:
                    self.logger.debug("Invalid timestamp format", timestamp=mem.get("timestamp"), error=str(e))

        if len(timestamps) < 2:
            return None

        # Sort and check for gaps
        timestamps.sort()
        max_gap_days = 0

        for i in range(1, len(timestamps)):
            gap_days = (timestamps[i] - timestamps[i-1]).days
            max_gap_days = max(max_gap_days, gap_days)

        # If gap > 30 days, flag it
        if max_gap_days > 30:
            return ContextGap(
                gap_type=ContextGapType.TEMPORAL_GAP,
                description=f"Large temporal gap in context ({max_gap_days} days)",
                severity=0.6,
                suggested_exploration=["recent", "current", "update"],
                probing_questions=[
                    "Has anything changed recently?",
                    f"What's happened in the last {max_gap_days} days?",
                    "Is this still current?"
                ]
            )

        return None

    def _has_emotional_markers(self, query: str) -> bool:
        """Check if query has emotional/urgent tone."""
        emotional_markers = [
            "urgent", "asap", "help", "worried", "stressed",
            "panic", "crisis", "now", "immediately", "desperate"
        ]
        return any(marker in query.lower() for marker in emotional_markers)

    def _is_decision_query(self, query: str) -> bool:
        """Check if query is asking for decision guidance."""
        decision_markers = [
            "should i", "what should", "help me decide",
            "which option", "what's the best", "advice on"
        ]
        return any(marker in query.lower() for marker in decision_markers)


class StrategicFrameworkEngine:
    """
    Generates multi-framework strategic analysis for decisions.

    Why: Every decision looks different through different lenses.
    Machiavelli sees power moves, Boy Scouts see integrity tests.
    Showing multiple perspectives reveals blind spots and trade-offs.
    """

    def __init__(self, anthropic_api_key: str):
        """
        Args:
            anthropic_api_key: API key for Claude

        Why: Framework analysis requires sophisticated reasoning.
        We use Claude to generate the multi-perspective analysis.
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

    async def generate_strategic_analysis(
        self,
        query: str,
        context_memories: List[Dict[str, Any]],
        frameworks: List[StrategicFramework]
    ) -> StrategicAnalysis:
        """
        Generate multi-framework analysis of decision.

        Why: Uses Claude to reason through the same decision from
        multiple strategic perspectives, then synthesize a nuanced
        recommendation that acknowledges trade-offs.
        """
        logger.info(f"Generating strategic analysis with {len(frameworks)} frameworks")

        # Build context from memories
        context_text = self._build_context_text(context_memories)

        # Build framework descriptions
        framework_descriptions = {
            StrategicFramework.MACHIAVELLIAN: "Machiavellian lens: Focus on power, leverage, positioning, and effectiveness. Ethics are secondary to outcomes. What move increases your influence?",
            StrategicFramework.BOY_SCOUT: "Boy Scout lens: Focus on integrity, trust, transparency, and long-term relationships. Do the right thing even when it's hard. What builds trust?",
            StrategicFramework.PRAGMATIC_HYBRID: "Pragmatic lens: Balance effectiveness and integrity. Strategic honesty, calibrated approach. What's wise given your values AND constraints?",
            StrategicFramework.UTILITARIAN: "Utilitarian lens: Greatest good for greatest number. Maximize overall welfare. What creates most value for all stakeholders?",
            StrategicFramework.DEONTOLOGICAL: "Deontological lens: Focus on rules, duties, principles. Some things are right/wrong regardless of consequences. What's your obligation?",
            StrategicFramework.VIRTUE_ETHICS: "Virtue Ethics lens: Focus on character, excellence, flourishing. What would the best version of yourself do?"
        }

        # Build prompt
        prompt = f"""You are a strategic advisor providing multi-perspective analysis.

QUERY:
{query}

CONTEXT:
{context_text}

TASK:
Analyze this decision through {len(frameworks)} different strategic frameworks. For each framework, provide:
1. The core analysis from that perspective
2. Recommended action
3. Risks and trade-offs

FRAMEWORKS:
{chr(10).join(f"- {fw.value}: {framework_descriptions[fw]}" for fw in frameworks)}

Then synthesize across frameworks to provide integrated guidance that acknowledges tensions.

Format your response as:

## Framework: [Name]
Analysis: [detailed analysis]
Recommendation: [what to do]
Risks: [what could go wrong]

[Repeat for each framework]

## Synthesis
[Integrated recommendation that acknowledges trade-offs]

## Key Tensions
[Where frameworks conflict and how to navigate]
"""

        try:
            # Call Claude for analysis
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-5-sonnet-20241022",  # Update to latest model when available
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            analysis_text = response.content[0].text

            # Parse response (simplified - would be more robust in production)
            framework_analyses = {}
            synthesis = ""
            tensions = []

            # Simple parsing logic
            for fw in frameworks:
                if fw.value in analysis_text.lower():
                    # Extract analysis for this framework
                    # (In production, would use more robust parsing)
                    framework_analyses[fw] = f"Analysis from {fw.value} perspective"

            # Extract synthesis section
            if "## Synthesis" in analysis_text:
                synthesis_idx = analysis_text.index("## Synthesis")
                tensions_idx = analysis_text.index("## Key Tensions") if "## Key Tensions" in analysis_text else len(analysis_text)
                synthesis = analysis_text[synthesis_idx:tensions_idx].strip()

            return StrategicAnalysis(
                query=query,
                frameworks=framework_analyses,
                tensions=tensions,
                synthesis=synthesis or analysis_text,  # Fall back to full text
                confidence=0.8
            )

        except Exception as e:
            logger.error(f"Strategic analysis failed: {e}")
            # Return fallback analysis
            return StrategicAnalysis(
                query=query,
                frameworks={fw: f"Analysis unavailable due to error" for fw in frameworks},
                tensions=[],
                synthesis="Unable to generate strategic analysis due to API error",
                confidence=0.0
            )

    def _build_context_text(self, memories: List[Dict[str, Any]]) -> str:
        """Build formatted context from memories."""
        if not memories:
            return "No context available"

        context_parts = []
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", "")
            context_parts.append(f"{i}. {content}")

        return "\n".join(context_parts)


class ConversationDriver:
    """
    Manages multi-turn proactive exploration.

    Why: Sometimes one retrieval isn't enough. We need to chain through
    memory communities, follow conceptual threads, and build complete
    context before responding. This manages that exploration process.
    """

    def __init__(
        self,
        max_chain_depth: int = 3,
        community_threshold: int = 2
    ):
        """
        Args:
            max_chain_depth: Maximum hops in memory chain
            community_threshold: Min co-access count for community membership

        Why: Prevents infinite exploration while allowing sufficient depth
        to build complete context.
        """
        self.max_chain_depth = max_chain_depth
        self.community_threshold = community_threshold

    async def explore_memory_chains(
        self,
        root_memories: List[str],
        memory_manager: Any,  # Would be actual MemoryManager instance
        gaps: List[ContextGap]
    ) -> List[MemoryChain]:
        """
        Chain through memory communities to fill context gaps.

        Why: Memories form communities through co-access. If we need
        more context, traverse the graph to find related memories that
        provide missing pieces.
        """
        chains = []

        for root_id in root_memories[:3]:  # Explore top 3 roots
            chain = await self._explore_from_root(
                root_id,
                memory_manager,
                gaps
            )
            if chain:
                chains.append(chain)

        logger.info(f"Explored {len(chains)} memory chains")
        return chains

    async def _explore_from_root(
        self,
        root_id: str,
        memory_manager: Any,
        gaps: List[ContextGap]
    ) -> Optional[MemoryChain]:
        """
        Explore memory chain starting from root.

        Why: Depth-first exploration with relevance-guided traversal.
        Stop when we hit max depth, find gap-filling context, or
        coherence drops too low.
        """
        visited = {root_id}
        chain = [root_id]
        relevance_scores = [1.0]  # Root has max relevance

        current_id = root_id
        depth = 0

        while depth < self.max_chain_depth:
            # Get neighbors (co-accessed memories)
            neighbors = await self._get_memory_neighbors(current_id, memory_manager)

            if not neighbors:
                break

            # Filter unvisited, score by relevance to gaps
            candidates = [
                (nid, score) for nid, score in neighbors
                if nid not in visited
            ]

            if not candidates:
                break

            # Pick most relevant
            next_id, relevance = max(candidates, key=lambda x: x[1])

            # Check coherence
            if relevance < 0.3:  # Coherence threshold
                break

            visited.add(next_id)
            chain.append(next_id)
            relevance_scores.append(relevance)
            current_id = next_id
            depth += 1

        # Calculate chain coherence
        coherence = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        return MemoryChain(
            root_memory_id=root_id,
            chain=chain,
            relevance_scores=relevance_scores,
            chain_coherence=coherence,
            termination_reason=f"Depth limit reached" if depth >= self.max_chain_depth else "Low relevance"
        )

    async def _get_memory_neighbors(
        self,
        memory_id: str,
        memory_manager: Any
    ) -> List[Tuple[str, float]]:
        """
        Get neighboring memories from community graph.

        Why: In production, this would query the memory thermodynamics
        co-access graph. For now, returns mock neighbors.

        INTEGRATION POINT: wires into MemoryThermodynamics co_access_graph
        See COGNITIVE_AGENT_INTEGRATION_GUIDE.md section 3
        """
        # ===================================================================
        # STUB: Replace with actual co-access graph traversal
        # ===================================================================
        # When ready to integrate, wire to thermodynamics co-access graph:
        #
        # if memory_manager and hasattr(memory_manager, 'mirror'):
        #     co_access_counts = memory_manager.mirror.thermodynamics.co_access_graph.get(
        #         memory_id,
        #         {}
        #     )
        #
        #     if co_access_counts:
        #         # Normalize counts to relevance scores (0-1)
        #         max_count = max(co_access_counts.values())
        #         neighbors = [
        #             (mid, count / max_count)
        #             for mid, count in co_access_counts.items()
        #         ]
        #         # Return top 10 by relevance
        #         return sorted(neighbors, key=lambda x: x[1], reverse=True)[:10]
        # ===================================================================

        # Mock implementation - returns empty for now
        logger.debug(f"Getting neighbors for {memory_id} (MOCK - not wired to co-access graph)")
        return []

    def generate_probing_questions(
        self,
        gaps: List[ContextGap],
        cognitive_phase: CognitivePhase
    ) -> List[str]:
        """
        Generate questions to fill detected gaps.

        Why: Proactive conversation driving. Instead of just noting
        gaps, we ask questions that fill them. Adapt question style
        to cognitive phase.
        """
        questions = []

        # Collect questions from gaps
        for gap in sorted(gaps, key=lambda g: g.severity, reverse=True)[:3]:
            questions.extend(gap.probing_questions[:2])  # Top 2 per gap

        # Adapt to cognitive phase
        if cognitive_phase == CognitivePhase.CRISIS:
            # Grounding questions
            questions = [
                "Let's start with the basics - what's happening right now?",
                "What's the most immediate concern?",
                "What do you need most urgently?"
            ][:2]

        elif cognitive_phase == CognitivePhase.EXPLORATION:
            # Expansive questions
            questions.extend([
                "What other aspects should we consider?",
                "How does this connect to your broader goals?"
            ])

        return questions[:5]  # Max 5 questions


class CognitiveAgent:
    """
    Autonomous conversation-driving agent.

    Why: The orchestrator that brings everything together. Makes the
    decision: answer directly or explore proactively. Manages the full
    cognitive loop from query to synthesis.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        anthropic_api_key: str,
        memory_manager: Any = None  # Would be actual MemoryManager
    ):
        """
        Initialize cognitive agent.

        Args:
            config: Configuration dict
            anthropic_api_key: API key for Claude
            memory_manager: Memory retrieval system

        Why: Config-driven for easy experimentation. Integrates all
        cognitive components into unified decision-making system.
        """
        self.config = config
        self.memory_manager = memory_manager

        # Initialize cognitive infrastructure
        self.mirror = MetacognitiveMirror(config.get("mirror", {}))
        self.gap_detector = ContextGapDetector(
            gap_threshold=config.get("gap_threshold", 0.3)
        )
        self.framework_engine = StrategicFrameworkEngine(anthropic_api_key)
        self.conversation_driver = ConversationDriver(
            max_chain_depth=config.get("max_chain_depth", 3)
        )

        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

        logger.info("CognitiveAgent initialized - autonomous mode active")

    async def process_query(
        self,
        query: str,
        query_embedding: NDArray[np.float32],
        user_context: Optional[Dict[str, Any]] = None
    ) -> ProactiveResponse:
        """
        Main entry point - process query with full cognitive pipeline.

        Why: The complete autonomous loop:
        1. Detect cognitive state
        2. Retrieve memories
        3. Detect gaps
        4. Decide: answer or explore
        5. Explore if needed
        6. Synthesize with LLM
        7. Generate proactive guidance

        This is AI that thinks before answering.
        """
        start_time = datetime.now()

        logger.info(f"Processing query: {query[:100]}...")

        # Step 1: Get cognitive state from mirror
        insights = self.mirror.get_real_time_insights()
        cognitive_phase = CognitivePhase(insights["cognitive_phase"])

        logger.info(f"Cognitive phase: {cognitive_phase.value}")

        # Step 2: Retrieve initial memories
        primary_memories = await self._retrieve_memories(query, query_embedding)

        # Step 3: Detect context gaps
        gaps = await self.gap_detector.detect_gaps(
            query,
            query_embedding,
            primary_memories,
            cognitive_phase
        )

        gap_severity = (
            sum(g.severity for g in gaps) / len(gaps)
            if gaps else 0.0
        )

        logger.info(f"Detected {len(gaps)} gaps with severity {gap_severity:.2f}")

        # Step 4: Decide response mode
        response_mode = self._decide_response_mode(
            query,
            cognitive_phase,
            gaps,
            gap_severity
        )

        logger.info(f"Response mode: {response_mode.value}")

        # Step 5: Explore if needed
        explored_chains = []
        if response_mode in [ResponseMode.DEEP_EXPLORE, ResponseMode.SHALLOW_EXPLORE]:
            memory_ids = [mem.get("id", f"mem_{i}") for i, mem in enumerate(primary_memories)]
            explored_chains = await self.conversation_driver.explore_memory_chains(
                memory_ids,
                self.memory_manager,
                gaps
            )

        # Step 6: Strategic analysis if decision query
        strategic_analysis = None
        if response_mode == ResponseMode.FRAMEWORK_INJECTION:
            frameworks = [
                StrategicFramework.MACHIAVELLIAN,
                StrategicFramework.BOY_SCOUT,
                StrategicFramework.PRAGMATIC_HYBRID
            ]
            strategic_analysis = await self.framework_engine.generate_strategic_analysis(
                query,
                primary_memories,
                frameworks
            )

        # Step 7: Synthesize response with Claude
        synthesized_response = await self._synthesize_response(
            query,
            primary_memories,
            explored_chains,
            gaps,
            cognitive_phase,
            strategic_analysis
        )

        # Step 8: Generate proactive questions
        probing_questions = self.conversation_driver.generate_probing_questions(
            gaps,
            cognitive_phase
        )

        # Step 9: Suggest next topics
        suggested_topics = self._suggest_next_topics(
            primary_memories,
            explored_chains,
            gaps
        )

        # Calculate metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        confidence = self._calculate_confidence(gap_severity, cognitive_phase)

        # Record query in mirror
        query_event = QueryEvent(
            timestamp=datetime.now(),
            query_text=query,
            query_embedding=query_embedding,
            retrieved_memory_ids=[mem.get("id", f"mem_{i}") for i, mem in enumerate(primary_memories)],
            retrieval_scores=[mem.get("score", 0.0) for mem in primary_memories],
            execution_time_ms=processing_time,
            result_count=len(primary_memories),
            semantic_gate_passed=True
        )
        self.mirror.record_query(query_event)

        return ProactiveResponse(
            query=query,
            response_mode=response_mode,
            cognitive_phase=cognitive_phase,
            primary_memories=primary_memories,
            explored_chains=explored_chains,
            detected_gaps=gaps,
            gap_severity=gap_severity,
            strategic_analysis=strategic_analysis,
            synthesized_response=synthesized_response,
            probing_questions=probing_questions,
            suggested_next_topics=suggested_topics,
            timestamp=datetime.now(),
            exploration_depth=len(explored_chains),
            confidence=confidence,
            processing_time_ms=processing_time
        )

    async def _retrieve_memories(
        self,
        query: str,
        query_embedding: NDArray[np.float32]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories.

        Why: In production, this calls the actual memory retrieval system.
        For now, returns mock memories for testing.

        INTEGRATION POINT: wires into retrieval_api.py or memory_manager
        See COGNITIVE_AGENT_INTEGRATION_GUIDE.md section 4
        """
        # ===================================================================
        # STUB: Replace with actual memory retrieval
        # ===================================================================
        # When ready to integrate, wire to your actual retrieval system:
        #
        # if self.memory_manager is not None:
        #     results = await self.memory_manager.retrieve(
        #         query=query,
        #         query_embedding=query_embedding,
        #         top_k=10
        #     )
        #     return [
        #         {
        #             "id": r.memory_id,
        #             "content": r.content,
        #             "score": r.similarity_score,
        #             "timestamp": r.created_at.isoformat(),
        #         }
        #         for r in results
        #     ]
        # ===================================================================

        # Mock implementation for standalone testing
        logger.info("Retrieving memories (MOCK - not wired to actual retrieval)")
        return [
            {
                "id": "mem_1",
                "content": "Mock memory content 1",
                "score": 0.85,
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "mem_2",
                "content": "Mock memory content 2",
                "score": 0.72,
                "timestamp": datetime.now().isoformat()
            }
        ]

    def _decide_response_mode(
        self,
        query: str,
        cognitive_phase: CognitivePhase,
        gaps: List[ContextGap],
        gap_severity: float
    ) -> ResponseMode:
        """
        Decide how to respond based on state and gaps.

        Why: The critical decision point. High gaps → explore.
        Crisis phase → intervention. Decision query → framework injection.
        This is where autonomy happens.
        """
        # Crisis phase gets intervention
        if cognitive_phase == CognitivePhase.CRISIS:
            return ResponseMode.CRISIS_INTERVENTION

        # Semantic collapse gets pattern interrupt
        drift_mag, drift_signal = self.mirror.archaeologist.detect_semantic_drift()
        if drift_signal == DriftSignal.SEMANTIC_COLLAPSE:
            return ResponseMode.PATTERN_INTERRUPT

        # High severity gaps → deep exploration
        if gap_severity > 0.7:
            return ResponseMode.DEEP_EXPLORE

        # Medium gaps → shallow exploration
        if gap_severity > 0.4:
            return ResponseMode.SHALLOW_EXPLORE

        # Decision queries get framework injection
        if self._is_decision_query(query):
            return ResponseMode.FRAMEWORK_INJECTION

        # Default: direct answer
        return ResponseMode.DIRECT_ANSWER

    def _is_decision_query(self, query: str) -> bool:
        """Check if query is asking for decision guidance."""
        decision_markers = [
            "should i", "what should", "help me decide",
            "which option", "what's the best", "advice on"
        ]
        return any(marker in query.lower() for marker in decision_markers)

    async def _synthesize_response(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        chains: List[MemoryChain],
        gaps: List[ContextGap],
        phase: CognitivePhase,
        strategic_analysis: Optional[StrategicAnalysis]
    ) -> str:
        """
        Synthesize final response using Claude.

        Why: Bring together all context, gap analysis, strategic
        frameworks, and cognitive state into coherent response.
        This is where autonomous exploration becomes useful output.
        """
        # Build comprehensive context
        context_parts = []

        # Add primary memories
        context_parts.append("PRIMARY CONTEXT:")
        for mem in memories[:5]:
            context_parts.append(f"- {mem.get('content', 'N/A')}")

        # Add explored chains
        if chains:
            context_parts.append("\nEXPLORED MEMORY CHAINS:")
            for chain in chains:
                context_parts.append(f"- Chain from {chain.root_memory_id} (coherence: {chain.chain_coherence:.2f})")

        # Add gap analysis
        if gaps:
            context_parts.append("\nDETECTED CONTEXT GAPS:")
            for gap in gaps[:3]:
                context_parts.append(f"- {gap.description} (severity: {gap.severity:.2f})")

        # Add strategic analysis if present
        if strategic_analysis:
            context_parts.append("\nSTRATEGIC ANALYSIS:")
            context_parts.append(strategic_analysis.synthesis)

        context_text = "\n".join(context_parts)

        # Adapt prompt to cognitive phase
        phase_instructions = {
            CognitivePhase.CRISIS: "User is in crisis mode. Provide grounding, simplified guidance. Focus on immediate stabilization.",
            CognitivePhase.EXPLORATION: "User is exploring. Be expansive, make connections, follow threads.",
            CognitivePhase.EXPLOITATION: "User wants focused answer. Be direct and efficient.",
            CognitivePhase.CONSOLIDATION: "User is consolidating. Help synthesize and connect past context.",
            CognitivePhase.LEARNING: "User is learning. Provide structure, teach concepts.",
            CognitivePhase.IDLE: "Low activity. Provide thorough response."
        }

        prompt = f"""You are an autonomous cognitive agent with deep context awareness.

COGNITIVE STATE: {phase.value}
INSTRUCTION: {phase_instructions.get(phase, "Provide helpful response")}

USER QUERY:
{query}

ASSEMBLED CONTEXT:
{context_text}

TASK:
Synthesize a response that:
1. Directly addresses the query
2. Acknowledges any context gaps we detected
3. Incorporates strategic analysis if provided
4. Adapts tone/depth to cognitive phase
5. Is proactive - don't just answer, guide the conversation forward

Respond naturally and helpfully."""

        try:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-5-sonnet-20241022",  # Update to latest model when available
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"I encountered an error synthesizing a response. Based on the context, here's what I can say: {context_text[:500]}"

    def _suggest_next_topics(
        self,
        memories: List[Dict[str, Any]],
        chains: List[MemoryChain],
        gaps: List[ContextGap]
    ) -> List[str]:
        """
        Suggest where conversation could go next.

        Why: Proactive guidance. Show the user what they might want
        to explore based on current context and detected gaps.
        """
        topics = []

        # Topics from gaps
        for gap in gaps[:2]:
            if gap.suggested_exploration:
                topics.append(f"Explore {gap.suggested_exploration[0]} context")

        # Topics from memory chains
        for chain in chains[:2]:
            topics.append(f"Follow chain from {chain.root_memory_id}")

        return topics[:5]

    def _calculate_confidence(
        self,
        gap_severity: float,
        phase: CognitivePhase
    ) -> float:
        """
        Calculate confidence in response quality.

        Why: Self-awareness about response quality. High gaps → low
        confidence. Crisis phase → lower confidence. Be honest about
        limitations.
        """
        base_confidence = 1.0 - (gap_severity * 0.5)

        # Adjust for phase
        phase_modifiers = {
            CognitivePhase.CRISIS: 0.7,  # Lower confidence in crisis
            CognitivePhase.EXPLORATION: 0.9,
            CognitivePhase.EXPLOITATION: 1.0,
            CognitivePhase.CONSOLIDATION: 0.95,
            CognitivePhase.LEARNING: 0.9,
            CognitivePhase.IDLE: 0.85
        }

        phase_modifier = phase_modifiers.get(phase, 0.9)

        return base_confidence * phase_modifier


# Example usage
if __name__ == "__main__":
    """
    Demonstration of cognitive agent capabilities.

    Why: Shows integration of all components and autonomous behavior.
    """
    print("CognitiveAgent - Autonomous Conversation Driver")
    print("=" * 60)
    print("\nThis agent doesn't just answer - it THINKS about what it doesn't know.")
    print("\nKey capabilities:")
    print("  1. Detects cognitive state (crisis/exploration/etc)")
    print("  2. Identifies context gaps")
    print("  3. Decides: answer directly or explore proactively")
    print("  4. Chains through memory communities")
    print("  5. Injects strategic frameworks for decisions")
    print("  6. Generates probing questions")
    print("  7. Drives conversation toward insight")
    print("\n" + "=" * 60)
    print("\nTo run:")
    print("  agent = CognitiveAgent(config, api_key, memory_manager)")
    print("  response = await agent.process_query(query, embedding)")
    print("\nIntegration needed:")
    print("  - Connect to actual memory retrieval system")
    print("  - Hook up thermodynamics co-access graph")
    print("  - Add to API layer for production use")
    print("  - Configure via config.yaml")
