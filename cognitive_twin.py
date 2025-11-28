"""
Cognitive Twin - Personalized digital twin that thinks like YOU think.

Learns your values, priorities, decision patterns, and communication style.
Frameworks evolve based on your feedback. Self-corrects when uncertain.

Enhancements over CognitiveAgent:
  - User profile system (values, priorities, communication style)
  - Framework evolution (learns which frameworks you prefer)
  - Self-correction loops (re-explores when confidence < threshold)
  - Enhanced NER for gap detection
  - Chain visualization output
  - Crisis mode escalation
  - Preference learning from feedback

Version: 1.0.0 (cog_twin)
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray

# Import from local modules (no agi_engine dependency)
from cognitive_agent import (
    CognitiveAgent,
    StrategicFramework,
    ProactiveResponse
)

from metacognitive_mirror import (
    CognitivePhase
)

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """
    Your cognitive fingerprint - values, priorities, patterns.

    Why: A digital twin needs to know WHO it's mirroring. Your values,
    decision-making patterns, communication style, risk tolerance.
    This profile evolves as the twin learns from interactions.
    """
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Core values and priorities
    values: Dict[str, float] = field(default_factory=dict)  # value -> importance (0-1)
    priorities: List[str] = field(default_factory=list)  # Ordered list
    constraints: List[str] = field(default_factory=list)  # Hard boundaries

    # Decision-making patterns
    risk_tolerance: float = 0.5  # 0=risk-averse, 1=risk-seeking
    time_horizon: str = "medium"  # short, medium, long
    decision_speed: str = "deliberate"  # quick, deliberate, slow

    # Framework preferences (learned from feedback)
    framework_scores: Dict[str, float] = field(default_factory=dict)  # framework -> preference
    framework_usage_count: Dict[str, int] = field(default_factory=dict)

    # Communication style
    preferred_depth: str = "detailed"  # concise, balanced, detailed
    preferred_tone: str = "direct"  # direct, diplomatic, casual
    likes_probing_questions: bool = True

    # Crisis handling preferences
    crisis_escalation_threshold: float = 0.8  # When to recommend human help
    crisis_communication_style: str = "grounding"  # grounding, solution-focused

    # Learning metadata
    total_interactions: int = 0
    feedback_received: int = 0
    profile_confidence: float = 0.3  # How confident we are in this profile

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile for storage."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "values": self.values,
            "priorities": self.priorities,
            "constraints": self.constraints,
            "risk_tolerance": self.risk_tolerance,
            "time_horizon": self.time_horizon,
            "decision_speed": self.decision_speed,
            "framework_scores": self.framework_scores,
            "framework_usage_count": self.framework_usage_count,
            "preferred_depth": self.preferred_depth,
            "preferred_tone": self.preferred_tone,
            "likes_probing_questions": self.likes_probing_questions,
            "crisis_escalation_threshold": self.crisis_escalation_threshold,
            "crisis_communication_style": self.crisis_communication_style,
            "total_interactions": self.total_interactions,
            "feedback_received": self.feedback_received,
            "profile_confidence": self.profile_confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Deserialize profile from storage."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


@dataclass
class InteractionFeedback:
    """
    Feedback on a specific interaction.

    Why: The twin learns from your reactions. Explicit feedback ("that
    was helpful") and implicit feedback (you followed the advice) both
    shape how the twin evolves.
    """
    timestamp: datetime
    query_id: str  # Links to original query
    response_mode_used: str
    frameworks_used: List[str]

    # Explicit feedback
    was_helpful: Optional[bool] = None
    rating: Optional[int] = None  # 1-5 scale

    # Implicit feedback signals
    followed_advice: Optional[bool] = None
    explored_suggested_topics: bool = False
    asked_follow_up: bool = False

    # What to learn
    preferred_framework: Optional[str] = None  # Which framework resonated
    feedback_text: Optional[str] = None


@dataclass
class ChainVisualization:
    """
    Graph representation of memory exploration chains.

    Why: To visualize how the twin thinks - what memories it chained
    through, why it made those connections. Makes the thought process
    transparent and debuggable.
    """
    query: str
    nodes: List[Dict[str, Any]]  # memory_id, content, type
    edges: List[Dict[str, Any]]  # source, target, relation, weight
    root_node: str
    terminal_nodes: List[str]
    chain_depth: int
    exploration_path: List[str]  # Ordered traversal

    def to_networkx_dict(self) -> Dict[str, Any]:
        """Export as NetworkX-compatible dict."""
        return {
            "directed": True,
            "graph": {
                "query": self.query,
                "depth": self.chain_depth
            },
            "nodes": self.nodes,
            "links": self.edges  # NetworkX uses "links" instead of "edges"
        }


class EnhancedNER:
    """
    Enhanced Named Entity Recognition for gap detection.

    Why: Better entity extraction = better gap detection. Uses pattern
    matching and (when available) spaCy for production NER. Detects
    people, organizations, concepts, values, constraints.
    """

    def __init__(self, use_spacy: bool = False):
        """
        Args:
            use_spacy: If True, try to use spaCy for NER

        Why: Pattern matching works standalone, spaCy adds accuracy.
        Graceful degradation if spaCy not available.
        """
        self.use_spacy = use_spacy
        self.nlp = None

        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy NER loaded")
            except Exception as e:
                logger.warning(f"spaCy not available: {e}. Using pattern matching.")
                self.use_spacy = False

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.

        Returns: Dict mapping entity type to list of entities
        """
        if self.use_spacy and self.nlp:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_patterns(text)

    def _extract_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy."""
        doc = self.nlp(text)

        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)

        return dict(entities)

    def _extract_with_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using pattern matching.

        Why: Works standalone without dependencies. Detects common
        patterns like capitalized names, value indicators, etc.
        """
        entities = defaultdict(list)

        # Detect people (capitalized words that might be names)
        words = text.split()
        for i, word in enumerate(words):
            # Simple heuristic: capitalized word not at sentence start
            if (word[0].isupper() and
                i > 0 and
                words[i-1][-1] not in '.!?' and
                word.lower() not in ['i', 'the', 'a', 'an']):
                entities['PERSON'].append(word)

        # Detect value indicators
        value_patterns = [
            "important", "priority", "value", "care about",
            "matter", "principle", "believe"
        ]
        for pattern in value_patterns:
            if pattern in text.lower():
                # Extract context around pattern
                idx = text.lower().index(pattern)
                context_start = max(0, idx - 50)
                context_end = min(len(text), idx + 50)
                entities['VALUE'].append(text[context_start:context_end])

        # Detect constraint indicators
        constraint_patterns = [
            "can't", "cannot", "must not", "limited by",
            "constraint", "requirement", "have to"
        ]
        for pattern in constraint_patterns:
            if pattern in text.lower():
                idx = text.lower().index(pattern)
                context_start = max(0, idx - 50)
                context_end = min(len(text), idx + 50)
                entities['CONSTRAINT'].append(text[context_start:context_end])

        return dict(entities)


class ProfileLearner:
    """
    Learns and updates user profile from interactions.

    Why: The digital twin evolves. Every interaction reveals something
    about your values, preferences, patterns. This component extracts
    that signal and updates the profile.
    """

    def __init__(self, ner: EnhancedNER):
        self.ner = ner

    async def learn_from_query(
        self,
        query: str,
        profile: UserProfile
    ) -> None:
        """
        Extract profile signals from query.

        Why: Queries reveal values and priorities. "I need to decide
        between speed and quality" → values both but in conflict.
        """
        entities = self.ner.extract_entities(query)

        # Extract values
        if 'VALUE' in entities:
            for value_context in entities['VALUE']:
                # Simple extraction - could be more sophisticated
                for value_word in ['integrity', 'speed', 'quality', 'trust', 'growth']:
                    if value_word in value_context.lower():
                        # Increment value importance
                        current = profile.values.get(value_word, 0.5)
                        profile.values[value_word] = min(1.0, current + 0.05)

        # Extract constraints
        if 'CONSTRAINT' in entities:
            for constraint in entities['CONSTRAINT']:
                if constraint not in profile.constraints:
                    profile.constraints.append(constraint)

        profile.last_updated = datetime.now()

    async def learn_from_feedback(
        self,
        feedback: InteractionFeedback,
        profile: UserProfile
    ) -> None:
        """
        Update profile based on feedback.

        Why: Explicit feedback is gold. If you rate a machiavellian
        analysis highly, the twin learns you value that lens.
        """
        # Update framework preferences
        if feedback.preferred_framework:
            current = profile.framework_scores.get(feedback.preferred_framework, 0.5)
            # Boost if positive feedback
            if feedback.was_helpful or (feedback.rating and feedback.rating >= 4):
                profile.framework_scores[feedback.preferred_framework] = min(1.0, current + 0.1)
            else:
                profile.framework_scores[feedback.preferred_framework] = max(0.0, current - 0.1)

        # Update communication preferences based on implicit signals
        if feedback.asked_follow_up:
            profile.likes_probing_questions = True

        # Increase profile confidence with more feedback
        profile.feedback_received += 1
        profile.profile_confidence = min(
            1.0,
            0.3 + (profile.feedback_received * 0.05)
        )

        profile.last_updated = datetime.now()


class CognitiveTwin(CognitiveAgent):
    """
    Personalized digital twin - thinks like YOU think.

    Why: Extends CognitiveAgent with personalization, learning, and
    self-correction. Knows your values, learns from feedback, evolves
    its frameworks based on your preferences.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        api_key: str,
        memory_manager: Any = None,
        user_profile: Optional[UserProfile] = None,
        profile_path: Optional[Path] = None
    ):
        """
        Initialize cognitive twin.

        Args:
            config: Configuration dict
            api_key: API key for LLM (Anthropic, Grok, etc)
            memory_manager: Memory retrieval system
            user_profile: Existing user profile (if loading)
            profile_path: Path to save/load profile

        Why: Adds profile management to base cognitive agent. Profile
        persists across sessions, learning accumulates.
        """
        super().__init__(config, api_key, memory_manager)

        # Load or create user profile
        self.profile_path = profile_path or Path("./data/user_profiles/default_profile.json")

        if user_profile:
            self.profile = user_profile
        elif self.profile_path.exists():
            self.profile = self._load_profile()
            logger.info(f"Loaded user profile (confidence: {self.profile.profile_confidence:.2f})")
        else:
            self.profile = UserProfile(user_id="default")
            logger.info("Created new user profile")

        # Enhanced components
        self.ner = EnhancedNER(use_spacy=config.get("use_spacy", False))
        self.learner = ProfileLearner(self.ner)

        # Self-correction config
        self.self_correction_enabled = config.get("self_correction_enabled", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)

        # Feedback tracking
        self.feedback_history: List[InteractionFeedback] = []

        logger.info(f"CognitiveTwin initialized (profile confidence: {self.profile.profile_confidence:.2f})")

    async def process_query(
        self,
        query: str,
        query_embedding: NDArray[np.float32],
        user_context: Optional[Dict[str, Any]] = None
    ) -> ProactiveResponse:
        """
        Process query with personalization and self-correction.

        Why: Overrides base process_query to add:
        1. Profile learning from query
        2. Personalized framework selection
        3. Self-correction loop if confidence low
        4. Crisis escalation if needed
        """
        # Learn from query
        await self.learner.learn_from_query(query, self.profile)

        # Get base response
        response = await super().process_query(query, query_embedding, user_context)

        # Self-correction loop
        if (self.self_correction_enabled and
            response.confidence < self.confidence_threshold):

            logger.info(
                f"Low confidence ({response.confidence:.2f}), "
                f"triggering self-correction"
            )
            response = await self._self_correct(query, query_embedding, response)

        # Crisis escalation check
        if (response.cognitive_phase == CognitivePhase.CRISIS and
            response.gap_severity > self.profile.crisis_escalation_threshold):

            logger.warning(f"Crisis escalation threshold exceeded: {response.gap_severity:.2f}")
            response = await self._escalate_crisis(query, response)

        # Personalize response based on profile
        response = self._personalize_response(response)

        # Update profile
        self.profile.total_interactions += 1
        self._save_profile()

        return response

    async def _self_correct(
        self,
        query: str,
        query_embedding: NDArray[np.float32],
        initial_response: ProactiveResponse
    ) -> ProactiveResponse:
        """
        Self-correction loop when confidence is low.

        Why: If the twin isn't confident, it should explore deeper,
        ask clarifying questions, or explicitly note uncertainty.
        Don't pretend certainty when you don't have it.
        """
        logger.info("Initiating self-correction: deeper exploration")

        # Force deeper exploration
        old_max_depth = self.conversation_driver.max_chain_depth
        self.conversation_driver.max_chain_depth = old_max_depth + 2

        # Re-explore with deeper chains
        memory_ids = [mem.get("id", f"mem_{i}") for i, mem in enumerate(initial_response.primary_memories)]
        additional_chains = await self.conversation_driver.explore_memory_chains(
            memory_ids,
            self.memory_manager,
            initial_response.detected_gaps
        )

        # Restore original depth
        self.conversation_driver.max_chain_depth = old_max_depth

        # Re-synthesize with additional context
        all_chains = initial_response.explored_chains + additional_chains

        corrected_response_text = await self._synthesize_response(
            query,
            initial_response.primary_memories,
            all_chains,
            initial_response.detected_gaps,
            initial_response.cognitive_phase,
            initial_response.strategic_analysis
        )

        # Prepend uncertainty disclaimer
        corrected_response_text = (
            "[SELF-CORRECTION ACTIVE] Initial confidence was low, so I explored deeper. "
            "Here's what I found:\n\n" + corrected_response_text
        )

        # Update response
        initial_response.synthesized_response = corrected_response_text
        initial_response.explored_chains = all_chains
        initial_response.exploration_depth = len(all_chains)
        initial_response.confidence = min(1.0, initial_response.confidence + 0.2)

        return initial_response

    async def _escalate_crisis(
        self,
        query: str,
        response: ProactiveResponse
    ) -> ProactiveResponse:
        """
        Escalate crisis situations to human oversight.

        Why: When gap severity exceeds threshold in crisis mode,
        the twin recognizes its limitations. Better to escalate
        than give bad advice in high-stakes situations.
        """
        escalation_message = (
            f"\n\n CRISIS ESCALATION \n"
            f"Gap severity ({response.gap_severity:.2f}) exceeds your threshold "
            f"({self.profile.crisis_escalation_threshold:.2f}).\n\n"
            f"I'm detecting a high-stakes situation with significant context gaps. "
            f"I recommend:\n"
            f"1. Taking a moment to ground yourself\n"
            f"2. Gathering more context before deciding\n"
            f"3. Consulting a trusted human advisor\n\n"
            f"Missing context:\n"
        )

        for gap in response.detected_gaps[:3]:
            escalation_message += f"  - {gap.description}\n"

        response.synthesized_response = escalation_message + "\n" + response.synthesized_response

        return response

    def _personalize_response(self, response: ProactiveResponse) -> ProactiveResponse:
        """
        Adapt response to user's communication preferences.

        Why: Your twin should talk like you want to be talked to.
        Some people want concise, others want detailed. Some want
        probing questions, others find them annoying.
        """
        # Adjust probing questions based on preference
        if not self.profile.likes_probing_questions:
            response.probing_questions = response.probing_questions[:1]  # Just one

        # Adjust depth based on preference
        if self.profile.preferred_depth == "concise":
            # Truncate response
            if len(response.synthesized_response) > 500:
                response.synthesized_response = (
                    response.synthesized_response[:500] +
                    "\n\n[Concise mode: Full analysis available if needed]"
                )

        # Adjust tone (would require LLM rewrite in production)
        # For now, just add prefix based on tone preference
        tone_prefixes = {
            "direct": "",  # No prefix
            "diplomatic": "I want to share some thoughts on this: ",
            "casual": "Hey, so here's what I'm thinking: "
        }

        prefix = tone_prefixes.get(self.profile.preferred_tone, "")
        if prefix:
            response.synthesized_response = prefix + response.synthesized_response

        return response

    async def provide_feedback(
        self,
        query_id: str,
        feedback: InteractionFeedback
    ) -> None:
        """
        Receive and learn from feedback.

        Why: The twin evolves through feedback. Tell it what worked,
        what didn't, which frameworks resonated. It learns and adapts.
        """
        self.feedback_history.append(feedback)
        await self.learner.learn_from_feedback(feedback, self.profile)
        self._save_profile()

        logger.info(
            f"Feedback received. Profile confidence: {self.profile.profile_confidence:.2f}"
        )

    def visualize_exploration(
        self,
        response: ProactiveResponse
    ) -> ChainVisualization:
        """
        Generate visualization of memory exploration chains.

        Why: Makes thought process transparent. See what memories the
        twin chained through, why it made those connections, where
        exploration terminated.
        """
        nodes = []
        edges = []
        all_memory_ids = set()

        # Add primary memories as nodes
        for mem in response.primary_memories:
            mem_id = mem.get("id", f"mem_{len(nodes)}")
            all_memory_ids.add(mem_id)
            nodes.append({
                "id": mem_id,
                "content": mem.get("content", "")[:100],
                "type": "primary",
                "score": mem.get("score", 0.0)
            })

        # Add explored chains
        root_node = None
        terminal_nodes = []

        for chain in response.explored_chains:
            if not root_node:
                root_node = chain.root_memory_id

            for i in range(len(chain.chain) - 1):
                source = chain.chain[i]
                target = chain.chain[i + 1]

                # Add nodes if not already added
                if source not in all_memory_ids:
                    all_memory_ids.add(source)
                    nodes.append({
                        "id": source,
                        "content": f"Memory {source}",
                        "type": "explored",
                        "score": chain.relevance_scores[i] if i < len(chain.relevance_scores) else 0.0
                    })

                if target not in all_memory_ids:
                    all_memory_ids.add(target)
                    nodes.append({
                        "id": target,
                        "content": f"Memory {target}",
                        "type": "explored",
                        "score": chain.relevance_scores[i+1] if i+1 < len(chain.relevance_scores) else 0.0
                    })

                # Add edge
                edges.append({
                    "source": source,
                    "target": target,
                    "relation": "co_accessed",
                    "weight": chain.relevance_scores[i+1] if i+1 < len(chain.relevance_scores) else 0.0
                })

            # Last node is terminal
            if chain.chain:
                terminal_nodes.append(chain.chain[-1])

        # Get full exploration path (flatten all chains)
        exploration_path = []
        for chain in response.explored_chains:
            exploration_path.extend(chain.chain)

        return ChainVisualization(
            query=response.query,
            nodes=nodes,
            edges=edges,
            root_node=root_node or "unknown",
            terminal_nodes=terminal_nodes,
            chain_depth=response.exploration_depth,
            exploration_path=exploration_path
        )

    def get_preferred_frameworks(self, top_k: int = 3) -> List[StrategicFramework]:
        """
        Get user's preferred frameworks based on learning.

        Why: Over time, the twin learns which lenses you prefer.
        Machiavellian? Boy Scout? Pragmatic? Use what resonates.
        """
        if not self.profile.framework_scores:
            # Default to balanced set
            return [
                StrategicFramework.PRAGMATIC_HYBRID,
                StrategicFramework.UTILITARIAN,
                StrategicFramework.BOY_SCOUT
            ]

        # Sort by score
        sorted_frameworks = sorted(
            self.profile.framework_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Convert to enum
        preferred = []
        for fw_name, score in sorted_frameworks[:top_k]:
            try:
                preferred.append(StrategicFramework[fw_name.upper()])
            except KeyError:
                continue

        return preferred or [StrategicFramework.PRAGMATIC_HYBRID]

    def export_profile(self, path: Optional[Path] = None) -> Path:
        """Export user profile to JSON."""
        export_path = path or self.profile_path
        export_path.parent.mkdir(parents=True, exist_ok=True)

        with open(export_path, 'w') as f:
            json.dump(self.profile.to_dict(), f, indent=2)

        logger.info(f"Profile exported to {export_path}")
        return export_path

    def _save_profile(self) -> None:
        """Auto-save profile after updates."""
        self.export_profile()

    def _load_profile(self) -> UserProfile:
        """Load profile from disk."""
        with open(self.profile_path, 'r') as f:
            data = json.load(f)

        return UserProfile.from_dict(data)

    def export_visualization(
        self,
        viz: ChainVisualization,
        path: Path
    ) -> None:
        """
        Export chain visualization to JSON (NetworkX-compatible).

        Why: For external visualization tools. Can be loaded into
        NetworkX, D3.js, or other graph visualization libraries.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(viz.to_networkx_dict(), f, indent=2)

        logger.info(f"Visualization exported to {path}")


# Helper function to initialize twin from vault memories
async def initialize_twin_from_vault(
    vault_path: Path,
    api_key: str,
    config: Dict[str, Any]
) -> CognitiveTwin:
    """
    Bootstrap a cognitive twin from vault export.

    Why: Seed the twin with your existing memories. Parse your YAML
    frontmatter, extract values/priorities, build initial profile.

    INTEGRATION POINT: wires into vault parser
    """
    logger.info(f"Initializing twin from vault: {vault_path}")

    # Create base profile
    profile = UserProfile(user_id="vault_user")

    # TODO: Parse vault files and extract:
    # - Values from decision documents
    # - Priorities from project plans
    # - Communication style from writing samples
    # - Framework preferences from past analyses

    # For now, return twin with empty profile
    twin = CognitiveTwin(
        config=config,
        api_key=api_key,
        user_profile=profile
    )

    logger.info("Twin initialized from vault")
    return twin


# Example usage
if __name__ == "__main__":
    print("CognitiveTwin - Your Personalized Digital Cognitive Mirror")
    print("=" * 70)
    print("\nEnhancements over CognitiveAgent:")
    print("  • User profile system (values, priorities, style)")
    print("  • Framework evolution (learns your preferences)")
    print("  • Self-correction loops (re-explores when uncertain)")
    print("  • Enhanced NER for better gap detection")
    print("  • Chain visualization (see how the twin thinks)")
    print("  • Crisis escalation (knows its limitations)")
    print("  • Preference learning (evolves with feedback)")
    print("\nThis is YOUR cognitive twin - it thinks like YOU think.")
    print("\nIntegration points:")
    print("  1. initialize_twin_from_vault() - seed from your vault")
    print("  2. Enhanced NER - wire spaCy for production")
    print("  3. Profile persistence - auto-saves to disk")
    print("  4. Feedback loop - provide_feedback() after each interaction")
    print("\nReady for gradual integration alongside cognitive_agent.py")
