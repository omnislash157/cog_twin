"""
Cognitive Twin - Unified System

ONE brain. ONE system. ONE voice.

This is the integration layer that unifies:
- MetacognitiveMirror (the brain - self-monitoring cognition)
- CognitiveAgent components (nervous system - gap detection, exploration)
- MemoryPipeline (recursive memory - outputs become inputs)
- VenomVoice (the voice - how we speak through the API)

The context window is just a suggestion. The snake eats its tail.
Every thought becomes retrievable. 50 nodes in 0.3ms.

We solved statelessness.

Architecture:
                    User Input
                         |
                         v
    +--------------------------------------------+
    |            COGNITIVE TWIN                  |
    |                                            |
    |  MetacognitiveMirror -----> Cognitive State|
    |  (brain)                    - phase        |
    |                             - temperature  |
    |                             - drift        |
    |                                            |
    |  ContextGapDetector ------> Gap Analysis   |
    |  (nervous system)           - missing info |
    |                             - severity     |
    |                                            |
    |  DualRetriever -----------> Memories       |
    |  (long-term memory)         - process      |
    |                             - episodic     |
    |                                            |
    |  MemoryPipeline ----------> Session State  |
    |  (recursive loop)           - recent outputs|
    |                             - clustering   |
    |                                            |
    |  VenomVoice --------------> System Prompt  |
    |  (the voice)                - formatted    |
    |                                            |
    +--------------------------------------------+
                         |
                         v
                    API Call (Claude/GPT/etc)
                         |
                         v
    +--------------------------------------------+
    |  Parse Output -> Actions -> Memory Ingest  |
    +--------------------------------------------+
                         |
                         v
                   Response to User
                         +
              Memory Pipeline Ingest (async)

Version: 2.0.0 (unified)
"""

# Load .env FIRST - before any imports that might use env vars
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
import hashlib
import time
import re

from model_adapter import create_adapter
import numpy as np
from numpy.typing import NDArray

# Core cognitive components
from metacognitive_mirror import (
    MetacognitiveMirror,
    QueryEvent,
    CognitivePhase,
    DriftSignal,
)

# Memory and retrieval
from retrieval import DualRetriever
from embedder import AsyncEmbedder

# New unified components
from memory_pipeline import (
    MemoryPipeline,
    CognitiveOutput,
    ThoughtType,
    create_response_output,
    create_reflection_output,
    create_gap_detection_output,
)

from venom_voice import (
    VenomVoice,
    VoiceContext,
    StreamingVoice,
    OutputAction,
    ParsedOutput,
)

# Configuration
from config import cfg, get_config, setup_logging

# Reasoning trace
from reasoning_trace import CognitiveTracer, StepType, ReasoningTrace

# Scoring
from scoring import ResponseScore, TrainingModeUI

# Chat memory - Phase 6.2
from chat_memory import ChatMemoryStore

# Squirrel tool - Phase 6.3
from squirrel import SquirrelTool, SquirrelQuery

logger = logging.getLogger(__name__)


# ===== Minimal ResponseMode enum (replaces cognitive_agent version) =====
from enum import Enum

class ResponseMode(Enum):
    """Simplified response modes without agent scaffolding."""
    DIRECT_ANSWER = "direct_answer"
    SHALLOW_EXPLORE = "shallow_explore"
    DEEP_EXPLORE = "deep_explore"
    FRAMEWORK_INJECTION = "framework_injection"
    CRISIS_INTERVENTION = "crisis_intervention"
    PATTERN_INTERRUPT = "pattern_interrupt"


@dataclass
class TwinState:
    """
    Runtime state of the Cognitive Twin.
    
    Tracks session activity, token usage, and operational mode.
    """
    session_id: str
    started_at: datetime
    
    # Interaction tracking
    total_queries: int = 0
    total_tokens_used: int = 0
    
    # Autonomous mode
    is_autonomous: bool = False
    current_task: Optional[str] = None
    autonomous_iterations: int = 0
    
    # HITL tracking
    pending_code_proposals: List[Dict[str, Any]] = field(default_factory=list)


@dataclass 
class TwinResponse:
    """
    Complete response from the Cognitive Twin.
    
    Contains not just the answer, but the full cognitive
    context: what memories were used, what gaps were detected,
    what the cognitive state was, etc.
    """
    # Core response
    content: str
    parsed_output: ParsedOutput
    
    # Cognitive context
    cognitive_phase: CognitivePhase
    response_mode: ResponseMode
    confidence: float
    
    # Memory context
    process_memories_used: int
    episodic_memories_used: int
    session_memories_used: int
    
    # Gap analysis
    gaps_detected: int
    gap_severity: float
    probing_questions: List[str]
    
    # Performance
    retrieval_time_ms: float
    total_time_ms: float
    tokens_used: int
    
    # Metadata
    query_id: str
    timestamp: datetime


class CogTwin:
    """
    The Unified Cognitive Twin - Standalone.

    Core components:
    - MetacognitiveMirror (brain - cognitive state monitoring)
    - DualRetriever (long-term memory - process + episodic)
    - MemoryPipeline (recursive memory loop)
    - VenomVoice (the voice layer)
    - Reasoning traces + feedback learning

    No agent scaffolding. Grok 4.1 Fast IS the cognitive agent.

    This is the complete system. One unified cognitive architecture.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the unified Cognitive Twin.

        Args:
            data_dir: Path to data directory (or from config.yaml)
            api_key: API key (or from env - XAI_API_KEY for Grok, ANTHROPIC_API_KEY for Claude)
            model: Model to use for generation (or from config.yaml)
        """
        # Load from config.yaml with argument overrides
        self.data_dir = Path(data_dir) if data_dir else Path(cfg("paths.data_dir", "./data"))
        self.model = model or cfg("model.name", "grok-4-1-fast-reasoning")

        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        # Load retriever FIRST
        logger.info("Loading memory system...")
        self.retriever = DualRetriever.load(self.data_dir)
        self.memory_count = len(self.retriever.process.nodes)
        logger.info(f"Loaded {self.memory_count} memory nodes")

        # Initialize MetacognitiveMirror directly (no parent class)
        mirror_config = {
            "query_window_size": cfg("cognitive.query_window_size", 100),
            "query_cluster_epsilon": cfg("cognitive.cluster_epsilon", 0.3),
        }
        self.mirror = MetacognitiveMirror(mirror_config)
        logger.info("MetacognitiveMirror initialized")

        # Initialize LLM client via adapter (supports multiple providers)
        provider = cfg("model.provider", "xai")  # Default to Grok

        if provider == "xai":
            model_api_key = os.getenv("XAI_API_KEY")
        else:
            model_api_key = resolved_api_key  # Anthropic key

        self.client = create_adapter(
            provider=provider,
            api_key=model_api_key,
            model=self.model,
        )

        # Initialize memory pipeline (recursive loop) - CogTwin specific
        self.memory_pipeline = MemoryPipeline(
            embedder=self.retriever.embedder,
            data_dir=self.data_dir,
            batch_interval=cfg("memory_pipeline.batch_interval", 5.0),
            max_batch_size=cfg("memory_pipeline.max_batch_size", 10),
        )

        # Initialize voice - CogTwin specific
        self.voice = VenomVoice(memory_count=self.memory_count)

        # Initialize reasoning tracer - CogTwin specific
        # Phase 5: Pass memory_pipeline for live streaming
        self.tracer = CognitiveTracer(self.data_dir, memory_pipeline=self.memory_pipeline)
        self.last_trace_id: Optional[str] = None

        # Initialize chat memory store - Phase 6.2
        self.chat_memory = ChatMemoryStore(self.data_dir)
        self._last_exchange_id: Optional[str] = None

        # Initialize squirrel tool - Phase 6.3
        self.squirrel = SquirrelTool(self.chat_memory)

        # Initialize training mode - Phase 3 scoring
        self.training_mode = TrainingModeUI(
            enabled=cfg("training.enabled", True),
            quick_mode=cfg("training.quick_mode", True),
        )

        # Runtime state - CogTwin specific
        self.state = TwinState(
            session_id=f"twin_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            started_at=datetime.now(),
        )

        # HITL callback
        self.on_code_proposal: Optional[Callable[[Dict], bool]] = None

        # Track last response for feedback/viz commands
        self._last_response: Optional[Any] = None
        self._last_query_id: Optional[str] = None

        # Phase 5.5: Store grep results for provenance tracking
        self._last_grep_results: List[Dict[str, Any]] = []

        logger.info(f"CogTwin initialized: {self.memory_count} memories loaded")
    
    async def start(self):
        """Start the twin and memory pipeline."""
        await self.memory_pipeline.start()
        await self._hydrate_mirror(max_events=500)
        logger.info(f"CogTwin started: {self.state.session_id}")
    
    async def stop(self):
        """Stop the twin and flush memories."""
        await self.memory_pipeline.stop()
        logger.info(
            f"CogTwin stopped. Queries: {self.state.total_queries}, "
            f"Tokens: {self.state.total_tokens_used}"
        )

    async def _hydrate_mirror(self, max_events: int = 500):
        """Hydrate MetacognitiveMirror from persistent chat history."""
        logger.info(f"Hydrating mirror from up to {max_events} historical events...")

        exchanges = self.chat_memory.query_recent(n=max_events)
        if not exchanges:
            logger.info("No historical exchanges - mirror starts fresh")
            return

        hydrated = 0
        for exchange in reversed(exchanges):  # Oldest first
            if not exchange.trace_id:
                continue
            trace = self.tracer.load_trace(exchange.trace_id)
            if not trace:
                continue
            try:
                embedding = await self.retriever.embedder.embed_single(exchange.user_query)
                query_event = QueryEvent(
                    timestamp=exchange.timestamp,
                    query_text=exchange.user_query,
                    query_embedding=embedding,
                    retrieved_memory_ids=trace.memories_retrieved,
                    retrieval_scores=trace.retrieval_scores or [],
                    execution_time_ms=exchange.retrieval_time_ms or 0.0,
                    result_count=len(trace.memories_retrieved),
                    semantic_gate_passed=True,
                )
                self.mirror.record_query(query_event)
                hydrated += 1
            except Exception as e:
                logger.warning(f"Failed to hydrate {exchange.id}: {e}")

        logger.info(f"Mirror hydration complete: {hydrated} events")
        insights = self.mirror.get_real_time_insights()
        logger.info(f"Mirror state: phase={insights['cognitive_phase']}, temp={insights['temperature']:.2f}")

    async def think(
        self,
        user_input: str,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Process user input and generate response.

        This is the main entry point. The complete cognitive loop:
        1. Get cognitive state from mirror
        2. Retrieve relevant memories
        3. Detect context gaps
        4. Decide response mode
        5. Explore memory chains if needed
        6. Build voice context
        7. Generate through API
        8. Self-correct if confidence low (inherited from CognitiveTwin)
        9. Escalate crisis if needed (inherited from CognitiveTwin)
        10. Personalize response (inherited from CognitiveTwin)
        11. Parse output and extract actions
        12. Ingest to memory pipeline
        13. Yield response

        Args:
            user_input: The user's query
            stream: Whether to stream response chunks

        Yields:
            Response chunks (if streaming) or full response
        """
        start_time = time.time()
        query_id = hashlib.sha256(
            f"{self.state.session_id}_{self.state.total_queries}_{time.time()}".encode()
        ).hexdigest()[:16]

        self._last_query_id = query_id
        self.state.total_queries += 1

        # ===== STEP 1: Get cognitive state from mirror =====
        insights = self.mirror.get_real_time_insights()
        cognitive_phase = CognitivePhase(insights["cognitive_phase"])

        logger.info(f"Cognitive phase: {cognitive_phase.value}")

        # ===== STEP 2: Embed query and retrieve memories =====
        retrieval_start = time.time()

        query_embedding = await self.retriever.embedder.embed_single(user_input)

        retrieval_result = await self.retriever.retrieve(
            user_input,
            process_top_k=cfg("retrieval.process_top_k", 10),
            episodic_top_k=cfg("retrieval.episodic_top_k", 5),
        )

        retrieval_time = (time.time() - retrieval_start) * 1000
        logger.info(f"Retrieval: {retrieval_time:.1f}ms")

        # Start reasoning trace
        trace_id = self.tracer.start_trace(
            query=user_input,
            retrieved_memory_ids=[m.id for m in retrieval_result.process_memories],
            retrieval_scores=list(retrieval_result.process_scores),
            # query_embedding omitted - not needed for keyword-based trace search
            cognitive_phase=cognitive_phase.value if hasattr(cognitive_phase, 'value') else str(cognitive_phase),
            response_mode="",  # Will update after decision
        )

        # Get session memories (recursive loop)
        session_memories = self.memory_pipeline.search_session(
            query_embedding,
            top_k=cfg("retrieval.session_top_k", 5)
        )

        # ===== STEP 2.25: Auto-inject last 1h of session context =====
        # USER TRUTH > TOOL TRUTH - recent conversation is ground truth
        # Skip SQUIRREL hot context if session_outputs already has recent entries (< 5 min)
        # to avoid redundancy - session_outputs is already HIGHEST trust
        hot_context = ""
        has_recent_session = False
        if self.memory_pipeline.session_outputs:
            now = datetime.now()
            for output in self.memory_pipeline.session_outputs[-5:]:
                if hasattr(output, 'timestamp') and output.timestamp:
                    age_seconds = (now - output.timestamp).total_seconds()
                    if age_seconds < 300:  # 5 minutes
                        has_recent_session = True
                        break

        if not has_recent_session:
            hot_context = self.squirrel.execute(
                SquirrelQuery(timeframe="-60min"),
                limit=15
            )
            logger.debug("SQUIRREL hot context loaded (no recent session outputs)")

        # ===== STEP 2.5: Get high-scored reasoning exemplars (Phase 4) =====
        exemplar_traces: List[ReasoningTrace] = []
        if cfg("feedback_injection.enabled", True):
            exemplar_traces = self._get_reasoning_exemplars(
                user_input,
                top_k=cfg("feedback_injection.exemplar_top_k", 3),
                min_score=cfg("feedback_injection.exemplar_min_score", 0.7),
            )
            if exemplar_traces:
                logger.info(f"Injecting {len(exemplar_traces)} exemplar traces")
                for t in exemplar_traces:
                    logger.info(f"  - {t.id[:8]}: score={t.score.get('overall', 0):.2f}")

        # ===== STEP 3: Detect context gaps =====
        # (Deprecated: Grok handles gap detection natively via prompt engineering)
        process_memories_dicts = [
            {
                "id": m.id,
                "content": m.combined_content,
                "human_content": m.human_content,
                "assistant_content": m.assistant_content,
                "created_at": m.created_at,
                "score": s,
            }
            for m, s in zip(
                retrieval_result.process_memories,
                retrieval_result.process_scores
            )
        ]

        gaps = []  # Stubbed - Grok handles gap detection
        gap_severity = 0.0

        logger.info(f"Gap detection: stubbed (Grok handles natively)")

        # ===== STEP 4: Decide response mode =====
        response_mode = self._decide_response_mode(
            user_input,
            cognitive_phase,
            gaps,
            gap_severity
        )

        logger.info(f"Response mode: {response_mode.value}")

        # ===== STEP 5: Explore memory chains if needed =====
        # (Deprecated: Grok explores memory context natively)
        explored_chains = []  # Stubbed - Grok handles exploration
        logger.info(f"Memory chain exploration: stubbed (Grok handles natively)")

        # ===== STEP 6: Strategic analysis if needed =====
        # (Deprecated: Grok handles multi-framework analysis natively)
        strategic_analysis = None  # Stubbed - Grok handles frameworks
        logger.info(f"Strategic analysis: stubbed (Grok handles natively)")

        # ===== STEP 7: Build voice context =====
        voice_context = VoiceContext(
            user_profile={},  # Stubbed - profile deprecated
            cognitive_phase=cognitive_phase.value,
            temperature=insights["temperature"],
            focus_score=insights["focus_score"],
            drift_signal=insights.get("drift_signal"),
            process_memories=[
                {
                    "human_content": m.human_content,
                    "assistant_content": m.assistant_content,
                    "timestamp": m.created_at,
                    "score": s,
                }
                for m, s in zip(
                    retrieval_result.process_memories[:5],
                    retrieval_result.process_scores[:5]
                )
            ],
            episodic_memories=[
                {
                    "title": e.title,
                    "summary": e.summary_text,
                    "start_time": e.created_at,
                    "score": s,
                }
                for e, s in zip(
                    retrieval_result.episodic_memories[:3],
                    retrieval_result.episodic_scores[:3]
                )
            ],
            session_outputs=[
                o.to_dict() for o, _ in session_memories
            ],
            detected_gaps=[
                {
                    "gap_type": g.gap_type.value,
                    "description": g.description,
                    "severity": g.severity,
                    "probing_questions": g.probing_questions,
                }
                for g in gaps
            ],
            gap_severity=gap_severity,
            strategic_analysis=strategic_analysis.synthesis if strategic_analysis else None,
            response_mode=response_mode.value,
            past_traces=[
                {
                    "query": t.query,
                    "score": t.score,
                    "feedback_notes": t.feedback_notes,
                    "response": t.response[:500],  # Truncate for context efficiency
                    "cognitive_phase": t.cognitive_phase,
                }
                for t in exemplar_traces
            ],
            hot_context=hot_context,  # Last 1h of session - highest trust
            analytics_block=self.format_analytics_block(),  # Visible AI self-awareness
            show_analytics=cfg("analytics.show_in_prompt", True),
        )

        # Build system prompt with retrieval mode
        retrieval_mode = cfg("retrieval_mode", "inject")
        system_prompt = self.voice.build_system_prompt(voice_context, retrieval_mode=retrieval_mode)

        # ===== STEP 8: Generate through API =====
        full_response = ""
        streaming_voice = StreamingVoice(self.voice)

        if stream:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=cfg("model.max_tokens", 4096),
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}],
            ) as stream_response:
                for chunk in stream_response.text_stream:
                    full_response += chunk
                    clean_chunk = streaming_voice.process_chunk(chunk)
                    yield clean_chunk

            response_obj = stream_response.get_final_message()
            tokens_used = response_obj.usage.input_tokens + response_obj.usage.output_tokens
        else:
            response_obj = self.client.messages.create(
                model=self.model,
                max_tokens=cfg("model.max_tokens", 4096),
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}],
            )
            full_response = response_obj.content[0].text
            tokens_used = response_obj.usage.input_tokens + response_obj.usage.output_tokens
            yield full_response

        self.state.total_tokens_used += tokens_used

        # ===== STEP 8.5: UNIFIED TOOL EXECUTION (Phase 8: Single Synthesis) =====
        # Collect ALL tool results first, then ONE synthesis call
        self._last_grep_results = []  # Reset for this query
        tool_results = {}  # Collect all results here
        seen_memory_ids = set()  # Track seen IDs for deduplication across tools

        # --- GREP (now uses HybridSearch if available) ---
        if "[GREP" in full_response:
            grep_matches = re.findall(r'\[GREP term="([^"]+)"\]', full_response)
            if grep_matches:
                all_grep_contexts = []

                for term in grep_matches:
                    # Use hybrid search if available (semantic + keyword)
                    if self.retriever.hybrid:
                        logger.info(f"HYBRID search invoked for: {term}")
                        hybrid_result = await self.retriever.hybrid.search(term, top_k=10)
                        grep_context = self.retriever.hybrid.format_for_context(hybrid_result)
                        all_grep_contexts.append(grep_context)

                        # Record in trace
                        self.tracer.record_grep(
                            term,
                            len(hybrid_result.hits),
                            hybrid_result.keyword_count,
                        )

                        # Track seen IDs from hybrid results
                        for hit in hybrid_result.hits:
                            seen_memory_ids.add(hit.memory_id)

                        # Store structured results for provenance tracking
                        self._last_grep_results.append({
                            "term": term,
                            "total_occurrences": len(hybrid_result.hits),
                            "unique_memories": len(hybrid_result.hits),
                            "semantic_count": hybrid_result.semantic_count,
                            "keyword_count": hybrid_result.keyword_count,
                            "overlap_count": hybrid_result.overlap_count,
                            "hits": [
                                {
                                    "memory_id": h.memory_id,
                                    "semantic_score": h.semantic_score,
                                    "keyword_count": h.keyword_count,
                                    "rrf_score": h.rrf_score,
                                    "found_by": h.found_by,
                                    "timestamp": h.timestamp.isoformat() if h.timestamp else None,
                                }
                                for h in hybrid_result.hits[:5]
                            ],
                        })

                    # Fall back to BM25 grep if hybrid not available
                    elif self.retriever.grep:
                        logger.info(f"GREP tool invoked for: {term}")
                        grep_result = self.retriever.grep.grep(term)
                        grep_context = self.retriever.grep.format_for_context(grep_result)
                        all_grep_contexts.append(grep_context)

                        # Record grep in trace
                        self.tracer.record_grep(term, grep_result.total_occurrences, grep_result.unique_memories)

                        # Store structured grep results for provenance tracking
                        self._last_grep_results.append({
                            "term": term,
                            "total_occurrences": grep_result.total_occurrences,
                            "unique_memories": grep_result.unique_memories,
                            "temporal_distribution": dict(grep_result.temporal_distribution) if grep_result.temporal_distribution else {},
                            "co_occurring_terms": grep_result.co_occurring_terms[:10] if grep_result.co_occurring_terms else [],
                            "hits": [
                                {
                                    "snippet": h.snippet[:200] if h.snippet else "",
                                    "timestamp": h.timestamp.isoformat() if hasattr(h.timestamp, 'isoformat') else str(h.timestamp),
                                    "memory_id": h.memory_id,
                                }
                                for h in (grep_result.hits[:5] if grep_result.hits else [])
                            ],
                        })

                if all_grep_contexts:
                    tool_results['grep'] = "\n\n---\n\n".join(all_grep_contexts)
                    logger.info(f"GREP/HYBRID collected: {len(grep_matches)} terms")

        # --- SQUIRREL ---
        if "[SQUIRREL" in full_response:
            squirrel_matches = re.findall(r'\[SQUIRREL\s+([^\]]+)\]', full_response)

            if squirrel_matches:
                all_squirrel_contexts = []

                for match_content in squirrel_matches:
                    logger.info(f"SQUIRREL tool invoked: {match_content}")

                    # Parse the query
                    query = SquirrelQuery.parse(match_content)

                    # Execute and get formatted results
                    squirrel_context = self.squirrel.execute(query, limit=10)
                    all_squirrel_contexts.append(squirrel_context)

                    # Record in trace
                    self.tracer.record_step(
                        StepType.RETRIEVE,
                        f"SQUIRREL: timeframe={query.timeframe}, back={query.back_n}, search={query.search_term}",
                    )

                tool_results['squirrel'] = "\n\n".join(all_squirrel_contexts)
                logger.info(f"SQUIRREL collected: {len(squirrel_matches)} queries")

        # --- VECTOR ---
        if "[VECTOR" in full_response:
            vector_matches = re.findall(r'\[VECTOR query="([^"]+)"\]', full_response)
            if vector_matches and self.retriever.process:
                logger.info(f"VECTOR tool invoked for: {vector_matches}")

                vector_results = []  # List of (node, score) tuples
                for query in vector_matches:
                    query_embedding = await self.retriever.embedder.embed_single(query)
                    nodes, scores = self.retriever.process.retrieve(query_embedding, top_k=5)
                    # Dedupe: only add nodes we haven't seen
                    for node, score in zip(nodes, scores):
                        if node.id not in seen_memory_ids:
                            seen_memory_ids.add(node.id)
                            vector_results.append((node, score))

                # Format results with scores
                formatted_results = []
                for mem, score in vector_results[:5]:
                    ts = mem.created_at.strftime("%Y-%m-%d") if hasattr(mem.created_at, 'strftime') else str(mem.created_at)
                    human_preview = mem.human_content[:200] if mem.human_content else ""
                    assistant_preview = mem.assistant_content[:300] if mem.assistant_content else ""
                    formatted_results.append(f"[{ts}] (relevance: {score:.2f})\nQ: {human_preview}...\nA: {assistant_preview}...")

                tool_results['vector'] = "\n\n".join(formatted_results)
                logger.info(f"VECTOR collected: {len(vector_matches)} queries, {len(vector_results)} unique results")

        # --- EPISODIC ---
        if "[EPISODIC" in full_response:
            episodic_matches = re.findall(r'\[EPISODIC query="([^"]+)"(?:\s+timeframe="([^"]*)")?\]', full_response)
            if episodic_matches and self.retriever.episodic:
                logger.info(f"EPISODIC tool invoked for: {episodic_matches}")

                episodic_results = []
                for query, timeframe in episodic_matches:
                    query_embedding = await self.retriever.embedder.embed_single(query)
                    # Retrieve more candidates if timeframe filtering will apply
                    retrieve_k = 10 if timeframe else 5
                    nodes, scores = self.retriever.episodic.retrieve(query, query_embedding, top_k=retrieve_k)

                    # Apply timeframe filter if specified
                    if timeframe and timeframe.strip():
                        now = datetime.now()
                        filtered_nodes = []
                        for node in nodes:
                            node_ts = getattr(node, 'created_at', None)
                            if node_ts is None:
                                continue
                            # Parse timeframe: "7d", "30d", "3m", "1y", "all"
                            tf = timeframe.strip().lower()
                            if tf == "all":
                                filtered_nodes.append(node)
                            elif tf.endswith('d'):
                                days = int(tf[:-1])
                                if hasattr(node_ts, 'timestamp'):
                                    age_days = (now - node_ts).days
                                else:
                                    age_days = 0
                                if age_days <= days:
                                    filtered_nodes.append(node)
                            elif tf.endswith('m'):
                                months = int(tf[:-1])
                                if hasattr(node_ts, 'timestamp'):
                                    age_days = (now - node_ts).days
                                else:
                                    age_days = 0
                                if age_days <= months * 30:
                                    filtered_nodes.append(node)
                            elif tf.endswith('y'):
                                years = int(tf[:-1])
                                if hasattr(node_ts, 'timestamp'):
                                    age_days = (now - node_ts).days
                                else:
                                    age_days = 0
                                if age_days <= years * 365:
                                    filtered_nodes.append(node)
                            else:
                                # Unknown format, include all
                                filtered_nodes.append(node)
                        nodes = filtered_nodes
                        logger.info(f"EPISODIC timeframe filter '{timeframe}': {len(filtered_nodes)} results")

                    # Dedupe: only add episodes we haven't seen
                    for node in nodes:
                        ep_id = getattr(node, 'id', None)
                        if ep_id and ep_id not in seen_memory_ids:
                            seen_memory_ids.add(ep_id)
                            episodic_results.append(node)

                # Format results (limit to 3)
                formatted_results = []
                for ep in episodic_results[:3]:
                    title = ep.title if hasattr(ep, 'title') else "Untitled"
                    ts = ep.created_at.strftime("%Y-%m-%d") if hasattr(ep, 'created_at') and hasattr(ep.created_at, 'strftime') else str(getattr(ep, 'created_at', 'unknown'))
                    summary = ep.summary[:200] if hasattr(ep, 'summary') and ep.summary else ""
                    formatted_results.append(f'[{ts}] "{title}"\nSummary: {summary}...')

                tool_results['episodic'] = "\n\n".join(formatted_results)
                logger.info(f"EPISODIC collected: {len(episodic_matches)} queries, {len(episodic_results)} unique results")

        # ===== SINGLE SYNTHESIS CALL (if any tools were invoked) =====
        if tool_results:
            # Build combined context from all tool results
            synthesis_sections = []
            
            if 'grep' in tool_results:
                synthesis_sections.append(f"=== GREP RESULTS (keyword frequency) ===\n{tool_results['grep']}")
            
            if 'squirrel' in tool_results:
                synthesis_sections.append(f"=== SQUIRREL RESULTS (temporal recall) ===\n{tool_results['squirrel']}")
            
            if 'vector' in tool_results:
                synthesis_sections.append(f"=== VECTOR RESULTS (semantic similarity) ===\n{tool_results['vector']}")
            
            if 'episodic' in tool_results:
                synthesis_sections.append(f"=== EPISODIC RESULTS (conversation arcs) ===\n{tool_results['episodic']}")

            combined_tool_context = "\n\n".join(synthesis_sections)
            tools_used = ", ".join(tool_results.keys()).upper()

            # Update voice context with grep results for provenance
            voice_context.grep_results = self._last_grep_results
            followup_system_prompt = self.voice.build_system_prompt(voice_context, retrieval_mode=retrieval_mode)

            # ONE synthesis call with ALL results
            followup_messages = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": full_response},
                {"role": "user", "content": f"SYSTEM: All tool searches completed ({tools_used}). Here are the combined results:\n\n{combined_tool_context}\n\nSynthesize ALL results into a single coherent response. Cross-reference sources - if GREP found zero but VECTOR/EPISODIC found hits, note the semantic vs keyword mismatch. Do NOT report each tool separately - give ONE unified answer."},
            ]

            followup_response = self.client.messages.create(
                model=self.model,
                max_tokens=cfg("model.max_tokens", 4096),
                system=followup_system_prompt,
                messages=followup_messages,
            )

            followup_text = followup_response.content[0].text
            followup_tokens = followup_response.usage.input_tokens + followup_response.usage.output_tokens
            self.state.total_tokens_used += followup_tokens
            tokens_used += followup_tokens

            # Append followup to response
            full_response += "\n\n" + followup_text

            # Yield the followup if streaming
            if stream:
                yield "\n\n" + followup_text

            logger.info(f"UNIFIED SYNTHESIS complete: {tools_used}")

        # Record synthesis step
        self.tracer.record_step(
            StepType.SYNTHESIZE,
            f"Generated response ({len(full_response)} chars)",
            memories_touched=[m.id for m in retrieval_result.process_memories[:5]],
        )

        # Finalize trace (Phase 5: now async, streams to memory pipeline)
        trace = await self.tracer.finalize_trace(
            response=full_response,
            confidence=0.8,  # Will be updated after parsing
            tokens_used=tokens_used,
        )
        self.last_trace_id = trace.id

        # ===== STEP 9: Parse output =====
        parsed = self.voice.parse_output(full_response)

        # Calculate initial confidence from parsed output
        initial_confidence = parsed.confidence_stated or 0.8

        # Generate probing questions from gaps
        probing_questions = []

        # ===== STEP 10: Build response metadata (ProactiveResponse removed) =====
        total_time = (time.time() - start_time) * 1000

        # Simple metadata dict instead of ProactiveResponse
        response_metadata = {
            "query": user_input,
            "response_mode": response_mode,
            "cognitive_phase": cognitive_phase,
            "confidence": initial_confidence,
            "processing_time_ms": total_time,
            "synthesized_response": full_response,
        }

        # Store for commands
        self._last_response = response_metadata

        # ===== STEP 11: Self-correction if confidence low (inherited) =====
        # Stubbed - self-correction removed with cognitive_agent

        # ===== STEP 12: Crisis escalation if needed (inherited) =====
        # Stubbed - crisis escalation removed with cognitive_agent

        # ===== STEP 13: Personalize response (inherited) =====
        # Stubbed - personalization removed with cognitive_agent

        # ===== STEP 14: Handle actions =====
        await self._handle_actions(parsed, user_input, cognitive_phase)

        # ===== STEP 15: Ingest to memory pipeline =====
        cognitive_output = create_response_output(
            content=full_response,
            source_memory_ids=[m.id for m in retrieval_result.process_memories[:5]],
            cognitive_phase=cognitive_phase.value,
            confidence=initial_confidence
        )
        await self.memory_pipeline.ingest(cognitive_output)

        # ===== STEP 15.5: Record to chat memory (Phase 6.2) =====
        # Full triplet: query + trace + response for temporal retrieval
        trace_content = None
        if trace and trace.steps:
            # Format trace steps as readable summary
            trace_content = "; ".join([
                f"{s.step_type.value}: {s.content[:100]}"
                for s in trace.steps[:5]
            ])

        self._last_exchange_id = self.chat_memory.record_exchange(
            session_id=self.state.session_id,
            user_query=user_input,
            model_response=full_response,
            model_trace=trace_content,
            cognitive_phase=cognitive_phase.value,
            response_confidence=initial_confidence,
            tokens_used=tokens_used,
            retrieval_time_ms=retrieval_result.retrieval_time_ms,
            trace_id=trace.id if trace else None,
        )

        # ===== STEP 16: Record query in mirror =====
        query_event = QueryEvent(
            timestamp=datetime.now(),
            query_text=user_input,
            query_embedding=query_embedding,
            retrieved_memory_ids=[m.id for m in retrieval_result.process_memories],
            retrieval_scores=list(retrieval_result.process_scores),
            execution_time_ms=total_time,
            result_count=len(retrieval_result.process_memories),
            semantic_gate_passed=True,
        )
        self.mirror.record_query(query_event)

        # Generate prediction for next likely memories (Phase 8: predictive prefetching)
        current_memories = [m.id for m in retrieval_result.process_memories[:5]]
        if current_memories:
            prediction = self.mirror.prefetcher.predict_next_memories(current_memories, top_k=3)
            if prediction.predicted_memory_ids:
                logger.debug(f"Prefetcher predicts next: {prediction.predicted_memory_ids[:3]}")
                # Store for potential use in next query
                self._predicted_memories = prediction.predicted_memory_ids

        logger.info(
            f"Query complete: {total_time:.1f}ms, "
            f"{tokens_used} tokens, "
            f"confidence: {initial_confidence:.2f}, "
            f"action: {parsed.primary_action.value}"
        )
    
    def _decide_response_mode(
        self,
        query: str,
        phase: CognitivePhase,
        gaps: List[Any],  # Stubbed - was ContextGap
        gap_severity: float
    ) -> ResponseMode:
        """Decide how to respond based on cognitive state (simplified - no agent scaffolding)."""
        # Crisis gets intervention
        if phase == CognitivePhase.CRISIS:
            return ResponseMode.CRISIS_INTERVENTION
        
        # Check for semantic collapse
        drift_mag, drift_signal = self.mirror.archaeologist.detect_semantic_drift()
        if drift_signal == DriftSignal.SEMANTIC_COLLAPSE:
            return ResponseMode.PATTERN_INTERRUPT
        
        # High gaps -> deep explore
        if gap_severity > 0.7:
            return ResponseMode.DEEP_EXPLORE
        
        # Medium gaps -> shallow explore
        if gap_severity > 0.4:
            return ResponseMode.SHALLOW_EXPLORE
        
        # Decision query -> framework injection
        if self._is_decision_query(query):
            return ResponseMode.FRAMEWORK_INJECTION
        
        return ResponseMode.DIRECT_ANSWER
    
    def _is_decision_query(self, query: str) -> bool:
        """Check if query is asking for decision guidance."""
        markers = [
            "should i", "what should", "help me decide",
            "which option", "what's the best", "advice on",
            "recommend", "choose between"
        ]
        return any(m in query.lower() for m in markers)

    def _get_reasoning_exemplars(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.7,
    ) -> List[ReasoningTrace]:
        """
        Get high-scored past reasoning for similar queries.

        Priority:
        1. Similar query + high score
        2. Same cognitive phase + high score
        3. High score (general exemplars)

        Used for Phase 4 feedback injection - the model learns
        from its own graded history.
        """
        # Get similar traces by query keyword overlap
        similar = self.tracer.search_traces(query, top_k=10)

        # Filter by score - using our 3-dimension scoring
        # overall is computed from accuracy, temporal, tone
        high_scored = [
            t for t in similar
            if t.score and t.score.get("overall", 0) >= min_score
        ]

        if len(high_scored) >= top_k:
            return high_scored[:top_k]

        # Backfill with generally high-scored traces
        general_high = self.tracer.get_high_scored_traces(min_score, top_k=5)

        # Combine and dedupe
        seen = {t.id for t in high_scored}
        for t in general_high:
            if t.id not in seen and len(high_scored) < top_k:
                high_scored.append(t)
                seen.add(t.id)

        return high_scored[:top_k]

    async def _handle_actions(
        self,
        parsed: ParsedOutput,
        query: str,
        phase: CognitivePhase
    ):
        """Handle extracted actions from output."""
        for action, content in parsed.extracted_actions:
            if action == OutputAction.REMEMBER:
                # Explicit memory request - create insight output
                insight = CognitiveOutput.create(
                    thought_type=ThoughtType.INSIGHT,
                    content=content,
                    reasoning=f"Explicit REMEMBER from query: {query[:50]}",
                    cognitive_phase=phase.value,
                )
                await self.memory_pipeline.ingest(insight)
                logger.info(f"Stored explicit memory: {content[:50]}...")
            
            elif action == OutputAction.REFLECT:
                # Metacognitive reflection
                reflection = create_reflection_output(
                    content=content,
                    reasoning=f"Metacognitive interrupt during: {query[:50]}",
                    cognitive_phase=phase.value,
                )
                await self.memory_pipeline.ingest(reflection)
                logger.info(f"Stored reflection: {content[:50]}...")
            
            elif action == OutputAction.CODE_PROPOSAL:
                # Code proposal needs HITL
                proposal = {
                    "content": content,
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                }
                self.state.pending_code_proposals.append(proposal)
                
                if self.on_code_proposal:
                    approved = self.on_code_proposal(proposal)
                    proposal["approved"] = approved
                    logger.info(f"Code proposal {'approved' if approved else 'rejected'}")
            
            elif action == OutputAction.ESCALATE:
                logger.warning(f"ESCALATION requested: {content[:100]}")
    
    
    # ===== Convenience methods =====
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state summary."""
        insights = self.mirror.get_real_time_insights()
        return {
            "phase": insights["cognitive_phase"],
            "temperature": insights["temperature"],
            "focus_score": insights["focus_score"],
            "drift_signal": insights.get("drift_signal"),
            "session_outputs": len(self.memory_pipeline.session_outputs),
            "total_queries": self.state.total_queries,
        }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        pipeline_stats = self.memory_pipeline.get_session_stats()
        return {
            "session_id": self.state.session_id,
            "started_at": self.state.started_at.isoformat(),
            "total_queries": self.state.total_queries,
            "total_tokens": self.state.total_tokens_used,
            "memory_count": self.memory_count,
            **pipeline_stats,
        }
    
    async def run_health_check(self) -> List[Dict[str, Any]]:
        """Run metacognitive health check."""
        return self.mirror.run_health_check()

    def get_session_analytics(self) -> Dict[str, Any]:
        """
        Get rich session analytics for user-facing display.

        This is the "visible AI self-awareness" that differentiates us.
        Returns data formatted for [Session Analytics] block in responses.
        """
        insights = self.mirror.get_real_time_insights()

        # Get recurring patterns
        patterns = self.mirror.archaeologist.detect_recurring_patterns()

        # Get stability score
        stability = self.mirror.seismograph.calculate_cognitive_stability()

        # Get phase transitions (recent)
        transitions = self.mirror.seismograph.detect_phase_transitions()
        recent_transitions = transitions[-3:] if transitions else []

        # Get hotspot memories
        hotspots = self.mirror.thermodynamics.detect_hotspots(top_k=5)

        # Get bursts (sudden activity)
        bursts = self.mirror.thermodynamics.detect_bursts()

        # Get prediction accuracy
        pred_perf = self.mirror.prefetcher.calculate_prediction_performance()

        # Calculate session duration
        session_duration = (datetime.now() - self.state.started_at).total_seconds() / 60

        # Generate actionable suggestion based on phase
        phase = insights["cognitive_phase"]
        suggestion = self._generate_phase_suggestion(phase, stability, patterns)

        return {
            # Core metrics
            "phase": phase,
            "phase_description": self._get_phase_description(phase),
            "stability": round(stability, 2),
            "temperature": round(insights["temperature"], 2),
            "focus_score": round(insights["focus_score"], 2),

            # Drift detection
            "drift_signal": insights.get("drift_signal"),
            "drift_magnitude": round(insights.get("drift_magnitude", 0), 3),

            # Patterns
            "recurring_patterns": [
                {
                    "topic": p[0][:50] + "..." if len(p[0]) > 50 else p[0],
                    "frequency": p[1],
                    "recency": round(p[2], 2),
                }
                for p in patterns[:5]
            ],

            # Hotspots
            "hotspot_topics": [
                {"memory_id": mid, "temperature": round(temp, 2)}
                for mid, temp in hotspots[:3]
            ],

            # Bursts (emerging topics)
            "emerging_topics": [
                {"memory_id": mid, "burst_intensity": round(intensity, 2)}
                for mid, intensity in bursts[:3]
            ],

            # Session stats
            "session_duration_minutes": round(session_duration, 1),
            "total_queries": self.state.total_queries,
            "total_tokens": self.state.total_tokens_used,

            # Prediction performance
            "prediction_accuracy": round(pred_perf.get("accuracy_mean", 0), 2),

            # Predicted next memories (if available)
            "predicted_next": getattr(self, '_predicted_memories', [])[:3],

            # Actionable suggestion
            "suggestion": suggestion,

            # Phase transitions
            "recent_transitions": [
                {
                    "timestamp": ts.isoformat(),
                    "from": old.value,
                    "to": new.value,
                }
                for ts, old, new in recent_transitions
            ],
        }

    def _get_phase_description(self, phase: str) -> str:
        """Get human-readable phase description."""
        descriptions = {
            "exploration": "Wide-ranging queries, discovering new areas",
            "exploitation": "Focused queries, deep-diving on specific topics",
            "learning": "Building new knowledge structures",
            "consolidation": "Reviewing and connecting existing knowledge",
            "idle": "Low activity, background processing",
            "crisis": "Rapid, unfocused queries - may indicate confusion",
        }
        return descriptions.get(phase, "Unknown cognitive state")

    def _generate_phase_suggestion(
        self,
        phase: str,
        stability: float,
        patterns: List,
    ) -> str:
        """Generate actionable suggestion based on cognitive state."""
        # Crisis mode - highest priority
        if phase == "crisis":
            return "Slow down. Let's clarify what you're trying to accomplish."

        # Low stability - frequent context switches
        if stability < 0.4:
            return "Frequent context switches detected. Consider focusing on one thread."

        # Consolidation with high repetition
        if phase == "consolidation" and patterns:
            top_pattern = patterns[0][0][:30]
            freq = patterns[0][1]
            if freq >= 4:
                return f"You've revisited '{top_pattern}' {freq}x. Ready to ship or need to pivot?"

        # Exploitation mode - deep focus
        if phase == "exploitation":
            return "Deep focus mode. I'll prioritize precision over exploration."

        # Exploration mode
        if phase == "exploration":
            return "Exploration mode. I'll surface diverse connections."

        # Learning mode
        if phase == "learning":
            return "Learning mode. Building new knowledge structures."

        return "Nominal cognitive state."

    def format_analytics_block(self) -> str:
        """
        Format analytics as a block for injection into system prompt.

        This is the user-facing "visible AI self-awareness".
        """
        analytics = self.get_session_analytics()

        lines = [
            "[Session Analytics]",
            f"Phase: {analytics['phase'].upper()} ({analytics['phase_description']})",
        ]

        # Add patterns if present
        if analytics["recurring_patterns"]:
            top = analytics["recurring_patterns"][0]
            lines.append(f"Pattern: {top['topic']} (visited {top['frequency']}x)")

        # Add stability with interpretation
        stability = analytics["stability"]
        stability_desc = "stable" if stability > 0.7 else "moderate" if stability > 0.4 else "unstable"
        lines.append(f"Stability: {stability} ({stability_desc})")

        # Add emerging topics if any
        if analytics["emerging_topics"]:
            burst = analytics["emerging_topics"][0]
            lines.append(f"Emerging: Memory burst detected (intensity {burst['burst_intensity']})")

        # Add suggestion
        lines.append(f"Insight: {analytics['suggestion']}")

        return "\n".join(lines)

    async def autonomous_loop(
        self,
        task: str,
        check_interval: float = 60.0,
        max_iterations: int = 100,
    ):
        """
        Run autonomously on a task.
        
        The twin will work on the task, store progress,
        and continue until complete or max iterations.
        """
        self.state.is_autonomous = True
        self.state.current_task = task
        
        logger.info(f"Autonomous mode: {task}")
        
        for i in range(max_iterations):
            self.state.autonomous_iterations = i + 1
            
            prompt = f"""We're working autonomously on: {task}

Iteration: {i + 1}/{max_iterations}
Runtime: {(datetime.now() - self.state.started_at).total_seconds() / 60:.1f} minutes

What should we do next? Check session context for progress.
Emit [TASK_COMPLETE] when done, [SLEEP] to pause."""
            
            async for _ in self.think(prompt, stream=False):
                pass
            
            # Check for completion/sleep
            if self.memory_pipeline.session_outputs:
                last = self.memory_pipeline.session_outputs[-1]
                if last.thought_type == ThoughtType.TASK_COMPLETE:
                    logger.info(f"Task complete at iteration {i + 1}")
                    break
                if "[SLEEP]" in last.content:
                    logger.info(f"Sleeping for {check_interval}s")
                    await asyncio.sleep(check_interval)
            
            await asyncio.sleep(1.0)
        
        self.state.is_autonomous = False
        self.state.current_task = None


# ===== CLI =====

async def main():
    """Interactive Cognitive Twin CLI."""
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging()

    import sys

    # Data dir from CLI arg or config.yaml
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    
    print("=" * 60)
    print("  COGNITIVE TWIN - Unified System")
    print("  One brain. One memory. One voice.")
    print("=" * 60)
    
    try:
        twin = CogTwin(data_dir=data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run ingest.py first to process chat exports.")
        return
    
    # HITL callback
    def review_proposal(proposal: Dict) -> bool:
        print("\n" + "=" * 60)
        print("CODE PROPOSAL - HUMAN REVIEW REQUIRED")
        print("=" * 60)
        print(proposal["content"][:500])
        response = input("\nApprove? (y/n): ").strip().lower()
        return response == "y"
    
    twin.on_code_proposal = review_proposal
    
    await twin.start()
    
    print(f"\nLoaded {twin.memory_count} memories")
    print("\nCommands:")
    print("  /status  - Show cognitive state")
    print("  /health  - Run health check")
    print("  /feedback <1-5> - Rate last response")
    print("  /viz     - Show exploration chain visualization")
    print("  /ingest-traces - Batch ingest reasoning traces to memory")
    print("  /auto <task> - Autonomous mode")
    print("  /quit    - Exit\n")
    
    try:
        while True:
            try:
                user_input = input("You> ").strip()
                if not user_input:
                    continue
                
                if user_input == "/quit":
                    break
                
                if user_input == "/status":
                    state = twin.get_cognitive_state()
                    stats = twin.get_session_stats()
                    print(f"\nPhase: {state['phase']}")
                    print(f"Temperature: {state['temperature']:.2f}")
                    print(f"Focus: {state['focus_score']:.2f}")
                    print(f"Session outputs: {state['session_outputs']}")
                    print(f"Total queries: {stats['total_queries']}")
                    print(f"Tokens used: {stats['total_tokens']}\n")
                    continue
                
                if user_input == "/health":
                    print("\nRunning health check...")
                    insights = await twin.run_health_check()
                    for insight in insights:
                        print(f"  [{insight['severity']}] {insight['type']}: {insight['description']}")
                    print()
                    continue
                
                if user_input.startswith("/feedback"):
                    parts = user_input.split()
                    if len(parts) < 2:
                        print("\nUsage: /feedback <1-5> [optional comment]")
                        print("Example: /feedback 5 Great analysis!")
                        continue

                    try:
                        rating = int(parts[1])
                        if rating < 1 or rating > 5:
                            raise ValueError("Rating must be 1-5")
                    except ValueError as e:
                        print(f"\nError: {e}")
                        continue

                    comment = " ".join(parts[2:]) if len(parts) > 2 else None

                    if not twin._last_query_id:
                        print("\nNo previous query to provide feedback for.")
                        continue

                    # Create feedback object
                    # InteractionFeedback and provide_feedback removed with cognitive_agent
                    # Feedback now stored directly in traces and chat memory

                    # Also save to reasoning trace
                    if twin.last_trace_id:
                        normalized = rating / 5.0  # Normalize 1-5 to 0-1
                        twin.tracer.add_score_to_trace(
                            twin.last_trace_id,
                            {"overall": normalized},
                            {"comment": comment} if comment else None
                        )
                        print(f"\n[Score {rating}/5 saved to trace {twin.last_trace_id[:8]}]")

                    # Also save to chat memory (Phase 6.2)
                    if twin._last_exchange_id:
                        twin.chat_memory.add_rating(
                            twin._last_exchange_id,
                            overall=normalized,
                            notes=comment,
                        )

                    print(f"Feedback recorded (rating: {rating}/5)\n")
                    continue

                if user_input == "/viz":
                    print("\n/viz command deprecated (visualize_exploration removed with cognitive_agent)\n")
                    continue

                if user_input.startswith("/auto "):
                    task = user_input[6:]
                    print(f"\nAutonomous mode: {task}")
                    print("(Ctrl+C to interrupt)\n")
                    await twin.autonomous_loop(task, max_iterations=10)
                    continue

                if user_input == "/ingest-traces":
                    print("\nIngesting reasoning traces into memory...")
                    from ingest import ingest_reasoning_traces
                    from dedup import DedupBatch

                    traces_dir = twin.data_dir / "reasoning_traces"

                    with DedupBatch(twin.data_dir) as dedup:
                        stats = ingest_reasoning_traces(traces_dir, twin.data_dir, dedup)

                    print(f"Trace ingestion complete:")
                    print(f"  Found: {stats['total_found']}")
                    print(f"  Already ingested: {stats['already_ingested']}")
                    print(f"  Newly ingested: {stats['newly_ingested']}")
                    print(f"  Failed: {stats['failed']}")

                    # Reload retriever to pick up new episodes
                    print("Reloading retriever...")
                    twin.retriever = DualRetriever.load(twin.data_dir)
                    twin.memory_count = len(twin.retriever.process.nodes)
                    twin.voice.memory_count = twin.memory_count
                    print(f"Done. {twin.memory_count} memories loaded.\n")
                    continue

                # Normal query
                print("\nTwin> ", end="", flush=True)
                async for chunk in twin.think(user_input):
                    print(chunk, end="", flush=True)
                print("\n")

                # Training mode scoring
                if twin.training_mode.enabled and twin.last_trace_id:
                    score = twin.training_mode.prompt_for_score()
                    if score:
                        twin.tracer.add_score_to_trace(
                            twin.last_trace_id,
                            score.to_dict(),
                            score.get_feedback_notes(),
                        )
                        print(f"[Score saved: overall={score.overall:.2f}]")

                        # Also save to chat memory (Phase 6.2)
                        if twin._last_exchange_id:
                            twin.chat_memory.add_rating(
                                twin._last_exchange_id,
                                overall=score.overall,
                                accuracy=score.accuracy,
                                temporal=score.context_use,
                                tone=score.tone_match,
                                notes="; ".join(
                                    f"{k}: {v}"
                                    for k, v in score.get_feedback_notes().items()
                                ) if score.get_feedback_notes() else None,
                            )
                
            except KeyboardInterrupt:
                print("\n[Interrupted]")
                if twin.state.is_autonomous:
                    twin.state.is_autonomous = False
                    print("Autonomous mode stopped.")
    
    finally:
        await twin.stop()
        print("\nMemories persist. Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())