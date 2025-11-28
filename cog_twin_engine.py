"""
Cog Twin Engine - Unified interface for external memory and cognition.

The main entry point that wires together:
- Dual retrieval (Process + Episodic memory)
- Metacognitive mirror (system watching itself think)
- Cognitive twin (personalized digital twin)
- Venom voice (we/us/our answer style)

Usage:
    engine = CogTwinEngine("./data")
    response = await engine.query("What did we work on with FastAPI?")
    print(response.venom_response)  # "We were debugging async handlers..."

Version: 1.0.0 (cog_twin)
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

# Local imports
from retrieval import DualRetriever, RetrievalResult
from embedder import AsyncEmbedder
from metacognitive_mirror import (
    MetacognitiveMirror,
    QueryEvent,
    CognitivePhase
)

logger = logging.getLogger(__name__)


@dataclass
class VenomResponse:
    """
    Response in Venom voice - we/us/our plural.

    Why: The cognitive twin speaks as "we" because it represents the
    shared cognition between you and your external memory. "We were
    debugging X because Y" not "You were debugging X".
    """
    query: str
    venom_response: str  # "We were working on..."

    # Retrieved context
    process_memories: List[Dict[str, Any]]  # What/How memories
    episodic_memories: List[Dict[str, Any]]  # Why/When memories

    # Metacognitive state
    cognitive_phase: CognitivePhase
    confidence: float

    # Follow-up suggestions
    suggested_queries: List[str] = field(default_factory=list)
    related_domains: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    retrieval_time_ms: float = 0.0

    def __str__(self) -> str:
        return self.venom_response


class CogTwinEngine:
    """
    Main interface for the cognitive twin memory engine.

    Provides a simple query interface that:
    1. Embeds the query
    2. Retrieves from both process and episodic memory
    3. Builds context with Venom voice
    4. Tracks metacognitive state

    Design: Thin orchestration layer over retrieval + metacognition.
    All the heavy lifting happens in specialized modules.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the cognitive twin engine.

        Args:
            data_dir: Path to data directory with vectors/nodes/indexes
            config: Optional configuration overrides
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}

        # Initialize embedder (for queries)
        self.embedder = AsyncEmbedder(
            cache_dir=self.data_dir / "embedding_cache"
        )

        # Initialize dual retriever
        self.retriever = DualRetriever(
            data_dir=self.data_dir,
            process_top_k=self.config.get("process_top_k", 10),
            episodic_top_k=self.config.get("episodic_top_k", 5),
        )

        # Initialize metacognitive mirror
        self.mirror = MetacognitiveMirror(
            config={
                "query_window_size": self.config.get("query_window_size", 100),
                "query_cluster_epsilon": self.config.get("cluster_epsilon", 0.3),
            }
        )

        # Track query history for context
        self.query_history: List[str] = []

        logger.info(f"CogTwinEngine initialized with data from {self.data_dir}")

    async def query(
        self,
        query_text: str,
        include_episodic: bool = True,
        min_score: float = 0.3,
    ) -> VenomResponse:
        """
        Query the cognitive twin.

        Args:
            query_text: Natural language query
            include_episodic: Whether to search episodic memory too
            min_score: Minimum similarity score threshold

        Returns:
            VenomResponse with retrieved context in "we" voice
        """
        start_time = datetime.now()

        # Embed the query
        query_embedding = await self.embedder.embed_single(query_text)

        # Retrieve from dual memory
        retrieval_result = await self.retriever.query(
            query_text=query_text,
            query_embedding=query_embedding,
            include_episodic=include_episodic,
            min_score=min_score,
        )

        # Build Venom response
        venom_text = self._build_venom_response(query_text, retrieval_result)

        # Record for metacognitive tracking
        query_event = QueryEvent(
            timestamp=datetime.now(),
            query_text=query_text,
            query_embedding=query_embedding,
            retrieved_memory_ids=[
                m.get("id", f"mem_{i}")
                for i, m in enumerate(retrieval_result.process_results)
            ],
            retrieval_scores=[
                m.get("score", 0.0)
                for m in retrieval_result.process_results
            ],
            execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            result_count=len(retrieval_result.process_results) + len(retrieval_result.episodic_results),
            semantic_gate_passed=len(retrieval_result.process_results) > 0,
        )
        self.mirror.record_query(query_event)

        # Get metacognitive insights
        insights = self.mirror.get_real_time_insights()
        cognitive_phase = CognitivePhase(insights.get("cognitive_phase", "idle"))

        # Track history
        self.query_history.append(query_text)

        # Generate follow-up suggestions
        suggestions = self._generate_suggestions(retrieval_result, insights)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return VenomResponse(
            query=query_text,
            venom_response=venom_text,
            process_memories=retrieval_result.process_results,
            episodic_memories=retrieval_result.episodic_results,
            cognitive_phase=cognitive_phase,
            confidence=insights.get("focus_score", 0.5),
            suggested_queries=suggestions,
            related_domains=list(set(
                m.get("domain", "UNKNOWN")
                for m in retrieval_result.process_results
            )),
            retrieval_time_ms=elapsed_ms,
        )

    def _build_venom_response(
        self,
        query: str,
        result: RetrievalResult
    ) -> str:
        """
        Build response in Venom voice (we/us/our).

        Why: The cognitive twin is a shared entity. When it speaks,
        it uses "we" to represent the symbiosis between you and your
        external memory. "We were working on X" not "You were working on X".
        """
        if not result.process_results and not result.episodic_results:
            return "We don't have memories matching that query. Perhaps we should explore this topic together?"

        parts = []

        # Process memories (What/How)
        if result.process_results:
            parts.append("**From our process memory** (what we've learned):")
            for i, mem in enumerate(result.process_results[:3]):
                human = mem.get("human_content", "")[:150]
                assistant = mem.get("assistant_content", "")[:200]
                score = mem.get("score", 0.0)
                domain = mem.get("domain", "UNKNOWN")

                parts.append(f"\n[{domain}] (relevance: {score:.2f})")
                if human:
                    parts.append(f"  We asked: {human}...")
                if assistant:
                    parts.append(f"  We found: {assistant}...")

        # Episodic memories (Why/When)
        if result.episodic_results:
            parts.append("\n\n**From our episodic memory** (when & why):")
            for ep in result.episodic_results[:2]:
                title = ep.get("title", "Untitled conversation")
                date = ep.get("created_at", "Unknown date")
                msg_count = ep.get("message_count", 0)
                summary = ep.get("summary", "")[:200]

                parts.append(f"\n[{date}] {title} ({msg_count} messages)")
                if summary:
                    parts.append(f"  Summary: {summary}...")

        return "\n".join(parts)

    def _generate_suggestions(
        self,
        result: RetrievalResult,
        insights: Dict[str, Any]
    ) -> List[str]:
        """Generate follow-up query suggestions."""
        suggestions = []

        # Based on recurring patterns
        patterns = insights.get("recurring_patterns", [])
        for pattern in patterns[:2]:
            if pattern.get("pattern"):
                suggestions.append(f"Tell me more about: {pattern['pattern'][:50]}...")

        # Based on related domains
        domains = set(m.get("domain", "UNKNOWN") for m in result.process_results)
        for domain in list(domains)[:2]:
            if domain != "UNKNOWN":
                suggestions.append(f"What else do we know about {domain.lower().replace('_', ' ')}?")

        return suggestions[:5]

    def get_health(self) -> Dict[str, Any]:
        """Get system health and metacognitive insights."""
        insights = self.mirror.get_real_time_insights()
        health_check = self.mirror.run_health_check()

        return {
            "status": "healthy",
            "cognitive_phase": insights.get("cognitive_phase"),
            "temperature": insights.get("temperature"),
            "focus_score": insights.get("focus_score"),
            "query_count": len(self.query_history),
            "insights": health_check,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            "embedder": self.embedder.get_stats(),
            "mirror": self.mirror.get_real_time_insights(),
            "query_history_size": len(self.query_history),
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

async def interactive_session(data_dir: str = "./data"):
    """
    Interactive query session with the cognitive twin.

    Usage: python cog_twin_engine.py [data_dir]
    """
    print("=" * 60)
    print("  COG TWIN ENGINE - We Are Venom")
    print("=" * 60)
    print("\nInitializing...")

    try:
        engine = CogTwinEngine(data_dir)
        print(f"Loaded data from {data_dir}")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        print("Make sure you've run ingest.py first to populate the data directory.")
        return

    print("\nReady! Type your queries. Commands:")
    print("  /health - Show system health")
    print("  /stats  - Show statistics")
    print("  /quit   - Exit")
    print()

    while True:
        try:
            query = input("\n[YOU] > ").strip()

            if not query:
                continue

            if query.lower() == "/quit":
                print("Goodbye from us!")
                break

            if query.lower() == "/health":
                health = engine.get_health()
                print(f"\nHealth: {health['status']}")
                print(f"Cognitive Phase: {health['cognitive_phase']}")
                print(f"Focus Score: {health['focus_score']:.2f}")
                continue

            if query.lower() == "/stats":
                stats = engine.get_stats()
                print(f"\nEmbedder: {stats['embedder']}")
                print(f"Query History: {stats['query_history_size']} queries")
                continue

            # Process query
            print("\n[VENOM] Searching our memories...")
            response = await engine.query(query)

            print(f"\n[VENOM] ({response.retrieval_time_ms:.0f}ms, {response.cognitive_phase.value})")
            print("-" * 40)
            print(response.venom_response)

            if response.suggested_queries:
                print("\n[VENOM] We might also explore:")
                for suggestion in response.suggested_queries:
                    print(f"  - {suggestion}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye from us!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.exception("Query error")


async def main():
    """Entry point."""
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    await interactive_session(data_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
