"""
hybrid_search.py - Semantic + Keyword Hybrid Search

Replaces BM25-only GREP with a hybrid approach:
1. Semantic search: Embed query, find similar memories via FAISS
2. Keyword search: Keep BM25 for exact term matching
3. Merge via RRF: Reciprocal Rank Fusion combines both rankings

This gives us:
- "vitamin" finds "supplement", "nutrition", "health" (semantic)
- "vitamin" still finds exact mentions (keyword)
- Results ranked by combined relevance

The model gets BOTH semantic matches AND keyword hits, with clear provenance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import numpy as np

from memory_grep import MemoryGrep, GrepResult, GrepHit

logger = logging.getLogger(__name__)


@dataclass
class HybridHit:
    """A single search result with provenance from both search methods."""
    memory_id: str
    node: Any  # MemoryNode

    # Scores from each method (None if not found by that method)
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    keyword_count: int = 0  # Exact keyword occurrences

    # Combined ranking
    rrf_score: float = 0.0

    # Provenance
    found_by: List[str] = field(default_factory=list)  # ["semantic", "keyword"]
    snippet: str = ""
    timestamp: Optional[datetime] = None


@dataclass
class HybridResult:
    """Combined result from hybrid search."""
    query: str
    hits: List[HybridHit]

    # Stats
    semantic_count: int = 0  # Unique hits from semantic
    keyword_count: int = 0   # Unique hits from keyword
    overlap_count: int = 0   # Found by both

    # Original results for debugging
    grep_result: Optional[GrepResult] = None


class HybridSearch:
    """
    Hybrid semantic + keyword search over memory nodes.

    Uses FAISS for semantic similarity and BM25 for keyword matching,
    then merges results with Reciprocal Rank Fusion.

    Trust hierarchy remains:
    - Keyword hits get a boost (precision signal)
    - Semantic hits expand recall (find related concepts)
    - RRF balances both rankings
    """

    def __init__(
        self,
        nodes: List[Any],
        embeddings: np.ndarray,
        faiss_index: Any,
        node_map: Dict[str, Any],
        embedder: Any,  # AsyncEmbedder for query embedding
        grep: Optional[MemoryGrep] = None,
    ):
        """
        Initialize hybrid search.

        Args:
            nodes: List of MemoryNode objects
            embeddings: Pre-computed embeddings (N x 1024)
            faiss_index: FAISS index for semantic search
            node_map: Dict mapping node ID to node object
            embedder: AsyncEmbedder for embedding search queries
            grep: Optional MemoryGrep instance (will create if not provided)
        """
        self.nodes = nodes
        self.embeddings = embeddings
        self.faiss_index = faiss_index
        self.node_map = node_map
        self.embedder = embedder

        # Create or use provided grep
        self.grep = grep if grep else MemoryGrep(nodes)

        logger.info(f"HybridSearch initialized: {len(nodes)} nodes, FAISS + BM25")

    async def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_k: int = 20,
        keyword_k: int = 20,
        rrf_k: int = 60,
        keyword_boost: float = 1.5,
        min_semantic_score: float = 0.3,
    ) -> HybridResult:
        """
        Hybrid search combining semantic and keyword matching.

        Args:
            query: Search query (single word or phrase)
            top_k: Final results to return
            semantic_k: Candidates from semantic search
            keyword_k: Candidates from keyword search
            rrf_k: RRF ranking parameter (higher = more weight to top ranks)
            keyword_boost: Multiplier for keyword matches in RRF
            min_semantic_score: Minimum cosine similarity for semantic hits

        Returns:
            HybridResult with merged, deduplicated results
        """
        # 1. Semantic search via FAISS
        semantic_hits = await self._semantic_search(query, semantic_k, min_semantic_score)

        # 2. Keyword search via BM25/inverted index
        keyword_hits, grep_result = self._keyword_search(query, keyword_k)

        # 3. Merge with RRF
        merged = self._rrf_merge(
            semantic_hits,
            keyword_hits,
            rrf_k=rrf_k,
            keyword_boost=keyword_boost,
        )

        # 4. Build final results
        hits = list(merged.values())
        hits.sort(key=lambda h: h.rrf_score, reverse=True)
        hits = hits[:top_k]

        # Calculate stats
        semantic_only = sum(1 for h in hits if h.found_by == ["semantic"])
        keyword_only = sum(1 for h in hits if h.found_by == ["keyword"])
        both = sum(1 for h in hits if len(h.found_by) == 2)

        return HybridResult(
            query=query,
            hits=hits,
            semantic_count=len(semantic_hits),
            keyword_count=len(keyword_hits),
            overlap_count=both,
            grep_result=grep_result,
        )

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
    ) -> List[Tuple[str, float, Any]]:
        """
        Semantic search via FAISS.

        Returns list of (memory_id, score, node) tuples.
        """
        # Embed query
        query_embedding = await self.embedder.embed_single(query)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_norm = query_norm.astype(np.float32).reshape(1, -1)

        # FAISS search
        try:
            distances, indices = self.faiss_index.search(query_norm, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.nodes):
                    continue

                score = float(dist)  # Cosine similarity (IndexFlatIP)
                if score < min_score:
                    continue

                node = self.nodes[idx]
                results.append((node.id, score, node))

            logger.debug(f"Semantic search '{query}': {len(results)} hits (min_score={min_score})")
            return results

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> Tuple[List[Tuple[str, float, int, Any]], Optional[GrepResult]]:
        """
        Keyword search via BM25.

        Returns:
            Tuple of (results, grep_result) where results is list of
            (memory_id, bm25_score, keyword_count, node) tuples
        """
        # Use grep for exact keyword matching
        grep_result = self.grep.grep(query, max_hits=top_k)

        results = []
        for hit in grep_result.hits:
            node = self.node_map.get(hit.memory_id)
            if not node:
                continue

            # Use keyword count as a proxy for relevance
            # Normalize by dividing by max to get 0-1 range
            max_count = max(h.count for h in grep_result.hits) if grep_result.hits else 1
            score = hit.count / max_count

            results.append((hit.memory_id, score, hit.count, node))

        logger.debug(f"Keyword search '{query}': {len(results)} hits, {grep_result.total_occurrences} occurrences")
        return results, grep_result

    def _rrf_merge(
        self,
        semantic_hits: List[Tuple[str, float, Any]],
        keyword_hits: List[Tuple[str, float, int, Any]],
        rrf_k: int = 60,
        keyword_boost: float = 1.5,
    ) -> Dict[str, HybridHit]:
        """
        Merge results using Reciprocal Rank Fusion.

        RRF formula: score = sum(1 / (k + rank_i)) for each ranking

        Args:
            semantic_hits: (memory_id, score, node) from semantic search
            keyword_hits: (memory_id, score, count, node) from keyword search
            rrf_k: Ranking parameter (60 is standard)
            keyword_boost: Multiplier for keyword ranking contribution

        Returns:
            Dict mapping memory_id to HybridHit with combined score
        """
        merged: Dict[str, HybridHit] = {}

        # Process semantic results
        for rank, (mem_id, score, node) in enumerate(semantic_hits, start=1):
            if mem_id not in merged:
                merged[mem_id] = HybridHit(
                    memory_id=mem_id,
                    node=node,
                    timestamp=getattr(node, 'created_at', None),
                )

            merged[mem_id].semantic_score = score
            merged[mem_id].found_by.append("semantic")
            merged[mem_id].rrf_score += 1.0 / (rrf_k + rank)

        # Process keyword results (with boost)
        for rank, (mem_id, score, count, node) in enumerate(keyword_hits, start=1):
            if mem_id not in merged:
                merged[mem_id] = HybridHit(
                    memory_id=mem_id,
                    node=node,
                    timestamp=getattr(node, 'created_at', None),
                )

            merged[mem_id].keyword_score = score
            merged[mem_id].keyword_count = count
            merged[mem_id].found_by.append("keyword")

            # Keyword matches get boosted contribution
            merged[mem_id].rrf_score += keyword_boost / (rrf_k + rank)

            # Extract snippet if we have it
            content = getattr(node, 'combined_content', '') or ''
            merged[mem_id].snippet = self._extract_snippet(content, 100)

        return merged

    def _extract_snippet(self, content: str, max_len: int = 100) -> str:
        """Extract a preview snippet from content."""
        if len(content) <= max_len:
            return content
        return content[:max_len] + "..."

    def format_for_context(self, result: HybridResult) -> str:
        """
        Format hybrid search result for injection into LLM context.

        Clearly shows provenance: which results came from semantic vs keyword.
        """
        if not result.hits:
            return f"HYBRID SEARCH: '{result.query}' - No results found."

        lines = [
            f"HYBRID SEARCH: '{result.query}'",
            f"Found: {len(result.hits)} results (semantic: {result.semantic_count}, keyword: {result.keyword_count}, overlap: {result.overlap_count})",
            "",
        ]

        # Group by provenance for clarity
        semantic_only = [h for h in result.hits if h.found_by == ["semantic"]]
        keyword_only = [h for h in result.hits if h.found_by == ["keyword"]]
        both = [h for h in result.hits if len(h.found_by) == 2]

        if both:
            lines.append("=== FOUND BY BOTH (highest confidence) ===")
            for hit in both[:3]:
                self._format_hit(hit, lines)
            lines.append("")

        if keyword_only:
            lines.append("=== EXACT KEYWORD MATCHES ===")
            for hit in keyword_only[:3]:
                self._format_hit(hit, lines)
            lines.append("")

        if semantic_only:
            lines.append("=== SEMANTIC MATCHES (related concepts) ===")
            for hit in semantic_only[:3]:
                self._format_hit(hit, lines)

        return "\n".join(lines)

    def _format_hit(self, hit: HybridHit, lines: List[str]) -> None:
        """Format a single hit for context output."""
        ts = hit.timestamp.strftime("%Y-%m-%d") if hit.timestamp else "unknown"

        provenance = []
        if hit.semantic_score is not None:
            provenance.append(f"semantic={hit.semantic_score:.2f}")
        if hit.keyword_count > 0:
            provenance.append(f"keyword_count={hit.keyword_count}")

        prov_str = ", ".join(provenance)

        # Get content preview
        node = hit.node
        human = getattr(node, 'human_content', '')[:150] if node else ""
        assistant = getattr(node, 'assistant_content', '')[:200] if node else ""

        lines.append(f"[{ts}] ({prov_str}) RRF={hit.rrf_score:.3f}")
        if human:
            lines.append(f"  Q: {human}...")
        if assistant:
            lines.append(f"  A: {assistant}...")


def create_hybrid_search(
    nodes: List[Any],
    embeddings: np.ndarray,
    faiss_index: Any,
    embedder: Any,
    grep: Optional[MemoryGrep] = None,
) -> HybridSearch:
    """
    Factory function to create HybridSearch.

    Args:
        nodes: List of MemoryNode objects
        embeddings: Pre-computed embeddings (N x 1024)
        faiss_index: FAISS index
        embedder: AsyncEmbedder instance
        grep: Optional MemoryGrep instance

    Returns:
        Configured HybridSearch instance
    """
    node_map = {n.id: n for n in nodes}

    return HybridSearch(
        nodes=nodes,
        embeddings=embeddings,
        faiss_index=faiss_index,
        node_map=node_map,
        embedder=embedder,
        grep=grep,
    )
