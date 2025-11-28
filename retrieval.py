"""
Dual Retrieval Engine - Process Memory + Episodic Memory

The heart of the Venom system: parallel retrieval from two memory types.

Process Memory (What/How):
- NumPy cosine similarity on clustered memory nodes
- Fast geometric search, cluster-aware boosting
- Returns: "Here's HOW we solved similar problems"

Episodic Memory (Why/When):
- FAISS index + heuristic pre-filtering
- Optional LLM reranking for final pass
- Returns: "We were doing this BECAUSE..."

Combined: "We were working on X using Y approach because Z context."

Usage:
    retriever = DualRetriever.load("./data")
    results = await retriever.retrieve("How do we handle async errors?", top_k=5)
    print(results.build_venom_context())

Version: 1.0.0 (cog_twin)
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from schemas import (
    MemoryNode, EpisodicMemory, RetrievalResult,
    Source, IntentType, Complexity, Urgency, EmotionalValence
)
from embedder import AsyncEmbedder
from heuristic_enricher import HeuristicEnricher

# Optional FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessMemoryRetriever:
    """
    Retriever for clustered memory nodes (What/How).

    Pure NumPy cosine similarity with cluster-aware boosting.
    Fast, simple, geometric.
    """

    def __init__(
        self,
        nodes: List[MemoryNode],
        embeddings: np.ndarray,
        cluster_info: Dict[int, List[int]],
    ):
        """
        Initialize process memory retriever.

        Args:
            nodes: List of memory nodes
            embeddings: Node embeddings (N x D)
            cluster_info: Cluster ID to node indices mapping
        """
        self.nodes = nodes
        self.embeddings = embeddings
        self.cluster_info = cluster_info

        # Pre-normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.normalized = embeddings / (norms + 1e-8)

        # Build cluster centroids for cluster-level search
        self.cluster_centroids: Dict[int, np.ndarray] = {}
        for cluster_id, indices in cluster_info.items():
            if cluster_id == -1:  # Skip noise
                continue
            cluster_vectors = self.normalized[indices]
            centroid = cluster_vectors.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            self.cluster_centroids[cluster_id] = centroid

        # Build index mapping for non-noise nodes
        self.valid_indices = [i for i, n in enumerate(nodes) if n.cluster_id != -1]

        logger.info(f"ProcessMemoryRetriever: {len(self.valid_indices)} valid nodes, "
                    f"{len(self.cluster_centroids)} clusters")

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        relevance_threshold: float = 0.5,
        cluster_boost: float = 0.1,
    ) -> Tuple[List[MemoryNode], List[float]]:
        """
        Retrieve relevant memory nodes.

        Args:
            query_embedding: Query vector (D,)
            top_k: Max results to return
            relevance_threshold: Minimum similarity score
            cluster_boost: Bonus for nodes in high-similarity clusters

        Returns:
            Tuple of (nodes, scores)
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Compute similarities to all valid nodes
        valid_embeddings = self.normalized[self.valid_indices]
        similarities = valid_embeddings @ query_norm

        # Cluster boosting: if query is similar to cluster centroid,
        # boost all members of that cluster
        if self.cluster_centroids:
            cluster_sims = {
                cid: float(centroid @ query_norm)
                for cid, centroid in self.cluster_centroids.items()
            }

            # Boost nodes in relevant clusters
            boosts = np.zeros(len(self.valid_indices))
            for i, idx in enumerate(self.valid_indices):
                cluster_id = self.nodes[idx].cluster_id
                if cluster_id in cluster_sims and cluster_sims[cluster_id] > relevance_threshold:
                    boosts[i] = cluster_boost * cluster_sims[cluster_id]

            similarities = similarities + boosts

        # Get top-k above threshold
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        scores = []
        for i in sorted_indices:
            if len(results) >= top_k:
                break
            if similarities[i] < relevance_threshold:
                break

            node_idx = self.valid_indices[i]
            results.append(self.nodes[node_idx])
            scores.append(float(similarities[i]))

        return results, scores

    def retrieve_by_cluster(
        self,
        query_embedding: np.ndarray,
        top_clusters: int = 3,
        nodes_per_cluster: int = 3,
    ) -> Tuple[List[MemoryNode], List[float]]:
        """
        Retrieve by first finding relevant clusters, then nodes within them.

        Good for exploring related topics.
        """
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Find top clusters by centroid similarity
        cluster_scores = [
            (cid, float(centroid @ query_norm))
            for cid, centroid in self.cluster_centroids.items()
        ]
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        top_cluster_ids = [cid for cid, _ in cluster_scores[:top_clusters]]

        # Get top nodes from each cluster
        results = []
        scores = []

        for cluster_id in top_cluster_ids:
            cluster_indices = self.cluster_info[cluster_id]
            cluster_embeddings = self.normalized[cluster_indices]
            cluster_sims = cluster_embeddings @ query_norm

            sorted_local = np.argsort(cluster_sims)[::-1][:nodes_per_cluster]

            for local_idx in sorted_local:
                global_idx = cluster_indices[local_idx]
                results.append(self.nodes[global_idx])
                scores.append(float(cluster_sims[local_idx]))

        return results, scores


class EpisodicMemoryRetriever:
    """
    Retriever for full conversations (Why/When).

    FAISS index with heuristic pre-filtering.
    Optional LLM reranking for precision.
    """

    def __init__(
        self,
        episodes: List[EpisodicMemory],
        embeddings: np.ndarray,
        faiss_index: Optional[Any] = None,
    ):
        """
        Initialize episodic memory retriever.

        Args:
            episodes: List of episodic memories
            embeddings: Episode embeddings (N x D)
            faiss_index: Pre-built FAISS index (or None to build)
        """
        self.episodes = episodes
        self.embeddings = embeddings

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.normalized = (embeddings / (norms + 1e-8)).astype(np.float32)

        # Build or use FAISS index
        if faiss_index is not None:
            self.index = faiss_index
        elif FAISS_AVAILABLE:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.normalized)
        else:
            self.index = None
            logger.warning("FAISS not available, using NumPy fallback")

        self.enricher = HeuristicEnricher()

        logger.info(f"EpisodicMemoryRetriever: {len(episodes)} episodes")

    def _apply_heuristic_filter(
        self,
        query: str,
        candidate_indices: List[int],
    ) -> List[int]:
        """
        Filter candidates using heuristic matching.

        Args:
            query: Query string
            candidate_indices: Indices to filter

        Returns:
            Filtered indices (same or fewer)
        """
        query_signals = self.enricher.extract_all(query)
        query_intent = query_signals.get("intent_type", "question")
        query_urgency = query_signals.get("urgency", "low")
        query_domain = query_signals.get("primary_domain", "general")

        # Score candidates by heuristic match
        scored = []
        for idx in candidate_indices:
            ep = self.episodes[idx]
            score = 0

            # Domain match
            if query_domain in ep.llm_tags.get("domains", []):
                score += 2
            if query_domain != "general":
                # Check if any message contains domain keywords
                for msg in ep.messages[:5]:  # Check first few messages
                    content = msg.get("content", "").lower()
                    if query_domain.lower() in content:
                        score += 1
                        break

            # Intent alignment
            if query_intent == "question" and ep.dominant_intent in [IntentType.QUESTION, IntentType.REQUEST]:
                score += 1

            # Recency boost (recent episodes more likely relevant)
            # Could add time-based scoring here

            scored.append((idx, score))

        # Keep all but sort by heuristic score (secondary to embedding score)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored]

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        pre_filter_k: int = 50,
        relevance_threshold: float = 0.3,
    ) -> Tuple[List[EpisodicMemory], List[float]]:
        """
        Retrieve relevant episodic memories.

        Args:
            query: Query string (for heuristic filtering)
            query_embedding: Query vector (D,)
            top_k: Max results to return
            pre_filter_k: Candidates to consider before heuristic filter
            relevance_threshold: Minimum similarity score

        Returns:
            Tuple of (episodes, scores)
        """
        # Normalize query
        query_norm = (query_embedding / (np.linalg.norm(query_embedding) + 1e-8)).astype(np.float32)

        # FAISS search or NumPy fallback
        if self.index is not None and FAISS_AVAILABLE:
            distances, indices = self.index.search(
                query_norm.reshape(1, -1),
                pre_filter_k
            )
            candidates = indices[0].tolist()
            candidate_scores = distances[0].tolist()
        else:
            # NumPy fallback
            similarities = self.normalized @ query_norm
            sorted_indices = np.argsort(similarities)[::-1][:pre_filter_k]
            candidates = sorted_indices.tolist()
            candidate_scores = [float(similarities[i]) for i in candidates]

        # Apply heuristic filter
        filtered_candidates = self._apply_heuristic_filter(query, candidates)

        # Build results
        results = []
        scores = []

        for idx in filtered_candidates:
            if len(results) >= top_k:
                break

            # Get original score
            try:
                score = candidate_scores[candidates.index(idx)]
            except ValueError:
                continue

            if score < relevance_threshold:
                continue

            results.append(self.episodes[idx])
            scores.append(score)

        return results, scores


class DualRetriever:
    """
    Combined retriever that queries both memory systems.

    This is the main interface for the Venom memory engine.
    """

    def __init__(
        self,
        process_retriever: ProcessMemoryRetriever,
        episodic_retriever: EpisodicMemoryRetriever,
        embedder: AsyncEmbedder,
    ):
        """
        Initialize dual retriever.

        Args:
            process_retriever: Memory node retriever
            episodic_retriever: Episode retriever
            embedder: Async embedder for query embedding
        """
        self.process = process_retriever
        self.episodic = episodic_retriever
        self.embedder = embedder

    @classmethod
    def load(cls, data_dir: Path, manifest_file: Optional[str] = None) -> "DualRetriever":
        """
        Load retriever from saved data.

        Args:
            data_dir: Directory containing saved data
            manifest_file: Specific manifest to load (or latest)

        Returns:
            Initialized DualRetriever
        """
        data_dir = Path(data_dir)

        # Find manifest
        if manifest_file:
            manifest_path = data_dir / manifest_file
        else:
            manifests = list(data_dir.glob("manifest_*.json"))
            if not manifests:
                raise FileNotFoundError(f"No manifest found in {data_dir}")
            manifest_path = max(manifests, key=lambda p: p.stat().st_mtime)

        logger.info(f"Loading from manifest: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Load memory nodes
        nodes_file = data_dir / "memory_nodes" / manifest["nodes_file"]
        with open(nodes_file) as f:
            nodes_data = json.load(f)
        nodes = [MemoryNode.from_dict(d) for d in nodes_data]

        # Load episodes
        episodes_file = data_dir / "episodic_memories" / manifest["episodes_file"]
        with open(episodes_file) as f:
            episodes_data = json.load(f)
        episodes = [EpisodicMemory.from_dict(d) for d in episodes_data]

        # Load embeddings
        node_emb_file = data_dir / "vectors" / manifest["node_embeddings_file"]
        node_embeddings = np.load(node_emb_file)

        episode_emb_file = data_dir / "vectors" / manifest["episode_embeddings_file"]
        episode_embeddings = np.load(episode_emb_file)

        # Load cluster info
        clusters_file = data_dir / "indexes" / manifest["clusters_file"]
        with open(clusters_file) as f:
            cluster_info = json.load(f)
        cluster_info = {int(k): v for k, v in cluster_info.items()}

        # Load FAISS index if available
        faiss_index = None
        if manifest.get("faiss_index_file") and FAISS_AVAILABLE:
            faiss_file = data_dir / "indexes" / manifest["faiss_index_file"]
            if faiss_file.exists():
                faiss_index = faiss.read_index(str(faiss_file))

        # Build retrievers
        process_retriever = ProcessMemoryRetriever(nodes, node_embeddings, cluster_info)
        episodic_retriever = EpisodicMemoryRetriever(episodes, episode_embeddings, faiss_index)

        # Build embedder
        embedder = AsyncEmbedder(cache_dir=data_dir / "embedding_cache")

        return cls(process_retriever, episodic_retriever, embedder)

    async def retrieve(
        self,
        query: str,
        process_top_k: int = 5,
        episodic_top_k: int = 3,
        process_threshold: float = 0.5,
        episodic_threshold: float = 0.3,
    ) -> RetrievalResult:
        """
        Retrieve from both memory systems.

        Args:
            query: Query string
            process_top_k: Max process memories to return
            episodic_top_k: Max episodic memories to return
            process_threshold: Min relevance for process memories
            episodic_threshold: Min relevance for episodic memories

        Returns:
            RetrievalResult with both memory types
        """
        import time
        start = time.time()

        # Embed query
        query_embedding = await self.embedder.embed_single(query)

        # Retrieve from both systems in parallel (they're sync but fast)
        process_results, process_scores = self.process.retrieve(
            query_embedding,
            top_k=process_top_k,
            relevance_threshold=process_threshold,
        )

        episodic_results, episodic_scores = self.episodic.retrieve(
            query,
            query_embedding,
            top_k=episodic_top_k,
            relevance_threshold=episodic_threshold,
        )

        elapsed = (time.time() - start) * 1000

        result = RetrievalResult(
            query=query,
            process_memories=process_results,
            process_scores=process_scores,
            episodic_memories=episodic_results,
            episodic_scores=episodic_scores,
            retrieval_time_ms=elapsed,
            process_candidates_scanned=len(self.process.valid_indices),
            episodic_candidates_scanned=len(self.episodic.episodes),
        )

        # Build merged context
        result.build_venom_context()

        return result

    async def retrieve_with_context(
        self,
        query: str,
        context_queries: List[str] = None,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve with additional context queries for better matching.

        Useful when you have follow-up context.
        """
        # Embed all queries and average
        queries = [query] + (context_queries or [])
        embeddings = await self.embedder.embed_batch(queries, show_progress=False)

        # Weighted average (primary query has more weight)
        weights = [2.0] + [1.0] * len(context_queries or [])
        weighted = embeddings * np.array(weights).reshape(-1, 1)
        combined = weighted.sum(axis=0) / sum(weights)

        # Use combined embedding
        import time
        start = time.time()

        process_results, process_scores = self.process.retrieve(
            combined,
            top_k=kwargs.get("process_top_k", 5),
            relevance_threshold=kwargs.get("process_threshold", 0.5),
        )

        episodic_results, episodic_scores = self.episodic.retrieve(
            query,
            combined,
            top_k=kwargs.get("episodic_top_k", 3),
            relevance_threshold=kwargs.get("episodic_threshold", 0.3),
        )

        elapsed = (time.time() - start) * 1000

        result = RetrievalResult(
            query=query,
            process_memories=process_results,
            process_scores=process_scores,
            episodic_memories=episodic_results,
            episodic_scores=episodic_scores,
            retrieval_time_ms=elapsed,
        )

        result.build_venom_context()
        return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Interactive retrieval CLI."""
    from dotenv import load_dotenv
    load_dotenv()

    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data")

    print("Dual Retriever - We Are Venom")
    print("=" * 60)

    try:
        retriever = DualRetriever.load(data_dir)
        print(f"Loaded from {data_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run ingest.py first to process chat exports.")
        return

    print("\nEnter queries (Ctrl+C to exit):\n")

    while True:
        try:
            query = input("Query> ").strip()
            if not query:
                continue

            result = await retriever.retrieve(query)

            print(f"\n[Retrieved in {result.retrieval_time_ms:.1f}ms]")
            print(result.merged_context)
            print("\n" + "-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
