"""
Ingest Pipeline - Raw Chat Exports to Dual Memory Store

The main orchestrator that takes raw JSON exports and produces:
1. Memory Nodes: 1:1 Q/A pairs → clustered → NumPy vectors
2. Episodic Memories: Full conversations → FAISS index → LLM tagged

Pipeline:
    Raw JSON → Parse → Split → Enrich → Embed → Cluster/Index → Store

Usage:
    python ingest.py ./chat_exports/anthropic/conversations.json
    python ingest.py ./chat_exports/ --recursive

Version: 1.0.0 (cog_twin)
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Local imports
from chat_parser_agnostic import ChatParserFactory
from heuristic_enricher import HeuristicEnricher, enrich_nodes_batch
from schemas import (
    Source, MemoryNode, EpisodicMemory,
    conversation_to_nodes, conversation_to_episode,
    IntentType, Complexity, ConversationMode, Urgency, EmotionalValence
)
from embedder import AsyncEmbedder, embed_memory_nodes, embed_episodes

# Optional imports
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed. Clustering will be skipped.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. FAISS indexing will be skipped.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    Main ingest pipeline for processing chat exports into memory stores.

    Handles both memory node extraction (for process memory) and
    episodic memory preservation (for context memory).
    """

    def __init__(
        self,
        output_dir: Path = Path("./data"),
        deepinfra_api_key: Optional[str] = None,
        embedding_batch_size: int = 32,
        embedding_concurrency: int = 8,
    ):
        """
        Initialize the ingest pipeline.

        Args:
            output_dir: Base directory for all output
            deepinfra_api_key: API key for embeddings
            embedding_batch_size: Batch size for embedding API calls
            embedding_concurrency: Max concurrent embedding requests
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.nodes_dir = output_dir / "memory_nodes"
        self.episodes_dir = output_dir / "episodic_memories"
        self.vectors_dir = output_dir / "vectors"
        self.index_dir = output_dir / "indexes"

        for d in [self.nodes_dir, self.episodes_dir, self.vectors_dir, self.index_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Components
        self.parser = ChatParserFactory()
        self.enricher = HeuristicEnricher()
        self.embedder = AsyncEmbedder(
            api_key=deepinfra_api_key,
            cache_dir=output_dir / "embedding_cache",
        )

        # Config
        self.embedding_batch_size = embedding_batch_size
        self.embedding_concurrency = embedding_concurrency

        # Stats
        self.stats = {
            "files_processed": 0,
            "conversations_parsed": 0,
            "memory_nodes_created": 0,
            "episodes_created": 0,
            "nodes_clustered": 0,
            "noise_nodes_filtered": 0,
            "total_time_seconds": 0,
        }

    def _source_from_string(self, source_str: str) -> Source:
        """Convert source string to enum."""
        mapping = {
            "anthropic": Source.ANTHROPIC,
            "openai": Source.OPENAI,
            "grok": Source.GROK,
            "gemini": Source.GEMINI,
        }
        return mapping.get(source_str.lower(), Source.ANTHROPIC)

    def _enrich_node(self, node: MemoryNode) -> MemoryNode:
        """Apply heuristic enrichment to a memory node."""
        signals = self.enricher.extract_all(node.combined_content)

        # Map signals to node fields
        node.intent_type = IntentType(signals.get("intent_type", "statement"))
        node.complexity = Complexity(signals.get("complexity", "simple"))
        node.technical_depth = signals.get("technical_depth", 0)
        node.emotional_valence = EmotionalValence(signals.get("emotional_valence", "neutral"))
        node.urgency = Urgency(signals.get("urgency", "low"))
        node.conversation_mode = ConversationMode(signals.get("conversation_mode", "chat"))
        node.action_required = signals.get("action_required", False)
        node.has_code = signals.get("has_code", False)
        node.has_error = signals.get("has_error", False)

        # Dynamic tags from heuristics
        node.tags["domains"] = [signals.get("primary_domain", "general")]
        node.tags["entities"] = signals.get("entity_hints", [])

        return node

    def _enrich_episode(self, episode: EpisodicMemory) -> EpisodicMemory:
        """Apply heuristic enrichment to an episodic memory."""
        # Aggregate signals from all messages
        intents = []
        complexities = []
        tech_depths = []
        valences = []
        urgencies = []

        for msg in episode.messages:
            content = msg.get("content", "")
            if not content:
                continue

            signals = self.enricher.extract_all(content)
            intents.append(signals.get("intent_type", "statement"))
            complexities.append(signals.get("complexity", "simple"))
            tech_depths.append(signals.get("technical_depth", 0))
            valences.append(signals.get("emotional_valence", "neutral"))
            urgencies.append(signals.get("urgency", "low"))

            if signals.get("has_code"):
                episode.has_code = True
            if signals.get("has_error"):
                episode.has_errors = True

        # Set aggregated fields
        if intents:
            from collections import Counter
            episode.dominant_intent = IntentType(Counter(intents).most_common(1)[0][0])

        if complexities:
            complexity_scores = {"simple": 1, "moderate": 2, "complex": 3}
            avg_score = sum(complexity_scores.get(c, 1) for c in complexities) / len(complexities)
            episode.avg_complexity = avg_score

        if tech_depths:
            episode.avg_technical_depth = sum(tech_depths) / len(tech_depths)

        if valences:
            # Emotional arc: first → last
            first_valence = valences[0] if valences else "neutral"
            last_valence = valences[-1] if valences else "neutral"
            episode.emotional_arc = f"{first_valence}→{last_valence}"

        if urgencies:
            urgency_order = {"low": 0, "medium": 1, "high": 2}
            max_urgency = max(urgencies, key=lambda u: urgency_order.get(u, 0))
            episode.urgency_max = Urgency(max_urgency)

        return episode

    async def process_file(
        self,
        filepath: Path,
        provider: str = "auto",
    ) -> Tuple[List[MemoryNode], List[EpisodicMemory]]:
        """
        Process a single chat export file.

        Args:
            filepath: Path to JSON export
            provider: Provider hint or "auto" for detection

        Returns:
            Tuple of (memory_nodes, episodic_memories)
        """
        logger.info(f"Processing: {filepath}")

        # Parse
        conversations = self.parser.parse(str(filepath), provider=provider)
        source = self._source_from_string(self.parser.last_provider or "anthropic")

        self.stats["conversations_parsed"] += len(conversations)
        logger.info(f"Parsed {len(conversations)} conversations")

        # Convert to nodes and episodes
        all_nodes = []
        all_episodes = []

        for conv in conversations:
            # Create memory nodes (1:1 Q/A pairs)
            nodes = conversation_to_nodes(conv, source)
            for node in nodes:
                node = self._enrich_node(node)
            all_nodes.extend(nodes)

            # Create episodic memory (full conversation)
            episode = conversation_to_episode(conv, source)
            episode = self._enrich_episode(episode)

            # Link nodes to episode
            episode.memory_node_ids = [n.id for n in nodes]
            all_episodes.append(episode)

        self.stats["memory_nodes_created"] += len(all_nodes)
        self.stats["episodes_created"] += len(all_episodes)
        self.stats["files_processed"] += 1

        logger.info(f"Created {len(all_nodes)} memory nodes, {len(all_episodes)} episodes")

        return all_nodes, all_episodes

    async def embed_all(
        self,
        nodes: List[MemoryNode],
        episodes: List[EpisodicMemory],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed all nodes and episodes.

        Args:
            nodes: List of memory nodes
            episodes: List of episodic memories

        Returns:
            Tuple of (node_embeddings, episode_embeddings)
        """
        logger.info(f"Embedding {len(nodes)} nodes and {len(episodes)} episodes...")

        # Embed nodes
        node_texts = [n.combined_content for n in nodes]
        node_embeddings = await self.embedder.embed_batch(
            node_texts,
            batch_size=self.embedding_batch_size,
            max_concurrent=self.embedding_concurrency,
        )

        # Embed episodes (truncated full text)
        episode_texts = []
        for ep in episodes:
            text = ep.full_text
            if len(text) > 16000:
                text = text[:16000] + "\n\n[truncated]"
            episode_texts.append(text)

        episode_embeddings = await self.embedder.embed_batch(
            episode_texts,
            batch_size=16,  # Smaller batches for longer content
            max_concurrent=4,
        )

        logger.info(f"Embeddings complete: nodes={node_embeddings.shape}, episodes={episode_embeddings.shape}")

        return node_embeddings, episode_embeddings

    def cluster_nodes(
        self,
        nodes: List[MemoryNode],
        embeddings: np.ndarray,
        min_cluster_size: int = 3,
        min_samples: int = 1,
    ) -> Tuple[List[MemoryNode], Dict[int, List[int]]]:
        """
        Cluster memory nodes using HDBSCAN.

        Args:
            nodes: List of memory nodes
            embeddings: Node embeddings (N x D)
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples for core point

        Returns:
            Tuple of (nodes_with_cluster_ids, cluster_to_indices)
        """
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available, skipping clustering")
            return nodes, {}

        logger.info(f"Clustering {len(nodes)} nodes with HDBSCAN...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )

        # Normalize embeddings for better clustering
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        labels = clusterer.fit_predict(normalized)
        probabilities = clusterer.probabilities_

        # Assign cluster info to nodes
        cluster_to_indices: Dict[int, List[int]] = {}

        for i, (node, label, prob) in enumerate(zip(nodes, labels, probabilities)):
            node.cluster_id = int(label)
            node.cluster_confidence = float(prob)

            if label not in cluster_to_indices:
                cluster_to_indices[label] = []
            cluster_to_indices[label].append(i)

        # Stats
        n_clusters = len([k for k in cluster_to_indices.keys() if k != -1])
        n_noise = len(cluster_to_indices.get(-1, []))

        self.stats["nodes_clustered"] = len(nodes) - n_noise
        self.stats["noise_nodes_filtered"] = n_noise

        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(nodes):.1f}%)")

        return nodes, cluster_to_indices

    def build_faiss_index(
        self,
        embeddings: np.ndarray,
    ) -> Any:
        """
        Build FAISS index for episodic memory search.

        Args:
            embeddings: Episode embeddings (N x D)

        Returns:
            FAISS index
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index build")
            return None

        logger.info(f"Building FAISS index for {len(embeddings)} episodes...")

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = (embeddings / (norms + 1e-8)).astype(np.float32)

        # Use IndexFlatIP for exact inner product (cosine on normalized vectors)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(normalized)

        logger.info(f"FAISS index built: {index.ntotal} vectors")

        return index

    def save_all(
        self,
        nodes: List[MemoryNode],
        episodes: List[EpisodicMemory],
        node_embeddings: np.ndarray,
        episode_embeddings: np.ndarray,
        faiss_index: Any,
        cluster_info: Dict[int, List[int]],
    ) -> None:
        """
        Save all data to disk.

        Args:
            nodes: Memory nodes with cluster assignments
            episodes: Episodic memories
            node_embeddings: Node embedding matrix
            episode_embeddings: Episode embedding matrix
            faiss_index: FAISS index for episodes
            cluster_info: Cluster ID to node indices mapping
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Saving data with timestamp {timestamp}...")

        # Save memory nodes (JSON)
        nodes_file = self.nodes_dir / f"nodes_{timestamp}.json"
        nodes_data = [n.to_dict() for n in nodes]
        with open(nodes_file, "w") as f:
            json.dump(nodes_data, f, indent=2)
        logger.info(f"Saved {len(nodes)} nodes to {nodes_file}")

        # Save episodic memories (JSON)
        episodes_file = self.episodes_dir / f"episodes_{timestamp}.json"
        episodes_data = [e.to_dict() for e in episodes]
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f, indent=2)
        logger.info(f"Saved {len(episodes)} episodes to {episodes_file}")

        # Save embeddings (NumPy)
        node_emb_file = self.vectors_dir / f"node_embeddings_{timestamp}.npy"
        np.save(node_emb_file, node_embeddings)
        logger.info(f"Saved node embeddings to {node_emb_file}")

        episode_emb_file = self.vectors_dir / f"episode_embeddings_{timestamp}.npy"
        np.save(episode_emb_file, episode_embeddings)
        logger.info(f"Saved episode embeddings to {episode_emb_file}")

        # Save FAISS index
        if faiss_index is not None and FAISS_AVAILABLE:
            faiss_file = self.index_dir / f"episode_index_{timestamp}.faiss"
            faiss.write_index(faiss_index, str(faiss_file))
            logger.info(f"Saved FAISS index to {faiss_file}")

        # Save cluster info
        cluster_file = self.index_dir / f"clusters_{timestamp}.json"
        # Convert int keys to strings for JSON
        cluster_data = {str(k): v for k, v in cluster_info.items()}
        with open(cluster_file, "w") as f:
            json.dump(cluster_data, f, indent=2)
        logger.info(f"Saved cluster info to {cluster_file}")

        # Save manifest
        manifest = {
            "timestamp": timestamp,
            "nodes_file": nodes_file.name,
            "episodes_file": episodes_file.name,
            "node_embeddings_file": node_emb_file.name,
            "episode_embeddings_file": episode_emb_file.name,
            "faiss_index_file": f"episode_index_{timestamp}.faiss" if faiss_index else None,
            "clusters_file": cluster_file.name,
            "stats": self.stats,
        }
        manifest_file = self.output_dir / f"manifest_{timestamp}.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved manifest to {manifest_file}")

    async def run(
        self,
        input_paths: List[Path],
        provider: str = "auto",
    ) -> None:
        """
        Run the full ingest pipeline.

        Args:
            input_paths: List of file or directory paths
            provider: Provider hint or "auto"
        """
        start_time = time.time()

        # Collect all files to process
        files_to_process = []
        for path in input_paths:
            if path.is_file():
                files_to_process.append(path)
            elif path.is_dir():
                files_to_process.extend(path.glob("**/*.json"))

        logger.info(f"Found {len(files_to_process)} files to process")

        # Process all files
        all_nodes = []
        all_episodes = []

        for filepath in files_to_process:
            nodes, episodes = await self.process_file(filepath, provider)
            all_nodes.extend(nodes)
            all_episodes.extend(episodes)

        logger.info(f"Total: {len(all_nodes)} nodes, {len(all_episodes)} episodes")

        if not all_nodes:
            logger.warning("No data to process!")
            return

        # Embed everything
        node_embeddings, episode_embeddings = await self.embed_all(all_nodes, all_episodes)

        # Cluster nodes
        all_nodes, cluster_info = self.cluster_nodes(all_nodes, node_embeddings)

        # Build FAISS index for episodes
        faiss_index = self.build_faiss_index(episode_embeddings)

        # Save everything
        self.save_all(
            all_nodes, all_episodes,
            node_embeddings, episode_embeddings,
            faiss_index, cluster_info,
        )

        # Final stats
        self.stats["total_time_seconds"] = time.time() - start_time

        print("\n" + "=" * 60)
        print("INGEST COMPLETE")
        print("=" * 60)
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """CLI entry point."""
    from dotenv import load_dotenv
    load_dotenv(override=True)  # Override system env vars with .env

    if len(sys.argv) < 2:
        print("Ingest Pipeline - Process Chat Exports")
        print("=" * 50)
        print("\nUsage:")
        print("  python ingest.py <path> [path2] [--provider=auto]")
        print("\nExamples:")
        print("  python ingest.py ./chat_exports/anthropic/conversations.json")
        print("  python ingest.py ./chat_exports/ --recursive")
        print("  python ingest.py ./export.json --provider=openai")
        return

    # Parse args
    paths = []
    provider = "auto"

    for arg in sys.argv[1:]:
        if arg.startswith("--provider="):
            provider = arg.split("=")[1]
        else:
            paths.append(Path(arg))

    # Run pipeline
    pipeline = IngestPipeline(
        output_dir=Path("./data"),
        embedding_batch_size=32,
        embedding_concurrency=8,
    )

    await pipeline.run(paths, provider=provider)


if __name__ == "__main__":
    asyncio.run(main())
