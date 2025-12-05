"""
Ingest Pipeline - Raw Chat Exports to Dual Memory Store (Unified/Incremental)

The main orchestrator that takes raw JSON exports and produces:
1. Memory Nodes: 1:1 Q/A pairs → clustered → NumPy vectors
2. Episodic Memories: Full conversations → FAISS index → LLM tagged

Pipeline (Incremental):
    Raw JSON → Parse → DEDUP CHECK → Embed ONLY NEW → Merge → Re-cluster → Store

Key Features:
- Dedup BEFORE embedding (saves API costs)
- Incremental merge with existing corpus
- Single unified output files (no timestamps)
- Full re-cluster on merged corpus

File Structure (unified):
    data/
    ├── corpus/
    │   ├── nodes.json           # Single unified nodes file
    │   ├── episodes.json        # Single unified episodes file
    │   └── dedup_index.json     # Content hashes for dedup
    ├── vectors/
    │   ├── nodes.npy            # Single embeddings array
    │   └── episodes.npy         # Single episode embeddings
    ├── indexes/
    │   ├── faiss.index          # Single FAISS index
    │   └── clusters.json        # Cluster assignments
    └── manifest.json            # Single manifest

Usage:
    python ingest.py ./chat_exports/anthropic/conversations.json
    python ingest.py ./chat_exports/ --recursive
    python ingest.py ./data --traces   # Ingest reasoning traces (Phase 5)

Version: 2.0.0 (cog_twin - Unified incremental ingestion)
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

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


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash for content-based dedup."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class IngestPipeline:
    """
    Main ingest pipeline for processing chat exports into memory stores.

    Handles both memory node extraction (for process memory) and
    episodic memory preservation (for context memory).

    Version 2.0: Unified incremental ingestion with dedup-before-embedding.
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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Unified directory structure
        self.corpus_dir = self.output_dir / "corpus"
        self.vectors_dir = self.output_dir / "vectors"
        self.index_dir = self.output_dir / "indexes"

        for d in [self.corpus_dir, self.vectors_dir, self.index_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Legacy directories (for backward compatibility during migration)
        self.nodes_dir = self.output_dir / "memory_nodes"
        self.episodes_dir = self.output_dir / "episodic_memories"

        # Components
        self.parser = ChatParserFactory()
        self.enricher = HeuristicEnricher()
        self.embedder = AsyncEmbedder(
            api_key=deepinfra_api_key,
            cache_dir=self.output_dir / "embedding_cache",
        )

        # Config
        self.embedding_batch_size = embedding_batch_size
        self.embedding_concurrency = embedding_concurrency

        # Load existing corpus if present (for incremental merge)
        self.existing_nodes: List[MemoryNode] = []
        self.existing_episodes: List[EpisodicMemory] = []
        self.existing_node_embeddings: Optional[np.ndarray] = None
        self.existing_episode_embeddings: Optional[np.ndarray] = None
        self.dedup_index: Set[str] = set()

        self._load_existing_corpus()

        # Stats
        self.stats = {
            "files_processed": 0,
            "conversations_parsed": 0,
            "memory_nodes_created": 0,
            "episodes_created": 0,
            "nodes_clustered": 0,
            "noise_nodes_filtered": 0,
            "duplicates_skipped": 0,
            "existing_nodes_loaded": len(self.existing_nodes),
            "existing_episodes_loaded": len(self.existing_episodes),
            "total_time_seconds": 0,
        }

    def _load_existing_corpus(self) -> None:
        """Load existing unified corpus if present."""
        # Check for unified structure first
        nodes_file = self.corpus_dir / "nodes.json"
        episodes_file = self.corpus_dir / "episodes.json"
        dedup_file = self.corpus_dir / "dedup_index.json"
        node_emb_file = self.vectors_dir / "nodes.npy"
        episode_emb_file = self.vectors_dir / "episodes.npy"

        # Load nodes
        if nodes_file.exists():
            try:
                with open(nodes_file) as f:
                    nodes_data = json.load(f)
                self.existing_nodes = [MemoryNode.from_dict(d) for d in nodes_data]
                logger.info(f"Loaded {len(self.existing_nodes)} existing nodes from unified corpus")
            except Exception as e:
                logger.warning(f"Failed to load existing nodes: {e}")

        # Load episodes
        if episodes_file.exists():
            try:
                with open(episodes_file) as f:
                    episodes_data = json.load(f)
                self.existing_episodes = [EpisodicMemory.from_dict(d) for d in episodes_data]
                logger.info(f"Loaded {len(self.existing_episodes)} existing episodes from unified corpus")
            except Exception as e:
                logger.warning(f"Failed to load existing episodes: {e}")

        # Load dedup index
        if dedup_file.exists():
            try:
                with open(dedup_file) as f:
                    data = json.load(f)
                self.dedup_index = set(data.get("ingested_ids", []))
                logger.info(f"Loaded dedup index with {len(self.dedup_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load dedup index: {e}")

        # Load embeddings
        if node_emb_file.exists():
            try:
                self.existing_node_embeddings = np.load(node_emb_file)
                logger.info(f"Loaded existing node embeddings: {self.existing_node_embeddings.shape}")
            except Exception as e:
                logger.warning(f"Failed to load node embeddings: {e}")

        if episode_emb_file.exists():
            try:
                self.existing_episode_embeddings = np.load(episode_emb_file)
                logger.info(f"Loaded existing episode embeddings: {self.existing_episode_embeddings.shape}")
            except Exception as e:
                logger.warning(f"Failed to load episode embeddings: {e}")

    def _filter_duplicates(self, nodes: List[MemoryNode]) -> List[MemoryNode]:
        """
        Filter out nodes that already exist in corpus. Check BEFORE embedding.

        Args:
            nodes: List of new nodes to check

        Returns:
            List of nodes that are NOT duplicates
        """
        new_nodes = []
        duplicates = 0

        for node in nodes:
            # Check by ID
            if node.id in self.dedup_index:
                duplicates += 1
                continue

            # Check by content hash
            content_hash = compute_content_hash(node.combined_content)
            if content_hash in self.dedup_index:
                duplicates += 1
                continue

            # Not a duplicate - add to new list and register
            new_nodes.append(node)
            self.dedup_index.add(node.id)
            self.dedup_index.add(content_hash)

        self.stats["duplicates_skipped"] += duplicates
        logger.info(f"Dedup: {len(nodes)} input → {len(new_nodes)} new ({duplicates} duplicates skipped)")
        return new_nodes

    def _filter_duplicate_episodes(self, episodes: List[EpisodicMemory]) -> List[EpisodicMemory]:
        """Filter out episodes that already exist in corpus."""
        new_episodes = []
        duplicates = 0

        for ep in episodes:
            if ep.id in self.dedup_index:
                duplicates += 1
                continue

            new_episodes.append(ep)
            self.dedup_index.add(ep.id)

        logger.info(f"Episode dedup: {len(episodes)} input → {len(new_episodes)} new ({duplicates} duplicates skipped)")
        return new_episodes

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

    def _save_unified(
        self,
        nodes: List[MemoryNode],
        episodes: List[EpisodicMemory],
        node_embeddings: np.ndarray,
        episode_embeddings: np.ndarray,
        faiss_index: Any,
        cluster_info: Dict[int, List[int]],
    ) -> None:
        """
        Save to unified files (no timestamps, single source of truth).

        Args:
            nodes: Memory nodes with cluster assignments
            episodes: Episodic memories
            node_embeddings: Node embedding matrix
            episode_embeddings: Episode embedding matrix
            faiss_index: FAISS index for episodes
            cluster_info: Cluster ID to node indices mapping
        """
        logger.info("Saving unified corpus...")

        # Save memory nodes (JSON) - single file
        nodes_file = self.corpus_dir / "nodes.json"
        nodes_data = [n.to_dict() for n in nodes]
        with open(nodes_file, "w") as f:
            json.dump(nodes_data, f, indent=2, default=str)
        logger.info(f"Saved {len(nodes)} nodes to {nodes_file}")

        # Save episodic memories (JSON) - single file
        episodes_file = self.corpus_dir / "episodes.json"
        episodes_data = [e.to_dict() for e in episodes]
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f, indent=2, default=str)
        logger.info(f"Saved {len(episodes)} episodes to {episodes_file}")

        # Save dedup index
        dedup_file = self.corpus_dir / "dedup_index.json"
        with open(dedup_file, "w") as f:
            json.dump({"ingested_ids": sorted(list(self.dedup_index))}, f, indent=2)
        logger.info(f"Saved dedup index with {len(self.dedup_index)} entries")

        # Save embeddings (NumPy) - single files
        node_emb_file = self.vectors_dir / "nodes.npy"
        np.save(node_emb_file, node_embeddings)
        logger.info(f"Saved node embeddings {node_embeddings.shape} to {node_emb_file}")

        episode_emb_file = self.vectors_dir / "episodes.npy"
        np.save(episode_emb_file, episode_embeddings)
        logger.info(f"Saved episode embeddings {episode_embeddings.shape} to {episode_emb_file}")

        # Save FAISS index - single file
        if faiss_index is not None and FAISS_AVAILABLE:
            faiss_file = self.index_dir / "faiss.index"
            faiss.write_index(faiss_index, str(faiss_file))
            logger.info(f"Saved FAISS index to {faiss_file}")

        # Save cluster info - single file
        cluster_file = self.index_dir / "clusters.json"
        cluster_data = {str(k): v for k, v in cluster_info.items()}
        with open(cluster_file, "w") as f:
            json.dump(cluster_data, f, indent=2)
        logger.info(f"Saved cluster info to {cluster_file}")

        # Save unified manifest - single file (overwrites)
        manifest = {
            "version": "2.0.0",
            "structure": "unified",
            "updated_at": datetime.now().isoformat(),
            "corpus": {
                "nodes_file": "corpus/nodes.json",
                "episodes_file": "corpus/episodes.json",
                "dedup_index_file": "corpus/dedup_index.json",
            },
            "vectors": {
                "nodes_file": "vectors/nodes.npy",
                "episodes_file": "vectors/episodes.npy",
            },
            "indexes": {
                "clusters_file": "indexes/clusters.json",
                "faiss_file": "indexes/faiss.index" if faiss_index else None,
            },
            "stats": {
                "total_nodes": len(nodes),
                "total_episodes": len(episodes),
                "n_clusters": len([k for k in cluster_info if k != -1]),
                "n_noise": len(cluster_info.get(-1, [])),
                "embedding_dim": node_embeddings.shape[1] if len(node_embeddings) > 0 else 0,
                "ingest_stats": self.stats,
            },
        }
        manifest_file = self.output_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved unified manifest to {manifest_file}")

    async def run(
        self,
        input_paths: List[Path],
        provider: str = "auto",
    ) -> None:
        """
        Run the incremental ingest pipeline.

        Flow:
        1. Parse new exports
        2. Dedup BEFORE embedding (saves API costs)
        3. Embed ONLY new nodes
        4. Merge with existing corpus
        5. Re-cluster entire corpus (HDBSCAN)
        6. Rebuild FAISS index
        7. Save unified corpus (overwrites, not appends)

        Args:
            input_paths: List of file or directory paths
            provider: Provider hint or "auto"
        """
        start_time = time.time()

        # Collect all files to process
        files_to_process = []
        for path in input_paths:
            path = Path(path)
            if path.is_file():
                files_to_process.append(path)
            elif path.is_dir():
                files_to_process.extend(path.glob("**/*.json"))

        logger.info(f"Found {len(files_to_process)} files to process")

        # 1. Parse all new exports
        new_nodes = []
        new_episodes = []

        for filepath in files_to_process:
            nodes, episodes = await self.process_file(filepath, provider)
            new_nodes.extend(nodes)
            new_episodes.extend(episodes)

        logger.info(f"Parsed: {len(new_nodes)} new nodes, {len(new_episodes)} new episodes")

        # 2. Dedup BEFORE embedding (saves API costs)
        new_nodes = self._filter_duplicates(new_nodes)
        new_episodes = self._filter_duplicate_episodes(new_episodes)

        if not new_nodes and not new_episodes:
            logger.info("No new content to ingest (all duplicates)")
            self.stats["total_time_seconds"] = time.time() - start_time
            print("\n" + "=" * 60)
            print("INGEST COMPLETE - No new content")
            print("=" * 60)
            print(f"  Duplicates skipped: {self.stats['duplicates_skipped']}")
            print("=" * 60)
            return

        # 3. Embed ONLY new nodes (not the entire corpus!)
        logger.info(f"Embedding {len(new_nodes)} new nodes and {len(new_episodes)} new episodes...")
        new_node_embeddings, new_episode_embeddings = await self.embed_all(new_nodes, new_episodes)

        # 4. Merge with existing corpus
        merged_nodes = self.existing_nodes + new_nodes
        merged_episodes = self.existing_episodes + new_episodes

        # Merge embeddings
        if self.existing_node_embeddings is not None and len(self.existing_node_embeddings) > 0:
            merged_node_embeddings = np.concatenate([self.existing_node_embeddings, new_node_embeddings])
        else:
            merged_node_embeddings = new_node_embeddings

        if self.existing_episode_embeddings is not None and len(self.existing_episode_embeddings) > 0:
            merged_episode_embeddings = np.concatenate([self.existing_episode_embeddings, new_episode_embeddings])
        else:
            merged_episode_embeddings = new_episode_embeddings

        logger.info(f"Merged corpus: {len(merged_nodes)} nodes, {len(merged_episodes)} episodes")

        # 5. Re-cluster entire merged corpus (HDBSCAN)
        # This is fast enough for 25K nodes (~2 seconds)
        merged_nodes, cluster_info = self.cluster_nodes(merged_nodes, merged_node_embeddings)

        # 6. Rebuild FAISS index for episodes
        faiss_index = self.build_faiss_index(merged_episode_embeddings)

        # 7. Save unified corpus (overwrites previous)
        self._save_unified(
            merged_nodes, merged_episodes,
            merged_node_embeddings, merged_episode_embeddings,
            faiss_index, cluster_info,
        )

        # Final stats
        self.stats["total_time_seconds"] = time.time() - start_time
        self.stats["final_node_count"] = len(merged_nodes)
        self.stats["final_episode_count"] = len(merged_episodes)
        self.stats["new_nodes_added"] = len(new_nodes)
        self.stats["new_episodes_added"] = len(new_episodes)

        print("\n" + "=" * 60)
        print("INGEST COMPLETE (Incremental)")
        print("=" * 60)
        print(f"  Existing nodes loaded: {self.stats['existing_nodes_loaded']}")
        print(f"  New nodes added: {len(new_nodes)}")
        print(f"  Duplicates skipped: {self.stats['duplicates_skipped']}")
        print(f"  Final node count: {len(merged_nodes)}")
        print(f"  Final episode count: {len(merged_episodes)}")
        print(f"  Clusters: {len([k for k in cluster_info if k != -1])}")
        print(f"  Time: {self.stats['total_time_seconds']:.1f}s")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5: Reasoning Trace Ingestion
# ═══════════════════════════════════════════════════════════════════════════

def ingest_reasoning_traces(traces_dir: Path, data_dir: Path, dedup_batch) -> dict:
    """
    Batch ingest reasoning traces as episodic memories.

    Phase 5: Makes traces searchable in FAISS and retrievable through
    the dual retriever. This is the batch path for traces that weren't
    live-streamed to memory_pipeline.

    Args:
        traces_dir: Path to reasoning_traces folder
        data_dir: Path to main data directory
        dedup_batch: DedupBatch context manager instance

    Returns:
        Stats dict with counts
    """
    stats = {
        "total_found": 0,
        "already_ingested": 0,
        "newly_ingested": 0,
        "failed": 0,
    }

    traces_dir = Path(traces_dir)
    data_dir = Path(data_dir)

    if not traces_dir.exists():
        logger.warning(f"Traces directory not found: {traces_dir}")
        return stats

    trace_files = list(traces_dir.glob("trace_*.json"))
    stats["total_found"] = len(trace_files)

    new_episodes = []

    for trace_file in trace_files:
        try:
            with open(trace_file) as f:
                trace_data = json.load(f)

            trace_id = trace_data.get("id", "")

            # Dedup check - check both raw ID and prefixed versions
            if dedup_batch.is_duplicate(trace_id) or dedup_batch.is_duplicate(f"trace_{trace_id}"):
                stats["already_ingested"] += 1
                continue

            # Also check by content hash (in case ID changed)
            content = trace_data.get("response", "")
            if content and dedup_batch.is_duplicate(f"trace_content_{trace_id}", content[:500]):
                stats["already_ingested"] += 1
                continue

            # Convert to EpisodicMemory format
            query = trace_data.get("query", "")
            response = trace_data.get("response", "")
            timestamp = trace_data.get("timestamp", datetime.now().isoformat())

            episode = {
                "id": f"ep_trace_{trace_id}",
                "title": f"Reasoning: {query[:50]}...",
                "summary": response[:500] if len(response) > 500 else response,
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response},
                ],
                "created_at": timestamp,
                "dominant_intent": "reasoning_trace",
                "metadata": {
                    "trace_id": trace_id,
                    "cognitive_phase": trace_data.get("cognitive_phase"),
                    "score": trace_data.get("score"),
                    "feedback_notes": trace_data.get("feedback_notes"),
                    "memories_retrieved": trace_data.get("memories_retrieved", []),
                    "memories_cited": trace_data.get("memories_cited", []),
                    "total_duration_ms": trace_data.get("total_duration_ms"),
                    "tokens_used": trace_data.get("tokens_used"),
                },
            }

            new_episodes.append(episode)
            dedup_batch.register(trace_id)
            dedup_batch.register(f"trace_{trace_id}")
            dedup_batch.register(f"ep_trace_{trace_id}")
            stats["newly_ingested"] += 1

        except Exception as e:
            logger.warning(f"Failed to ingest {trace_file}: {e}")
            stats["failed"] += 1

    # Save new episodes
    if new_episodes:
        output_file = data_dir / "episodic_memories" / f"trace_episodes_{int(time.time())}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(new_episodes, f, indent=2, default=str)
        logger.info(f"Saved {len(new_episodes)} trace episodes to {output_file}")

    return stats


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
        print("  python ingest.py <data_dir> --traces")
        print("\nExamples:")
        print("  python ingest.py ./chat_exports/anthropic/conversations.json")
        print("  python ingest.py ./chat_exports/ --recursive")
        print("  python ingest.py ./export.json --provider=openai")
        print("  python ingest.py ./data --traces  # Ingest reasoning traces")
        return

    # Check for trace ingestion mode
    if "--traces" in sys.argv:
        from dedup import DedupBatch

        # Find data dir (first non-flag argument)
        data_dir = None
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                data_dir = Path(arg)
                break

        if not data_dir:
            print("Error: Please specify data directory")
            print("Usage: python ingest.py <data_dir> --traces")
            return

        traces_dir = data_dir / "reasoning_traces"

        print(f"Ingesting reasoning traces from {traces_dir}...")
        with DedupBatch(data_dir) as dedup:
            stats = ingest_reasoning_traces(traces_dir, data_dir, dedup)

        print(f"\nTrace ingestion complete:")
        print(f"  Found: {stats['total_found']}")
        print(f"  Already ingested: {stats['already_ingested']}")
        print(f"  Newly ingested: {stats['newly_ingested']}")
        print(f"  Failed: {stats['failed']}")
        return

    # Parse args for chat ingestion
    paths = []
    provider = "auto"

    for arg in sys.argv[1:]:
        if arg.startswith("--provider="):
            provider = arg.split("=")[1]
        elif not arg.startswith("--"):
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
