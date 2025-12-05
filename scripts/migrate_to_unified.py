"""
Migration Script: Consolidate scattered data into unified corpus structure.

Migrates from:
    data/
    ├── memory_nodes/nodes_*.json, session_nodes_*.json, session_outputs_*.json
    ├── episodic_memories/episodes_*.json
    ├── vectors/node_embeddings_*.npy, episode_embeddings_*.npy, session_embeddings_*.npy
    ├── indexes/clusters_*.json, *.faiss
    └── manifest_*.json (multiple, orphaned)

To:
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
    ├── manifest.json            # Single manifest
    └── archive/                 # Old files moved here

Usage:
    python scripts/migrate_to_unified.py ./data
    python scripts/migrate_to_unified.py ./data --dry-run
    python scripts/migrate_to_unified.py ./data --skip-archive

Version: 1.0.0
"""

import json
import logging
import shutil
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

import numpy as np

# Optional FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Index will not be rebuilt.")

# Optional HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: HDBSCAN not available. Clustering will be skipped.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash for content-based dedup."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_json_safe(path: Path) -> Optional[Any]:
    """Load JSON file with error handling."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def load_all_nodes(data_dir: Path) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Load all node files and their embeddings.

    Returns:
        (nodes_list, {filename: embeddings_array})
    """
    nodes_dir = data_dir / "memory_nodes"
    vectors_dir = data_dir / "vectors"

    all_nodes = []
    embeddings_map = {}

    if not nodes_dir.exists():
        logger.warning(f"Nodes directory not found: {nodes_dir}")
        return all_nodes, embeddings_map

    # Load main node files (nodes_*.json)
    for nodes_file in sorted(nodes_dir.glob("nodes_*.json")):
        logger.info(f"Loading nodes from {nodes_file.name}")
        nodes = load_json_safe(nodes_file)
        if nodes:
            # Extract timestamp to find matching embeddings
            timestamp = nodes_file.stem.replace("nodes_", "")
            emb_file = vectors_dir / f"node_embeddings_{timestamp}.npy"

            if emb_file.exists():
                embeddings = np.load(emb_file)
                logger.info(f"  Loaded {len(nodes)} nodes + {embeddings.shape} embeddings")

                # Tag nodes with their embedding index range
                start_idx = sum(len(n) for n in [all_nodes])
                for i, node in enumerate(nodes):
                    node['_emb_source'] = str(emb_file)
                    node['_emb_idx'] = i

                embeddings_map[str(emb_file)] = embeddings
            else:
                logger.warning(f"  No embeddings found for {nodes_file.name}")

            all_nodes.extend(nodes)

    # Load session node files (session_nodes_*.json)
    for session_file in sorted(nodes_dir.glob("session_nodes_*.json")):
        logger.info(f"Loading session nodes from {session_file.name}")
        nodes = load_json_safe(session_file)
        if nodes:
            timestamp = session_file.stem.replace("session_nodes_", "")
            emb_file = vectors_dir / f"session_embeddings_{timestamp}.npy"

            if emb_file.exists():
                embeddings = np.load(emb_file)
                logger.info(f"  Loaded {len(nodes)} session nodes + {embeddings.shape} embeddings")

                for i, node in enumerate(nodes):
                    node['_emb_source'] = str(emb_file)
                    node['_emb_idx'] = i

                embeddings_map[str(emb_file)] = embeddings
            else:
                logger.warning(f"  No embeddings found for {session_file.name}")

            all_nodes.extend(nodes)

    # Load session output files (session_outputs_*.json) - these are CognitiveOutputs
    for output_file in sorted(nodes_dir.glob("session_outputs_*.json")):
        logger.info(f"Loading session outputs from {output_file.name}")
        outputs = load_json_safe(output_file)
        if outputs:
            # Convert CognitiveOutput to MemoryNode format
            for output in outputs:
                node = convert_output_to_node(output)
                if node:
                    node['_emb_source'] = None  # Will need embedding
                    node['_emb_idx'] = None
                    all_nodes.append(node)
            logger.info(f"  Converted {len(outputs)} outputs to nodes")

    return all_nodes, embeddings_map


def convert_output_to_node(output: Dict) -> Optional[Dict]:
    """Convert a CognitiveOutput to MemoryNode format."""
    try:
        content = output.get('content', '')
        if not content:
            return None

        return {
            'id': output.get('id', f"converted_{compute_content_hash(content)}"),
            'conversation_id': output.get('conversation_id', 'session'),
            'created_at': output.get('timestamp', datetime.now().isoformat()),
            'human_content': '',  # Session outputs are assistant-only
            'assistant_content': content,
            'tags': {
                'thought_type': output.get('thought_type', 'unknown'),
                'source': 'session_output',
                'original_source_ids': output.get('source_memory_ids', []),
            },
            'metadata': {
                'confidence': output.get('confidence', 0.0),
                'converted_from': 'cognitive_output',
            }
        }
    except Exception as e:
        logger.warning(f"Failed to convert output: {e}")
        return None


def load_all_episodes(data_dir: Path) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """Load all episode files and their embeddings."""
    episodes_dir = data_dir / "episodic_memories"
    vectors_dir = data_dir / "vectors"

    all_episodes = []
    embeddings_map = {}

    if not episodes_dir.exists():
        logger.warning(f"Episodes directory not found: {episodes_dir}")
        return all_episodes, embeddings_map

    for ep_file in sorted(episodes_dir.glob("episodes_*.json")):
        logger.info(f"Loading episodes from {ep_file.name}")
        episodes = load_json_safe(ep_file)
        if episodes:
            timestamp = ep_file.stem.replace("episodes_", "")
            emb_file = vectors_dir / f"episode_embeddings_{timestamp}.npy"

            if emb_file.exists():
                embeddings = np.load(emb_file)
                logger.info(f"  Loaded {len(episodes)} episodes + {embeddings.shape} embeddings")

                for i, ep in enumerate(episodes):
                    ep['_emb_source'] = str(emb_file)
                    ep['_emb_idx'] = i

                embeddings_map[str(emb_file)] = embeddings
            else:
                logger.warning(f"  No embeddings found for {ep_file.name}")

            all_episodes.extend(episodes)

    # Also load trace episodes if present
    for trace_ep_file in sorted(episodes_dir.glob("trace_episodes_*.json")):
        logger.info(f"Loading trace episodes from {trace_ep_file.name}")
        episodes = load_json_safe(trace_ep_file)
        if episodes:
            for ep in episodes:
                ep['_emb_source'] = None  # Will need embedding
                ep['_emb_idx'] = None
            all_episodes.extend(episodes)
            logger.info(f"  Loaded {len(episodes)} trace episodes (need embedding)")

    return all_episodes, embeddings_map


def deduplicate_nodes(nodes: List[Dict]) -> Tuple[List[Dict], Set[str], int]:
    """
    Deduplicate nodes by ID and content hash.

    Returns:
        (deduplicated_nodes, dedup_index, duplicates_removed)
    """
    seen_ids: Set[str] = set()
    seen_hashes: Set[str] = set()
    dedup_index: Set[str] = set()
    unique_nodes = []
    duplicates = 0

    for node in nodes:
        node_id = node.get('id', '')

        # Check by ID
        if node_id and node_id in seen_ids:
            duplicates += 1
            continue

        # Check by content hash
        content = node.get('assistant_content', '') or node.get('content', '')
        if content:
            content_hash = compute_content_hash(content)
            if content_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(content_hash)
            dedup_index.add(content_hash)

        if node_id:
            seen_ids.add(node_id)
            dedup_index.add(node_id)

        unique_nodes.append(node)

    return unique_nodes, dedup_index, duplicates


def deduplicate_episodes(episodes: List[Dict]) -> Tuple[List[Dict], int]:
    """Deduplicate episodes by ID."""
    seen_ids: Set[str] = set()
    unique_episodes = []
    duplicates = 0

    for ep in episodes:
        ep_id = ep.get('id', '')
        if ep_id and ep_id in seen_ids:
            duplicates += 1
            continue

        if ep_id:
            seen_ids.add(ep_id)
        unique_episodes.append(ep)

    return unique_episodes, duplicates


def merge_embeddings(
    items: List[Dict],
    embeddings_map: Dict[str, np.ndarray],
    embedding_dim: int = 1024,
) -> np.ndarray:
    """
    Merge embeddings for deduplicated items.

    Items without embeddings get zero vectors (to be filled by incremental ingest).
    """
    merged = []
    needs_embedding = 0

    for item in items:
        emb_source = item.get('_emb_source')
        emb_idx = item.get('_emb_idx')

        if emb_source and emb_idx is not None and emb_source in embeddings_map:
            emb_array = embeddings_map[emb_source]
            if emb_idx < len(emb_array):
                merged.append(emb_array[emb_idx])
                continue

        # No embedding found - use zero vector as placeholder
        merged.append(np.zeros(embedding_dim))
        needs_embedding += 1

    if needs_embedding > 0:
        logger.warning(f"  {needs_embedding} items have no embeddings (will need re-embedding)")

    return np.array(merged, dtype=np.float32)


def run_hdbscan_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]:
    """
    Run HDBSCAN clustering on embeddings.

    Returns:
        (labels, probabilities, cluster_to_indices)
    """
    if not HDBSCAN_AVAILABLE:
        logger.warning("HDBSCAN not available, skipping clustering")
        labels = np.full(len(embeddings), -1)
        probs = np.zeros(len(embeddings))
        return labels, probs, {-1: list(range(len(embeddings)))}

    logger.info(f"Running HDBSCAN on {len(embeddings)} vectors...")

    # Normalize for clustering
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
    )

    labels = clusterer.fit_predict(normalized)
    probabilities = clusterer.probabilities_

    # Build cluster index
    cluster_to_indices: Dict[int, List[int]] = {}
    for i, label in enumerate(labels):
        label = int(label)
        if label not in cluster_to_indices:
            cluster_to_indices[label] = []
        cluster_to_indices[label].append(i)

    n_clusters = len([k for k in cluster_to_indices if k != -1])
    n_noise = len(cluster_to_indices.get(-1, []))
    logger.info(f"  Found {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(embeddings):.1f}%)")

    return labels, probabilities, cluster_to_indices


def build_faiss_index(embeddings: np.ndarray) -> Optional[Any]:
    """Build FAISS index from embeddings."""
    if not FAISS_AVAILABLE:
        return None

    logger.info(f"Building FAISS index for {len(embeddings)} vectors...")

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = (embeddings / (norms + 1e-8)).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)

    logger.info(f"  FAISS index built: {index.ntotal} vectors")
    return index


def archive_old_files(data_dir: Path, dry_run: bool = False):
    """Move old timestamped files to archive."""
    archive_dir = data_dir / "archive"

    if not dry_run:
        archive_dir.mkdir(exist_ok=True)

    files_to_archive = []

    # Old manifests
    for f in data_dir.glob("manifest_*.json"):
        files_to_archive.append(f)

    # Old node files
    nodes_dir = data_dir / "memory_nodes"
    if nodes_dir.exists():
        for f in nodes_dir.glob("*.json"):
            files_to_archive.append(f)

    # Old episode files
    episodes_dir = data_dir / "episodic_memories"
    if episodes_dir.exists():
        for f in episodes_dir.glob("*.json"):
            files_to_archive.append(f)

    # Old vector files
    vectors_dir = data_dir / "vectors"
    if vectors_dir.exists():
        for f in vectors_dir.glob("*.npy"):
            files_to_archive.append(f)

    # Old index files
    indexes_dir = data_dir / "indexes"
    if indexes_dir.exists():
        for f in indexes_dir.glob("clusters_*.json"):
            files_to_archive.append(f)
        for f in indexes_dir.glob("*.faiss"):
            files_to_archive.append(f)

    logger.info(f"Archiving {len(files_to_archive)} old files...")

    for f in files_to_archive:
        rel_path = f.relative_to(data_dir)
        archive_path = archive_dir / rel_path

        if dry_run:
            logger.info(f"  [DRY RUN] Would move: {rel_path}")
        else:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(archive_path))
            logger.info(f"  Archived: {rel_path}")


def save_unified_corpus(
    data_dir: Path,
    nodes: List[Dict],
    episodes: List[Dict],
    node_embeddings: np.ndarray,
    episode_embeddings: np.ndarray,
    dedup_index: Set[str],
    cluster_info: Dict[int, List[int]],
    node_labels: np.ndarray,
    node_probs: np.ndarray,
    faiss_index: Optional[Any],
    dry_run: bool = False,
):
    """Save the unified corpus to new structure."""
    corpus_dir = data_dir / "corpus"
    vectors_dir = data_dir / "vectors"
    indexes_dir = data_dir / "indexes"

    if dry_run:
        logger.info("[DRY RUN] Would save unified corpus:")
        logger.info(f"  - corpus/nodes.json: {len(nodes)} nodes")
        logger.info(f"  - corpus/episodes.json: {len(episodes)} episodes")
        logger.info(f"  - vectors/nodes.npy: {node_embeddings.shape}")
        logger.info(f"  - vectors/episodes.npy: {episode_embeddings.shape}")
        logger.info(f"  - corpus/dedup_index.json: {len(dedup_index)} entries")
        return

    # Create directories
    corpus_dir.mkdir(exist_ok=True)
    vectors_dir.mkdir(exist_ok=True)
    indexes_dir.mkdir(exist_ok=True)

    # Remove internal tracking fields and add cluster info
    clean_nodes = []
    for i, node in enumerate(nodes):
        clean = {k: v for k, v in node.items() if not k.startswith('_')}
        clean['cluster_id'] = int(node_labels[i])
        clean['cluster_confidence'] = float(node_probs[i])
        clean_nodes.append(clean)

    clean_episodes = []
    for ep in episodes:
        clean = {k: v for k, v in ep.items() if not k.startswith('_')}
        clean_episodes.append(clean)

    # Save nodes
    nodes_file = corpus_dir / "nodes.json"
    with open(nodes_file, 'w') as f:
        json.dump(clean_nodes, f, indent=2, default=str)
    logger.info(f"Saved {len(clean_nodes)} nodes to {nodes_file}")

    # Save episodes
    episodes_file = corpus_dir / "episodes.json"
    with open(episodes_file, 'w') as f:
        json.dump(clean_episodes, f, indent=2, default=str)
    logger.info(f"Saved {len(clean_episodes)} episodes to {episodes_file}")

    # Save dedup index
    dedup_file = corpus_dir / "dedup_index.json"
    with open(dedup_file, 'w') as f:
        json.dump({"ingested_ids": sorted(list(dedup_index))}, f, indent=2)
    logger.info(f"Saved dedup index with {len(dedup_index)} entries")

    # Save embeddings
    nodes_emb_file = vectors_dir / "nodes.npy"
    np.save(nodes_emb_file, node_embeddings)
    logger.info(f"Saved node embeddings {node_embeddings.shape} to {nodes_emb_file}")

    episodes_emb_file = vectors_dir / "episodes.npy"
    np.save(episodes_emb_file, episode_embeddings)
    logger.info(f"Saved episode embeddings {episode_embeddings.shape} to {episodes_emb_file}")

    # Save cluster info
    clusters_file = indexes_dir / "clusters.json"
    cluster_data = {str(k): v for k, v in cluster_info.items()}
    with open(clusters_file, 'w') as f:
        json.dump(cluster_data, f, indent=2)
    logger.info(f"Saved cluster info to {clusters_file}")

    # Save FAISS index
    if faiss_index is not None and FAISS_AVAILABLE:
        faiss_file = indexes_dir / "faiss.index"
        faiss.write_index(faiss_index, str(faiss_file))
        logger.info(f"Saved FAISS index to {faiss_file}")

    # Save unified manifest
    manifest = {
        "version": "2.0.0",
        "structure": "unified",
        "created_at": datetime.now().isoformat(),
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
        },
    }

    manifest_file = data_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved unified manifest to {manifest_file}")


def migrate(data_dir: Path, dry_run: bool = False, skip_archive: bool = False):
    """Run the full migration."""
    logger.info("=" * 60)
    logger.info("CogTwin Migration: Scattered Files → Unified Corpus")
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN MODE - no files will be modified]")

    # Load all nodes
    logger.info("\n[1/7] Loading all node files...")
    nodes, node_emb_map = load_all_nodes(data_dir)
    logger.info(f"  Total nodes loaded: {len(nodes)}")

    # Load all episodes
    logger.info("\n[2/7] Loading all episode files...")
    episodes, episode_emb_map = load_all_episodes(data_dir)
    logger.info(f"  Total episodes loaded: {len(episodes)}")

    # Deduplicate
    logger.info("\n[3/7] Deduplicating nodes...")
    nodes, dedup_index, node_dups = deduplicate_nodes(nodes)
    logger.info(f"  Unique nodes: {len(nodes)} ({node_dups} duplicates removed)")

    logger.info("\n[4/7] Deduplicating episodes...")
    episodes, ep_dups = deduplicate_episodes(episodes)
    logger.info(f"  Unique episodes: {len(episodes)} ({ep_dups} duplicates removed)")

    # Merge embeddings
    logger.info("\n[5/7] Merging embeddings...")
    node_embeddings = merge_embeddings(nodes, node_emb_map)
    episode_embeddings = merge_embeddings(episodes, episode_emb_map)
    logger.info(f"  Node embeddings: {node_embeddings.shape}")
    logger.info(f"  Episode embeddings: {episode_embeddings.shape}")

    # Cluster
    logger.info("\n[6/7] Running HDBSCAN clustering...")
    node_labels, node_probs, cluster_info = run_hdbscan_clustering(node_embeddings)

    # Build FAISS
    faiss_index = build_faiss_index(episode_embeddings)

    # Archive old files
    if not skip_archive:
        logger.info("\n[7/7] Archiving old files...")
        archive_old_files(data_dir, dry_run=dry_run)
    else:
        logger.info("\n[7/7] Skipping archive (--skip-archive)")

    # Save unified corpus
    logger.info("\nSaving unified corpus...")
    save_unified_corpus(
        data_dir,
        nodes,
        episodes,
        node_embeddings,
        episode_embeddings,
        dedup_index,
        cluster_info,
        node_labels,
        node_probs,
        faiss_index,
        dry_run=dry_run,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Nodes: {len(nodes)}")
    logger.info(f"  Episodes: {len(episodes)}")
    logger.info(f"  Clusters: {len([k for k in cluster_info if k != -1])}")
    logger.info(f"  Dedup index: {len(dedup_index)} entries")
    if dry_run:
        logger.info("\n[DRY RUN] No files were modified. Run without --dry-run to apply changes.")
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python scripts/migrate_to_unified.py <data_dir> [--dry-run] [--skip-archive]")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv
    skip_archive = "--skip-archive" in sys.argv

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    migrate(data_dir, dry_run=dry_run, skip_archive=skip_archive)


if __name__ == "__main__":
    main()
