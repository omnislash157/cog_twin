"""
dedup.py - Global deduplication for all ingestion paths.

Prevents duplicate memories regardless of source (chat exports, reasoning traces,
session outputs). Uses both ID-based and content-hash deduplication.

The learning loop needs dedup because:
- Live streaming: traces go to memory_pipeline immediately
- Batch ingest: traces can be re-imported as episodic memories
- Without dedup: same trace appears multiple times = noise

Usage:
    from dedup import DedupBatch, load_dedup_index

    # For single operations
    index = load_dedup_index(data_dir)
    is_dup, index = check_and_register(data_dir, item_id, content, index)

    # For batch operations
    with DedupBatch(data_dir) as dedup:
        for item in items:
            if not dedup.is_duplicate(item.id, item.content):
                process(item)
                dedup.register(item.id, item.content)

Version: 1.0.0 (Phase 5)
"""
import json
import hashlib
from pathlib import Path
from typing import Set, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

DEDUP_INDEX_FILE = "dedup_index.json"


def load_dedup_index(data_dir: Path) -> Set[str]:
    """
    Load the global deduplication index.

    Returns set of content hashes / IDs that have been ingested.
    """
    index_file = Path(data_dir) / DEDUP_INDEX_FILE

    if index_file.exists():
        try:
            with open(index_file) as f:
                data = json.load(f)
                return set(data.get("ingested_ids", []))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load dedup index: {e}")
            return set()

    return set()


def save_dedup_index(data_dir: Path, index: Set[str]):
    """Save the deduplication index."""
    index_file = Path(data_dir) / DEDUP_INDEX_FILE

    try:
        with open(index_file, "w") as f:
            json.dump({"ingested_ids": sorted(list(index))}, f, indent=2)

        logger.info(f"Saved dedup index with {len(index)} entries")
    except IOError as e:
        logger.error(f"Failed to save dedup index: {e}")


def compute_content_hash(content: str) -> str:
    """
    Compute a hash for content-based deduplication.

    Uses first 16 chars of SHA256 - collision probability is negligible
    for our use case (< 100K items).
    """
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def check_and_register(
    data_dir: Path,
    item_id: str,
    content: Optional[str] = None,
    index: Optional[Set[str]] = None
) -> Tuple[bool, Set[str]]:
    """
    Check if item is duplicate, register if not.

    Args:
        data_dir: Data directory path
        item_id: Unique ID of the item
        content: Optional content for hash-based dedup
        index: Optional pre-loaded index (avoids repeated file reads)

    Returns:
        (is_duplicate, updated_index)
    """
    if index is None:
        index = load_dedup_index(data_dir)

    # Check by ID
    if item_id in index:
        return True, index

    # Check by content hash if provided
    if content:
        content_hash = compute_content_hash(content)
        if content_hash in index:
            return True, index
        index.add(content_hash)

    index.add(item_id)
    return False, index


class DedupBatch:
    """
    Context manager for batched dedup operations.

    Loads index once at start, saves once at end - much more efficient
    for batch operations that process many items.

    Usage:
        with DedupBatch(data_dir) as dedup:
            for item in items:
                if not dedup.is_duplicate(item.id):
                    process(item)
                    dedup.register(item.id)
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.index: Set[str] = set()
        self.dirty = False
        self._initial_size = 0

    def __enter__(self):
        self.index = load_dedup_index(self.data_dir)
        self._initial_size = len(self.index)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dirty:
            save_dedup_index(self.data_dir, self.index)
        return False  # Don't suppress exceptions

    def is_duplicate(self, item_id: str, content: Optional[str] = None) -> bool:
        """
        Check if item is duplicate.

        Args:
            item_id: Unique identifier
            content: Optional content for hash-based dedup

        Returns:
            True if duplicate, False if new
        """
        if item_id in self.index:
            return True
        if content:
            content_hash = compute_content_hash(content)
            if content_hash in self.index:
                return True
        return False

    def register(self, item_id: str, content: Optional[str] = None):
        """
        Register item as ingested.

        Args:
            item_id: Unique identifier to register
            content: Optional content for hash-based dedup
        """
        self.index.add(item_id)
        if content:
            self.index.add(compute_content_hash(content))
        self.dirty = True

    def stats(self) -> Dict[str, int]:
        """Return index statistics."""
        return {
            "total_registered": len(self.index),
            "new_this_batch": len(self.index) - self._initial_size,
        }


def build_dedup_index_from_existing(data_dir: Path) -> Set[str]:
    """
    Build dedup index from existing data files.

    Scans all memory nodes, episodic memories, and reasoning traces
    to build a complete dedup index. Use for recovery or first-time setup.

    Args:
        data_dir: Data directory to scan

    Returns:
        Set of all existing IDs and content hashes
    """
    index: Set[str] = set()
    data_dir = Path(data_dir)

    # Scan memory nodes
    nodes_dir = data_dir / "memory_nodes"
    if nodes_dir.exists():
        for nodes_file in nodes_dir.glob("*.json"):
            try:
                with open(nodes_file) as f:
                    nodes = json.load(f)
                    if isinstance(nodes, list):
                        for node in nodes:
                            if "id" in node:
                                index.add(node["id"])
                            # Also hash content for content-based dedup
                            content = node.get("assistant_content", "") or node.get("content", "")
                            if content:
                                index.add(compute_content_hash(content[:1000]))  # First 1K chars
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to scan {nodes_file}: {e}")

    # Scan episodic memories
    episodes_dir = data_dir / "episodic_memories"
    if episodes_dir.exists():
        for ep_file in episodes_dir.glob("*.json"):
            try:
                with open(ep_file) as f:
                    episodes = json.load(f)
                    if isinstance(episodes, list):
                        for ep in episodes:
                            if "id" in ep:
                                index.add(ep["id"])
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to scan {ep_file}: {e}")

    # Scan reasoning traces
    traces_dir = data_dir / "reasoning_traces"
    if traces_dir.exists():
        for trace_file in traces_dir.glob("trace_*.json"):
            try:
                with open(trace_file) as f:
                    trace = json.load(f)
                    if "id" in trace:
                        index.add(trace["id"])
                        # Also add prefixed version for when traces become memories
                        index.add(f"trace_{trace['id']}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to scan {trace_file}: {e}")

    logger.info(f"Built dedup index with {len(index)} entries from existing data")
    return index


if __name__ == "__main__":
    """CLI for dedup operations."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python dedup.py <data_dir> --build    Build index from existing data")
        print("  python dedup.py <data_dir> --stats    Show index statistics")
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    if "--build" in sys.argv:
        print(f"Building dedup index from {data_dir}...")
        index = build_dedup_index_from_existing(data_dir)
        save_dedup_index(data_dir, index)
        print(f"Done. Index contains {len(index)} entries.")

    elif "--stats" in sys.argv:
        index = load_dedup_index(data_dir)
        print(f"Dedup index: {len(index)} entries")

        # Sample some entries
        sample = list(index)[:10]
        print(f"Sample entries: {sample}")

    else:
        print("Unknown command. Use --build or --stats")
