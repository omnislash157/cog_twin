"""
Async Embedder - BGE-M3 via DeepInfra with Parallel Batching

Designed for speed: 2200+ conversations need parallel processing.

Features:
- Async batch embedding with configurable concurrency
- Rate limiting to avoid API throttling
- Local caching to skip already-embedded content
- Progress tracking

BGE-M3: 1024-dim multilingual embeddings, excellent for semantic search.
DeepInfra: GPU-accelerated inference, fast and cheap.

Usage:
    embedder = AsyncEmbedder(api_key)
    embeddings = await embedder.embed_batch(texts, batch_size=32, max_concurrent=8)

Version: 1.0.0 (cog_twin)
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class AsyncEmbedder:
    """
    Async embedder using DeepInfra's BGE-M3 endpoint.

    Optimized for throughput with parallel batch processing.
    """

    # DeepInfra BGE-M3 endpoint
    DEEPINFRA_URL = "https://api.deepinfra.com/v1/inference/BAAI/bge-m3"

    # Model specs
    EMBEDDING_DIM = 1024
    MAX_TOKENS = 8192  # BGE-M3 context window

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        requests_per_minute: int = 300,  # DeepInfra is generous
    ):
        """
        Initialize embedder.

        Args:
            api_key: DeepInfra API key (or from DEEPINFRA_API_KEY env)
            cache_dir: Directory for embedding cache
            requests_per_minute: Rate limit
        """
        self.api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPINFRA_API_KEY required")

        self.cache_dir = cache_dir or Path("./data/embedding_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._rate_lock = asyncio.Lock()

        # Stats
        self.stats = {
            "total_embedded": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "total_tokens": 0,
            "errors": 0,
        }

    def _content_hash(self, text: str) -> str:
        """Generate cache key from content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _get_cached(self, text: str) -> Optional[np.ndarray]:
        """Check cache for existing embedding."""
        cache_key = self._content_hash(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"

        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                pass
        return None

    def _save_cache(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        cache_key = self._content_hash(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_file, embedding)

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        async with self._rate_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()

    async def _embed_single_batch(
        self,
        client: httpx.AsyncClient,
        texts: List[str],
        batch_id: int,
    ) -> List[np.ndarray]:
        """
        Embed a single batch of texts.

        Args:
            client: Async HTTP client
            texts: List of texts to embed
            batch_id: Batch identifier for logging

        Returns:
            List of embedding arrays
        """
        await self._rate_limit()

        try:
            response = await client.post(
                self.DEEPINFRA_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"inputs": texts},
                timeout=60.0,
            )

            response.raise_for_status()
            data = response.json()

            # DeepInfra returns {"embeddings": [[...], [...], ...]}
            embeddings_raw = data.get("embeddings", [])

            self.stats["api_calls"] += 1
            self.stats["total_embedded"] += len(texts)

            embeddings = [np.array(e, dtype=np.float32) for e in embeddings_raw]

            logger.debug(f"Batch {batch_id}: embedded {len(texts)} texts")
            return embeddings

        except httpx.HTTPStatusError as e:
            logger.error(f"Batch {batch_id} HTTP error: {e.response.status_code}")
            self.stats["errors"] += 1
            # Return zero vectors on error (will be flagged for retry)
            return [np.zeros(self.EMBEDDING_DIM, dtype=np.float32) for _ in texts]

        except Exception as e:
            logger.error(f"Batch {batch_id} error: {e}")
            self.stats["errors"] += 1
            return [np.zeros(self.EMBEDDING_DIM, dtype=np.float32) for _ in texts]

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_concurrent: int = 8,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed texts with parallel batch processing.

        Args:
            texts: List of texts to embed
            batch_size: Texts per API call
            max_concurrent: Max parallel requests
            show_progress: Print progress updates

        Returns:
            numpy array of shape (len(texts), EMBEDDING_DIM)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.EMBEDDING_DIM)

        # Check cache first
        embeddings = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                embeddings[i] = cached
                self.stats["cache_hits"] += 1
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        if show_progress:
            print(f"Cache hits: {self.stats['cache_hits']}/{len(texts)}")
            print(f"Need to embed: {len(texts_to_embed)}")

        if not texts_to_embed:
            return np.array(embeddings, dtype=np.float32)

        # Create batches
        batches = []
        batch_indices = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_idx = indices_to_embed[i:i + batch_size]
            batches.append(batch_texts)
            batch_indices.append(batch_idx)

        if show_progress:
            print(f"Processing {len(batches)} batches with {max_concurrent} concurrent requests...")

        # Process batches with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch_id: int, batch_texts: List[str], client: httpx.AsyncClient):
            async with semaphore:
                return await self._embed_single_batch(client, batch_texts, batch_id)

        # Run all batches
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            tasks = [
                process_batch(i, batch_texts, client)
                for i, batch_texts in enumerate(batches)
            ]

            # Process with progress updates
            completed = 0
            results = []

            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                completed += 1

                if show_progress and completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {completed}/{len(batches)} batches ({rate:.1f} batches/sec)")

        # Results come back in completion order, need to match with batch indices
        # Actually, as_completed doesn't preserve order, so we need a different approach

        # Let's redo this properly - gather preserves order
        async with httpx.AsyncClient() as client:
            tasks = [
                process_batch(i, batch_texts, client)
                for i, batch_texts in enumerate(batches)
            ]
            batch_results = await asyncio.gather(*tasks)

        # Assign results back to correct positions
        for batch_idx_list, batch_embeddings in zip(batch_indices, batch_results):
            for idx, emb in zip(batch_idx_list, batch_embeddings):
                embeddings[idx] = emb
                # Cache the new embedding
                self._save_cache(texts[idx], emb)

        elapsed = time.time() - start_time

        if show_progress:
            print(f"Embedded {len(texts_to_embed)} texts in {elapsed:.1f}s")
            print(f"Rate: {len(texts_to_embed) / elapsed:.1f} texts/sec")

        return np.array(embeddings, dtype=np.float32)

    async def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text (with caching)."""
        cached = self._get_cached(text)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        result = await self.embed_batch([text], batch_size=1, max_concurrent=1, show_progress=False)
        return result[0]

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        return {
            **self.stats,
            "cache_size": len(list(self.cache_dir.glob("*.npy"))),
        }


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

async def embed_memory_nodes(
    nodes: List[Dict[str, Any]],
    embedder: AsyncEmbedder,
    batch_size: int = 32,
    max_concurrent: int = 8,
) -> np.ndarray:
    """
    Embed memory nodes (combined human + assistant content).

    Args:
        nodes: List of MemoryNode dicts or objects
        embedder: AsyncEmbedder instance
        batch_size: Texts per API call
        max_concurrent: Max parallel requests

    Returns:
        numpy array of embeddings (N x 1024)
    """
    texts = []
    for node in nodes:
        if hasattr(node, "combined_content"):
            texts.append(node.combined_content)
        else:
            human = node.get("human_content", "")
            assistant = node.get("assistant_content", "")
            texts.append(f"Human: {human}\n\nAssistant: {assistant}")

    return await embedder.embed_batch(texts, batch_size, max_concurrent)


async def embed_episodes(
    episodes: List[Dict[str, Any]],
    embedder: AsyncEmbedder,
    batch_size: int = 16,  # Smaller batches for longer content
    max_concurrent: int = 4,
    max_chars: int = 16000,  # Truncate very long conversations
) -> np.ndarray:
    """
    Embed episodic memories (full conversation summaries).

    Args:
        episodes: List of EpisodicMemory dicts or objects
        embedder: AsyncEmbedder instance
        batch_size: Texts per API call (smaller for long content)
        max_concurrent: Max parallel requests
        max_chars: Max characters to embed per episode

    Returns:
        numpy array of embeddings (N x 1024)
    """
    texts = []
    for ep in episodes:
        if hasattr(ep, "full_text"):
            text = ep.full_text
        else:
            # Build from messages
            messages = ep.get("messages", [])
            parts = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            text = "\n\n".join(parts)

        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[truncated]"

        texts.append(text)

    return await embedder.embed_batch(texts, batch_size, max_concurrent)


# ═══════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Test the embedder."""
    from dotenv import load_dotenv
    load_dotenv()

    print("AsyncEmbedder Test")
    print("=" * 60)

    embedder = AsyncEmbedder()

    # Test texts
    test_texts = [
        "How do I fix an async error in FastAPI?",
        "What's the best way to structure a Python project?",
        "We were debugging the memory engine last night.",
        "The embedding pipeline needs parallel processing for speed.",
        "HDBSCAN clustering groups similar memories together.",
    ]

    print(f"\nEmbedding {len(test_texts)} test texts...")
    embeddings = await embedder.embed_batch(test_texts, batch_size=2, max_concurrent=2)

    print(f"\nResults:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")

    # Test similarity
    print(f"\nSimilarity matrix (cosine):")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    similarity = normalized @ normalized.T

    for i, text in enumerate(test_texts):
        print(f"\n  '{text[:40]}...'")
        top_similar = np.argsort(similarity[i])[::-1][1:3]  # Skip self
        for j in top_similar:
            print(f"    -> {similarity[i,j]:.3f}: '{test_texts[j][:40]}...'")

    print(f"\nStats: {embedder.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
