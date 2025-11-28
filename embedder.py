"""
Async Embedder - Multi-Provider BGE-M3 Embedding Pipeline

Supports multiple backends:
- DeepInfra API (rate-limited, good for small scale)
- TEI (Text Embeddings Inference) - self-hosted GPU (RunPod/Modal)
- Cloudflare Workers AI (fallback)

Designed for scale: power users with 3+ years of LLM conversations.

Features:
- Provider abstraction (swap backends without code changes)
- Async batch embedding with configurable concurrency
- Local caching to skip already-embedded content
- Progress tracking

BGE-M3: 1024-dim multilingual embeddings, excellent for semantic search.

Usage:
    # DeepInfra (rate limited)
    embedder = AsyncEmbedder(provider="deepinfra")

    # Self-hosted TEI on RunPod (unlimited)
    embedder = AsyncEmbedder(provider="tei", tei_endpoint="http://your-runpod:8080")

    embeddings = await embedder.embed_batch(texts, batch_size=32, max_concurrent=8)

Version: 1.1.0 (cog_twin)
"""

import asyncio
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
import logging

import httpx
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PROVIDER PROTOCOL & IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    EMBEDDING_DIM: int

    async def embed_batch_raw(
        self,
        client: httpx.AsyncClient,
        texts: List[str],
        batch_id: int,
    ) -> List[np.ndarray]:
        """Embed a batch of texts without caching."""
        ...

    async def rate_limit(self) -> None:
        """Apply rate limiting if needed."""
        ...


class DeepInfraProvider:
    """
    DeepInfra BGE-M3 provider.

    Rate limited to 200 RPM. Good for small-scale or cached workloads.
    """

    EMBEDDING_DIM = 1024
    DEEPINFRA_URL = "https://api.deepinfra.com/v1/inference/BAAI/bge-m3"

    def __init__(
        self,
        api_key: Optional[str] = None,
        requests_per_minute: int = 180,
    ):
        from dotenv import load_dotenv
        load_dotenv(override=True)

        self.api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPINFRA_API_KEY required for DeepInfra provider")

        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._rate_lock = asyncio.Lock()

    async def rate_limit(self) -> None:
        """Enforce rate limiting."""
        async with self._rate_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()

    async def embed_batch_raw(
        self,
        client: httpx.AsyncClient,
        texts: List[str],
        batch_id: int,
    ) -> List[np.ndarray]:
        """Embed via DeepInfra API."""
        await self.rate_limit()

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

            embeddings_raw = data.get("embeddings", [])
            return [np.array(e, dtype=np.float32) for e in embeddings_raw]

        except httpx.HTTPStatusError as e:
            logger.error(f"Batch {batch_id} HTTP error: {e.response.status_code}")
            return [np.zeros(self.EMBEDDING_DIM, dtype=np.float32) for _ in texts]
        except Exception as e:
            logger.error(f"Batch {batch_id} error: {e}")
            return [np.zeros(self.EMBEDDING_DIM, dtype=np.float32) for _ in texts]


class TEIProvider:
    """
    Text Embeddings Inference (TEI) provider.

    Self-hosted on RunPod/Modal with no rate limits.
    Demolishes large workloads - 22k embeddings in ~2-3 minutes.

    Deploy with:
        docker run --gpus all -p 80:80 \\
          ghcr.io/huggingface/text-embeddings-inference:1.5 \\
          --model-id BAAI/bge-m3 \\
          --pooling cls \\
          --max-client-batch-size 128 \\
          --max-batch-tokens 16384
    """

    EMBEDDING_DIM = 1024

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,  # Some deployments use auth
    ):
        """
        Args:
            endpoint: TEI endpoint URL (e.g., http://runpod-xyz:8080)
            api_key: Optional API key for authenticated deployments
        """
        self.endpoint = endpoint.rstrip("/")
        self.embed_url = f"{self.endpoint}/embed"
        self.api_key = api_key

        logger.info(f"TEI provider initialized: {self.endpoint}")

    async def rate_limit(self) -> None:
        """No rate limiting for self-hosted."""
        pass

    async def embed_batch_raw(
        self,
        client: httpx.AsyncClient,
        texts: List[str],
        batch_id: int,
    ) -> List[np.ndarray]:
        """Embed via TEI endpoint."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # TEI expects {"inputs": ["text1", "text2", ...]}
            response = await client.post(
                self.embed_url,
                headers=headers,
                json={"inputs": texts},
                timeout=120.0,  # Longer timeout for large batches
            )
            response.raise_for_status()

            # TEI returns list of embeddings directly
            embeddings_raw = response.json()
            return [np.array(e, dtype=np.float32) for e in embeddings_raw]

        except httpx.HTTPStatusError as e:
            logger.error(f"Batch {batch_id} TEI HTTP error: {e.response.status_code}")
            return [np.zeros(self.EMBEDDING_DIM, dtype=np.float32) for _ in texts]
        except Exception as e:
            logger.error(f"Batch {batch_id} TEI error: {e}")
            return [np.zeros(self.EMBEDDING_DIM, dtype=np.float32) for _ in texts]


class CloudflareProvider:
    """
    Cloudflare Workers AI provider.

    Fallback option with reasonable rate limits.
    """

    EMBEDDING_DIM = 1024

    def __init__(
        self,
        account_id: str,
        api_token: Optional[str] = None,
        requests_per_minute: int = 300,
    ):
        from dotenv import load_dotenv
        load_dotenv(override=True)

        self.account_id = account_id
        self.api_token = api_token or os.getenv("CLOUDFLARE_API_TOKEN")
        if not self.api_token:
            raise ValueError("CLOUDFLARE_API_TOKEN required")

        self.embed_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-m3"

        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._rate_lock = asyncio.Lock()

    async def rate_limit(self) -> None:
        """Enforce rate limiting."""
        async with self._rate_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()

    async def embed_batch_raw(
        self,
        client: httpx.AsyncClient,
        texts: List[str],
        batch_id: int,
    ) -> List[np.ndarray]:
        """Embed via Cloudflare Workers AI."""
        await self.rate_limit()

        try:
            response = await client.post(
                self.embed_url,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                },
                json={"text": texts},  # Cloudflare uses "text" not "inputs"
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            # Cloudflare returns {"result": {"data": [...]}}
            embeddings_raw = data.get("result", {}).get("data", [])
            return [np.array(e, dtype=np.float32) for e in embeddings_raw]

        except httpx.HTTPStatusError as e:
            logger.error(f"Batch {batch_id} Cloudflare HTTP error: {e.response.status_code}")
            return [np.zeros(self.EMBEDDING_DIM, dtype=np.float32) for _ in texts]
        except Exception as e:
            logger.error(f"Batch {batch_id} Cloudflare error: {e}")
            return [np.zeros(self.EMBEDDING_DIM, dtype=np.float32) for _ in texts]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EMBEDDER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class AsyncEmbedder:
    """
    Async embedder with pluggable provider backends.

    Supports caching, batching, and progress tracking.
    """

    EMBEDDING_DIM = 1024
    MAX_TOKENS = 8192  # BGE-M3 context window

    def __init__(
        self,
        provider: str = "deepinfra",
        cache_dir: Optional[Path] = None,
        # DeepInfra options
        api_key: Optional[str] = None,
        requests_per_minute: int = 180,
        # TEI options
        tei_endpoint: Optional[str] = None,
        # Cloudflare options
        cloudflare_account_id: Optional[str] = None,
    ):
        """
        Initialize embedder with specified provider.

        Args:
            provider: "deepinfra", "tei", or "cloudflare"
            cache_dir: Directory for embedding cache
            api_key: API key for DeepInfra/Cloudflare
            requests_per_minute: Rate limit for API providers
            tei_endpoint: URL for self-hosted TEI instance
            cloudflare_account_id: Cloudflare account ID
        """
        self.cache_dir = cache_dir or Path("./data/embedding_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize provider
        self.provider_name = provider.lower()
        if self.provider_name == "deepinfra":
            self.provider = DeepInfraProvider(
                api_key=api_key,
                requests_per_minute=requests_per_minute,
            )
        elif self.provider_name == "tei":
            if not tei_endpoint:
                raise ValueError("tei_endpoint required for TEI provider")
            self.provider = TEIProvider(endpoint=tei_endpoint, api_key=api_key)
        elif self.provider_name == "cloudflare":
            if not cloudflare_account_id:
                raise ValueError("cloudflare_account_id required")
            self.provider = CloudflareProvider(
                account_id=cloudflare_account_id,
                api_token=api_key,
                requests_per_minute=requests_per_minute,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Stats
        self.stats = {
            "provider": self.provider_name,
            "total_embedded": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "errors": 0,
        }

        logger.info(f"AsyncEmbedder initialized with {self.provider_name} provider")

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
            batch_size: Texts per API call (TEI can handle 128+)
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
            print(f"Provider: {self.provider_name}")

        # Process batches with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        completed_count = [0]
        start_time = time.time()

        async def process_batch(batch_id: int, batch_texts: List[str], client: httpx.AsyncClient):
            async with semaphore:
                result = await self.provider.embed_batch_raw(client, batch_texts, batch_id)
                self.stats["api_calls"] += 1
                self.stats["total_embedded"] += len(batch_texts)

                completed_count[0] += 1
                if show_progress and completed_count[0] % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed_count[0] / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {completed_count[0]}/{len(batches)} batches ({rate:.1f} batches/sec)", flush=True)
                return result

        # Run all batches with gather (preserves order)
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
    batch_size: int = 16,
    max_concurrent: int = 4,
    max_chars: int = 16000,
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
            messages = ep.get("messages", [])
            parts = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            text = "\n\n".join(parts)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[truncated]"

        texts.append(text)

    return await embedder.embed_batch(texts, batch_size, max_concurrent)


def create_embedder(
    provider: str = "auto",
    tei_endpoint: Optional[str] = None,
    **kwargs
) -> AsyncEmbedder:
    """
    Factory function to create embedder with best available provider.

    Args:
        provider: "auto", "deepinfra", "tei", or "cloudflare"
        tei_endpoint: TEI endpoint URL (required if provider="tei")
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured AsyncEmbedder instance
    """
    if provider == "auto":
        # Try TEI first if endpoint provided
        if tei_endpoint:
            try:
                return AsyncEmbedder(provider="tei", tei_endpoint=tei_endpoint, **kwargs)
            except Exception as e:
                logger.warning(f"TEI provider failed: {e}, falling back to DeepInfra")

        # Fall back to DeepInfra
        return AsyncEmbedder(provider="deepinfra", **kwargs)

    return AsyncEmbedder(provider=provider, tei_endpoint=tei_endpoint, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Test the embedder."""
    from dotenv import load_dotenv
    load_dotenv(override=True)

    print("AsyncEmbedder Test")
    print("=" * 60)

    embedder = AsyncEmbedder()

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

    print(f"\nSimilarity matrix (cosine):")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    similarity = normalized @ normalized.T

    for i, text in enumerate(test_texts):
        print(f"\n  '{text[:40]}...'")
        top_similar = np.argsort(similarity[i])[::-1][1:3]
        for j in top_similar:
            print(f"    -> {similarity[i,j]:.3f}: '{test_texts[j][:40]}...'")

    print(f"\nStats: {embedder.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
