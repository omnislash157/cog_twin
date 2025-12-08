"""
AcidBurn Pipeline - Phase 2: Embedding

Reads chunked JSONL from Phase 1, embeds with BGE-M3, outputs to research_data/.

Usage:
    python embed_research.py --input chunks.jsonl --corpus gutenberg_general
    python embed_research.py --input pubmed_chunks.jsonl --corpus pubmed_central_oa

Input: chunks.jsonl from Phase 1 (ingest.py)
Output: research_data/{corpus_name}/
    |-- corpus.json        # Chunk metadata
    |-- vectors.npy        # (N, 1024) float32 embeddings
    |-- manifest.json      # Corpus info

Version: 1.0.0 (acidburn)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

import numpy as np

# Embedding setup
try:
    from embedder import AsyncEmbedder
    EMBEDDER_AVAILABLE = True
except ImportError:
    EMBEDDER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Batch size for embedding (tune for GPU memory)
EMBED_BATCH_SIZE = 32


def stream_chunks(input_path: Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Stream chunks from JSONL file."""
    count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if limit and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                count += 1
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line: {e}")
                continue


def batch_chunks(chunks: Iterator[Dict], batch_size: int) -> Iterator[List[Dict]]:
    """Batch chunks for efficient embedding."""
    batch = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


async def embed_batch(embedder, texts: List[str]) -> np.ndarray:
    """Embed a batch of texts."""
    embeddings = await embedder.embed_batch(texts)
    return np.array(embeddings, dtype=np.float32)


async def process_corpus(
    input_path: Path,
    output_dir: Path,
    corpus_name: str,
    embedder,
    limit: Optional[int] = None,
    batch_size: int = EMBED_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Process input JSONL into embedded corpus.
    
    Returns:
        Stats dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "chunks_processed": 0,
        "chunks_embedded": 0,
        "total_tokens": 0,
        "embedding_time_s": 0,
        "start_time": datetime.now().isoformat(),
    }
    
    all_chunks: List[Dict] = []
    all_embeddings: List[np.ndarray] = []
    
    chunk_stream = stream_chunks(input_path, limit=limit)
    
    embed_start = time.time()
    
    for batch_idx, batch in enumerate(batch_chunks(chunk_stream, batch_size)):
        # Extract texts for embedding
        texts = [chunk.get("text", "") for chunk in batch]
        
        # Skip empty texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not valid_indices:
            continue
        
        valid_texts = [texts[i] for i in valid_indices]
        valid_chunks = [batch[i] for i in valid_indices]
        
        # Embed batch
        try:
            embeddings = await embed_batch(embedder, valid_texts)
        except Exception as e:
            logger.error(f"Embedding failed for batch {batch_idx}: {e}")
            continue
        
        # Store results
        all_chunks.extend(valid_chunks)
        all_embeddings.append(embeddings)
        
        stats["chunks_embedded"] += len(valid_chunks)
        stats["total_tokens"] += sum(c.get("token_count", 0) for c in valid_chunks)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Embedded {stats['chunks_embedded']} chunks...", flush=True)
    
    stats["embedding_time_s"] = time.time() - embed_start
    stats["chunks_processed"] = stats["chunks_embedded"]
    
    if not all_embeddings:
        logger.error("No embeddings generated!")
        return stats
    
    # Concatenate all embeddings
    vectors = np.vstack(all_embeddings)
    logger.info(f"Final vectors shape: {vectors.shape}")
    
    # Save corpus.json (chunk metadata)
    corpus_path = output_dir / "corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved corpus: {corpus_path}")
    
    # Save vectors.npy
    vectors_path = output_dir / "vectors.npy"
    np.save(vectors_path, vectors)
    logger.info(f"Saved vectors: {vectors_path} ({vectors.shape})")
    
    # Save manifest.json
    manifest = {
        "corpus_name": corpus_name,
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "chunk_count": len(all_chunks),
        "vector_dim": vectors.shape[1],
        "total_tokens": stats["total_tokens"],
        "embedding_model": "bge-m3",
        "source_file": str(input_path.name),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest: {manifest_path}")
    
    stats["end_time"] = datetime.now().isoformat()
    return stats


async def main_async(args):
    """Async main for embedding."""
    if not EMBEDDER_AVAILABLE:
        print("ERROR: embedder module not available. Run from project directory.")
        sys.exit(1)
    
    # Initialize embedder
    print("Initializing embedder (BGE-M3)...")
    embedder = AsyncEmbedder()
    await embedder.initialize()
    
    # Setup output directory
    output_base = Path(args.output_dir) / "research_data" / args.corpus
    
    print(f"Processing: {args.input}")
    print(f"Output: {output_base}")
    print(f"Limit: {args.limit or 'unlimited'}")
    print()
    
    stats = await process_corpus(
        input_path=Path(args.input),
        output_dir=output_base,
        corpus_name=args.corpus,
        embedder=embedder,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    
    print()
    print("=" * 60)
    print("Complete!")
    print(f"  Chunks embedded: {stats['chunks_embedded']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Embedding time: {stats['embedding_time_s']:.1f}s")
    print(f"  Output: {output_base}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AcidBurn Pipeline Phase 2: Embed chunks"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input JSONL file from Phase 1",
    )
    parser.add_argument(
        "--corpus", "-c",
        type=str,
        required=True,
        help="Corpus name (e.g., gutenberg_general, pubmed_central_oa)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data",
        help="Base output directory (default: ./data)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Max chunks to embed (default: all)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=EMBED_BATCH_SIZE,
        help=f"Embedding batch size (default: {EMBED_BATCH_SIZE})",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 60)
    print("AcidBurn Pipeline - Phase 2: Embedding")
    print("=" * 60)
    
    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
