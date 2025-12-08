#!/usr/bin/env python3
"""
AcidBurn Pipeline - Phase 2: GPU Embedding

Embeds chunks using sentence-transformers directly on GPU.
Designed for Vast.ai/RunPod with RTX 6000+ VRAM.

Usage (on GPU machine):
    pip install sentence-transformers pyarrow
    python embed_gpu.py --input chunks.jsonl --output embeddings.parquet

Output: Parquet with columns:
    chunk_id, book_id, title, authors, text, token_count, embedding (1024-dim)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


BATCH_SIZE = 128  # BGE-M3 handles large batches well on 97GB VRAM
EMBEDDING_DIM = 1024


def load_model():
    """Load BGE-M3 model."""
    from sentence_transformers import SentenceTransformer
    print("Loading BAAI/bge-m3 model...")
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"Model loaded on: {model.device}")
    return model


def embed_all(
    model,
    chunks: List[Dict[str, Any]],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed all chunks."""
    texts = [c["text"] for c in chunks]
    n = len(texts)

    print(f"Embedding {n} chunks with batch_size={batch_size}...")
    start_time = time.time()

    # sentence-transformers handles batching internally
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine sim
    )

    elapsed = time.time() - start_time
    print(f"Done! {n} embeddings in {elapsed:.1f}s ({n/elapsed:.1f} chunks/sec)")

    return embeddings.astype(np.float32)


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    """Load chunks from jsonl."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def save_parquet(chunks: List[Dict[str, Any]], embeddings: np.ndarray, path: Path):
    """Save chunks + embeddings to parquet."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow required. Install with: pip install pyarrow")
        sys.exit(1)

    # Build table
    data = {
        "chunk_id": [c["chunk_id"] for c in chunks],
        "book_id": [c["book_id"] for c in chunks],
        "title": [c["title"] for c in chunks],
        "authors": [c["authors"] for c in chunks],
        "text": [c["text"] for c in chunks],
        "token_count": [c["token_count"] for c in chunks],
        "embedding": [emb.tolist() for emb in embeddings],
    }

    table = pa.table(data)
    pq.write_table(table, path, compression="snappy")

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved: {path} ({size_mb:.1f} MB)")


def test_gpu():
    """Quick GPU check."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {name} ({mem:.0f}GB)")
            return True
        else:
            print("CUDA not available!")
            return False
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AcidBurn Phase 2: GPU Embedding with sentence-transformers"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input chunks.jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("embeddings.parquet"),
        help="Output parquet file (default: embeddings.parquet)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Just test GPU availability",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AcidBurn Pipeline - Phase 2: GPU Embedding")
    print("=" * 60)

    # Check GPU
    if not test_gpu():
        print("WARNING: No GPU detected, will be slow!")

    if args.test:
        print("GPU test complete!")
        sys.exit(0)

    # Load model
    model = load_model()

    # Load chunks
    print(f"\nLoading {args.input}...")
    chunks = load_chunks(args.input)
    print(f"Loaded {len(chunks)} chunks")

    # Embed
    print()
    embeddings = embed_all(model, chunks, batch_size=args.batch_size)

    # Save
    print()
    save_parquet(chunks, embeddings, args.output)

    print()
    print("=" * 60)
    print("Complete! Download embeddings.parquet via Jupyter file browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
