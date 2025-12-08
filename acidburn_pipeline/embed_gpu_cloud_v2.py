#!/usr/bin/env python3
"""
AcidBurn Pipeline - Phase 2B: Cloud-Native GPU Embedding (V2 - Crash Safe)

Runs entirely on Vast.ai:
    HuggingFace (stream) → chunk → embed (BGE-M3) → B2 (incremental saves)

KEY FEATURES:
    - Saves checkpoint every 50K chunks (survives crashes!)
    - Resume support: skips already-processed chunk_ids
    - Numbered parquet shards: embeddings_000.parquet, embeddings_001.parquet, etc.
    - Final merge at end

Usage:
    # Set B2 env vars
    export B2_KEY_ID="005723da756488b0000000002"
    export B2_APP_KEY="K005n9fRnHG/Ht0vW5gkW7CMu8mrtpE"
    export B2_ENDPOINT="https://s3.us-east-005.backblazeb2.com"
    export B2_BUCKET="cogtwinHarvardBooks"

    # Full run with checkpoints
    python embed_gpu_cloud_v2.py --b2-only --batch-size 512

    # Resume after crash (auto-detects existing shards)
    python embed_gpu_cloud_v2.py --b2-only --batch-size 512 --resume
"""

import argparse
import glob as globlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Set

import numpy as np


# Chunking parameters
CHUNK_SIZE = 2000  # tokens
CHUNK_OVERLAP = 200  # tokens
EMBEDDING_DIM = 1024

# Checkpoint parameters - save incrementally to survive crashes
CHECKPOINT_INTERVAL = 50000  # Save every 50K chunks
CHECKPOINT_DIR = "/tmp/acidburn_checkpoints"


def get_tokenizer():
    """Get tiktoken encoder."""
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def chunk_document(text: str, tokenizer, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterator[tuple[str, int]]:
    """Chunk text into overlapping segments."""
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return
    if total_tokens <= chunk_size:
        yield text, total_tokens
        return

    start = 0
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        decoded = tokenizer.decode(chunk_tokens)
        yield decoded, len(chunk_tokens)
        start += chunk_size - overlap
        if end >= total_tokens:
            break


def generate_chunk_id(text: str) -> str:
    """Deterministic chunk ID from content hash."""
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def stream_gutenberg(limit: Optional[int] = None) -> Iterator[dict]:
    """Stream books from HuggingFace."""
    from datasets import load_dataset

    print("Loading Gutenberg dataset (streaming)...")
    dataset = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)

    count = 0
    for record in dataset:
        if limit and count >= limit:
            break

        text = record.get("TEXT", "")
        metadata_str = record.get("METADATA", "{}")
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            metadata = {}

        yield {
            "id": str(metadata.get("text_id", count)),
            "title": metadata.get("title", "Unknown"),
            "author": metadata.get("author", "Unknown"),
            "text": text,
        }
        count += 1
        if count % 100 == 0:
            print(f"  Streamed {count} books...")


def load_model():
    """Load BGE-M3 model."""
    from sentence_transformers import SentenceTransformer
    print("Loading BAAI/bge-m3 model...")
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"Model loaded on: {model.device}")
    return model


def embed_batch(model, texts: List[str], batch_size: int = 128) -> np.ndarray:
    """Embed texts."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


class B2Client:
    """Simple B2 blob storage client."""

    def __init__(self):
        import boto3

        key_id = os.environ.get("B2_KEY_ID")
        app_key = os.environ.get("B2_APP_KEY")
        self.bucket_name = os.environ.get("B2_BUCKET")
        endpoint = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")

        if not key_id or not app_key or not self.bucket_name:
            raise ValueError("B2_KEY_ID, B2_APP_KEY, B2_BUCKET required")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
        )
        print(f"Connected to B2: {self.bucket_name} @ {endpoint}")

    def upload_text(self, key: str, text: str) -> str:
        """Upload text to B2, return path."""
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType="text/plain",
        )
        return f"b2://{self.bucket_name}/{key}"

    def upload_batch(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Upload batch of chunks, return blob paths."""
        paths = []
        for chunk in chunks:
            key = f"chunks/{chunk['chunk_id']}.txt"
            path = self.upload_text(key, chunk["text"])
            paths.append(path)
        return paths

    def upload_parquet(self, local_path: str, remote_key: str):
        """Upload parquet file to B2."""
        from boto3.s3.transfer import TransferConfig

        file_size = Path(local_path).stat().st_size
        size_mb = file_size / (1024 * 1024)

        print(f"Uploading {local_path} ({size_mb:.1f} MB) to {remote_key}...")

        # Use multipart for large files
        config = TransferConfig(
            multipart_threshold=50 * 1024 * 1024,
            max_concurrency=10,
            multipart_chunksize=50 * 1024 * 1024,
        )

        self.s3.upload_file(local_path, self.bucket_name, remote_key, Config=config)
        return f"b2://{self.bucket_name}/{remote_key}"

    def list_shards(self, prefix: str = "embeddings/shards/") -> List[str]:
        """List existing shard files in B2."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if "Contents" not in response:
                return []
            return [obj["Key"] for obj in response["Contents"]]
        except Exception as e:
            print(f"Warning: Could not list shards: {e}")
            return []

    def download_file(self, remote_key: str, local_path: str):
        """Download file from B2."""
        self.s3.download_file(self.bucket_name, remote_key, local_path)


def load_existing_chunk_ids(checkpoint_dir: str, b2: Optional['B2Client'] = None) -> Set[str]:
    """Load chunk_ids from existing local shards and B2 shards for resume support."""
    import pyarrow.parquet as pq

    existing_ids: Set[str] = set()
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Check local shards
    local_shards = sorted(globlib.glob(f"{checkpoint_dir}/embeddings_*.parquet"))
    for shard_path in local_shards:
        try:
            table = pq.read_table(shard_path, columns=["chunk_id"])
            ids = table.column("chunk_id").to_pylist()
            existing_ids.update(ids)
            print(f"  Loaded {len(ids)} chunk_ids from {shard_path}")
        except Exception as e:
            print(f"  Warning: Could not read {shard_path}: {e}")

    # Check B2 shards if client provided
    if b2:
        b2_shards = b2.list_shards()
        for remote_key in b2_shards:
            if not remote_key.endswith(".parquet"):
                continue
            local_temp = f"{checkpoint_dir}/_temp_shard.parquet"
            try:
                b2.download_file(remote_key, local_temp)
                table = pq.read_table(local_temp, columns=["chunk_id"])
                ids = table.column("chunk_id").to_pylist()
                existing_ids.update(ids)
                print(f"  Loaded {len(ids)} chunk_ids from B2:{remote_key}")
                os.remove(local_temp)
            except Exception as e:
                print(f"  Warning: Could not read B2:{remote_key}: {e}")

    return existing_ids


def save_shard(rows: List[Dict[str, Any]], shard_num: int, checkpoint_dir: str, b2: Optional['B2Client'] = None) -> str:
    """Save a shard to local disk and optionally upload to B2."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    shard_path = f"{checkpoint_dir}/embeddings_{shard_num:03d}.parquet"

    data = {
        "chunk_id": [r["chunk_id"] for r in rows],
        "book_id": [r["book_id"] for r in rows],
        "title": [r["title"] for r in rows],
        "authors": [r["authors"] for r in rows],
        "blob_path": [r["blob_path"] for r in rows],
        "token_count": [r["token_count"] for r in rows],
        "embedding": [r["embedding"] for r in rows],
    }
    table = pa.table(data)
    pq.write_table(table, shard_path, compression="snappy")

    size_mb = Path(shard_path).stat().st_size / (1024 * 1024)
    print(f"  Saved shard {shard_num}: {shard_path} ({size_mb:.1f} MB, {len(rows)} chunks)")

    # Upload to B2 immediately for crash safety
    if b2:
        remote_key = f"embeddings/shards/embeddings_{shard_num:03d}.parquet"
        b2.upload_parquet(shard_path, remote_key)
        print(f"  Uploaded shard to B2: {remote_key}")

    return shard_path


def merge_shards(checkpoint_dir: str, output_path: str) -> int:
    """Merge all shards into single parquet file."""
    import pyarrow.parquet as pq

    shard_files = sorted(globlib.glob(f"{checkpoint_dir}/embeddings_*.parquet"))
    if not shard_files:
        print("No shards to merge")
        return 0

    print(f"\nMerging {len(shard_files)} shards into {output_path}...")

    # Read and concatenate all shards
    tables = []
    total_rows = 0
    for shard_path in shard_files:
        table = pq.read_table(shard_path)
        tables.append(table)
        total_rows += len(table)
        print(f"  Read {shard_path}: {len(table)} rows")

    import pyarrow as pa
    merged = pa.concat_tables(tables)
    pq.write_table(merged, output_path, compression="snappy")

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Merged: {output_path} ({size_mb:.1f} MB, {total_rows} total chunks)")

    return total_rows


def process_books_b2_incremental(
    model,
    b2: B2Client,
    limit: Optional[int] = None,
    batch_size: int = 500,
    embed_batch_size: int = 128,
    resume: bool = False,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
) -> dict:
    """Process books with incremental saves every checkpoint_interval chunks."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow required. Install with: pip install pyarrow")
        sys.exit(1)

    tokenizer = get_tokenizer()

    # Resume support: load existing chunk_ids
    existing_ids: Set[str] = set()
    shard_num = 0
    if resume:
        print("\nChecking for existing chunks (resume mode)...")
        existing_ids = load_existing_chunk_ids(CHECKPOINT_DIR, b2)
        print(f"Found {len(existing_ids)} already-processed chunks")

        # Find next shard number
        existing_shards = sorted(globlib.glob(f"{CHECKPOINT_DIR}/embeddings_*.parquet"))
        if existing_shards:
            last_shard = existing_shards[-1]
            # Extract number from embeddings_NNN.parquet
            shard_num = int(Path(last_shard).stem.split("_")[1]) + 1
            print(f"Will start at shard {shard_num}")

    stats = {
        "books_processed": 0,
        "chunks_created": 0,
        "chunks_skipped": 0,
        "total_tokens": 0,
        "shards_saved": 0,
    }

    shard_rows: List[Dict[str, Any]] = []
    chunk_buffer: List[Dict[str, Any]] = []
    start_time = time.time()

    def flush_buffer():
        """Embed and prepare rows from buffer."""
        nonlocal shard_rows
        if not chunk_buffer:
            return

        texts = [c["text"] for c in chunk_buffer]
        embeddings = embed_batch(model, texts, batch_size=embed_batch_size)

        # Upload chunks to B2
        blob_paths = b2.upload_batch(chunk_buffer)

        for chunk, emb, blob_path in zip(chunk_buffer, embeddings, blob_paths):
            shard_rows.append({
                "chunk_id": chunk["chunk_id"],
                "book_id": chunk["book_id"],
                "title": chunk["title"],
                "authors": chunk["authors"],
                "blob_path": blob_path,
                "token_count": chunk["token_count"],
                "embedding": emb.tolist(),
            })

        elapsed = time.time() - start_time
        rate = stats["chunks_created"] / elapsed if elapsed > 0 else 0
        print(f"  Pushed {len(chunk_buffer)} chunks | Total: {stats['chunks_created']} | Rate: {rate:.1f}/sec")
        chunk_buffer.clear()

    def save_checkpoint():
        """Save current shard to disk and B2."""
        nonlocal shard_num, shard_rows
        if not shard_rows:
            return

        save_shard(shard_rows, shard_num, CHECKPOINT_DIR, b2)
        stats["shards_saved"] += 1
        shard_num += 1
        shard_rows = []

    for book in stream_gutenberg(limit=limit):
        text = book["text"]
        if not text or len(text.strip()) < 100:
            continue

        for chunk_str, token_count in chunk_document(text, tokenizer):
            chunk_id = generate_chunk_id(chunk_str)

            # Skip if already processed (resume support)
            if chunk_id in existing_ids:
                stats["chunks_skipped"] += 1
                continue

            chunk_buffer.append({
                "chunk_id": chunk_id,
                "book_id": book["id"],
                "title": book["title"],
                "authors": book["author"],
                "text": chunk_str,
                "token_count": token_count,
            })
            stats["chunks_created"] += 1
            stats["total_tokens"] += token_count

            # Flush embedding buffer
            if len(chunk_buffer) >= batch_size:
                flush_buffer()

            # Save checkpoint every checkpoint_interval chunks
            if len(shard_rows) >= checkpoint_interval:
                print(f"\n*** CHECKPOINT: Saving shard {shard_num} ({len(shard_rows)} chunks) ***")
                save_checkpoint()

        stats["books_processed"] += 1

    # Final flush
    flush_buffer()

    # Save any remaining rows
    if shard_rows:
        print(f"\n*** FINAL: Saving shard {shard_num} ({len(shard_rows)} chunks) ***")
        save_checkpoint()

    # Merge all shards into final file
    final_path = "/tmp/embeddings.parquet"
    total_merged = merge_shards(CHECKPOINT_DIR, final_path)

    # Upload merged file to B2
    if total_merged > 0:
        remote_key = "embeddings/embeddings.parquet"
        print(f"\nUploading merged file to B2: {remote_key}...")
        b2.upload_parquet(final_path, remote_key)
        print(f"Uploaded: b2://{b2.bucket_name}/{remote_key}")

    return stats


def process_books_local(
    model,
    output_path: str,
    limit: Optional[int] = None,
    batch_size: int = 500,
    embed_batch_size: int = 128,
) -> dict:
    """Process books: chunk → embed → save to parquet (no cloud)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow required. Install with: pip install pyarrow")
        sys.exit(1)

    tokenizer = get_tokenizer()

    stats = {
        "books_processed": 0,
        "chunks_created": 0,
        "total_tokens": 0,
    }

    all_rows: List[Dict[str, Any]] = []
    chunk_buffer: List[Dict[str, Any]] = []
    start_time = time.time()

    def flush_buffer():
        if not chunk_buffer:
            return

        texts = [c["text"] for c in chunk_buffer]
        embeddings = embed_batch(model, texts, batch_size=embed_batch_size)

        for chunk, emb in zip(chunk_buffer, embeddings):
            all_rows.append({
                "chunk_id": chunk["chunk_id"],
                "book_id": chunk["book_id"],
                "title": chunk["title"],
                "authors": chunk["authors"],
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "embedding": emb.tolist(),
            })

        elapsed = time.time() - start_time
        rate = stats["chunks_created"] / elapsed if elapsed > 0 else 0
        print(f"  Embedded {len(chunk_buffer)} chunks | Total: {stats['chunks_created']} | Rate: {rate:.1f}/sec")
        chunk_buffer.clear()

    for book in stream_gutenberg(limit=limit):
        text = book["text"]
        if not text or len(text.strip()) < 100:
            continue

        for chunk_str, token_count in chunk_document(text, tokenizer):
            chunk_id = generate_chunk_id(chunk_str)
            chunk_buffer.append({
                "chunk_id": chunk_id,
                "book_id": book["id"],
                "title": book["title"],
                "authors": book["author"],
                "text": chunk_str,
                "token_count": token_count,
            })
            stats["chunks_created"] += 1
            stats["total_tokens"] += token_count

            if len(chunk_buffer) >= batch_size:
                flush_buffer()

        stats["books_processed"] += 1

    flush_buffer()

    # Save to parquet
    print(f"\nSaving {len(all_rows)} chunks to {output_path}...")
    data = {
        "chunk_id": [r["chunk_id"] for r in all_rows],
        "book_id": [r["book_id"] for r in all_rows],
        "title": [r["title"] for r in all_rows],
        "authors": [r["authors"] for r in all_rows],
        "text": [r["text"] for r in all_rows],
        "token_count": [r["token_count"] for r in all_rows],
        "embedding": [r["embedding"] for r in all_rows],
    }
    table = pa.table(data)
    pq.write_table(table, output_path, compression="snappy")

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="AcidBurn Phase 2B: Cloud-Native GPU Embedding (V2 - Crash Safe)")
    parser.add_argument("--limit", "-l", type=int, help="Max books to process")
    parser.add_argument("--batch-size", "-b", type=int, default=500, help="Chunks per batch (default: 500)")
    parser.add_argument("--embed-batch-size", type=int, default=128, help="Embedding batch size (default: 128)")
    parser.add_argument("--local-only", action="store_true", help="Save to parquet, skip cloud push")
    parser.add_argument("--b2-only", action="store_true", help="Push to B2 only, skip Supabase")
    parser.add_argument("--output", "-o", type=str, default="embeddings.parquet", help="Output file for --local-only")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from existing checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
                        help=f"Chunks per checkpoint (default: {CHECKPOINT_INTERVAL})")

    args = parser.parse_args()

    print("=" * 60)
    print("AcidBurn Pipeline - Phase 2B: Cloud-Native GPU Embedding")
    print("  V2 - CRASH SAFE (incremental saves)")
    print("=" * 60)
    print(f"Checkpoint interval: {args.checkpoint_interval} chunks")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: No GPU detected, will be slow!")
    except Exception as e:
        print(f"GPU check failed: {e}")

    # Local-only mode
    if args.local_only:
        print("\nLOCAL-ONLY MODE: Saving to parquet, no cloud push")
        model = load_model()
        stats = process_books_local(
            model=model,
            output_path=args.output,
            limit=args.limit,
            batch_size=args.batch_size,
            embed_batch_size=args.embed_batch_size,
        )
        print()
        print("=" * 60)
        print("Complete!")
        print(f"  Books processed: {stats['books_processed']}")
        print(f"  Chunks created: {stats['chunks_created']}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Output: {args.output}")
        print("=" * 60)
        sys.exit(0)

    # B2-only mode with incremental saves
    if args.b2_only:
        print("\nB2-ONLY MODE: Pushing to B2 with incremental checkpoints")
        if args.resume:
            print("RESUME MODE: Will skip already-processed chunks")
        try:
            b2 = B2Client()
        except Exception as e:
            print(f"B2 error: {e}")
            sys.exit(1)

        model = load_model()
        stats = process_books_b2_incremental(
            model=model,
            b2=b2,
            limit=args.limit,
            batch_size=args.batch_size,
            embed_batch_size=args.embed_batch_size,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval,
        )
        print()
        print("=" * 60)
        print("Complete!")
        print(f"  Books processed: {stats['books_processed']}")
        print(f"  Chunks created: {stats['chunks_created']}")
        print(f"  Chunks skipped (resume): {stats['chunks_skipped']}")
        print(f"  Shards saved: {stats['shards_saved']}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Final parquet: b2://cogtwinHarvardBooks/embeddings/embeddings.parquet")
        print("=" * 60)
        sys.exit(0)

    # Default: need B2 credentials
    print("\nERROR: Please use --b2-only or --local-only mode")
    print("  --b2-only   : Push to B2 (recommended for Vast.ai)")
    print("  --local-only: Save locally only")
    sys.exit(1)


if __name__ == "__main__":
    main()
