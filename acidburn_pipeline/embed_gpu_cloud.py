#!/usr/bin/env python3
"""
AcidBurn Pipeline - Phase 2B: Cloud-Native GPU Embedding

Runs entirely on Vast.ai:
    HuggingFace (stream) → chunk → embed (BGE-M3) → Supabase + B2

Usage:
    # Local-only mode (saves to parquet, no cloud)
    python embed_gpu_cloud.py --limit 100 --local-only
    python embed_gpu_cloud.py --local-only --batch-size 512

    # Cloud mode (requires env vars)
    export SUPABASE_HOST=db.xxx.supabase.co
    export SUPABASE_PASSWORD=xxx
    export B2_KEY_ID=xxx
    export B2_APP_KEY=xxx
    export B2_BUCKET=xxx
    python embed_gpu_cloud.py --limit 100
    python embed_gpu_cloud.py --batch-size 512

No local files needed - streams from HuggingFace, pushes to cloud.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Iterator, List, Dict, Any, Optional

import numpy as np


# Chunking parameters
CHUNK_SIZE = 2000  # tokens
CHUNK_OVERLAP = 200  # tokens
EMBEDDING_DIM = 1024


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


class SupabaseClient:
    """Simple Supabase pgvector client."""

    def __init__(self):
        import psycopg2

        host = os.environ.get("SUPABASE_HOST")
        password = os.environ.get("SUPABASE_PASSWORD")

        if not host or not password:
            raise ValueError("SUPABASE_HOST and SUPABASE_PASSWORD required")

        # Supabase connection string
        self.conn = psycopg2.connect(
            host=host,
            port=5432,
            database="postgres",
            user="postgres",
            password=password,
        )
        self.conn.autocommit = False
        print(f"Connected to Supabase: {host}")

    def ensure_table(self):
        """Create table if not exists."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    book_id TEXT,
                    title TEXT,
                    authors TEXT,
                    blob_path TEXT,
                    token_count INT,
                    embedding vector(1024),
                    cluster_id INT DEFAULT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_chunk_cluster ON chunk_embeddings (cluster_id);
            """)
            self.conn.commit()
        print("Table chunk_embeddings ready")

    def insert_batch(self, rows: List[Dict[str, Any]]):
        """Insert batch of embeddings."""
        if not rows:
            return

        with self.conn.cursor() as cur:
            for row in rows:
                # Convert embedding list to pgvector format
                emb_str = "[" + ",".join(str(x) for x in row["embedding"]) + "]"
                cur.execute("""
                    INSERT INTO chunk_embeddings (chunk_id, book_id, title, authors, blob_path, token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO NOTHING
                """, (
                    row["chunk_id"],
                    row["book_id"],
                    row["title"],
                    row["authors"],
                    row["blob_path"],
                    row["token_count"],
                    emb_str,
                ))
            self.conn.commit()

    def get_existing_chunks(self) -> set:
        """Get set of already-processed chunk IDs for resume support."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id FROM chunk_embeddings")
            return {row[0] for row in cur.fetchall()}

    def close(self):
        self.conn.close()


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
        with open(local_path, "rb") as f:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=remote_key,
                Body=f,
                ContentType="application/octet-stream",
            )
        return f"b2://{self.bucket_name}/{remote_key}"


def process_books_b2(
    model,
    b2: B2Client,
    limit: Optional[int] = None,
    batch_size: int = 500,
    embed_batch_size: int = 128,
) -> dict:
    """Process books: chunk → embed → push to B2 only (no Supabase)."""
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

        # Upload chunks to B2
        blob_paths = b2.upload_batch(chunk_buffer)

        for chunk, emb, blob_path in zip(chunk_buffer, embeddings, blob_paths):
            all_rows.append({
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
        print(f"  Pushed {len(chunk_buffer)} chunks to B2 | Total: {stats['chunks_created']} | Rate: {rate:.1f}/sec")
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

    # Save parquet locally then upload to B2
    local_parquet = "/tmp/embeddings.parquet"
    print(f"\nSaving {len(all_rows)} chunks to parquet...")
    data = {
        "chunk_id": [r["chunk_id"] for r in all_rows],
        "book_id": [r["book_id"] for r in all_rows],
        "title": [r["title"] for r in all_rows],
        "authors": [r["authors"] for r in all_rows],
        "blob_path": [r["blob_path"] for r in all_rows],
        "token_count": [r["token_count"] for r in all_rows],
        "embedding": [r["embedding"] for r in all_rows],
    }
    table = pa.table(data)
    pq.write_table(table, local_parquet, compression="snappy")

    from pathlib import Path
    size_mb = Path(local_parquet).stat().st_size / (1024 * 1024)
    print(f"Local parquet: {size_mb:.1f} MB")

    # Upload to B2
    remote_key = "embeddings/embeddings.parquet"
    print(f"Uploading to B2: {remote_key}...")
    b2_path = b2.upload_parquet(local_parquet, remote_key)
    print(f"Uploaded: {b2_path}")

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

    from pathlib import Path
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    return stats


def process_books(
    model,
    supabase: SupabaseClient,
    b2: B2Client,
    limit: Optional[int] = None,
    batch_size: int = 500,
    embed_batch_size: int = 128,
) -> dict:
    """Process books: chunk → embed → push to cloud."""
    tokenizer = get_tokenizer()

    # Get existing chunks for resume
    print("Checking for existing chunks (resume support)...")
    existing = supabase.get_existing_chunks()
    print(f"Found {len(existing)} existing chunks")

    stats = {
        "books_processed": 0,
        "chunks_created": 0,
        "chunks_skipped": 0,
        "total_tokens": 0,
    }

    # Buffer for batching
    chunk_buffer: List[Dict[str, Any]] = []
    start_time = time.time()

    def flush_buffer():
        """Process and push buffer to cloud."""
        if not chunk_buffer:
            return

        # Embed
        texts = [c["text"] for c in chunk_buffer]
        embeddings = embed_batch(model, texts, batch_size=embed_batch_size)

        # Upload to B2
        blob_paths = b2.upload_batch(chunk_buffer)

        # Prepare rows for Supabase
        rows = []
        for chunk, emb, blob_path in zip(chunk_buffer, embeddings, blob_paths):
            rows.append({
                "chunk_id": chunk["chunk_id"],
                "book_id": chunk["book_id"],
                "title": chunk["title"],
                "authors": chunk["authors"],
                "blob_path": blob_path,
                "token_count": chunk["token_count"],
                "embedding": emb.tolist(),
            })

        # Insert to Supabase
        supabase.insert_batch(rows)

        elapsed = time.time() - start_time
        rate = stats["chunks_created"] / elapsed if elapsed > 0 else 0
        print(f"  Pushed {len(chunk_buffer)} chunks | Total: {stats['chunks_created']} | Rate: {rate:.1f}/sec")

        chunk_buffer.clear()

    # Process books
    for book in stream_gutenberg(limit=limit):
        text = book["text"]
        if not text or len(text.strip()) < 100:
            continue

        for chunk_str, token_count in chunk_document(text, tokenizer):
            chunk_id = generate_chunk_id(chunk_str)

            # Skip if already processed
            if chunk_id in existing:
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

            # Flush when buffer is full
            if len(chunk_buffer) >= batch_size:
                flush_buffer()

        stats["books_processed"] += 1

    # Final flush
    flush_buffer()

    return stats


def main():
    parser = argparse.ArgumentParser(description="AcidBurn Phase 2B: Cloud-Native GPU Embedding")
    parser.add_argument("--limit", "-l", type=int, help="Max books to process")
    parser.add_argument("--batch-size", "-b", type=int, default=500, help="Chunks per batch (default: 500)")
    parser.add_argument("--embed-batch-size", type=int, default=128, help="Embedding batch size (default: 128)")
    parser.add_argument("--test", action="store_true", help="Test connections only")
    parser.add_argument("--local-only", action="store_true", help="Save to parquet, skip cloud push")
    parser.add_argument("--b2-only", action="store_true", help="Push to B2 only, skip Supabase")
    parser.add_argument("--output", "-o", type=str, default="embeddings.parquet", help="Output file for --local-only")

    args = parser.parse_args()

    print("=" * 60)
    print("AcidBurn Pipeline - Phase 2B: Cloud-Native GPU Embedding")
    print("=" * 60)

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

    # B2-only mode (no Supabase)
    if args.b2_only:
        print("\nB2-ONLY MODE: Pushing to B2, skipping Supabase")
        try:
            b2 = B2Client()
        except Exception as e:
            print(f"B2 error: {e}")
            sys.exit(1)

        model = load_model()
        stats = process_books_b2(
            model=model,
            b2=b2,
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
        print(f"  Parquet uploaded to: b2://cogtwinHarvardBooks/embeddings/embeddings.parquet")
        print("=" * 60)
        sys.exit(0)

    # Initialize clients (full mode: Supabase + B2)
    print("\nConnecting to cloud services...")
    try:
        supabase = SupabaseClient()
        supabase.ensure_table()
    except Exception as e:
        print(f"Supabase error: {e}")
        sys.exit(1)

    try:
        b2 = B2Client()
    except Exception as e:
        print(f"B2 error: {e}")
        sys.exit(1)

    if args.test:
        print("\nConnection test passed!")
        supabase.close()
        sys.exit(0)

    # Load model
    print()
    model = load_model()

    # Process
    print()
    print(f"Processing with batch_size={args.batch_size}, embed_batch_size={args.embed_batch_size}")
    if args.limit:
        print(f"Limit: {args.limit} books")
    print()

    stats = process_books(
        model=model,
        supabase=supabase,
        b2=b2,
        limit=args.limit,
        batch_size=args.batch_size,
        embed_batch_size=args.embed_batch_size,
    )

    supabase.close()

    print()
    print("=" * 60)
    print("Complete!")
    print(f"  Books processed: {stats['books_processed']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    print(f"  Chunks skipped (resume): {stats['chunks_skipped']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
