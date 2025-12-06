"""
AcidBurn Pipeline - Phase 1: Ingestion

Streams books from HuggingFace, chunks to ~2K tokens, outputs jsonl.

Usage:
    python -m acidburn_pipeline.ingest --limit 100 --output chunks.jsonl
    python -m acidburn_pipeline.ingest --output chunks.jsonl  # full run

Input: HuggingFace dataset sedthh/gutenberg_english (streaming)
Output: chunks.jsonl with format:
    {"chunk_id": "sha256[:24]", "book_id": "...", "title": "...", "authors": "...", "text": "...", "token_count": 1847}
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

import tiktoken


# Chunking parameters
CHUNK_SIZE = 2000  # tokens
CHUNK_OVERLAP = 200  # tokens


def get_tokenizer():
    """Get tiktoken encoder for cl100k_base (GPT-4/Claude tokenizer)."""
    return tiktoken.get_encoding("cl100k_base")


def chunk_document(
    text: str,
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Iterator[tuple[str, int]]:
    """
    Chunk text into overlapping segments by token count.

    Yields:
        (chunk_text, token_count) tuples
    """
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return

    # Single chunk if under limit
    if total_tokens <= chunk_size:
        yield text, total_tokens
        return

    # Sliding window with overlap
    start = 0
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        decoded_chunk = tokenizer.decode(chunk_tokens)

        yield decoded_chunk, len(chunk_tokens)

        # Move window forward
        start += chunk_size - overlap

        # Prevent infinite loop at end
        if end >= total_tokens:
            break


def generate_chunk_id(text: str) -> str:
    """Generate deterministic chunk ID from content hash."""
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def stream_gutenberg(limit: Optional[int] = None) -> Iterator[dict]:
    """
    Stream books from HuggingFace Gutenberg dataset.

    Args:
        limit: Max books to process (None for all)

    Yields:
        Book records with 'id', 'title', 'author', 'text' fields
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)

    print("Loading Gutenberg dataset (streaming)...")
    dataset = load_dataset(
        "sedthh/gutenberg_english",
        split="train",
        streaming=True,
    )

    count = 0
    for record in dataset:
        if limit and count >= limit:
            break

        # Dataset uses uppercase keys and JSON metadata
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
            print(f"  Streamed {count} books...", flush=True)


def process_books(
    output_path: Path,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """
    Process books into chunks and write to jsonl.

    Args:
        output_path: Path to output jsonl file
        limit: Max books to process
        verbose: Print per-book progress

    Returns:
        Stats dict with counts
    """
    tokenizer = get_tokenizer()

    stats = {
        "books_processed": 0,
        "chunks_created": 0,
        "total_tokens": 0,
        "skipped_empty": 0,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        for book in stream_gutenberg(limit=limit):
            book_id = book["id"]
            title = book["title"]
            authors = book["author"]
            text = book["text"]

            # Skip empty books
            if not text or len(text.strip()) < 100:
                stats["skipped_empty"] += 1
                continue

            book_chunks = 0
            for chunk_str, token_count in chunk_document(text, tokenizer):
                chunk_id = generate_chunk_id(chunk_str)

                record = {
                    "chunk_id": chunk_id,
                    "book_id": book_id,
                    "title": title,
                    "authors": authors,
                    "text": chunk_str,
                    "token_count": token_count,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                stats["chunks_created"] += 1
                stats["total_tokens"] += token_count
                book_chunks += 1

            stats["books_processed"] += 1

            if verbose:
                print(f"  [{book_id}] {title[:50]}: {book_chunks} chunks")
            elif stats["books_processed"] % 100 == 0:
                print(f"  Processed {stats['books_processed']} books, {stats['chunks_created']} chunks...", flush=True)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="AcidBurn Pipeline Phase 1: Ingest books to chunks"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("chunks.jsonl"),
        help="Output jsonl file (default: chunks.jsonl)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Max books to process (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-book progress",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AcidBurn Pipeline - Phase 1: Ingestion")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Limit: {args.limit or 'unlimited'}")
    print(f"Chunk size: {CHUNK_SIZE} tokens, overlap: {CHUNK_OVERLAP}")
    print()

    stats = process_books(
        output_path=args.output,
        limit=args.limit,
        verbose=args.verbose,
    )

    print()
    print("=" * 60)
    print("Complete!")
    print(f"  Books processed: {stats['books_processed']}")
    print(f"  Books skipped (empty): {stats['skipped_empty']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Avg tokens/chunk: {stats['total_tokens'] // max(stats['chunks_created'], 1)}")
    print(f"  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
