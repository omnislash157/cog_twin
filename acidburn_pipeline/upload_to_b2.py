#!/usr/bin/env python3
"""
AcidBurn Pipeline - Upload Results to B2

Uploads clustered.parquet and centroids.parquet directly from Vast.ai to B2.
Your laptop never touches the big files.

Usage:
    # Set env vars first
    export B2_KEY_ID="005723da756488b0000000002"
    export B2_APP_KEY="K005n9fRnHG/Ht0vW5gkW7CMu8mrtpE"
    export B2_ENDPOINT="https://s3.us-east-005.backblazeb2.com"
    export B2_BUCKET="cogtwinHarvardBooks"

    # Upload
    python upload_to_b2.py

    # Or specify files
    python upload_to_b2.py --clustered clustered.parquet --centroids centroids.parquet
"""

import argparse
import os
import sys
from pathlib import Path


def get_s3_client():
    """Create S3 client for B2."""
    import boto3

    endpoint = os.environ.get("B2_ENDPOINT")
    key_id = os.environ.get("B2_KEY_ID")
    app_key = os.environ.get("B2_APP_KEY")

    if not all([endpoint, key_id, app_key]):
        print("ERROR: Missing B2 environment variables!")
        print("Set these before running:")
        print('  export B2_KEY_ID="005723da756488b0000000002"')
        print('  export B2_APP_KEY="K005n9fRnHG/Ht0vW5gkW7CMu8mrtpE"')
        print('  export B2_ENDPOINT="https://s3.us-east-005.backblazeb2.com"')
        print('  export B2_BUCKET="cogtwinHarvardBooks"')
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
    )


def upload_file(s3, bucket: str, local_path: str, remote_key: str):
    """Upload a file to B2 with progress."""
    from boto3.s3.transfer import TransferConfig

    file_size = Path(local_path).stat().st_size
    size_mb = file_size / (1024 * 1024)

    print(f"Uploading {local_path} ({size_mb:.1f} MB)")
    print(f"  -> s3://{bucket}/{remote_key}")

    # Use multipart for large files
    config = TransferConfig(
        multipart_threshold=50 * 1024 * 1024,  # 50MB
        max_concurrency=10,
        multipart_chunksize=50 * 1024 * 1024,
    )

    # Progress callback
    uploaded = [0]

    def progress(bytes_transferred):
        uploaded[0] += bytes_transferred
        pct = 100 * uploaded[0] / file_size
        print(f"\r  Progress: {pct:.1f}% ({uploaded[0] / (1024*1024):.1f} MB)", end="", flush=True)

    s3.upload_file(
        local_path,
        bucket,
        remote_key,
        Config=config,
        Callback=progress,
    )

    print()  # newline after progress
    print(f"  Done!")


def main():
    parser = argparse.ArgumentParser(description="Upload clustering results to B2")
    parser.add_argument(
        "--clustered",
        type=str,
        default="clustered.parquet",
        help="Path to clustered.parquet",
    )
    parser.add_argument(
        "--centroids",
        type=str,
        default="centroids.parquet",
        help="Path to centroids.parquet",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Optional: also upload embeddings.parquet if not already uploaded",
    )

    args = parser.parse_args()

    bucket = os.environ.get("B2_BUCKET", "cogtwinHarvardBooks")

    print("=" * 60)
    print("AcidBurn Pipeline - Upload to B2")
    print("=" * 60)
    print(f"Bucket: {bucket}")
    print()

    s3 = get_s3_client()

    # Upload clustered.parquet
    if Path(args.clustered).exists():
        upload_file(s3, bucket, args.clustered, "embeddings/clustered.parquet")
    else:
        print(f"WARNING: {args.clustered} not found, skipping")

    # Upload centroids.parquet
    if Path(args.centroids).exists():
        upload_file(s3, bucket, args.centroids, "embeddings/centroids.parquet")
    else:
        print(f"WARNING: {args.centroids} not found, skipping")

    # Optionally upload embeddings if specified
    if args.embeddings and Path(args.embeddings).exists():
        upload_file(s3, bucket, args.embeddings, "embeddings/embeddings.parquet")

    print()
    print("=" * 60)
    print("Upload complete!")
    print()
    print("Files now in B2:")
    print(f"  s3://{bucket}/embeddings/clustered.parquet")
    print(f"  s3://{bucket}/embeddings/centroids.parquet")
    print()
    print("You can now DESTROY the Vast.ai instance!")
    print("=" * 60)


if __name__ == "__main__":
    main()
