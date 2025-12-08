#!/usr/bin/env python3
"""
AcidBurn Pipeline - Phase 3: HDBSCAN Clustering

Runs on Vast.ai after embedding completes.
Uses all available CPU cores for clustering.

Usage:
    pip install hdbscan
    python cluster_cpu.py --input embeddings.parquet --output clustered.parquet

Input: embeddings.parquet (from Phase 2)
Output:
    - clustered.parquet (original + cluster_id column)
    - centroids.parquet (cluster_id, centroid, node_count)
"""

import argparse
import time
from pathlib import Path

import numpy as np


def load_embeddings(path: str) -> tuple:
    """Load embeddings from parquet."""
    import pyarrow.parquet as pq

    print(f"Loading {path}...")
    table = pq.read_table(path)
    df = table.to_pandas()

    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Extract embeddings as numpy array
    embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)
    print(f"  Embedding shape: {embeddings.shape}")

    return df, embeddings


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> np.ndarray:
    """Run HDBSCAN clustering."""
    import hdbscan

    print(f"\nRunning HDBSCAN...")
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  min_samples: {min_samples}")
    print(f"  n_samples: {len(embeddings)}")

    start = time.time()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,  # Use all cores
        prediction_data=False,  # Don't need prediction, saves memory
    )

    labels = clusterer.fit_predict(embeddings)

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")

    # Stats
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")

    return labels


def compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute centroid for each cluster."""
    print("\nComputing centroids...")

    unique_labels = sorted(set(labels))
    # Remove noise label (-1)
    cluster_ids = [l for l in unique_labels if l >= 0]

    centroids = {}
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        centroid = cluster_embeddings.mean(axis=0)
        centroids[cluster_id] = {
            "centroid": centroid.tolist(),
            "node_count": int(mask.sum()),
        }

    print(f"  Computed {len(centroids)} centroids")
    return centroids


def save_clustered(df, labels: np.ndarray, output_path: str):
    """Save clustered parquet with cluster_id column."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"\nSaving clustered data to {output_path}...")

    # Add cluster_id column
    df["cluster_id"] = labels

    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="snappy")

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


def save_centroids(centroids: dict, output_path: str):
    """Save centroids to parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"Saving centroids to {output_path}...")

    data = {
        "cluster_id": list(centroids.keys()),
        "centroid": [c["centroid"] for c in centroids.values()],
        "node_count": [c["node_count"] for c in centroids.values()],
    }

    table = pa.table(data)
    pq.write_table(table, output_path, compression="snappy")

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="AcidBurn Phase 3: HDBSCAN Clustering")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input embeddings.parquet")
    parser.add_argument("--output", "-o", type=str, default="clustered.parquet", help="Output clustered.parquet")
    parser.add_argument("--centroids", "-c", type=str, default="centroids.parquet", help="Output centroids.parquet")
    parser.add_argument("--min-cluster-size", type=int, default=100, help="HDBSCAN min_cluster_size (default: 100, tuned for ~2M vectors)")
    parser.add_argument("--min-samples", type=int, default=20, help="HDBSCAN min_samples (default: 20, tuned for ~2M vectors)")

    args = parser.parse_args()

    print("=" * 60)
    print("AcidBurn Pipeline - Phase 3: HDBSCAN Clustering")
    print("=" * 60)

    # Check cores
    import os
    cores = os.cpu_count()
    print(f"CPU cores: {cores}")

    # Load
    df, embeddings = load_embeddings(args.input)

    # Cluster
    labels = run_hdbscan(
        embeddings,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    # Centroids
    centroids = compute_centroids(embeddings, labels)

    # Save
    save_clustered(df, labels, args.output)
    save_centroids(centroids, args.centroids)

    print()
    print("=" * 60)
    print("Complete!")
    print(f"  Clustered data: {args.output}")
    print(f"  Centroids: {args.centroids}")
    print("=" * 60)


if __name__ == "__main__":
    main()
