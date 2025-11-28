"""
Streaming Cluster Module - River + HDBSCAN Hybrid

Combines:
1. HDBSCAN (batch): Initial cluster structure from historical data
2. River DBSTREAM (streaming): Real-time cluster updates for new thoughts

The hybrid approach:
- Load pre-computed HDBSCAN clusters as "anchor" clusters
- Use River's DBSTREAM to assign new points to existing clusters OR detect new cluster formation
- Periodically re-batch with HDBSCAN for quality (optional)

"Our thoughts form constellations in real-time."

Version: 1.0.0 (cog_twin)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# River for streaming clustering
try:
    from river import cluster as river_cluster
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

# HDBSCAN for batch clustering and soft prediction
try:
    import hdbscan
    import joblib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ClusterAssignment:
    """Result of assigning a point to a cluster."""
    cluster_id: int
    confidence: float
    is_new_cluster: bool = False
    nearest_clusters: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class ClusterStats:
    """Statistics about the current cluster state."""
    n_clusters: int
    n_noise: int
    n_points: int
    largest_cluster_size: int
    avg_cluster_size: float
    new_clusters_this_session: int


class StreamingClusterEngine:
    """
    Hybrid streaming + batch clustering engine.

    Uses HDBSCAN's soft prediction for existing clusters,
    River's DBSTREAM for detecting new cluster formation.
    """

    def __init__(
        self,
        data_dir: Path,
        hdbscan_model_path: Optional[Path] = None,
        embedding_dim: int = 1024,
    ):
        """
        Initialize the streaming cluster engine.

        Args:
            data_dir: Directory containing cluster data
            hdbscan_model_path: Path to pre-trained HDBSCAN model
            embedding_dim: Dimension of embeddings
        """
        self.data_dir = Path(data_dir)
        self.embedding_dim = embedding_dim

        # Load HDBSCAN model if available
        self.hdbscan_model = None
        self.cluster_labels = None
        self.cluster_probabilities = None

        if hdbscan_model_path and hdbscan_model_path.exists():
            self._load_hdbscan_model(hdbscan_model_path)
        else:
            # Try default location
            default_path = self.data_dir / "indexes" / "hdbscan_model.joblib"
            if default_path.exists():
                self._load_hdbscan_model(default_path)

        # Initialize River DBSTREAM for streaming updates
        self.dbstream = None
        if RIVER_AVAILABLE:
            self.dbstream = river_cluster.DBSTREAM(
                clustering_threshold=0.5,  # Distance threshold for micro-clusters
                fading_factor=0.01,        # How fast old points fade
                cleanup_interval=100,       # Clean up every N points
                intersection_factor=0.3,   # Merge threshold
            )
            logger.info("River DBSTREAM initialized for streaming clustering")
        else:
            logger.warning("River not available - streaming clustering disabled")

        # Track new clusters formed this session
        self.session_clusters: Dict[int, List[np.ndarray]] = {}
        self.session_cluster_count = 0
        self.points_processed = 0

        # Cluster centroids (from HDBSCAN or computed)
        self.centroids: Dict[int, np.ndarray] = {}
        self._compute_centroids()

    def _load_hdbscan_model(self, model_path: Path):
        """Load pre-trained HDBSCAN model."""
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available")
            return

        try:
            self.hdbscan_model = joblib.load(model_path)
            logger.info(f"Loaded HDBSCAN model from {model_path}")

            # Load cluster assignments
            clusters_path = model_path.parent / "hdbscan_clusters.json"
            if clusters_path.exists():
                with open(clusters_path) as f:
                    data = json.load(f)
                    self.cluster_labels = np.array(data["labels"])
                    self.cluster_probabilities = np.array(data["probabilities"])
                    logger.info(f"Loaded {data['n_clusters']} clusters, {data['n_noise']} noise points")

        except Exception as e:
            logger.error(f"Failed to load HDBSCAN model: {e}")

    def _compute_centroids(self):
        """Compute cluster centroids from HDBSCAN results."""
        if self.cluster_labels is None:
            return

        # Load embeddings
        vectors_dir = self.data_dir / "vectors"
        emb_files = list(vectors_dir.glob("node_embeddings_*.npy"))
        if not emb_files:
            return

        embeddings = np.load(max(emb_files, key=lambda p: p.stat().st_mtime))

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        # Compute centroids for each cluster
        unique_labels = set(self.cluster_labels)
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue

            mask = self.cluster_labels == label
            cluster_vectors = normalized[mask]
            centroid = cluster_vectors.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            self.centroids[label] = centroid

        logger.info(f"Computed {len(self.centroids)} cluster centroids")

    def assign_point(
        self,
        embedding: np.ndarray,
        use_soft_prediction: bool = True,
    ) -> ClusterAssignment:
        """
        Assign a new point to a cluster.

        Uses HDBSCAN soft prediction if available, falls back to
        nearest centroid, and uses River to detect new cluster formation.

        Args:
            embedding: The embedding vector to assign
            use_soft_prediction: Use HDBSCAN approximate_predict if available

        Returns:
            ClusterAssignment with cluster ID and confidence
        """
        # Normalize input
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Try HDBSCAN soft prediction first (most accurate for existing clusters)
        if use_soft_prediction and self.hdbscan_model is not None and HDBSCAN_AVAILABLE:
            try:
                labels, strengths = hdbscan.approximate_predict(
                    self.hdbscan_model,
                    embedding.reshape(1, -1)
                )
                cluster_id = int(labels[0])
                confidence = float(strengths[0])

                # If confidently assigned to existing cluster
                if cluster_id != -1 and confidence > 0.5:
                    return ClusterAssignment(
                        cluster_id=cluster_id,
                        confidence=confidence,
                        is_new_cluster=False,
                        nearest_clusters=self._find_nearest_clusters(embedding, top_k=3),
                    )
            except Exception as e:
                logger.warning(f"HDBSCAN soft prediction failed: {e}")

        # Fall back to nearest centroid
        nearest = self._find_nearest_clusters(embedding, top_k=5)

        if nearest and nearest[0][1] > 0.7:  # High similarity to existing cluster
            return ClusterAssignment(
                cluster_id=nearest[0][0],
                confidence=nearest[0][1],
                is_new_cluster=False,
                nearest_clusters=nearest,
            )

        # Use River DBSTREAM to potentially form new cluster
        if self.dbstream is not None:
            # River expects dict format
            point_dict = {f"dim_{i}": float(v) for i, v in enumerate(embedding)}
            self.dbstream.learn_one(point_dict)

            # Check if new micro-cluster formed
            n_clusters = len(self.dbstream.centers)
            if n_clusters > self.session_cluster_count:
                self.session_cluster_count = n_clusters
                new_cluster_id = 10000 + n_clusters  # Offset to avoid collision with HDBSCAN IDs
                self.session_clusters[new_cluster_id] = [embedding]

                return ClusterAssignment(
                    cluster_id=new_cluster_id,
                    confidence=0.6,  # New clusters have moderate confidence
                    is_new_cluster=True,
                    nearest_clusters=nearest,
                )

        # No clear cluster assignment - mark as noise
        return ClusterAssignment(
            cluster_id=-1,
            confidence=0.0,
            is_new_cluster=False,
            nearest_clusters=nearest,
        )

    def _find_nearest_clusters(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[int, float]]:
        """Find nearest clusters by centroid similarity."""
        if not self.centroids:
            return []

        similarities = []
        for cluster_id, centroid in self.centroids.items():
            sim = float(embedding @ centroid)
            similarities.append((cluster_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def batch_assign(
        self,
        embeddings: np.ndarray,
    ) -> List[ClusterAssignment]:
        """
        Assign multiple points in batch.

        More efficient for bulk processing.
        """
        assignments = []
        for emb in embeddings:
            assignment = self.assign_point(emb)
            assignments.append(assignment)
            self.points_processed += 1

        return assignments

    def get_cluster_context(
        self,
        cluster_id: int,
        max_examples: int = 5,
    ) -> Dict[str, Any]:
        """
        Get context about a cluster for the agent.

        Returns representative information about what this cluster contains.
        """
        if cluster_id == -1:
            return {"type": "noise", "description": "Unclustered point"}

        if cluster_id >= 10000:
            # Session cluster
            return {
                "type": "new_session_cluster",
                "description": "New cluster formed this session",
                "n_points": len(self.session_clusters.get(cluster_id, [])),
            }

        # Historical cluster
        if self.cluster_labels is None:
            return {"type": "unknown", "cluster_id": cluster_id}

        mask = self.cluster_labels == cluster_id
        n_points = mask.sum()

        return {
            "type": "historical_cluster",
            "cluster_id": cluster_id,
            "n_points": int(n_points),
            "centroid_available": cluster_id in self.centroids,
        }

    def get_stats(self) -> ClusterStats:
        """Get current clustering statistics."""
        n_clusters = len(self.centroids)
        n_noise = 0
        n_points = 0

        if self.cluster_labels is not None:
            n_noise = int((self.cluster_labels == -1).sum())
            n_points = len(self.cluster_labels)

        # Add session clusters
        n_clusters += len(self.session_clusters)
        n_points += self.points_processed

        # Cluster sizes
        sizes = []
        if self.cluster_labels is not None:
            from collections import Counter
            label_counts = Counter(self.cluster_labels)
            sizes = [c for l, c in label_counts.items() if l != -1]

        return ClusterStats(
            n_clusters=n_clusters,
            n_noise=n_noise,
            n_points=n_points,
            largest_cluster_size=max(sizes) if sizes else 0,
            avg_cluster_size=sum(sizes) / len(sizes) if sizes else 0,
            new_clusters_this_session=len(self.session_clusters),
        )

    def save_session_clusters(self):
        """Save session clusters for persistence."""
        if not self.session_clusters:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file = self.data_dir / "indexes" / f"session_clusters_{timestamp}.json"

        data = {
            "timestamp": timestamp,
            "n_clusters": len(self.session_clusters),
            "points_processed": self.points_processed,
            "cluster_ids": list(self.session_clusters.keys()),
        }

        with open(session_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.session_clusters)} session clusters to {session_file}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Test the streaming cluster engine."""
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data")

    print("Streaming Cluster Engine")
    print("=" * 60)

    engine = StreamingClusterEngine(data_dir)

    stats = engine.get_stats()
    print(f"Clusters: {stats.n_clusters}")
    print(f"Noise points: {stats.n_noise}")
    print(f"Total points: {stats.n_points}")
    print(f"Largest cluster: {stats.largest_cluster_size}")
    print(f"Avg cluster size: {stats.avg_cluster_size:.1f}")

    # Test with random vector
    print("\nTesting point assignment...")
    test_vector = np.random.randn(1024)
    assignment = engine.assign_point(test_vector)
    print(f"Assigned to cluster {assignment.cluster_id} (confidence: {assignment.confidence:.3f})")
    print(f"New cluster: {assignment.is_new_cluster}")
    print(f"Nearest clusters: {assignment.nearest_clusters[:3]}")


if __name__ == "__main__":
    main()
