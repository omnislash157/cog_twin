"""
Cluster Schema - Semantic Labels + Fast Navigation for Clusters

Gives the API model a semantic understanding of clusters:
- Each cluster gets a label, description, and top keywords
- NumPy-powered cluster retrieval for instant navigation
- Pre-computed cluster summaries for LLM context injection

Flow:
    1. Load clusters + nodes
    2. Generate labels by sampling representative nodes
    3. Build cluster index (numpy centroids + metadata)
    4. API model can browse clusters semantically

"The clusters aren't just numbers - they're constellations of thought."

Version: 1.0.0 (cog_twin)
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from heuristic_enricher import HeuristicEnricher

logger = logging.getLogger(__name__)


@dataclass
class ClusterProfile:
    """Semantic profile for a cluster."""
    cluster_id: int
    label: str  # Short label like "Python Async", "API Design", "Life Philosophy"
    description: str  # 1-2 sentence description
    size: int  # Number of nodes

    # Semantic fingerprint
    primary_domain: str
    domain_distribution: Dict[str, float]  # domain -> percentage
    avg_technical_depth: float
    top_keywords: List[str]

    # Representative samples
    sample_contents: List[str] = field(default_factory=list)

    # Centroid for fast lookup
    centroid: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "description": self.description,
            "size": self.size,
            "primary_domain": self.primary_domain,
            "domain_distribution": self.domain_distribution,
            "avg_technical_depth": self.avg_technical_depth,
            "top_keywords": self.top_keywords,
            "sample_contents": self.sample_contents,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterProfile":
        """Create from dict."""
        return cls(
            cluster_id=data["cluster_id"],
            label=data["label"],
            description=data["description"],
            size=data["size"],
            primary_domain=data["primary_domain"],
            domain_distribution=data["domain_distribution"],
            avg_technical_depth=data["avg_technical_depth"],
            top_keywords=data["top_keywords"],
            sample_contents=data.get("sample_contents", []),
        )

    def to_context_string(self) -> str:
        """Format for LLM context injection."""
        domains = ", ".join(f"{d}:{v:.0%}" for d, v in
                          sorted(self.domain_distribution.items(),
                                 key=lambda x: x[1], reverse=True)[:3])
        return (
            f"[Cluster {self.cluster_id}] {self.label}\n"
            f"  {self.description}\n"
            f"  Size: {self.size} nodes | Domains: {domains} | "
            f"Tech depth: {self.avg_technical_depth:.1f}/10\n"
            f"  Keywords: {', '.join(self.top_keywords[:5])}"
        )


class ClusterSchemaEngine:
    """
    Semantic schema engine for clusters.

    Generates labels, builds navigation index, provides
    LLM-friendly cluster browsing.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize cluster schema engine.

        Args:
            data_dir: Directory containing cluster data
        """
        self.data_dir = Path(data_dir)
        self.enricher = HeuristicEnricher()

        # Loaded data
        self.cluster_info: Dict[int, List[int]] = {}
        self.nodes: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.centroids: Dict[int, np.ndarray] = {}

        # Generated profiles
        self.profiles: Dict[int, ClusterProfile] = {}

        # Schema file path
        self.schema_file = self.data_dir / "indexes" / "cluster_schema.json"

    def load_clusters(self, manifest_file: Optional[str] = None):
        """Load cluster data from manifest."""
        # Find manifest
        if manifest_file:
            manifest_path = self.data_dir / manifest_file
        else:
            manifests = list(self.data_dir.glob("manifest_*.json"))
            if not manifests:
                raise FileNotFoundError(f"No manifest found in {self.data_dir}")
            manifest_path = max(manifests, key=lambda p: p.stat().st_mtime)

        logger.info(f"Loading from manifest: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Load nodes
        nodes_file = self.data_dir / "memory_nodes" / manifest["nodes_file"]
        with open(nodes_file) as f:
            self.nodes = json.load(f)

        # Load embeddings
        emb_file = self.data_dir / "vectors" / manifest["node_embeddings_file"]
        self.embeddings = np.load(emb_file)

        # Normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)

        # Load cluster assignments
        clusters_file = self.data_dir / "indexes" / manifest["clusters_file"]
        with open(clusters_file) as f:
            self.cluster_info = {int(k): v for k, v in json.load(f).items()}

        # Compute centroids
        for cluster_id, indices in self.cluster_info.items():
            if cluster_id == -1:  # Skip noise
                continue
            cluster_vectors = self.embeddings[indices]
            centroid = cluster_vectors.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            self.centroids[cluster_id] = centroid

        logger.info(f"Loaded {len(self.nodes)} nodes, {len(self.centroids)} clusters")

    def generate_profile(self, cluster_id: int, sample_size: int = 5) -> ClusterProfile:
        """
        Generate semantic profile for a cluster.

        Args:
            cluster_id: Cluster to profile
            sample_size: Number of nodes to sample for analysis

        Returns:
            ClusterProfile with semantic information
        """
        if cluster_id not in self.cluster_info:
            raise ValueError(f"Unknown cluster: {cluster_id}")

        indices = self.cluster_info[cluster_id]
        cluster_nodes = [self.nodes[i] for i in indices if i < len(self.nodes)]

        if not cluster_nodes:
            return ClusterProfile(
                cluster_id=cluster_id,
                label="Empty Cluster",
                description="No nodes in this cluster",
                size=0,
                primary_domain="unknown",
                domain_distribution={},
                avg_technical_depth=0,
                top_keywords=[],
            )

        # Analyze all nodes for aggregate signals
        domain_counts = Counter()
        tech_depths = []
        all_keywords = []

        for node in cluster_nodes:
            content = node.get("human_content", "")
            signals = self.enricher.extract_all(content)

            domain_counts[signals["primary_domain"]] += 1
            tech_depths.append(signals["technical_depth"])
            all_keywords.extend(signals.get("keyword_set", [])[:10])

        # Compute aggregates
        total = len(cluster_nodes)
        domain_distribution = {d: c / total for d, c in domain_counts.items()}
        primary_domain = domain_counts.most_common(1)[0][0]
        avg_tech_depth = sum(tech_depths) / len(tech_depths) if tech_depths else 0

        # Top keywords by frequency
        keyword_counts = Counter(all_keywords)
        top_keywords = [k for k, _ in keyword_counts.most_common(10)]

        # Sample representative nodes (closest to centroid)
        if cluster_id in self.centroids:
            centroid = self.centroids[cluster_id]
            cluster_embeddings = self.embeddings[indices]
            similarities = cluster_embeddings @ centroid
            top_indices = np.argsort(similarities)[::-1][:sample_size]
            samples = [
                cluster_nodes[i].get("human_content", "")[:200]
                for i in top_indices if i < len(cluster_nodes)
            ]
        else:
            samples = [n.get("human_content", "")[:200] for n in cluster_nodes[:sample_size]]

        # Generate label from domain + top keywords
        label = self._generate_label(primary_domain, top_keywords)
        description = self._generate_description(
            primary_domain, domain_distribution, avg_tech_depth, top_keywords, total
        )

        return ClusterProfile(
            cluster_id=cluster_id,
            label=label,
            description=description,
            size=total,
            primary_domain=primary_domain,
            domain_distribution=domain_distribution,
            avg_technical_depth=avg_tech_depth,
            top_keywords=top_keywords,
            sample_contents=samples,
            centroid=self.centroids.get(cluster_id),
        )

    def _generate_label(self, domain: str, keywords: List[str]) -> str:
        """Generate short label from domain and keywords."""
        # Domain-specific label patterns
        domain_labels = {
            "code": "Code",
            "ml": "ML/AI",
            "api": "API",
            "data": "Data",
            "infra": "Infrastructure",
            "architecture": "Architecture",
            "testing": "Testing",
            "config": "Config",
            "security": "Security",
            "ops": "DevOps",
            "frontend": "Frontend",
            "business": "Business",
            "exercise": "Exercise",
            "spiritual": "Spiritual",
            "psychology": "Psychology",
            "philosophy": "Philosophy",
            "relationship": "Relationships",
            "family": "Family",
            "creative": "Creative",
            "hobbies": "Hobbies",
            "employer": "Work",
            "general": "General",
        }

        base = domain_labels.get(domain, domain.title())

        # Add top keyword qualifier
        if keywords:
            qualifier = keywords[0].title()
            return f"{base}: {qualifier}"

        return base

    def _generate_description(
        self,
        domain: str,
        distribution: Dict[str, float],
        tech_depth: float,
        keywords: List[str],
        size: int,
    ) -> str:
        """Generate 1-2 sentence description."""
        # Determine character
        if tech_depth > 7:
            tech_char = "highly technical"
        elif tech_depth > 4:
            tech_char = "moderately technical"
        else:
            tech_char = "conversational"

        # Secondary domains
        secondary = [d for d, v in sorted(distribution.items(), key=lambda x: x[1], reverse=True)[1:3]
                    if v > 0.1]

        desc = f"{tech_char.title()} discussions primarily about {domain}"
        if secondary:
            desc += f", with elements of {', '.join(secondary)}"
        desc += f". {size} memories covering topics like {', '.join(keywords[:3])}."

        return desc

    def generate_all_profiles(self, max_clusters: Optional[int] = None) -> Dict[int, ClusterProfile]:
        """
        Generate profiles for all clusters.

        Args:
            max_clusters: Limit number of clusters (None = all)

        Returns:
            Dict of cluster_id -> ClusterProfile
        """
        cluster_ids = [cid for cid in self.cluster_info.keys() if cid != -1]

        if max_clusters:
            # Prioritize larger clusters
            cluster_ids.sort(key=lambda cid: len(self.cluster_info[cid]), reverse=True)
            cluster_ids = cluster_ids[:max_clusters]

        logger.info(f"Generating profiles for {len(cluster_ids)} clusters...")

        for i, cid in enumerate(cluster_ids):
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{len(cluster_ids)}")

            try:
                profile = self.generate_profile(cid)
                self.profiles[cid] = profile
            except Exception as e:
                logger.warning(f"Failed to profile cluster {cid}: {e}")

        logger.info(f"Generated {len(self.profiles)} cluster profiles")
        return self.profiles

    def save_schema(self):
        """Save cluster schema to JSON."""
        schema = {
            "generated_at": datetime.now().isoformat(),
            "n_clusters": len(self.profiles),
            "profiles": {str(k): v.to_dict() for k, v in self.profiles.items()},
        }

        with open(self.schema_file, "w") as f:
            json.dump(schema, f, indent=2)

        logger.info(f"Saved cluster schema to {self.schema_file}")

    def load_schema(self) -> bool:
        """Load existing schema from JSON."""
        if not self.schema_file.exists():
            return False

        with open(self.schema_file) as f:
            schema = json.load(f)

        self.profiles = {
            int(k): ClusterProfile.from_dict(v)
            for k, v in schema["profiles"].items()
        }

        logger.info(f"Loaded {len(self.profiles)} cluster profiles from schema")
        return True

    # =========================================================================
    # NAVIGATION API (for LLM/Agent use)
    # =========================================================================

    def find_clusters_by_query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[Tuple[int, float, ClusterProfile]]:
        """
        Find relevant clusters by embedding similarity.

        Args:
            query_embedding: Query vector
            top_k: Max clusters to return
            threshold: Min similarity

        Returns:
            List of (cluster_id, similarity, profile) tuples
        """
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        results = []
        for cid, centroid in self.centroids.items():
            sim = float(query_norm @ centroid)
            if sim >= threshold:
                profile = self.profiles.get(cid)
                if profile:
                    results.append((cid, sim, profile))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def find_clusters_by_domain(
        self,
        domain: str,
        min_percentage: float = 0.3,
    ) -> List[ClusterProfile]:
        """Find clusters dominated by a specific domain."""
        return [
            p for p in self.profiles.values()
            if p.domain_distribution.get(domain, 0) >= min_percentage
        ]

    def find_clusters_by_keyword(
        self,
        keyword: str,
    ) -> List[ClusterProfile]:
        """Find clusters containing a keyword."""
        keyword_lower = keyword.lower()
        return [
            p for p in self.profiles.values()
            if any(keyword_lower in k.lower() for k in p.top_keywords)
        ]

    def get_cluster_map(self, top_n: int = 50) -> str:
        """
        Get a text map of top clusters for LLM context.

        Returns markdown-formatted cluster overview.
        """
        # Sort by size
        sorted_profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.size,
            reverse=True
        )[:top_n]

        lines = ["# Cluster Map", ""]

        for profile in sorted_profiles:
            lines.append(profile.to_context_string())
            lines.append("")

        return "\n".join(lines)

    def get_cluster_for_context(
        self,
        cluster_ids: List[int],
        max_samples: int = 2,
    ) -> str:
        """
        Get formatted context for specific clusters.

        For injection into LLM prompts.
        """
        lines = []

        for cid in cluster_ids:
            profile = self.profiles.get(cid)
            if not profile:
                continue

            lines.append(f"## Cluster {cid}: {profile.label}")
            lines.append(profile.description)
            lines.append("")

            if profile.sample_contents:
                lines.append("**Sample memories:**")
                for i, sample in enumerate(profile.sample_contents[:max_samples], 1):
                    lines.append(f"  {i}. {sample}...")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Generate cluster schema."""
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data")

    print("Cluster Schema Generator")
    print("=" * 60)

    engine = ClusterSchemaEngine(data_dir)

    # Try to load existing schema
    if engine.load_schema():
        print(f"Loaded existing schema with {len(engine.profiles)} profiles")

        # Show sample
        print("\nSample profiles:")
        for i, profile in enumerate(list(engine.profiles.values())[:5]):
            print(f"\n{profile.to_context_string()}")

        return

    # Generate new schema
    print("No existing schema found, generating new one...")

    engine.load_clusters()
    engine.generate_all_profiles()
    engine.save_schema()

    print(f"\nGenerated {len(engine.profiles)} cluster profiles")

    # Show sample
    print("\nTop 5 clusters by size:")
    top_5 = sorted(engine.profiles.values(), key=lambda p: p.size, reverse=True)[:5]
    for profile in top_5:
        print(f"\n{profile.to_context_string()}")


if __name__ == "__main__":
    main()
