"""
AcidBurn Pipeline - Cluster-Assisted Retrieval for Massive Datasets

Named after Angelina Jolie's hacker tag in 'Hackers' (1995).

Architecture:
    Query → Embedder (threshold) → Grok Quality Gate → Results
                                        ↓ (low conf)
                                   Tools: get_cluster_labels()
                                        → retrieve_from_clusters()

Phases:
    1. ingest.py      - Chunk books → jsonl
    2. embed_gpu.py   - GPU embedding job
    3. cluster.py     - HDBSCAN clustering
    4. tag_clusters.py - Grok labels clusters
    5. retrieval.py   - Threshold-based search
    6. quality_gate.py + tools.py - Grok intervention on low confidence
"""

__version__ = "0.1.0"
