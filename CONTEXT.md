# CogTwin Context

**What**: Memory prosthetic for LLMs. 5-lane retrieval, 22K nodes, sub-second access. Agent outputs become searchable memory.

**Version**: 2.6.1 | **Model**: Grok 4.1 Fast (2M context) | **Stack**: Python/FastAPI + SvelteKit/Threlte

## Architecture

```
Query → DualRetriever → VenomVoice → LLM → Parse Actions → MemoryPipeline → Memory
         (5 lanes)      (prompts)   (Grok)  (GREP/SQUIRREL)  (recursive)
              |               ^
              v               |
       MetacognitiveMirror ---+---> Session Analytics -> WebSocket -> Frontend
```

## 5-Lane Retrieval
| Lane | File | Method | Purpose |
|------|------|--------|---------|
| Process | retrieval.py | FAISS/NumPy | 22K clustered nodes, semantic |
| Episodic | retrieval.py | FAISS | Conversation arcs |
| GREP | memory_grep.py | BM25 | Keyword frequency |
| SQUIRREL | squirrel.py | JSON temporal | "What was that 1hr ago?" |
| Traces | reasoning_trace.py | JSON + scores | Rated reasoning exemplars |

## File Map

### Entry Points
| File | Command |
|------|---------|
| cog_twin.py | `python cog_twin.py ./data` |
| ingest.py | `python ingest.py ./exports/` |
| backend/app/main.py | FastAPI + WebSocket |

### Hot Path (every query)
| File | Lines | Does | Imports | Imported By |
|------|-------|------|---------|-------------|
| cog_twin.py | ~1500 | Orchestrator | retrieval, venom_voice, memory_pipeline, model_adapter, reasoning_trace, chat_memory, squirrel, scoring, config, metacognitive_mirror | main.py |
| retrieval.py | 799 | Dual retrieval | schemas, embedder, memory_grep | cog_twin |
| venom_voice.py | 895 | Prompt construction, action parsing | none | cog_twin |
| memory_pipeline.py | 516 | Recursive memory loop | schemas, embedder, streaming_cluster | cog_twin, reasoning_trace |
| model_adapter.py | 378 | LLM abstraction (Grok/Claude) | none | cog_twin |
| schemas.py | 592 | MemoryNode, EpisodicMemory, RetrievalResult | none | 4+ files (RED ZONE) |
| embedder.py | 643 | BGE-M3 embeddings | none | 4 files |
| config.py | 204 | YAML config loader | none | cog_twin, backend |
| metacognitive_mirror.py | 1495 | Cognitive state monitoring | none | cog_twin |

### Tools (on-demand)
| File | Lines | Trigger | Does |
|------|-------|---------|------|
| reasoning_trace.py | 455 | Every query | Records reasoning chains, scores |
| chat_memory.py | 349 | Every exchange | Stores query+trace+response triplets |
| squirrel.py | 257 | `[SQUIRREL ...]` | Temporal recall |
| memory_grep.py | 376 | `[GREP term="..."]` | BM25 keyword search |
| scoring.py | 260 | Training mode | 3-dimension scoring UI |
| hybrid_search.py | 394 | Staged | Semantic + BM25 + RRF fusion (not yet wired) |

### Support (ingestion-time)
| File | Lines | Does |
|------|-------|------|
| heuristic_enricher.py | 472 | Fast signal extraction, 21 domains |
| streaming_cluster.py | 411 | HDBSCAN + River DBSTREAM |
| dedup.py | 283 | ID + content hash dedup |
| cluster_schema.py | 533 | Cluster labeling/navigation |

### Frontend
```
frontend/src/
├── routes/+page.svelte      # Main dashboard
└── lib/
    ├── stores/              # theme, websocket, session, artifacts, panels
    ├── threlte/             # 3D: Scene, CoreBrain, MemoryNode, ConnectionLines
    └── components/          # ArtifactPane, AnalyticsDashboard, FloatingPanel
```

## Data Structures (minimal)

```python
MemoryNode: id, conversation_id, created_at, human_content, assistant_content, cluster_id, tags
EpisodicMemory: id, title, summary, messages, created_at, dominant_intent
ReasoningTrace: id, query, memories_retrieved, memories_cited, response, score, feedback_notes
ChatExchange: id, timestamp, query, response, trace_id, rating
CognitiveOutput: id, timestamp, thought_type, content, source_memory_ids, confidence
```

## Data Layout (v2.0 Unified)
```
data/
├── corpus/                    # Unified corpus (single files)
│   ├── nodes.json            # All MemoryNodes
│   ├── episodes.json         # All EpisodicMemories
│   └── dedup_index.json      # Content hashes
├── vectors/
│   ├── nodes.npy             # (N, 1024) float32
│   └── episodes.npy
├── indexes/
│   ├── faiss.index
│   └── clusters.json
├── reasoning_traces/*.json   # Per-trace (live writes)
├── chat_exchanges/*.json     # SQUIRREL store
└── manifest.json             # v2.0 manifest

Note: Legacy structure still supported for migration.
```

## Config (config.yaml)
```yaml
provider: xai                    # or anthropic
model: grok-4-fast-reasoning
max_tokens: 8192
retrieval: {process_top_k: 10, episodic_top_k: 5, min_score: 0.3}
training: {enabled: true, quick_mode: true}
feedback_injection: {enabled: true, exemplar_top_k: 3, min_score: 0.7}
```

## Env
```
XAI_API_KEY=...        # Primary (Grok)
ANTHROPIC_API_KEY=...  # Fallback
OPENAI_API_KEY=...     # Embedding fallback
TEI_URL=...            # Optional local embedder
```

## Quick Test
```bash
python cog_twin.py ./data
# Commands: /status, /health, /feedback <1-5>, /quit
```

## Modification Safety
- **RED**: schemas.py (4+ importers, data foundation)
- **YELLOW**: heuristic_enricher.py (3 importers), embedder.py (4 importers)
- **GREEN**: Everything else (1-2 importers, clean interfaces)

## Current Phase
Phase 8: Metacognitive Dashboard. Analytics panel live, WebSocket broadcasting session state.
Next: Production hardening (Docker, Postgres, auth).
