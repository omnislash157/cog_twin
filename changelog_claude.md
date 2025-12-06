# Changelog

> 2-4 lines per session. Read CONTEXT.md for architecture.

## 2025-12-06 - stoic-fermat (AcidBurn Pipeline Phase 1)
- acidburn_pipeline/ingest.py: Stream HuggingFace Gutenberg → chunk to 2K tokens → jsonl
- Threshold-based retrieval pipeline for massive datasets (books, papers, company docs)
- Usage: `python -m acidburn_pipeline.ingest --limit 100 --output chunks.jsonl`
- Test: 100 books → 7,814 chunks, ~15.5M tokens

---

## 2025-12-06 - brave-boyd (Swarm Bugfixes + Pipeline Debug Tools)
- sandbox_executor.py: Added cd, npm, node, npx to COMMAND_WHITELIST (was blocking frontend commands)
- registry.py: Increased CONFIG/EXECUTOR max_tokens 4096/8192 → 16384 (was truncating large scaffolds)
- registry.py: Added BULK OPERATIONS instruction to CONFIG prompt (chunking for large files)
- Root cause: EXECUTOR JSON tool_calls were valid, but `cd frontend && npm install` failed whitelist check
- debug_pipeline.py: Trace what model sees (retrieval results, formatted memories, episodic section)
- read_traces.py: CLI to browse reasoning traces and chat exchanges
- agents/SWARM.md: AI cold start doc for swarm architecture

### read_traces.py Usage
```bash
python read_traces.py              # Last 5 exchanges
python read_traces.py -t           # Show traces instead
python read_traces.py -q "invoice" # Search by query
python read_traces.py -v -n 3      # Verbose, last 3
```

---

## 2025-12-05 - clever-lichterman (Phase 10f: Swarm UI Launcher)
- SwarmPanel.svelte: Added launcher form (project name, goal, tasks input) with submit to /api/swarm/start
- swarm.ts: Added handlers for swarm_diagnostic and swarm_consultation WebSocket events
- UI shows launcher when idle, hides when swarm active; all existing dashboard features preserved

---

## 2025-12-05 - gracious-archimedes (Phase 10e: Resilience Loop)
- diagnostic.py: CONFIG DIAGNOSTIC mode with agent awareness block, holder queries, diagnosis parsing
- consultation.py: Cross-agent consultation flow (EXECUTOR/REVIEWER/QUALITY_GATE input on failures)
- swarm_orchestrator.py: handle_tool_failure() decision tree, consultation aggregation, retry with context
- persistence.py: Diagnosis storage + mark_consultation_helpful() for learning which agents give useful advice
- test_diagnostic.py: 26 tests for diagnostic prompts, consultation parsing, confidence aggregation
- WebSocket events: swarm_diagnostic, swarm_consultation for live dashboard visibility

---

## 2025-12-05 - gracious-archimedes (Phase 10d: Sandbox Executor)
- sandbox_executor.py: Isolated execution with Python 3.11 venv (hdbscan compat), self-healing retry loop (max 3), HITL for unknown packages
- promote_cli.py: Human review workflow for Skynet Procedure (review/approve/reject/rollback)
- registry.py: EXECUTOR now emits <tool_calls> for write_file, run_command, etc.
- swarm_orchestrator.py: Wired sandbox execution between EXECUTOR and REVIEWER, promotion staging on PASS
- test_sandbox_executor.py: 18 tests covering tool parsing, sandbox writes, commands, promotions
- Note: Sandbox venv uses Python 3.11 specifically (auto-detects via py launcher or known paths)

---

## 2025-12-05 - intelligent-lamport (Phase 10c: Swarm Dashboard)
- swarm_orchestrator.py: WebSocket broadcasting (BroadcastCallback, _broadcast_turn, _broadcast_wave_start/end, _broadcast_failure)
- backend/app/main.py: /ws/swarm endpoint + SwarmConnectionManager + /api/swarm/start API
- frontend/src/lib/stores/swarm.ts: Full state management for live agent visualization
- frontend/src/lib/components/SwarmPanel.svelte: Agent cards, progress bar, reasoning display, code preview
- frontend/src/lib/threlte/AgentNode.svelte + SwarmSpace.svelte: 3D agent visualization with status pulses
- test_swarm_dashboard.py: Integration test for WebSocket broadcasts

---

## 2025-12-05 - ecstatic-wescoff (Phase 10b: Swarm Infrastructure)
- schemas.py: Unified data structures (OutboundTurn, InboundTurn, WaveSummary, Failure, Project)
- persistence.py: JSON storage for projects, waves, failures (agents/data/projects/)
- reasoning.py: Extract <reasoning> traces from model responses
- holders.py: CodeHolder + ConvoHolder for queryable context (get_file, search, summarize)
- swarm_orchestrator.py: Full persistence orchestrator with QUALITY_GATE agent
- registry.py: Added REASONING_INSTRUCTION to all prompts, new QUALITY_GATE role

---

## 2025-12-05 - busy-robinson (Phase 10a: Multi-Agent Swarm MVP)
- agents/ module: 4-agent wave loop (CONFIG → EXECUTOR → REVIEWER → ORCHESTRATOR)
- File reading: read_project_files() scans .py files, passes to CONFIG for context
- Surgical edits: extract_and_save_files() handles ADD_ENDPOINT, MODIFY_EXISTING, NEW_FILE
- Sandbox at agents/sandbox/ for safe testing (.gitignored)

---

## 2025-12-05 - graceful-turing (Phase 9a)
- Swipeable workspaces: tabs, dots, context menu (rename/delete)
- Panel layouts persist per workspace (localStorage)
- Fixed .gitignore (was hiding frontend/src/lib/)

---

## 2025-12-05 - practical-hawking (metacognitive dashboard)
- MetacognitiveMirror wired: get_session_analytics(), format_analytics_block()
- 7 new /api/analytics/* endpoints, WebSocket broadcasts session_analytics
- Frontend: AnalyticsDashboard.svelte with phase indicator, stability gauge, drift alerts, suggestions
- hybrid_search.py staged (Semantic + BM25 + RRF, not yet on hot path)

## 2025-12-03 - unified-corpus
- ingest.py v2.0: Unified incremental ingestion, dedup-before-embedding
- New structure: corpus/*.json, vectors/*.npy, indexes/*, manifest.json
- retrieval.py: Dual-mode loading (unified preferred, legacy fallback)

---

## 2025-12-02 - optimistic-wozniak (hot context + trust hierarchy + parallel tools)
- Phase 3 (Silent Orchestration) REMOVED: caused hallucinations when retrieval was sparse
- LLMs fabricated details (names, dates, percentages) instead of admitting gaps
- Rolled back all silent orchestration code from venom_voice.py, cog_twin.py, main.py, frontend
- Re-added `_parallel_tool_search()` as INFRASTRUCTURE only (no prompt policy changes)
- Tools now auto-fire on every query - Grok always has data without deciding to call tools

### Phase 2: Trust Hierarchy + Position Bias
- Auto-inject last 1h of session context into every query (USER TRUTH > TOOL TRUTH)
- Position bias: Reordered prompt - tool results FIRST, user context LAST (recency wins)
- Added explicit TRUST HIERARCHY block with conflict resolution pattern
- Aggressive labeling: VECTOR=LOW, EPISODIC=MEDIUM, GREP=LOWEST, USER=LAW

### Diff: cog_twin.py
```python
# After session_memories retrieval (~line 439):
+ hot_context = self.squirrel.execute(SquirrelQuery(timeframe="-60min"), limit=15)

# In VoiceContext construction (~line 550):
+ hot_context=hot_context,  # Last 1h of session - highest trust
```

### Diff: venom_voice.py - SYSTEM_TEMPLATE restructure
```python
# REORDERED for position bias (tools first, user last):
- {hot_context_section}  # Was after USER PROFILE
- RETRIEVED MEMORIES: {memories}
- SESSION CONTEXT: {session_context}

+ SEARCH RESULTS (UNVERIFIED - MAY BE STALE/WRONG): {memories}
+ SESSION CONTEXT: {session_context}
+ {hot_context_section}  # NOW at END for recency bias

# NEW TRUST HIERARCHY block after OPERATIONAL PARAMETERS:
+ TRUST HIERARCHY - ABSOLUTE (MEMORIZE THIS):
+ 1. What user said THIS SESSION → Absolute truth
+ 2. What user said LAST HOUR → Near-absolute truth
+ 3. SQUIRREL results → High trust (temporal recall)
+ 4. EPISODIC results → Medium trust (may be old context)
+ 5. VECTOR results → Low trust (topically similar ≠ factually relevant)
+ 6. GREP results → Verification only (word frequency ≠ meaning)
+
+ CONFLICT RESOLUTION PATTERN:
+ "You said [X]. Search found [Y]. Since you stated [X], I'll use that as ground truth."
```

### Diff: venom_voice.py - Aggressive labeling
```python
# _format_memories():
- "VECTOR RETRIEVAL (semantic similarity)"
+ "VECTOR RETRIEVAL (UNVERIFIED - TOPICAL MATCH ONLY)"
+ "Trust: LOW - topically similar ≠ factually relevant"
+ "DO NOT use these to override what user said THIS SESSION"

- "EPISODIC RETRIEVAL (conversation-level context)"
+ "EPISODIC RETRIEVAL (MEDIUM TRUST - MAY BE OUTDATED)"

# _format_grep_results():
- "GREP RESULTS (supplementary to your episodic context)"
+ "GREP RESULTS (VERIFICATION ONLY - WORD FREQUENCY ≠ MEANING)"
+ "Trust: LOWEST - word counts don't establish facts"
+ "WARNING: User said something? GREP cannot contradict it."

# _format_hot_context():
+ "═" * 60  # Double-line border for visual weight
+ "USER GROUND TRUTH (LAST 1 HOUR) - THIS IS LAW"
+ "CONFLICT RESOLUTION: If search says X but user said Y, USER WINS."
```

---

## 2025-12-02 - strange-bassi (production hardening)
- Unified corpus structure: corpus/, vectors/, indexes/, manifest.json
- Refactored ingest.py v2.0: dedup BEFORE embedding, incremental merge
- Created migrate_to_unified.py: consolidates 23K scattered nodes → single files
- Fixed retrieval.py: loads unified or legacy manifests

## 2025-12-01 - housekeeping
- Consolidated docs: COLD_START.md + 4 wiring maps → CONTEXT.md (~130 lines)
- Purged: cognitive_agent.py, cognitive_twin.py (metacognitive_mirror.py now wired)

## 2025-11-30 - brave-johnson + artifact-bank
- Fixed 12s latency: removed fake streaming, direct WebSocket chunks
- Phase 7 complete: Threlte 3D visualization operational

## 2025-11-30 - memory-lane-grok
- Phase 6 complete: SQUIRREL temporal tool, ChatMemoryStore, Grok 4.1 Fast swap
- 5-lane retrieval operational, 2M context window

## 2025-11-29 - exciting-mendel
- Grep demoted to supplementary tool, reasoning-first protocol enforced

## 2025-11-29 - peaceful-volhard  
- Phase 5.5: Retrieval provenance separation (LIVE > GREP > EPISODIC > VECTOR)
- dedup.py added, traces stream to memory_pipeline

---

*Older history archived. System locked for production prep.*
