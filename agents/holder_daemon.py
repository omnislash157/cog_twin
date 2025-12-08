"""
Holder Daemons - Persistent Grok sessions for project-duration context.

KEY INSIGHT: JSON logs are ground truth. Holders are VIEWS over that truth.

- CODE_HOLDER: Current codebase snapshot (wiped/reinjected per wave)
- CONVO_HOLDER: Queries JSON logs via persistence layer
- USER_HOLDER: Bridges to venom_voice with SQUIRREL tool
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .registry import spawn_agent
from .schemas import AgentRole, now_iso
from .persistence import SwarmPersistence


@dataclass
class DaemonMessage:
    """A message in the daemon conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=now_iso)


class CodeHolderDaemon:
    """
    Grok daemon holding current codebase snapshot.

    Lifecycle:
    - Wave start: refresh_from_sandbox() wipes and reinjects current files
    - During wave: inject_file() adds new files as EXECUTOR writes them
    - Wave end: (nothing - next wave will refresh)

    This keeps CODE_HOLDER seeing a consistent snapshot, not accumulated history.
    """

    def __init__(self, sandbox_root: Path):
        self.sandbox_root = Path(sandbox_root)
        self.files: Dict[str, str] = {}
        self.history: List[DaemonMessage] = []
        self.initialized = False
        self._lock = asyncio.Lock()

    async def refresh_from_sandbox(self) -> str:
        """
        Called at START of each wave.
        Wipes old context, injects current sandbox state.
        """
        async with self._lock:
            # Clear history (new Grok soldier)
            self.history = []
            self.files = {}

            # Load current files from sandbox
            exclude = {"__pycache__", ".git", ".venv", "node_modules", ".svelte-kit"}
            patterns = ["*.py", "*.ts", "*.svelte", "*.json", "*.yaml", "*.md", "*.toml"]

            for pattern in patterns:
                for filepath in self.sandbox_root.rglob(pattern):
                    if any(exc in str(filepath) for exc in exclude):
                        continue
                    try:
                        rel_path = str(filepath.relative_to(self.sandbox_root)).replace("\\", "/")
                        self.files[rel_path] = filepath.read_text(encoding="utf-8")
                    except Exception:
                        pass

            # Build structured injection
            manifest = self._build_manifest()
            core_modules = self._build_core_modules()

            # Inject manifest first
            await self._inject(manifest, "CODEBASE_MANIFEST")

            # Inject core modules
            if core_modules:
                await self._inject(core_modules, "CORE_MODULES")

            self.initialized = True
            print(f"[CODE_HOLDER] Refreshed: {len(self.files)} files")
            return f"CodeHolder initialized with {len(self.files)} files"

    def _build_manifest(self) -> str:
        """File list with sizes - cheap to inject."""
        lines = [f"CODEBASE MANIFEST ({len(self.files)} files):"]
        for path in sorted(self.files.keys()):
            line_count = len(self.files[path].split("\n"))
            lines.append(f"  {path} ({line_count} lines)")
        return "\n".join(lines)

    def _build_core_modules(self) -> str:
        """Inject key files in full. Others loaded on demand."""
        core_patterns = ["main.py", "config.py", "schemas.py", "registry.py",
                        "orchestrator.py", "app.py", "routes.py"]
        parts = []
        for path, file_content in self.files.items():
            if any(p in path for p in core_patterns):
                # Truncate very large files
                content = file_content
                if len(file_content) > 10000:
                    content = file_content[:10000] + "\n\n... [TRUNCATED - " + str(len(file_content)) + " total chars]"
                parts.append("=== " + path + " ===\n" + content)
        return "\n\n".join(parts) if parts else ""

    async def inject_file(self, path: str, content: str) -> str:
        """Inject a new or updated file (called when EXECUTOR writes)."""
        async with self._lock:
            self.files[path] = content
            await self._inject("=== " + path + " ===\n" + content, f"FILE_UPDATE:{path}")
            return f"Injected {path}"

    async def query(self, question: str) -> str:
        """Query about the codebase."""
        async with self._lock:
            # Check if asking about specific file not yet in context
            for path in self.files:
                if path in question and path not in self._get_injected_files():
                    await self._inject(
                        "=== " + path + " ===\n" + self.files[path],
                        f"LAZY_LOAD:{path}"
                    )

            return await self._query_grok(question)

    def _get_injected_files(self) -> set:
        """Track which files have been fully injected."""
        injected = set()
        for msg in self.history:
            if "===" in msg.content:
                matches = re.findall(r"=== (.+?) ===", msg.content)
                injected.update(matches)
        return injected

    async def _inject(self, content: str, label: str) -> None:
        """Inject content into Grok context."""
        self.history.append(DaemonMessage(role="user", content="[" + label + "]\n" + content))
        response = await self._query_grok(f"Acknowledge receipt of {label}. One line only.")
        self.history.append(DaemonMessage(role="assistant", content=response))

    async def _query_grok(self, question: str) -> str:
        """Query Grok with conversation history."""
        context_parts = []
        for msg in self.history[-30:]:
            context_parts.append("[" + msg.role.upper() + "]\n" + msg.content)
        context = "\n\n".join(context_parts) if context_parts else ""
        return await spawn_agent(AgentRole.CODE_HOLDER, question, context=context)

    def get_file_local(self, path: str) -> Optional[str]:
        """Quick local lookup without Grok."""
        return self.files.get(path)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": "CODE_HOLDER",
            "initialized": self.initialized,
            "file_count": len(self.files),
            "history_length": len(self.history)
        }



class ConvoHolderDaemon:
    """
    Grok daemon for querying conversation history.

    KEY INSIGHT: Does NOT hold raw logs. QUERIES JSON via persistence layer.
    """

    def __init__(self, persistence: SwarmPersistence, project_id: str):
        self.persistence = persistence
        self.project_id = project_id
        self.current_wave = "001"
        self.initialized = False

    async def init(self) -> str:
        """Initialize the daemon."""
        self.initialized = True
        print("[CONVO_HOLDER] Initialized (queries JSON logs)")
        return "ConvoHolder ready"

    def set_wave(self, wave: str) -> None:
        """Update current wave reference."""
        self.current_wave = wave

    async def query(self, question: str, wave: str = None, mode: str = "summary") -> str:
        """Query conversation history."""
        if wave is None or wave == "current":
            wave = self.current_wave
        elif wave == "last":
            wave = self._previous_wave()

        context = self._build_wave_context(wave, mode)
        return await spawn_agent(AgentRole.CONVO_HOLDER, question, context=context)

    def _previous_wave(self) -> str:
        """Get previous wave number."""
        waves = self.persistence.list_waves(self.project_id)
        if len(waves) < 2:
            return self.current_wave
        return waves[-2]

    def _build_wave_context(self, wave: str, mode: str) -> str:
        """Build structured context from JSON logs."""
        parts = [f"WAVE {wave} COMMUNICATIONS:"]

        turns = self.persistence.load_wave_turns(self.project_id, wave)

        if not turns:
            parts.append("(no communications recorded)")
        else:
            for turn in turns:
                seq = turn.get("sequence", "?")
                from_a = turn.get("from_agent", "?")
                to_a = turn.get("to_agent", "?")
                direction = turn.get("direction", "?")
                tokens = turn.get("tokens_out", turn.get("tokens_in", 0))

                if mode == "summary":
                    if direction == "outbound":
                        preview = turn.get("user_prompt", "")[:150]
                    else:
                        preview = turn.get("raw_response", "")[:150]
                    preview = preview.replace("\n", " ")
                    parts.append(f"[seq {seq}] {from_a} -> {to_a}: {preview}... ({tokens} tok)")
                else:
                    parts.append("\n=== SEQ " + str(seq) + ": " + from_a + " -> " + to_a + " ===")
                    if direction == "outbound":
                        parts.append("PROMPT:\n" + turn.get("user_prompt", "")[:2000])
                    else:
                        parts.append("RESPONSE:\n" + turn.get("raw_response", "")[:2000])
                        parsed = turn.get("parsed", {})
                        if parsed:
                            parts.append(f"PARSED: {json.dumps(parsed)}")

        failures = self.persistence.load_failures(self.project_id)
        wave_failures = [f for f in failures if f.wave == wave]

        if wave_failures:
            parts.append("\nFAILURES:")
            for f in wave_failures:
                parts.append(f"- {f.failure_type}: {f.root_cause}")
                parts.append(f"  Agent blamed: {f.agent_blamed}")
                parts.append(f"  Recommendation: {f.recommendation}")

        summary = self.persistence.load_wave_summary(self.project_id, wave)
        if summary:
            parts.append("\nWAVE SUMMARY:")
            parts.append(f"- Verdict: {summary.verdict}")
            parts.append(f"- Task: {summary.task}")
            parts.append(f"- Files modified: {summary.files_modified}")
            parts.append(f"- Summary: {summary.summary}")

        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        waves = self.persistence.list_waves(self.project_id)
        return {
            "name": "CONVO_HOLDER",
            "initialized": self.initialized,
            "project_id": self.project_id,
            "current_wave": self.current_wave,
            "total_waves": len(waves)
        }



class UserHolderDaemon:
    """
    USER_HOLDER - Bridge to venom_voice / CogTwin.

    Has SQUIRREL tool for temporal recency queries.
    Ground truth for user intent and preferences.
    """

    def __init__(self):
        self.initialized = False
        self.venom_endpoint = None

    async def init(self) -> str:
        """Initialize connection to venom_voice."""
        self.initialized = True
        print("[USER_HOLDER] Initialized (standalone mode)")
        return "UserHolder ready"

    async def query(self, question: str) -> str:
        """Query user intent/preferences."""
        context = """You have SQUIRREL access to check recent user statements.
For this query, consider:
- User stated preferences and patterns
- Recent decisions and directions
- Project goals and constraints

Answer definitively as the user voice."""

        return await spawn_agent(AgentRole.USER_HOLDER, question, context=context)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": "USER_HOLDER",
            "initialized": self.initialized,
            "mode": "standalone"
        }


class DaemonManager:
    """
    Manages lifecycle of all holder daemons for a project.

    Hides sharding/token management from Claude.
    Claude just sees: CODE_HOLDER, CONVO_HOLDER, USER_HOLDER
    """

    def __init__(self):
        self.code_holder: Optional[CodeHolderDaemon] = None
        self.convo_holder: Optional[ConvoHolderDaemon] = None
        self.user_holder: Optional[UserHolderDaemon] = None
        self.project_id: Optional[str] = None
        self.persistence: Optional[SwarmPersistence] = None

    async def start(
        self,
        sandbox_root: Path,
        persistence: SwarmPersistence,
        project_id: str
    ) -> Dict[str, str]:
        """
        Spin up all daemons for a project.
        Called once at project start.
        """
        self.project_id = project_id
        self.persistence = persistence
        results = {}

        self.code_holder = CodeHolderDaemon(sandbox_root)
        results["code_holder"] = await self.code_holder.refresh_from_sandbox()

        self.convo_holder = ConvoHolderDaemon(persistence, project_id)
        results["convo_holder"] = await self.convo_holder.init()

        self.user_holder = UserHolderDaemon()
        results["user_holder"] = await self.user_holder.init()

        print(f"[DAEMONS] All holders ready for project {project_id}")
        return results

    async def refresh_code_for_wave(self, wave: str) -> str:
        """Called at START of each wave."""
        if self.convo_holder:
            self.convo_holder.set_wave(wave)

        if self.code_holder:
            return await self.code_holder.refresh_from_sandbox()

        return "No code_holder to refresh"

    async def inject_file(self, path: str, content: str) -> str:
        """Inject new file into CODE_HOLDER (called when EXECUTOR writes)."""
        if self.code_holder:
            return await self.code_holder.inject_file(path, content)
        return "No code_holder"

    async def query_code(self, question: str) -> str:
        """Query CODE_HOLDER."""
        if self.code_holder:
            return await self.code_holder.query(question)
        return "CODE_HOLDER not initialized"

    async def query_convo(self, question: str, wave: str = None, mode: str = "summary") -> str:
        """Query CONVO_HOLDER."""
        if self.convo_holder:
            return await self.convo_holder.query(question, wave=wave, mode=mode)
        return "CONVO_HOLDER not initialized"

    async def query_user(self, question: str) -> str:
        """Query USER_HOLDER."""
        if self.user_holder:
            return await self.user_holder.query(question)
        return "USER_HOLDER not initialized"

    async def shutdown(self) -> None:
        """Shut down all daemons."""
        self.code_holder = None
        self.convo_holder = None
        self.user_holder = None
        print(f"[DAEMONS] Shut down for project {self.project_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get stats for dashboard."""
        return {
            "project_id": self.project_id,
            "code_holder": self.code_holder.get_stats() if self.code_holder else None,
            "convo_holder": self.convo_holder.get_stats() if self.convo_holder else None,
            "user_holder": self.user_holder.get_stats() if self.user_holder else None,
        }
