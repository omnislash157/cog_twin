"""
Holder agents with query protocol.
Orchestrator queries these instead of holding full context.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class HolderQuery:
    """Query to a holder agent."""
    action: str  # get_file, get_files, search, summarize, get_structure
    target: str | List[str]  # file path, search term, etc.
    truncate: str = "none"  # none, first_500_lines, summary_only, structure_only


@dataclass
class HolderResponse:
    """Response from a holder agent."""
    found: bool
    content: str
    metadata: Dict[str, Any]
    tokens_returned: int


class CodeHolder:
    """
    Holds full project codebase.
    Queryable for files, structure, summaries.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.files: Dict[str, str] = {}
        self.load_project()

    def load_project(self) -> None:
        """Load all project files into memory."""
        patterns = ["*.py", "*.ts", "*.svelte", "*.json", "*.yaml", "*.md"]
        exclude = ["node_modules", "__pycache__", ".git", "venv", ".venv", ".svelte-kit", "data"]

        for pattern in patterns:
            for filepath in self.project_root.rglob(pattern):
                if any(exc in str(filepath) for exc in exclude):
                    continue
                try:
                    rel_path = str(filepath.relative_to(self.project_root))
                    # Normalize path separators
                    rel_path = rel_path.replace("\\", "/")
                    self.files[rel_path] = filepath.read_text(encoding='utf-8')
                except Exception:
                    pass

    def query(self, q: HolderQuery) -> HolderResponse:
        """Handle a query."""
        if q.action == "get_file":
            return self._get_file(q.target, q.truncate)
        elif q.action == "get_files":
            return self._get_files(q.target, q.truncate)
        elif q.action == "search":
            return self._search(q.target)
        elif q.action == "summarize":
            return self._summarize(q.target, q.truncate)
        elif q.action == "get_structure":
            return self._get_structure()
        else:
            return HolderResponse(False, f"Unknown action: {q.action}", {}, 0)

    def _get_file(self, path: str, truncate: str) -> HolderResponse:
        # Normalize path
        path = path.replace("\\", "/")
        if path not in self.files:
            return HolderResponse(False, f"File not found: {path}", {}, 0)

        content = self.files[path]

        if truncate == "first_500_lines":
            lines = content.split('\n')[:500]
            content = '\n'.join(lines)
            if len(self.files[path].split('\n')) > 500:
                content += f"\n\n... [TRUNCATED - {len(self.files[path].split(chr(10)))} total lines]"
        elif truncate == "summary_only":
            content = self._summarize_file(path, content)

        return HolderResponse(
            found=True,
            content=content,
            metadata={"lines": len(content.split('\n')), "path": path},
            tokens_returned=len(content) // 4  # Rough estimate
        )

    def _get_files(self, paths: List[str], truncate: str) -> HolderResponse:
        results = []
        for path in paths:
            resp = self._get_file(path, truncate)
            if resp.found:
                results.append(f"=== {path} ===\n{resp.content}")

        content = "\n\n".join(results)
        return HolderResponse(True, content, {"file_count": len(results)}, len(content) // 4)

    def _search(self, term: str) -> HolderResponse:
        matches = []
        for path, content in self.files.items():
            if term.lower() in content.lower():
                # Find matching lines
                for i, line in enumerate(content.split('\n'), 1):
                    if term.lower() in line.lower():
                        matches.append(f"{path}:{i}: {line.strip()}")

        content = "\n".join(matches[:50])  # Limit results
        return HolderResponse(True, content, {"match_count": len(matches)}, len(content) // 4)

    def _summarize(self, target: str, truncate: str) -> HolderResponse:
        if target == "all" or target == "":
            return self._get_structure()

        # Summarize specific directory
        matching = {k: v for k, v in self.files.items() if k.startswith(target)}

        if truncate == "structure_only":
            content = "\n".join(sorted(matching.keys()))
        else:
            summaries = []
            for path, code in matching.items():
                summaries.append(f"=== {path} ===\n{self._summarize_file(path, code)}")
            content = "\n\n".join(summaries)

        return HolderResponse(True, content, {"file_count": len(matching)}, len(content) // 4)

    def _get_structure(self) -> HolderResponse:
        """Return project file tree."""
        tree = sorted(self.files.keys())
        content = "PROJECT STRUCTURE:\n" + "\n".join(tree)
        return HolderResponse(True, content, {"file_count": len(tree)}, len(content) // 4)

    def _summarize_file(self, path: str, content: str) -> str:
        """Extract function/class signatures from a file."""
        lines = content.split('\n')
        summary_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('async def '):
                summary_lines.append(line)
            elif stripped.startswith('class '):
                summary_lines.append(line)
            elif stripped.startswith('@'):
                summary_lines.append(line)
            elif stripped.startswith('from ') or stripped.startswith('import '):
                summary_lines.append(line)

        return '\n'.join(summary_lines) if summary_lines else "[No signatures found]"

    def update_file(self, path: str, content: str) -> None:
        """Update a file in memory and on disk."""
        # Normalize path
        path = path.replace("\\", "/")
        self.files[path] = content
        filepath = self.project_root / path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding='utf-8')

    def add_file(self, path: str, content: str) -> None:
        """Add a new file."""
        self.update_file(path, content)

    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        path = path.replace("\\", "/")
        return path in self.files

    def get_file_content(self, path: str) -> Optional[str]:
        """Get file content directly."""
        path = path.replace("\\", "/")
        return self.files.get(path)


class ConvoHolder:
    """
    Holds all agent conversation history.
    Queryable for wave summaries, search, etc.
    """

    def __init__(self, persistence, project_id: str):
        self.persistence = persistence
        self.project_id = project_id

    def query(self, q: HolderQuery) -> HolderResponse:
        if q.action == "get_wave":
            return self._get_wave(q.target)
        elif q.action == "get_failures":
            return self._get_failures()
        elif q.action == "summarize_all":
            return self._summarize_all(q.truncate)
        elif q.action == "search":
            return self._search(q.target)
        else:
            return HolderResponse(False, f"Unknown action: {q.action}", {}, 0)

    def _get_wave(self, wave: str) -> HolderResponse:
        summary = self.persistence.load_wave_summary(self.project_id, wave)
        if not summary:
            return HolderResponse(False, f"Wave {wave} not found", {}, 0)

        content = f"""Wave {wave}:
Task: {summary.task}
Verdict: {summary.verdict}
Files modified: {summary.files_modified}
Summary: {summary.summary}
"""
        return HolderResponse(True, content, {"wave": wave}, len(content) // 4)

    def _get_failures(self) -> HolderResponse:
        context = self.persistence.get_failure_context(self.project_id)
        return HolderResponse(True, context, {}, len(context) // 4)

    def _summarize_all(self, truncate: str) -> HolderResponse:
        waves = self.persistence.list_waves(self.project_id)

        summaries = []
        limit = 3 if truncate == "last_3_waves" else len(waves)

        for wave in waves[-limit:]:
            summary = self.persistence.load_wave_summary(self.project_id, wave)
            if summary:
                summaries.append(f"Wave {wave}: {summary.verdict.upper()} - {summary.summary}")

        content = "\n\n".join(summaries) if summaries else "No waves completed yet."
        return HolderResponse(True, content, {"wave_count": len(summaries)}, len(content) // 4)

    def _search(self, term: str) -> HolderResponse:
        # Search through wave summaries
        waves = self.persistence.list_waves(self.project_id)
        matches = []

        for wave in waves:
            summary = self.persistence.load_wave_summary(self.project_id, wave)
            if summary and term.lower() in summary.summary.lower():
                matches.append(f"Wave {wave}: {summary.summary}")

        content = "\n".join(matches) if matches else f"No matches for '{term}'"
        return HolderResponse(True, content, {"match_count": len(matches)}, len(content) // 4)
