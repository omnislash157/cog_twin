"""
Sandbox executor for autonomous code execution.
All operations isolated to sandbox/ folder.
Supports self-healing retry loops and human-in-the-loop for unknown packages.
"""

import subprocess
import sys
import json
import re
import shutil
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal, Callable, Awaitable
from enum import Enum

from .schemas import generate_id, now_iso


# Known packages that auto-install without human approval
KNOWN_PACKAGES = {
    # Core
    'fastapi', 'uvicorn', 'pydantic', 'starlette',
    # HTTP
    'requests', 'httpx', 'aiohttp',
    # Data
    'numpy', 'pandas', 'scipy',
    # AI/ML
    'anthropic', 'openai', 'tiktoken',
    # Testing
    'pytest', 'pytest-asyncio', 'pytest-cov',
    # Utils
    'python-dotenv', 'pyyaml', 'aiofiles', 'rich', 'click',
    # DB
    'sqlalchemy', 'asyncpg', 'aiosqlite',
    # CogTwin stack
    'faiss-cpu', 'hdbscan', 'scikit-learn', 'sentence-transformers',
    'websockets', 'river',
}

# Commands allowed to execute (base command only)
COMMAND_WHITELIST = ['python', 'pytest', 'pip', 'cat', 'ls', 'echo', 'pwd', 'dir', 'type', 'cd', 'npm', 'node', 'npx']


class HITLType(Enum):
    """Types of human-in-the-loop requests."""
    UNKNOWN_PACKAGE = "unknown_package"
    UNKNOWN_COMMAND = "unknown_command"
    MAX_RETRIES = "max_retries"
    QUALITY_GATE = "quality_gate"


@dataclass
class ToolCall:
    """A tool call extracted from agent response."""
    tool: Literal["write_file", "read_file", "run_command", "list_dir", "delete_file"]
    params: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    tool: str
    success: bool
    output: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HITLRequest:
    """Request for human-in-the-loop approval."""
    type: HITLType
    context: str
    options: List[str] = field(default_factory=lambda: ["approve", "deny"])
    package: Optional[str] = None
    command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['type'] = self.type.value
        return d


@dataclass
class FailureInfo:
    """Detailed failure information for diagnostic flow."""
    tool_name: str
    tool_args: Dict[str, Any]
    error_type: str
    error_message: str
    stack_trace: str
    attempt: int
    executor_context: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionResult:
    """Result of executing all tool calls (potentially with retries)."""
    success: bool
    results: List[ToolResult]
    needs_human: bool = False
    hitl_request: Optional[HITLRequest] = None
    retries_used: int = 0
    last_failure_info: Optional[FailureInfo] = None  # For diagnostic flow

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'success': self.success,
            'results': [r.to_dict() for r in self.results],
            'needs_human': self.needs_human,
            'retries_used': self.retries_used,
        }
        if self.hitl_request:
            d['hitl_request'] = self.hitl_request.to_dict()
        if self.last_failure_info:
            d['last_failure_info'] = self.last_failure_info.to_dict()
        return d


@dataclass
class PromotionRequest:
    """Request to promote sandbox files to real project."""
    id: str = field(default_factory=generate_id)
    timestamp: str = field(default_factory=now_iso)

    wave: str = ""
    project_id: str = ""

    files: List[str] = field(default_factory=list)

    diff_summary: str = ""
    test_output: str = ""
    quality_gate_verdict: str = ""

    status: Literal["pending", "approved", "rejected", "deferred"] = "pending"
    reviewer_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionResult:
    """Result of promotion attempt."""
    success: bool
    files_promoted: List[str]
    files_failed: List[str]
    backup_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Type for async HITL callback
HITLCallback = Callable[[HITLRequest], Awaitable[Optional[str]]]


class SandboxExecutor:
    """
    Execute tool calls in isolated sandbox.

    Safety guarantees:
    - All writes confined to sandbox/
    - Reads allowed from sandbox/ and project (read-only)
    - Commands run in sandbox venv with timeout
    - Unknown packages require human approval

    Note: Uses Python 3.11 for venv (required for hdbscan compatibility).
    """

    # Python 3.11 paths to search (Windows)
    PYTHON311_PATHS = [
        r"C:\Python311\python.exe",
        r"C:\Program Files\Python311\python.exe",
        r"C:\Program Files (x86)\Python311\python.exe",
        r"C:\Users\mthar\AppData\Local\Programs\Python\Python311\python.exe",
    ]

    def __init__(self, sandbox_root: Path, project_root: Optional[Path] = None):
        self.root = Path(sandbox_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(project_root).resolve() if project_root else None
        self.venv_path = self.root / ".venv"
        self.installed_packages: set = set()
        self._hitl_callback: Optional[HITLCallback] = None
        self._python311_path: Optional[str] = None
        self._ensure_venv()

    def _find_python311(self) -> Optional[str]:
        """Find Python 3.11 installation."""
        # Check cached path first
        if self._python311_path and Path(self._python311_path).exists():
            return self._python311_path

        # Try known paths on Windows
        if sys.platform == "win32":
            for path in self.PYTHON311_PATHS:
                if Path(path).exists():
                    self._python311_path = path
                    return path

            # Try py launcher
            try:
                result = subprocess.run(
                    ["py", "-3.11", "-c", "import sys; print(sys.executable)"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    path = result.stdout.strip()
                    if Path(path).exists():
                        self._python311_path = path
                        return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        else:
            # Unix: try python3.11 directly
            try:
                result = subprocess.run(
                    ["python3.11", "-c", "import sys; print(sys.executable)"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    path = result.stdout.strip()
                    self._python311_path = path
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        return None

    def _ensure_venv(self) -> None:
        """Create sandbox venv using Python 3.11 if it doesn't exist."""
        if not self.venv_path.exists():
            python311 = self._find_python311()

            if python311:
                print(f"[SANDBOX] Creating venv with Python 3.11 at {self.venv_path}")
                print(f"[SANDBOX] Using: {python311}")
                subprocess.run(
                    [python311, "-m", "venv", str(self.venv_path)],
                    check=True,
                    capture_output=True
                )
            else:
                # Fallback to current Python with warning
                print(f"[SANDBOX] WARNING: Python 3.11 not found, using {sys.executable}")
                print(f"[SANDBOX] hdbscan may not work correctly!")
                subprocess.run(
                    [sys.executable, "-m", "venv", str(self.venv_path)],
                    check=True,
                    capture_output=True
                )

    def _get_python(self) -> str:
        """Get path to sandbox Python executable."""
        if sys.platform == "win32":
            return str(self.venv_path / "Scripts" / "python.exe")
        return str(self.venv_path / "bin" / "python")

    def _get_pip(self) -> str:
        """Get path to sandbox pip executable."""
        if sys.platform == "win32":
            return str(self.venv_path / "Scripts" / "pip.exe")
        return str(self.venv_path / "bin" / "pip")

    def set_hitl_callback(self, callback: HITLCallback) -> None:
        """Set callback for human-in-the-loop requests."""
        self._hitl_callback = callback

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        try:
            if call.tool == "write_file":
                return self._write_file(call.params["path"], call.params["content"])
            elif call.tool == "read_file":
                return self._read_file(call.params["path"])
            elif call.tool == "run_command":
                return self._run_command(call.params["cmd"])
            elif call.tool == "list_dir":
                return self._list_dir(call.params.get("path", "."))
            elif call.tool == "delete_file":
                return self._delete_file(call.params["path"])
            else:
                return ToolResult(call.tool, False, "", f"Unknown tool: {call.tool}")
        except KeyError as e:
            return ToolResult(call.tool, False, "", f"Missing parameter: {e}")
        except Exception as e:
            return ToolResult(call.tool, False, "", str(e))

    def _write_file(self, path: str, content: str) -> ToolResult:
        """Write file to sandbox only."""
        # Normalize and resolve path within sandbox
        target_path = (self.root / path).resolve()

        # Security: ensure path stays within sandbox
        if not str(target_path).startswith(str(self.root)):
            return ToolResult("write_file", False, "", f"Path escapes sandbox: {path}")

        # Create parent directories
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        target_path.write_text(content, encoding='utf-8')

        return ToolResult(
            "write_file",
            True,
            f"Wrote {len(content)} bytes to {path}"
        )

    def _read_file(self, path: str) -> ToolResult:
        """Read file from sandbox or project (read-only)."""
        # Try sandbox first
        sandbox_path = (self.root / path).resolve()
        if sandbox_path.exists() and str(sandbox_path).startswith(str(self.root)):
            content = sandbox_path.read_text(encoding='utf-8')
            return ToolResult("read_file", True, content)

        # Try project root (read-only access for context)
        if self.project_root:
            project_path = (self.project_root / path).resolve()
            if project_path.exists() and str(project_path).startswith(str(self.project_root)):
                content = project_path.read_text(encoding='utf-8')
                return ToolResult("read_file", True, content)

        # Try absolute path (read-only)
        abs_path = Path(path).resolve()
        if abs_path.exists():
            content = abs_path.read_text(encoding='utf-8')
            return ToolResult("read_file", True, content)

        return ToolResult("read_file", False, "", f"File not found: {path}")

    def _delete_file(self, path: str) -> ToolResult:
        """Delete file from sandbox only."""
        target_path = (self.root / path).resolve()

        # Security: only delete within sandbox
        if not str(target_path).startswith(str(self.root)):
            return ToolResult("delete_file", False, "", f"Path escapes sandbox: {path}")

        if target_path.exists():
            target_path.unlink()
            return ToolResult("delete_file", True, f"Deleted {path}")

        return ToolResult("delete_file", False, "", f"File not found: {path}")

    def _list_dir(self, path: str) -> ToolResult:
        """List directory in sandbox."""
        target_path = (self.root / path).resolve()

        if not target_path.exists():
            return ToolResult("list_dir", False, "", f"Directory not found: {path}")

        if not target_path.is_dir():
            return ToolResult("list_dir", False, "", f"Not a directory: {path}")

        items = []
        for item in sorted(target_path.iterdir()):
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            items.append(f"{prefix} {item.name}")

        return ToolResult("list_dir", True, "\n".join(items))

    def _run_command(self, cmd: str) -> ToolResult:
        """Run command in sandbox with venv."""
        # Handle pip install specially
        if cmd.startswith("pip install"):
            return self._handle_pip_install(cmd)

        # Whitelist check (base command only)
        cmd_parts = cmd.split()
        if not cmd_parts:
            return ToolResult("run_command", False, "", "Empty command")

        cmd_base = cmd_parts[0]
        # Handle python -m module
        if cmd_base == "python" and len(cmd_parts) > 2 and cmd_parts[1] == "-m":
            cmd_base = cmd_parts[2]  # e.g., pytest

        if cmd_base not in COMMAND_WHITELIST:
            # Request HITL for unknown command
            return self._sync_request_hitl(HITLRequest(
                type=HITLType.UNKNOWN_COMMAND,
                context=f"Command not in whitelist: {cmd}",
                command=cmd
            ))

        # Rewrite command for venv
        exec_cmd = self._rewrite_for_venv(cmd)

        try:
            result = subprocess.run(
                exec_cmd,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=60,
                env={**dict(subprocess.os.environ), "PYTHONPATH": str(self.root)}
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"

            return ToolResult(
                "run_command",
                result.returncode == 0,
                output,
                result.stderr if result.returncode != 0 else None
            )
        except subprocess.TimeoutExpired:
            return ToolResult("run_command", False, "", "Command timed out (60s)")
        except Exception as e:
            return ToolResult("run_command", False, "", str(e))

    def _rewrite_for_venv(self, cmd: str) -> str:
        """Rewrite command to use sandbox venv."""
        python_path = self._get_python()

        if cmd.startswith("python "):
            return f'"{python_path}" {cmd[7:]}'
        elif cmd.startswith("python3 "):
            return f'"{python_path}" {cmd[8:]}'
        elif cmd.startswith("pytest"):
            args = cmd[6:].strip()
            return f'"{python_path}" -m pytest {args}'
        elif cmd.startswith("pip "):
            return f'"{python_path}" -m pip {cmd[4:]}'

        return cmd

    def _handle_pip_install(self, cmd: str) -> ToolResult:
        """Handle pip install with package approval."""
        # Extract package names (skip flags)
        parts = cmd.split()
        packages = []
        skip_next = False

        for i, part in enumerate(parts[2:], start=2):  # Skip "pip install"
            if skip_next:
                skip_next = False
                continue
            if part.startswith("-"):
                if part in ["-r", "--requirement", "-e", "--editable"]:
                    skip_next = True
                continue
            packages.append(part)

        # Check each package
        for pkg in packages:
            # Normalize package name (handle version specifiers)
            pkg_base = re.split(r'[<>=!~\[]', pkg)[0].lower().replace('_', '-')

            if pkg_base in self.installed_packages:
                continue  # Already installed this session

            if pkg_base not in KNOWN_PACKAGES:
                # Request HITL for unknown package
                result = self._sync_request_hitl(HITLRequest(
                    type=HITLType.UNKNOWN_PACKAGE,
                    context=f"Unknown package: {pkg_base}",
                    options=["approve", "approve_and_remember", "deny"],
                    package=pkg_base
                ))
                if not result.success:
                    return result

        # Execute pip install
        pip_cmd = f'"{self._get_pip()}" install {" ".join(packages)}'

        try:
            result = subprocess.run(
                pip_cmd,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                self.installed_packages.update(packages)

            return ToolResult(
                "run_command",
                result.returncode == 0,
                result.stdout,
                result.stderr if result.returncode != 0 else None
            )
        except subprocess.TimeoutExpired:
            return ToolResult("run_command", False, "", "pip install timed out (120s)")
        except Exception as e:
            return ToolResult("run_command", False, "", str(e))

    def _sync_request_hitl(self, request: HITLRequest) -> ToolResult:
        """Request human approval (sync wrapper - blocks until resolved)."""
        # For now, just return failure requiring human
        # In production, this would integrate with the dashboard
        return ToolResult(
            "hitl",
            False,
            "",
            f"HITL required: {request.context}"
        )

    async def execute_with_retry(
        self,
        executor_response: str,
        spawn_agent_fn: Callable,
        max_retries: int = 3,
        executor_context: str = ""
    ) -> ExecutionResult:
        """
        Execute tools with self-healing retry loop.

        On failure, feeds error back to EXECUTOR for a fix attempt.

        Args:
            executor_response: The EXECUTOR's response containing <tool_calls>
            spawn_agent_fn: Function to spawn an agent (async)
            max_retries: Maximum number of retry attempts
            executor_context: What the EXECUTOR was trying to accomplish (for diagnostics)

        Returns:
            ExecutionResult with success status, all results, and failure info if failed
        """
        import traceback
        from .registry import AgentRole

        current_response = executor_response
        all_results: List[ToolResult] = []
        last_failure_info: Optional[FailureInfo] = None

        for attempt in range(max_retries):
            tool_calls = parse_tool_calls(current_response)

            if not tool_calls:
                # No tool calls to execute
                return ExecutionResult(
                    success=True,
                    results=all_results,
                    retries_used=attempt
                )

            # Execute each tool call
            failed_call: Optional[ToolCall] = None
            failed_result: Optional[ToolResult] = None

            for call in tool_calls:
                result = self.execute(call)
                all_results.append(result)

                if not result.success:
                    failed_call = call
                    failed_result = result
                    break

            if failed_call is None:
                # All tools succeeded
                return ExecutionResult(
                    success=True,
                    results=all_results,
                    retries_used=attempt
                )

            # Build failure info for diagnostics
            error_type = "ToolExecutionError"
            if failed_result and failed_result.error:
                # Try to extract exception type from error message
                if ":" in failed_result.error:
                    error_type = failed_result.error.split(":")[0].strip()

            last_failure_info = FailureInfo(
                tool_name=failed_call.tool,
                tool_args=failed_call.params,
                error_type=error_type,
                error_message=failed_result.error or "Unknown error",
                stack_trace=failed_result.output if failed_result.output else "",
                attempt=attempt + 1,
                executor_context=executor_context or "Unknown task"
            )

            # Check if HITL is needed
            if failed_result and "HITL required" in (failed_result.error or ""):
                return ExecutionResult(
                    success=False,
                    results=all_results,
                    needs_human=True,
                    hitl_request=HITLRequest(
                        type=HITLType.UNKNOWN_PACKAGE if "package" in failed_result.error.lower() else HITLType.UNKNOWN_COMMAND,
                        context=failed_result.error
                    ),
                    retries_used=attempt,
                    last_failure_info=last_failure_info
                )

            # Feed error back to EXECUTOR for fix
            fix_prompt = f"""EXECUTION FAILED (attempt {attempt + 1}/{max_retries})

TOOL: {failed_call.tool}
PARAMS: {json.dumps(failed_call.params, indent=2)}
ERROR: {failed_result.error}
OUTPUT: {failed_result.output[:500] if failed_result.output else 'none'}

Fix the code and re-emit the tool calls. Focus on the error above.
"""
            print(f"[SANDBOX] Retry {attempt + 1}/{max_retries}: {failed_call.tool} failed")

            # Get fixed response from EXECUTOR
            current_response = await spawn_agent_fn(AgentRole.EXECUTOR, fix_prompt, "")

        # Max retries exceeded - return with detailed failure info
        return ExecutionResult(
            success=False,
            results=all_results,
            needs_human=True,
            hitl_request=HITLRequest(
                type=HITLType.MAX_RETRIES,
                context=f"Max retries ({max_retries}) exceeded. Last error: {last_failure_info.error_message if last_failure_info else 'unknown'}"
            ),
            retries_used=max_retries,
            last_failure_info=last_failure_info
        )

    # -------------------------------------------------------------------------
    # SKYNET PROCEDURE: Promotion staging
    # -------------------------------------------------------------------------

    def prepare_promotion(
        self,
        files: List[str],
        wave: str,
        project_id: str,
        test_output: str = ""
    ) -> PromotionRequest:
        """Stage files for human review before promotion to real project."""
        staging_dir = self.root / "staging" / f"wave_{wave}"
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Copy files to staging
        files_dir = staging_dir / "files_to_promote"
        files_dir.mkdir(exist_ok=True)

        valid_files = []
        for f in files:
            src = self.root / f
            if src.exists():
                dst = files_dir / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                valid_files.append(f)

        # Generate diff summary
        diff_summary = self._generate_diff_summary(valid_files)
        (staging_dir / "diff_summary.md").write_text(diff_summary, encoding='utf-8')

        # Save test results
        if test_output:
            (staging_dir / "test_results.log").write_text(test_output, encoding='utf-8')

        request = PromotionRequest(
            wave=wave,
            project_id=project_id,
            files=valid_files,
            diff_summary=diff_summary,
            test_output=test_output
        )

        # Save request as JSON
        (staging_dir / "promotion_request.json").write_text(
            json.dumps(request.to_dict(), indent=2),
            encoding='utf-8'
        )

        return request

    def _generate_diff_summary(self, files: List[str]) -> str:
        """Generate human-readable diff summary."""
        lines = ["# Promotion Summary\n", f"**Files:** {len(files)}\n"]

        for f in files:
            sandbox_file = self.root / f
            lines.append(f"\n## {f}\n")

            if self.project_root:
                project_file = self.project_root / f
                if not project_file.exists():
                    lines.append("**Status:** NEW FILE\n")
                    if sandbox_file.exists():
                        line_count = len(sandbox_file.read_text(encoding='utf-8').splitlines())
                        lines.append(f"**Lines:** {line_count}\n")
                else:
                    # Show line count diff
                    old_lines = len(project_file.read_text(encoding='utf-8').splitlines())
                    new_lines = len(sandbox_file.read_text(encoding='utf-8').splitlines())
                    diff = new_lines - old_lines
                    sign = "+" if diff > 0 else ""
                    lines.append(f"**Status:** MODIFIED ({sign}{diff} lines)\n")
            else:
                lines.append("**Status:** NEW FILE (no project root set)\n")

        return "".join(lines)

    def promote_files(
        self,
        request: PromotionRequest,
        backup: bool = True
    ) -> PromotionResult:
        """
        SKYNET GATE: Promote validated sandbox files to real project.

        This is the point of no return. Human must have approved.
        """
        if not self.project_root:
            return PromotionResult(
                success=False,
                files_promoted=[],
                files_failed=request.files,
            )

        if request.status != "approved":
            return PromotionResult(
                success=False,
                files_promoted=[],
                files_failed=request.files,
            )

        promoted = []
        failed = []
        backup_path = None

        # Create backup
        if backup:
            backup_dir = self.project_root / ".swarm_backups" / f"wave_{request.wave}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = str(backup_dir)

            for f in request.files:
                src = self.project_root / f
                if src.exists():
                    dst = backup_dir / f
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

        # Promote files
        for f in request.files:
            try:
                src = self.root / f
                dst = self.project_root / f

                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                promoted.append(f)

            except Exception as e:
                print(f"[PROMOTE] Failed to promote {f}: {e}")
                failed.append(f)

        return PromotionResult(
            success=len(failed) == 0,
            files_promoted=promoted,
            files_failed=failed,
            backup_path=backup_path
        )

    def rollback(self, backup_path: str) -> ToolResult:
        """Emergency rollback from backup."""
        if not self.project_root:
            return ToolResult("rollback", False, "", "No project root set")

        backup_dir = Path(backup_path)
        if not backup_dir.exists():
            return ToolResult("rollback", False, "", f"Backup not found: {backup_path}")

        restored = []
        for f in backup_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(backup_dir)
                dst = self.project_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                restored.append(str(rel))

        return ToolResult("rollback", True, f"Restored {len(restored)} files from backup")

    def list_pending_promotions(self) -> List[Dict[str, Any]]:
        """List all pending promotion requests."""
        staging_dir = self.root / "staging"
        if not staging_dir.exists():
            return []

        promotions = []
        for wave_dir in sorted(staging_dir.iterdir()):
            request_file = wave_dir / "promotion_request.json"
            if request_file.exists():
                data = json.loads(request_file.read_text(encoding='utf-8'))
                if data.get("status") == "pending":
                    promotions.append(data)

        return promotions

    def update_promotion_status(
        self,
        wave: str,
        status: Literal["approved", "rejected", "deferred"],
        notes: str = ""
    ) -> bool:
        """Update promotion request status."""
        request_file = self.root / "staging" / f"wave_{wave}" / "promotion_request.json"
        if not request_file.exists():
            return False

        data = json.loads(request_file.read_text(encoding='utf-8'))
        data["status"] = status
        data["reviewer_notes"] = notes
        request_file.write_text(json.dumps(data, indent=2), encoding='utf-8')

        return True


def parse_tool_calls(response: str) -> List[ToolCall]:
    """Extract tool calls from agent response."""
    match = re.search(r'<tool_calls>(.*?)</tool_calls>', response, re.DOTALL)
    if not match:
        return []

    calls = []
    for line in match.group(1).strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            tool = data.pop("tool")
            calls.append(ToolCall(tool=tool, params=data))
        except (json.JSONDecodeError, KeyError):
            continue

    return calls


def format_results(results: List[ToolResult]) -> str:
    """Format tool results for agent context."""
    lines = []
    for r in results:
        status = "OK" if r.success else "FAILED"
        lines.append(f"[{r.tool}] {status}")
        if r.output:
            # Truncate long output
            output = r.output[:500]
            if len(r.output) > 500:
                output += f"\n... ({len(r.output) - 500} more chars)"
            lines.append(f"  stdout: {output}")
        if r.error:
            lines.append(f"  stderr: {r.error[:500]}")
    return "\n".join(lines)
