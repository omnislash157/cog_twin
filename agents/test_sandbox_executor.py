"""
Tests for SandboxExecutor.
Run: python -m pytest agents/test_sandbox_executor.py -v
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from .sandbox_executor import (
    SandboxExecutor, ToolCall, parse_tool_calls, format_results,
    KNOWN_PACKAGES, COMMAND_WHITELIST
)


@pytest.fixture
def sandbox():
    """Create a temporary sandbox for testing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="sandbox_test_"))
    project_dir = temp_dir / "project"
    sandbox_dir = temp_dir / "sandbox"
    project_dir.mkdir()
    sandbox_dir.mkdir()

    # Create a sample project file
    (project_dir / "existing.py").write_text("# Existing file\nprint('hello')\n")

    executor = SandboxExecutor(sandbox_dir, project_dir)
    yield executor

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestToolCallParsing:
    """Test parsing tool calls from agent responses."""

    def test_parse_single_write(self):
        response = '''
Here's the code:
<tool_calls>
{"tool": "write_file", "path": "app/main.py", "content": "print('hello')"}
</tool_calls>
'''
        calls = parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].tool == "write_file"
        assert calls[0].params["path"] == "app/main.py"
        assert calls[0].params["content"] == "print('hello')"

    def test_parse_multiple_calls(self):
        response = '''
<tool_calls>
{"tool": "write_file", "path": "a.py", "content": "# a"}
{"tool": "run_command", "cmd": "python a.py"}
{"tool": "read_file", "path": "b.py"}
</tool_calls>
'''
        calls = parse_tool_calls(response)
        assert len(calls) == 3
        assert calls[0].tool == "write_file"
        assert calls[1].tool == "run_command"
        assert calls[2].tool == "read_file"

    def test_parse_no_tool_calls(self):
        response = "Just some text without tool calls."
        calls = parse_tool_calls(response)
        assert len(calls) == 0

    def test_parse_empty_tool_calls(self):
        response = "<tool_calls>\n\n</tool_calls>"
        calls = parse_tool_calls(response)
        assert len(calls) == 0


class TestSandboxWriteFile:
    """Test write_file tool."""

    def test_write_file_in_sandbox(self, sandbox):
        call = ToolCall(tool="write_file", params={
            "path": "test.py",
            "content": "print('test')"
        })
        result = sandbox.execute(call)

        assert result.success
        assert "Wrote" in result.output
        assert (sandbox.root / "test.py").exists()
        assert (sandbox.root / "test.py").read_text() == "print('test')"

    def test_write_file_creates_dirs(self, sandbox):
        call = ToolCall(tool="write_file", params={
            "path": "deep/nested/dir/test.py",
            "content": "# deep file"
        })
        result = sandbox.execute(call)

        assert result.success
        assert (sandbox.root / "deep/nested/dir/test.py").exists()

    def test_write_file_blocks_escape(self, sandbox):
        call = ToolCall(tool="write_file", params={
            "path": "../outside.py",
            "content": "# malicious"
        })
        result = sandbox.execute(call)

        assert not result.success
        assert "escapes sandbox" in result.error


class TestSandboxReadFile:
    """Test read_file tool."""

    def test_read_sandbox_file(self, sandbox):
        # First write a file
        (sandbox.root / "readme.txt").write_text("Hello world")

        call = ToolCall(tool="read_file", params={"path": "readme.txt"})
        result = sandbox.execute(call)

        assert result.success
        assert result.output == "Hello world"

    def test_read_project_file(self, sandbox):
        call = ToolCall(tool="read_file", params={"path": "existing.py"})
        result = sandbox.execute(call)

        assert result.success
        assert "Existing file" in result.output

    def test_read_missing_file(self, sandbox):
        call = ToolCall(tool="read_file", params={"path": "missing.py"})
        result = sandbox.execute(call)

        assert not result.success
        assert "not found" in result.error


class TestSandboxCommands:
    """Test run_command tool."""

    def test_python_version(self, sandbox):
        call = ToolCall(tool="run_command", params={"cmd": "python --version"})
        result = sandbox.execute(call)

        assert result.success
        assert "Python" in result.output or "Python" in (result.error or "")

    def test_blocked_command(self, sandbox):
        call = ToolCall(tool="run_command", params={"cmd": "curl http://example.com"})
        result = sandbox.execute(call)

        assert not result.success
        assert "HITL required" in result.error


class TestKnownPackages:
    """Test package approval logic."""

    def test_known_packages_defined(self):
        assert "fastapi" in KNOWN_PACKAGES
        assert "pytest" in KNOWN_PACKAGES
        assert "numpy" in KNOWN_PACKAGES

    def test_command_whitelist(self):
        assert "python" in COMMAND_WHITELIST
        assert "pytest" in COMMAND_WHITELIST
        assert "pip" in COMMAND_WHITELIST


class TestFormatResults:
    """Test result formatting for agent context."""

    def test_format_success(self):
        from .sandbox_executor import ToolResult
        results = [
            ToolResult("write_file", True, "Wrote 100 bytes"),
            ToolResult("run_command", True, "Tests passed"),
        ]
        output = format_results(results)

        assert "[write_file] OK" in output
        assert "[run_command] OK" in output
        assert "Wrote 100 bytes" in output

    def test_format_failure(self):
        from .sandbox_executor import ToolResult
        results = [
            ToolResult("run_command", False, "", "ModuleNotFoundError: No module named 'missing'"),
        ]
        output = format_results(results)

        assert "[run_command] FAILED" in output
        assert "ModuleNotFoundError" in output


class TestPromotion:
    """Test Skynet Procedure promotion staging."""

    def test_prepare_promotion(self, sandbox):
        # Write a file to sandbox
        (sandbox.root / "app" / "main.py").parent.mkdir(parents=True, exist_ok=True)
        (sandbox.root / "app" / "main.py").write_text("# New code")

        request = sandbox.prepare_promotion(
            files=["app/main.py"],
            wave="001",
            project_id="test_project",
            test_output="All tests passed"
        )

        assert request.status == "pending"
        assert "app/main.py" in request.files
        assert (sandbox.root / "staging" / "wave_001" / "diff_summary.md").exists()
        assert (sandbox.root / "staging" / "wave_001" / "test_results.log").exists()

    def test_update_promotion_status(self, sandbox):
        # First prepare
        (sandbox.root / "test.py").write_text("# test")
        sandbox.prepare_promotion(["test.py"], "002", "proj", "")

        # Update status
        result = sandbox.update_promotion_status("002", "approved", "LGTM")
        assert result is True

        # Verify
        import json
        request_file = sandbox.root / "staging" / "wave_002" / "promotion_request.json"
        data = json.loads(request_file.read_text())
        assert data["status"] == "approved"
        assert data["reviewer_notes"] == "LGTM"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
