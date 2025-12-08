"""
Tool executor for parsing and running EXECUTOR agent tool calls.
Parses <tool_calls> format and executes write_file, run_command, read_file.
"""
import re
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Single tool invocation."""
    name: str
    params: Dict[str, str]
    
    
@dataclass  
class ToolResult:
    """Result of tool execution."""
    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    file_path: Optional[str] = None
    

@dataclass
class ExecutionResult:
    """Complete execution result for a tool_calls block."""
    success: bool
    tool_results: List[ToolResult] = field(default_factory=list)
    files_written: List[str] = field(default_factory=list)
    error_summary: str = ""
    

def parse_tool_calls(response: str) -> List[ToolCall]:
    """
    Parse <tool_calls> block from EXECUTOR response.
    
    Expected format:
    <tool_calls>
    <tool name="write_file">
    <param name="path">file.py</param>
    <param name="content">...</param>
    </tool>
    </tool_calls>
    """
    tools = []
    
    # Find tool_calls block
    tc_match = re.search(r'<tool_calls>(.*?)</tool_calls>', response, re.DOTALL)
    if not tc_match:
        logger.warning("No <tool_calls> block found in response")
        return tools
    
    tc_content = tc_match.group(1)
    
    # Find each tool
    tool_pattern = r'<tool\s+name=["\'](\w+)["\']>(.*?)</tool>'
    for tool_match in re.finditer(tool_pattern, tc_content, re.DOTALL):
        tool_name = tool_match.group(1)
        tool_body = tool_match.group(2)
        
        # Extract params
        params = {}
        param_pattern = r'<param\s+name=["\'](\w+)["\']>(.*?)</param>'
        for param_match in re.finditer(param_pattern, tool_body, re.DOTALL):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            params[param_name] = param_value
            
        tools.append(ToolCall(name=tool_name, params=params))
        
    return tools


def fallback_code_block_parse(response: str) -> List[ToolCall]:
    """
    Fallback: parse raw code blocks if EXECUTOR doesn't use tool format.
    Converts ```python blocks with path comments to write_file tools.
    """
    tools = []
    
    # Find code blocks
    blocks = re.findall(r'```(?:python|py)?\n(.*?)```', response, re.DOTALL)
    
    for block in blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue
            
        # Check for path comment
        first_line = lines[0]
        if first_line.startswith('# ') and '.' in first_line:
            path = first_line[2:].strip()
            content = '\n'.join(lines[1:])
            tools.append(ToolCall(
                name="write_file",
                params={"path": path, "content": content}
            ))
            logger.info(f"Fallback parsed code block as write_file: {path}")
            
    return tools


class ToolExecutor:
    """
    Executes tool calls in a sandboxed environment.
    Files are written to sandbox_root, commands run with cwd=sandbox_root.
    """
    
    def __init__(
        self, 
        sandbox_root: Path,
        code_holder = None,
        max_retries: int = 3,
        allowed_commands: Optional[List[str]] = None
    ):
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.code_holder = code_holder
        self.max_retries = max_retries
        self.allowed_commands = allowed_commands or [
            'pip', 'python', 'pytest', 'ls', 'cat', 'head', 'tail', 'grep'
        ]
        
    def execute(self, response: str) -> ExecutionResult:
        """
        Parse and execute all tool calls from EXECUTOR response.
        Returns execution result with success/failure and details.
        """
        # Try tool_calls format first
        tools = parse_tool_calls(response)
        
        # Fallback to code blocks if no tools found
        if not tools:
            logger.info("No tool_calls found, trying fallback code block parse")
            tools = fallback_code_block_parse(response)
            
        if not tools:
            return ExecutionResult(
                success=False,
                error_summary="No tool calls or code blocks found in EXECUTOR output"
            )
        
        results = []
        files_written = []
        all_success = True
        
        for tool in tools:
            result = self._execute_tool(tool)
            results.append(result)
            
            if result.success and result.file_path:
                files_written.append(result.file_path)
            elif not result.success:
                all_success = False
                
        error_summary = ""
        if not all_success:
            failed = [r for r in results if not r.success]
            error_summary = "; ".join([f"{r.tool_name}: {r.error}" for r in failed])
            
        return ExecutionResult(
            success=all_success,
            tool_results=results,
            files_written=files_written,
            error_summary=error_summary
        )
    
    def _execute_tool(self, tool: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        try:
            if tool.name == "write_file":
                return self._write_file(tool.params)
            elif tool.name == "run_command":
                return self._run_command(tool.params)
            elif tool.name == "read_file":
                return self._read_file(tool.params)
            else:
                return ToolResult(
                    tool_name=tool.name,
                    success=False,
                    error=f"Unknown tool: {tool.name}"
                )
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ToolResult(
                tool_name=tool.name,
                success=False,
                error=str(e)
            )
    
    def _write_file(self, params: Dict[str, str]) -> ToolResult:
        """Write file to sandbox."""
        path = params.get("path", "")
        content = params.get("content", "")
        
        if not path:
            return ToolResult(
                tool_name="write_file",
                success=False,
                error="Missing 'path' parameter"
            )
            
        # Normalize path
        path = path.replace('\\', '/').lstrip('/')
        full_path = self.sandbox_root / path
        
        # Create parent dirs
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        full_path.write_text(content, encoding='utf-8')
        logger.info(f"[SANDBOX] Wrote: {full_path}")
        
        # Update code holder if available
        if self.code_holder:
            self.code_holder.update_file(path, content)
        
        return ToolResult(
            tool_name="write_file",
            success=True,
            output=f"Wrote {len(content)} bytes to {path}",
            file_path=path
        )
    
    def _run_command(self, params: Dict[str, str]) -> ToolResult:
        """Run shell command in sandbox."""
        cmd = params.get("cmd", "")
        
        if not cmd:
            return ToolResult(
                tool_name="run_command",
                success=False,
                error="Missing 'cmd' parameter"
            )
        
        # Security check - only allow certain commands
        cmd_name = cmd.split()[0]
        if cmd_name not in self.allowed_commands:
            return ToolResult(
                tool_name="run_command",
                success=False,
                error=f"Command '{cmd_name}' not in allowed list: {self.allowed_commands}"
            )
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.sandbox_root),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return ToolResult(
                    tool_name="run_command",
                    success=True,
                    output=result.stdout
                )
            else:
                return ToolResult(
                    tool_name="run_command",
                    success=False,
                    output=result.stdout,
                    error=result.stderr
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name="run_command",
                success=False,
                error="Command timed out after 60s"
            )
        except Exception as e:
            return ToolResult(
                tool_name="run_command",
                success=False,
                error=str(e)
            )
    
    def _read_file(self, params: Dict[str, str]) -> ToolResult:
        """Read file from sandbox or code holder."""
        path = params.get("path", "")
        
        if not path:
            return ToolResult(
                tool_name="read_file",
                success=False,
                error="Missing 'path' parameter"
            )
        
        # Try sandbox first
        full_path = self.sandbox_root / path
        if full_path.exists():
            content = full_path.read_text(encoding='utf-8')
            return ToolResult(
                tool_name="read_file",
                success=True,
                output=content
            )
        
        # Try code holder
        if self.code_holder:
            content = self.code_holder.get_file_content(path)
            if content:
                return ToolResult(
                    tool_name="read_file",
                    success=True,
                    output=content
                )
        
        return ToolResult(
            tool_name="read_file",
            success=False,
            error=f"File not found: {path}"
        )


def validate_executor_output(response: str) -> Tuple[bool, str]:
    """
    Validate EXECUTOR output format before execution.
    Returns (is_valid, error_message).
    """
    # Check for tool_calls
    if '<tool_calls>' in response:
        tools = parse_tool_calls(response)
        if not tools:
            return False, "Found <tool_calls> but failed to parse any tools"
        
        # Check each tool has required params
        for tool in tools:
            if tool.name == "write_file":
                if "path" not in tool.params:
                    return False, f"write_file missing 'path' param"
                if "content" not in tool.params:
                    return False, f"write_file missing 'content' param"
            elif tool.name == "run_command":
                if "cmd" not in tool.params:
                    return False, f"run_command missing 'cmd' param"
                    
        return True, ""
    
    # Check for fallback code blocks
    blocks = re.findall(r'```(?:python|py)?\n(.*?)```', response, re.DOTALL)
    if blocks:
        # Check at least one has path comment
        for block in blocks:
            lines = block.strip().split('\n')
            if lines and lines[0].startswith('# ') and '.' in lines[0]:
                return True, ""
        return False, "Code blocks found but none have path comment (# path/to/file.py)"
    
    return False, "No tool_calls or code blocks found"


# Quick test
if __name__ == "__main__":
    test_response = '''
<reasoning>
Step 1: Reading CONFIG instructions
Step 2: Merging changes into existing file
Step 3: Writing complete file via tool call
</reasoning>

<tool_calls>
<tool name="write_file">
<param name="path">app/main.py</param>
<param name="content">
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return {"uptime": 123, "queries": 456}
</param>
</tool>
</tool_calls>
'''
    
    tools = parse_tool_calls(test_response)
    print(f"Parsed {len(tools)} tools:")
    for t in tools:
        print(f"  - {t.name}: {list(t.params.keys())}")
    
    is_valid, error = validate_executor_output(test_response)
    print(f"Valid: {is_valid}, Error: {error}")