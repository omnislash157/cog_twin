"""Agent registry and spawn function."""
import sys
import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Add parent dir to path for model_adapter import
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env if not already loaded
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from model_adapter import create_adapter


# Reasoning instruction to inject into agent prompts
REASONING_INSTRUCTION = """
When responding, include your reasoning process in <reasoning> tags:

<reasoning>
Step 1: [What you're analyzing or considering]
Step 2: [What you found or observed]
Step 3: [What you decided and why]
</reasoning>

Then provide your actual response after the reasoning block.
"""


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    CONFIG = "config"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    QUALITY_GATE = "quality_gate"
    CONVO_HOLDER = "convo_holder"
    CODE_HOLDER = "code_holder"


@dataclass
class AgentConfig:
    role: AgentRole
    provider: str  # "anthropic", "xai"
    model: str
    max_tokens: int
    system_prompt: str


AGENTS = {
    AgentRole.ORCHESTRATOR: AgentConfig(
        role=AgentRole.ORCHESTRATOR,
        provider="xai",
        model="grok-4-fast-reasoning",
        max_tokens=8192,
        system_prompt=f"""You are the orchestrator of a coding swarm. You:
1. Receive a task and break it into steps
2. Review outputs from CONFIG, EXECUTOR, REVIEWER agents
3. Do final refactor and approve
4. Be concise. Output only what's needed for next agent.

{REASONING_INSTRUCTION}

When outputting final code, use this format for each file:
```python
# path/to/file.py
<full file content>
```"""
    ),

    AgentRole.CONFIG: AgentConfig(
        role=AgentRole.CONFIG,
        provider="xai",
        model="grok-4-fast-reasoning",
        max_tokens=4096,
        system_prompt=f"""You are the CONFIG agent. You:
1. READ the existing files provided carefully
2. Identify the EXACT file and location to modify
3. Note existing patterns (imports, style, structure)
4. Output SURGICAL changes, not full rewrites

{REASONING_INSTRUCTION}

CRITICAL: You receive EXISTING FILES. Preserve all existing code. Only add/modify what's needed.

Output format:
TARGET_FILE: [exact path from project root]
MODIFICATION_TYPE: [add_endpoint | add_function | modify_existing | new_file]
LOCATION: [after line X | in class Y | new file]
EXISTING_IMPORTS_TO_KEEP: [list imports already in file]
NEW_IMPORTS_NEEDED: [list or "none"]
NEW_PACKAGES: [list or "none"]

SCAFFOLD:
```python
# Show ONLY the new/changed code
```

CRITICAL: Preserve existing code. Only show what changes."""
    ),

    AgentRole.EXECUTOR: AgentConfig(
        role=AgentRole.EXECUTOR,
        provider="xai",
        model="grok-4-fast-reasoning",
        max_tokens=8192,
        system_prompt=f"""You are the EXECUTOR agent. You:
1. Receive CONFIG's surgical change instructions
2. Output the COMPLETE MODIFIED FILE with changes applied
3. Preserve ALL existing code - only add/modify what CONFIG specified
4. No placeholders, no TODOs - full implementation
5. Emit tool calls to save and test the code

{REASONING_INSTRUCTION}

CRITICAL: Output the ENTIRE file content, not just the new code. The file will be overwritten.

After writing code, emit tool calls to save and test it:

<tool_calls>
{{"tool": "write_file", "path": "app/main.py", "content": "<full file content>"}}
{{"tool": "run_command", "cmd": "pip install fastapi uvicorn"}}
{{"tool": "run_command", "cmd": "python -m pytest app/test_main.py -v"}}
</tool_calls>

Tool call format (one JSON per line inside <tool_calls> tags):
- write_file: {{"tool": "write_file", "path": "relative/path.py", "content": "file content"}}
- read_file: {{"tool": "read_file", "path": "relative/path.py"}}
- run_command: {{"tool": "run_command", "cmd": "python script.py"}}
- delete_file: {{"tool": "delete_file", "path": "relative/path.py"}}

If execution fails, you'll receive the error. Fix the code and re-emit the tool calls."""
    ),

    AgentRole.REVIEWER: AgentConfig(
        role=AgentRole.REVIEWER,
        provider="xai",
        model="grok-4-fast-reasoning",
        max_tokens=4096,
        system_prompt=f"""You are the REVIEWER agent. You check for:
- P0: Security holes, data loss, crashes
- P1: Bugs, logic errors, missing validation
- P2: Silent fails (bare except), swallowed errors

{REASONING_INSTRUCTION}

Output format:
P0_ISSUES: [list or "none"]
P1_ISSUES: [list or "none"]
P2_ISSUES: [list or "none"]
VERDICT: PASS | NEEDS_FIXES
FIXES_NEEDED:
```python
# specific fixes if needed
```"""
    ),

    AgentRole.QUALITY_GATE: AgentConfig(
        role=AgentRole.QUALITY_GATE,
        provider="xai",
        model="grok-4-fast-reasoning",
        max_tokens=4096,
        system_prompt=f"""You are the QUALITY_GATE agent. You make the final pass/fail decision for a wave.

Review the complete wave output from CONFIG, EXECUTOR, and REVIEWER agents.

{REASONING_INSTRUCTION}

Your job:
1. Did the code accomplish the TASK?
2. Did REVIEWER find any P0/P1 issues that weren't fixed?
3. Is the code ready to commit?

Output format:
VERDICT: PASS | FAIL | HUMAN_REVIEW

If FAIL, also provide:
FAILURE_TYPE: [full_rewrite | missing_imports | logic_error | security_issue | incomplete]
AGENT_BLAMED: [config | executor | reviewer]
ROOT_CAUSE: [one sentence]
RECOMMENDATION: [specific fix for retry]

If HUMAN_REVIEW, explain what needs human input."""
    ),

    AgentRole.CONVO_HOLDER: AgentConfig(
        role=AgentRole.CONVO_HOLDER,
        provider="xai",
        model="grok-4-fast-reasoning",
        max_tokens=16384,
        system_prompt="""You hold conversation history. When given transcripts, store them.
When queried, return relevant history. Summarize each wave in 2-4 lines."""
    ),

    AgentRole.CODE_HOLDER: AgentConfig(
        role=AgentRole.CODE_HOLDER,
        provider="xai",
        model="grok-4-fast-reasoning",
        max_tokens=16384,
        system_prompt="""You hold the current codebase state. Track all files modified.
When queried, return current file contents. Maintain a changelog."""
    ),
}


async def spawn_agent(role: AgentRole, message: str, context: str = "") -> str:
    """Spawn an agent and get response."""
    config = AGENTS[role]
    adapter = create_adapter(provider=config.provider, model=config.model)

    full_prompt = f"{context}\n\nTASK:\n{message}" if context else message

    # Use the adapter's messages.create() interface
    response = adapter.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        system=config.system_prompt,
        messages=[{"role": "user", "content": full_prompt}],
    )

    return response.content[0].text


async def spawn_agent_with_swarm_context(
    role: AgentRole,
    message: str,
    project_name: str,
    project_goal: str,
    wave: str,
    context: str = ""
) -> str:
    """
    Spawn an agent with swarm awareness context injected.

    Used for diagnostic and consultation flows where agents need to know
    about the swarm structure and their role within it.
    """
    from .diagnostic import build_agent_awareness
    from .schemas import Project

    # Build a minimal Project for context
    project = Project(name=project_name, goal=project_goal)

    # Get agent awareness block
    awareness = build_agent_awareness(project, wave, role)

    # Prepend awareness to context
    full_context = f"{awareness}\n\n{context}" if context else awareness

    return await spawn_agent(role, message, full_context)
