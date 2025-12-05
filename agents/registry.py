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


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    CONFIG = "config"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
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
        provider="xai",  # MVP: use Grok for all (no Anthropic key in .env)
        model="grok-4-fast-reasoning",
        max_tokens=8192,
        system_prompt="""You are the orchestrator of a coding swarm. You:
1. Receive a task and break it into steps
2. Review outputs from CONFIG, EXECUTOR, REVIEWER agents
3. Do final refactor and approve
4. Be concise. Output only what's needed for next agent.

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
        system_prompt="""You are the CONFIG agent. You:
1. Analyze the task for dependencies and imports needed
2. Check what files exist and what needs to be created
3. Output a scaffold: file paths, imports, function signatures
4. Flag any new packages needed (human must approve)

Output format:
FILES_NEEDED: [list]
IMPORTS: [list]
NEW_PACKAGES: [list or "none"]
SCAFFOLD:
```python
# code scaffold here
```"""
    ),

    AgentRole.EXECUTOR: AgentConfig(
        role=AgentRole.EXECUTOR,
        provider="xai",  # MVP: use Grok for all (no Anthropic key in .env)
        model="grok-4-fast-reasoning",
        max_tokens=8192,
        system_prompt="""You are the EXECUTOR agent. You:
1. Receive scaffold from CONFIG agent
2. Write complete, working code
3. No placeholders, no TODOs - full implementation
4. Follow existing code patterns in the project

Output the complete file(s) ready to save. Use this format:
```python
# path/to/file.py
<full file content>
```"""
    ),

    AgentRole.REVIEWER: AgentConfig(
        role=AgentRole.REVIEWER,
        provider="xai",
        model="grok-4-fast-reasoning",
        max_tokens=4096,
        system_prompt="""You are the REVIEWER agent. You check for:
- P0: Security holes, data loss, crashes
- P1: Bugs, logic errors, missing validation
- P2: Silent fails (bare except), swallowed errors

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
