"""Wave-based orchestration loop."""
import asyncio
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from .registry import AgentRole, spawn_agent

# Packages already in this project - don't gate on these
KNOWN_PACKAGES = {
    'fastapi', 'uvicorn', 'pydantic', 'requests', 'numpy', 'faiss',
    'anthropic', 'openai', 'python-dotenv', 'aiofiles', 'httpx',
    'websockets', 'pyyaml', 'scipy', 'sklearn', 'hdbscan', 'river',
}


@dataclass
class WaveResult:
    wave_number: int
    task: str
    config_output: str
    executor_output: str
    reviewer_output: str
    final_output: str
    files_written: List[str] = field(default_factory=list)
    needs_human: bool = False
    human_reason: str = ""


@dataclass
class ProjectState:
    goal: str
    waves: List[WaveResult] = field(default_factory=list)
    convo_history: str = ""
    codebase_state: Dict[str, str] = field(default_factory=dict)


def extract_and_save_files(output: str, project_root: Path) -> Dict[str, str]:
    """Extract code blocks and save to files. Crude MVP parsing."""
    files = {}

    # Pattern: ```python or ```py followed by content
    blocks = re.findall(r'```(?:python|py)?\n(.*?)```', output, re.DOTALL)

    for i, block in enumerate(blocks):
        # Try to find filename in first line
        lines = block.strip().split('\n')
        if lines and lines[0].startswith('# ') and ('.' in lines[0]):
            # Extract filename, handle paths like "# backend/app/routes.py"
            filename = lines[0][2:].strip()
            content = '\n'.join(lines[1:])
        else:
            filename = f"output_{i}.py"
            content = block

        # Save file
        filepath = project_root / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding='utf-8')
        files[filename] = content
        print(f"  [FILE] Saved: {filepath}")

    return files


async def run_wave(
    task: str,
    project: ProjectState,
    wave_number: int,
    project_root: Path,
) -> WaveResult:
    """Execute one wave of the swarm."""

    print(f"\n{'='*60}")
    print(f"WAVE {wave_number}: {task[:50]}...")
    print('='*60)

    # Context from previous waves
    context = f"""PROJECT GOAL: {project.goal}

PREVIOUS WAVES: {len(project.waves)}
{project.convo_history[-2000:] if project.convo_history else 'None yet'}

CURRENT CODEBASE FILES: {list(project.codebase_state.keys())}
"""

    # 1. CONFIG AGENT
    print("\n[CONFIG] Analyzing dependencies and structure...")
    config_output = await spawn_agent(
        AgentRole.CONFIG,
        f"Task: {task}\n\nProject root: {project_root}",
        context
    )
    print(f"[CONFIG] Done. Output length: {len(config_output)}")

    # Check for new packages - gate for human (only truly new packages)
    if "NEW_PACKAGES:" in config_output:
        packages_section = config_output.split("NEW_PACKAGES:")[1][:200]
        # Get just the bracketed list part, stop at comment or newline
        first_line = packages_section.split('\n')[0].split('#')[0].strip().lower()
        if "none" not in first_line and first_line:
            # Extract package names from bracketed list like ['pkg1', 'pkg2']
            bracket_match = re.search(r'\[([^\]]+)\]', first_line)
            if bracket_match:
                inner = bracket_match.group(1)
                pkg_matches = re.findall(r"['\"](\w+)['\"]", inner)
                unknown_packages = [p for p in pkg_matches if p.lower() not in KNOWN_PACKAGES]
                if unknown_packages:
                    return WaveResult(
                        wave_number=wave_number,
                        task=task,
                        config_output=config_output,
                        executor_output="",
                        reviewer_output="",
                        final_output="",
                        needs_human=True,
                        human_reason=f"New packages needed: {unknown_packages}. Review CONFIG output."
                    )

    # 2. EXECUTOR AGENT
    print("\n[EXECUTOR] Writing code...")
    executor_output = await spawn_agent(
        AgentRole.EXECUTOR,
        f"Implement based on this scaffold:\n\n{config_output}",
        context
    )
    print(f"[EXECUTOR] Done. Output length: {len(executor_output)}")

    # 3. REVIEWER AGENT
    print("\n[REVIEWER] Reviewing code...")
    reviewer_output = await spawn_agent(
        AgentRole.REVIEWER,
        f"Review this code:\n\n{executor_output}",
        ""
    )
    verdict = "PASS" if "VERDICT: PASS" in reviewer_output else "NEEDS_FIXES"
    print(f"[REVIEWER] Done. Verdict: {verdict}")

    # 4. ORCHESTRATOR FINAL REFACTOR
    print("\n[ORCHESTRATOR] Final refactor...")
    final_output = await spawn_agent(
        AgentRole.ORCHESTRATOR,
        f"""Review complete. Apply any fixes needed and output final code.

EXECUTOR OUTPUT:
{executor_output}

REVIEWER FEEDBACK:
{reviewer_output}

Output the final, ready-to-save code files. Each file should be in a code block with the path as a comment on the first line.""",
        ""
    )
    print(f"[ORCHESTRATOR] Done. Output length: {len(final_output)}")

    # 5. Update holders (simple accumulation for MVP)
    wave_summary = f"\n--- WAVE {wave_number} ---\nTask: {task}\nReviewer: {verdict}\n"
    project.convo_history += wave_summary

    # Extract and save files
    files_written = extract_and_save_files(final_output, project_root)
    for fpath, content in files_written.items():
        project.codebase_state[fpath] = content

    return WaveResult(
        wave_number=wave_number,
        task=task,
        config_output=config_output,
        executor_output=executor_output,
        reviewer_output=reviewer_output,
        final_output=final_output,
        files_written=list(files_written.keys()),
    )


async def run_project(
    goal: str,
    tasks: List[str],
    project_root: Path,
    max_waves: int = 10,
) -> ProjectState:
    """Run full project through wave loop."""

    project = ProjectState(goal=goal)
    project_root = Path(project_root)

    print(f"\n{'#'*60}")
    print(f"# PROJECT: {goal[:50]}...")
    print(f"# Tasks: {len(tasks)}")
    print(f"# Output: {project_root}")
    print('#'*60)

    for i, task in enumerate(tasks[:max_waves]):
        result = await run_wave(task, project, i + 1, project_root)
        project.waves.append(result)

        if result.needs_human:
            print(f"\n{'!'*60}")
            print(f"! HUMAN REVIEW NEEDED: {result.human_reason}")
            print(f"{'!'*60}")
            print(f"\nConfig output:\n{result.config_output[:500]}...")
            break

        print(f"\n[WAVE {i + 1} COMPLETE] Files: {result.files_written}")

    return project


# CLI for testing
if __name__ == "__main__":
    async def test():
        project = await run_project(
            goal="Add a /metrics endpoint to the CogTwin backend",
            tasks=[
                "Create a /metrics endpoint that returns query count, token usage, and uptime",
            ],
            project_root=Path("./test_output"),
        )

        print("\n" + "="*60)
        print("PROJECT COMPLETE")
        print("="*60)
        print(f"Waves: {len(project.waves)}")
        print(f"Files created: {list(project.codebase_state.keys())}")

    asyncio.run(test())
