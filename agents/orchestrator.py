"""
Wave-based orchestration with full persistence.
Every turn saved. Failures tracked. Holders queried.
"""
import asyncio
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from .registry import AgentRole, spawn_agent, AGENTS
from .schemas import (
    Project, WaveSummary, Failure,
    OutboundTurn, InboundTurn, FilesWritten, FileOperation,
    Verdict, generate_id, now_iso
)
from .persistence import SwarmPersistence
from .reasoning import extract_reasoning_trace, strip_reasoning_tags
from .holders import CodeHolder, ConvoHolder, HolderQuery

# Packages already in this project - don't gate on these
KNOWN_PACKAGES = {
    'fastapi', 'uvicorn', 'pydantic', 'requests', 'numpy', 'faiss',
    'anthropic', 'openai', 'python-dotenv', 'aiofiles', 'httpx',
    'websockets', 'pyyaml', 'scipy', 'sklearn', 'hdbscan', 'river',
}

# Directories to skip when reading project files
SKIP_DIRS = {'node_modules', '__pycache__', '.git', '.venv', 'venv', '.svelte-kit'}


def read_project_files(project_root: Path, patterns: List[str] = None) -> Dict[str, str]:
    """Read existing project files for context."""
    if patterns is None:
        patterns = ["*.py"]

    files = {}
    for pattern in patterns:
        for filepath in project_root.rglob(pattern):
            # Skip unwanted directories
            if any(skip in filepath.parts for skip in SKIP_DIRS):
                continue
            try:
                rel_path = str(filepath.relative_to(project_root))
                content = filepath.read_text(encoding='utf-8')
                files[rel_path] = content
            except Exception:
                pass
    return files


def format_file_context(files: Dict[str, str], max_chars_per_file: int = 2000) -> str:
    """Format files for context injection, truncating large files."""
    sections = []
    for path, content in sorted(files.items()):
        truncated = content[:max_chars_per_file]
        if len(content) > max_chars_per_file:
            truncated += f"\n... [truncated, {len(content)} total chars]"
        sections.append(f"=== {path} ===\n{truncated}")
    return "\n\n".join(sections)


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


def extract_and_save_files(
    output: str,
    project_root: Path,
    existing_files: Dict[str, str]
) -> Dict[str, str]:
    """Extract code and apply surgical edits or create new files."""
    saved = {}

    # Look for TARGET_FILE and MODIFICATION_TYPE markers
    target_match = re.search(r'TARGET_FILE:\s*(\S+)', output)
    mod_type_match = re.search(r'MODIFICATION_TYPE:\s*(\S+)', output)

    # Extract code blocks
    blocks = re.findall(r'```(?:python|py)?\n(.*?)```', output, re.DOTALL)

    if not blocks:
        print("  [FILE] No code blocks found")
        return saved

    # Use last code block (final refined version)
    code = blocks[-1]

    # Check if first line is a file path comment
    lines = code.strip().split('\n')
    if lines and lines[0].startswith('# ') and ('.' in lines[0]):
        # Full file output - extract filename from comment
        filename = lines[0][2:].strip()
        # Normalize path separators
        filename = filename.replace('\\', '/')
        content = '\n'.join(lines[1:])

        filepath = project_root / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding='utf-8')
        saved[filename] = content
        print(f"  [FILE] Saved: {filepath}")

    elif target_match and mod_type_match:
        # Surgical edit mode
        target_file = target_match.group(1)
        mod_type = mod_type_match.group(1).upper()

        if mod_type == "NEW_FILE":
            filepath = project_root / target_file
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(code, encoding='utf-8')
            saved[target_file] = code
            print(f"  [FILE] Created: {target_file}")

        elif target_file in existing_files and mod_type in ["ADD_ENDPOINT", "ADD_FUNCTION"]:
            # Append to existing file
            filepath = project_root / target_file
            existing = existing_files[target_file]
            new_content = existing.rstrip() + "\n\n\n" + code.strip() + "\n"
            filepath.write_text(new_content, encoding='utf-8')
            saved[target_file] = new_content
            print(f"  [FILE] Appended to: {target_file}")

        elif target_file in existing_files and mod_type == "MODIFY_EXISTING":
            # Save as .new for manual review
            new_path = f"{target_file}.new"
            filepath = project_root / new_path
            filepath.write_text(code, encoding='utf-8')
            saved[new_path] = code
            print(f"  [FILE] Created for review: {new_path}")
        else:
            # Fallback: save with target filename
            filepath = project_root / target_file
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(code, encoding='utf-8')
            saved[target_file] = code
            print(f"  [FILE] Saved (fallback): {target_file}")
    else:
        # No markers - save as output.py
        filepath = project_root / "output.py"
        filepath.write_text(code, encoding='utf-8')
        saved["output.py"] = code
        print(f"  [FILE] Saved (no markers): output.py")

    return saved


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

    # Read existing project files for context
    print("\n[FILES] Reading project files...")
    existing_files = read_project_files(project_root)
    file_context = format_file_context(existing_files)
    print(f"[FILES] Read {len(existing_files)} files")

    # Context from previous waves
    context = f"""PROJECT GOAL: {project.goal}

PREVIOUS WAVES: {len(project.waves)}
{project.convo_history[-2000:] if project.convo_history else 'None yet'}

CURRENT CODEBASE FILES: {list(project.codebase_state.keys())}
"""

    # 1. CONFIG AGENT - now with file context
    print("\n[CONFIG] Analyzing dependencies and structure...")
    config_output = await spawn_agent(
        AgentRole.CONFIG,
        f"Task: {task}\n\nEXISTING FILES:\n{file_context}",
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

    # 2. EXECUTOR AGENT - pass file context so it can apply surgical changes
    print("\n[EXECUTOR] Writing code...")
    executor_output = await spawn_agent(
        AgentRole.EXECUTOR,
        f"Apply these changes to the existing files:\n\nCONFIG INSTRUCTIONS:\n{config_output}\n\nEXISTING FILES:\n{file_context}",
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

    # Extract and save files (pass existing_files for surgical edits)
    files_written = extract_and_save_files(final_output, project_root, existing_files)
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
