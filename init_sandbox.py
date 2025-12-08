"""
NOTE FOR CLAUDE ORCHESTRATOR:
- Sandbox has NO venv. Use project's active venv.
- Project venv is already active when swarm starts.
- If you need deps, write to: agents/data/dep_requests.json
- Format: {"package": "name", "reason": "why", "timestamp": "iso"}
- Pipeline will crash loud on missing deps - check dep_requests.json
"""

import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import json

# File patterns to copy
COPY_PATTERNS = {
    '.py', '.ts', '.svelte', '.json', '.yaml', '.yml',
    '.md', '.txt', '.html', '.css', '.toml'
}

# Directories to exclude (never descend into)
EXCLUDE_DIRS = {
    '__pycache__', '.git', '.venv', 'venv', 'node_modules',
    '.svelte-kit', 'agents/sandbox', 'agents/data', '.mypy_cache',
    '.pytest_cache', 'dist', 'build', 'sandbox'
}

def should_copy_file(file_path: Path) -> bool:
    """Check if file should be copied based on extension."""
    return file_path.suffix in COPY_PATTERNS

def is_excluded_dir(dir_path: Path, project_root: Path) -> bool:
    """Check if directory should be excluded."""
    rel_path = str(dir_path.relative_to(project_root)).replace('\\', '/')

    # Check if any part of the path matches excluded directories
    parts = rel_path.split('/')
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True

    # Special check for agents/sandbox and agents/data
    if rel_path.startswith('agents/sandbox') or rel_path.startswith('agents/data'):
        return True

    return False

def should_copy(src: Path, dst: Path, incremental: bool) -> bool:
    """Determine if file should be copied."""
    if not incremental:
        return True

    if not dst.exists():
        return True

    # Copy if source is newer than destination
    src_mtime = src.stat().st_mtime
    dst_mtime = dst.stat().st_mtime
    return src_mtime > dst_mtime

def remove_sandbox_venv(sandbox_path: Path) -> bool:
    """Remove venv from sandbox if it exists."""
    venv_path = sandbox_path / '.venv'
    if venv_path.exists():
        shutil.rmtree(venv_path)
        return True
    return False

def collect_files(project_root: Path) -> list[tuple[Path, Path]]:
    """Collect files to copy from project root."""
    sandbox_path = project_root / 'agents' / 'sandbox'
    files_to_copy = []

    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)

        # Skip excluded directories
        if is_excluded_dir(root_path, project_root):
            dirs.clear()  # Don't descend into this directory
            continue

        # Remove excluded dirs from dirs list to prevent os.walk from descending
        dirs[:] = [d for d in dirs if not is_excluded_dir(root_path / d, project_root)]

        for file in files:
            src_file = root_path / file

            if should_copy_file(src_file):
                # Calculate relative path from project root
                rel_path = src_file.relative_to(project_root)
                dst_file = sandbox_path / rel_path
                files_to_copy.append((src_file, dst_file))

    return files_to_copy

def init_sandbox(clean: bool = False, dry_run: bool = False, incremental: bool = True):
    """Initialize sandbox directory with project files."""
    project_root = Path(__file__).parent
    sandbox_path = project_root / 'agents' / 'sandbox'
    data_path = project_root / 'agents' / 'data'

    # Ensure data directory exists
    data_path.mkdir(parents=True, exist_ok=True)

    # Clean mode: wipe sandbox first
    if clean and not dry_run:
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path)
            print(f"[SANDBOX] Cleaned: {sandbox_path}")

    # Create sandbox directory
    if not dry_run:
        sandbox_path.mkdir(parents=True, exist_ok=True)

    # Remove venv from sandbox if exists
    venv_removed = False
    if not dry_run and remove_sandbox_venv(sandbox_path):
        print("[SANDBOX] Removed stale venv from sandbox (use project venv)")
        venv_removed = True

    # Collect files to copy
    files_to_copy = collect_files(project_root)

    # Filter based on incremental mode
    if incremental and not clean:
        files_to_copy = [(src, dst) for src, dst in files_to_copy if should_copy(src, dst, incremental=True)]

    copied_count = 0
    skipped_count = 0
    total_files = len(collect_files(project_root))

    if dry_run:
        print("[SANDBOX] DRY RUN - No files will be copied")
        print(f"[SANDBOX] Would copy {len(files_to_copy)} files:")
        for src, dst in files_to_copy[:10]:  # Show first 10
            print(f"  {src.relative_to(project_root)} -> {dst.relative_to(sandbox_path)}")
        if len(files_to_copy) > 10:
            print(f"  ... and {len(files_to_copy) - 10} more files")
    else:
        # Copy files
        for src, dst in files_to_copy:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied_count += 1
            except Exception as e:
                print(f"[SANDBOX] Warning: Failed to copy {src}: {e}")

        skipped_count = total_files - copied_count

    # Print summary
    print(f"[SANDBOX] Initialized: {sandbox_path}")
    if dry_run:
        print(f"[SANDBOX] Would copy: {len(files_to_copy)} files")
    else:
        print(f"[SANDBOX] Copied: {copied_count} files")
        if incremental and not clean:
            print(f"[SANDBOX] Skipped: {skipped_count} files (unchanged)")

    excluded_list = ', '.join(sorted(EXCLUDE_DIRS))
    print(f"[SANDBOX] Excluded: {excluded_list}")

    if not dry_run:
        print("[SANDBOX] Ready for swarm execution")

def main():
    parser = argparse.ArgumentParser(
        description='Initialize sandbox directory for swarm execution'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Wipe sandbox first, then copy all files'
    )
    parser.add_argument(
        '--dry',
        action='store_true',
        help='Show what would be copied, don\'t actually copy'
    )

    args = parser.parse_args()

    # Incremental mode is default unless --clean is specified
    incremental = not args.clean

    init_sandbox(clean=args.clean, dry_run=args.dry, incremental=incremental)

if __name__ == '__main__':
    main()
