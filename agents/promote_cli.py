"""
Skynet Procedure - Swarm Promotion Control CLI

Human review workflow for promoting sandbox changes to real project.

Usage:
    python -m agents.promote_cli review <project_id>
    python -m agents.promote_cli decide <project_id> <wave> --approve/--reject
    python -m agents.promote_cli rollback <backup_path>
"""

import sys
import json
from pathlib import Path
from typing import Optional

# Handle both direct execution and module execution
try:
    from .sandbox_executor import SandboxExecutor, PromotionRequest
except ImportError:
    from sandbox_executor import SandboxExecutor, PromotionRequest


def get_sandbox_root() -> Path:
    """Get default sandbox root."""
    return Path(__file__).parent / "sandbox"


def get_project_root() -> Path:
    """Get project root (parent of agents/)."""
    return Path(__file__).parent.parent


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(text)
    print('='*60)


def print_divider() -> None:
    """Print section divider."""
    print('-'*60)


def cmd_review(project_id: Optional[str] = None) -> None:
    """Review pending promotions."""
    sandbox_root = get_sandbox_root()
    project_root = get_project_root()

    if not sandbox_root.exists():
        print(f"[ERROR] Sandbox not found: {sandbox_root}")
        return

    staging_dir = sandbox_root / "staging"
    if not staging_dir.exists():
        print("[INFO] No staging directory found. No promotions pending.")
        return

    found_any = False
    for wave_dir in sorted(staging_dir.iterdir()):
        if not wave_dir.is_dir():
            continue

        request_file = wave_dir / "promotion_request.json"
        summary_file = wave_dir / "diff_summary.md"
        test_file = wave_dir / "test_results.log"

        if not request_file.exists():
            continue

        data = json.loads(request_file.read_text(encoding='utf-8'))

        # Filter by project_id if specified
        if project_id and data.get("project_id") != project_id:
            continue

        found_any = True
        status = data.get("status", "unknown")
        status_icon = {
            "pending": "[PENDING]",
            "approved": "[APPROVED]",
            "rejected": "[REJECTED]",
            "deferred": "[DEFERRED]",
        }.get(status, f"[{status.upper()}]")

        print_header(f"WAVE: {wave_dir.name} {status_icon}")

        print(f"Project: {data.get('project_id', 'unknown')}")
        print(f"Files: {len(data.get('files', []))}")
        print(f"Timestamp: {data.get('timestamp', 'unknown')}")

        if summary_file.exists():
            print_divider()
            print("DIFF SUMMARY:")
            print(summary_file.read_text(encoding='utf-8'))

        if test_file.exists():
            print_divider()
            print("TEST RESULTS (last 50 lines):")
            lines = test_file.read_text(encoding='utf-8').splitlines()
            for line in lines[-50:]:
                print(f"  {line}")

        if data.get("reviewer_notes"):
            print_divider()
            print(f"REVIEWER NOTES: {data['reviewer_notes']}")

        print()

    if not found_any:
        if project_id:
            print(f"[INFO] No promotions found for project: {project_id}")
        else:
            print("[INFO] No promotions found.")


def cmd_decide(project_id: str, wave: str, approve: bool, reject: bool, notes: str = "") -> None:
    """Approve or reject a promotion."""
    if approve and reject:
        print("[ERROR] Cannot both approve and reject")
        return

    if not approve and not reject:
        print("[ERROR] Must specify --approve or --reject")
        return

    sandbox_root = get_sandbox_root()
    project_root = get_project_root()

    executor = SandboxExecutor(sandbox_root, project_root)

    # Find the promotion request
    wave_name = wave if wave.startswith("wave_") else f"wave_{wave}"
    request_file = sandbox_root / "staging" / wave_name / "promotion_request.json"

    if not request_file.exists():
        print(f"[ERROR] Promotion request not found: {request_file}")
        return

    data = json.loads(request_file.read_text(encoding='utf-8'))

    if data.get("project_id") != project_id:
        print(f"[ERROR] Project ID mismatch. Expected: {data.get('project_id')}, got: {project_id}")
        return

    if approve:
        print(f"\n[APPROVE] Wave {wave}")
        print("Promoting files to project...")

        # Update status
        executor.update_promotion_status(wave, "approved", notes)

        # Reload request with updated status
        data = json.loads(request_file.read_text(encoding='utf-8'))
        request = PromotionRequest(**{k: v for k, v in data.items() if k != 'status'})
        request.status = "approved"

        # Execute promotion
        result = executor.promote_files(request, backup=True)

        if result.success:
            print(f"\n[SUCCESS] Promoted {len(result.files_promoted)} files")
            for f in result.files_promoted:
                print(f"  + {f}")
            if result.backup_path:
                print(f"\n[BACKUP] Originals saved to: {result.backup_path}")
        else:
            print(f"\n[PARTIAL] Promoted {len(result.files_promoted)}, failed {len(result.files_failed)}")
            for f in result.files_failed:
                print(f"  ! {f}")

    elif reject:
        print(f"\n[REJECT] Wave {wave}")
        if notes:
            print(f"Notes: {notes}")

        executor.update_promotion_status(wave, "rejected", notes)
        print("[INFO] Rejection logged. Swarm will learn from feedback on next run.")


def cmd_rollback(backup_path: str) -> None:
    """Emergency rollback from backup."""
    project_root = get_project_root()
    sandbox_root = get_sandbox_root()

    # Handle relative paths
    if not Path(backup_path).is_absolute():
        # Check if it's in .swarm_backups
        if backup_path.startswith("wave_"):
            backup_path = str(project_root / ".swarm_backups" / backup_path)
        else:
            backup_path = str(project_root / ".swarm_backups" / f"wave_{backup_path}")

    backup_dir = Path(backup_path)
    if not backup_dir.exists():
        print(f"[ERROR] Backup not found: {backup_path}")
        print("\nAvailable backups:")
        backups_dir = project_root / ".swarm_backups"
        if backups_dir.exists():
            for d in sorted(backups_dir.iterdir()):
                if d.is_dir():
                    print(f"  - {d.name}")
        return

    print(f"\n[WARNING] ROLLBACK from: {backup_path}")
    print("This will restore the following files:")

    files_to_restore = list(backup_dir.rglob("*"))
    files_to_restore = [f for f in files_to_restore if f.is_file()]

    for f in files_to_restore:
        rel = f.relative_to(backup_dir)
        print(f"  <- {rel}")

    confirm = input("\nType 'ROLLBACK' to confirm: ")
    if confirm != "ROLLBACK":
        print("[ABORT] Rollback cancelled.")
        return

    executor = SandboxExecutor(sandbox_root, project_root)
    result = executor.rollback(backup_path)

    if result.success:
        print(f"\n[SUCCESS] {result.output}")
    else:
        print(f"\n[ERROR] {result.error}")


def cmd_list_files(wave: str) -> None:
    """List files in a promotion staging area."""
    sandbox_root = get_sandbox_root()
    wave_name = wave if wave.startswith("wave_") else f"wave_{wave}"
    files_dir = sandbox_root / "staging" / wave_name / "files_to_promote"

    if not files_dir.exists():
        print(f"[ERROR] Staging directory not found: {files_dir}")
        return

    print_header(f"Files in {wave_name}")

    for f in sorted(files_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(files_dir)
            size = f.stat().st_size
            print(f"  {rel} ({size} bytes)")


def cmd_show_file(wave: str, file_path: str) -> None:
    """Show contents of a staged file."""
    sandbox_root = get_sandbox_root()
    wave_name = wave if wave.startswith("wave_") else f"wave_{wave}"
    full_path = sandbox_root / "staging" / wave_name / "files_to_promote" / file_path

    if not full_path.exists():
        print(f"[ERROR] File not found: {full_path}")
        return

    print_header(f"File: {file_path}")
    print(full_path.read_text(encoding='utf-8'))


def print_usage() -> None:
    """Print usage information."""
    print("""
Skynet Procedure - Swarm Promotion Control

Usage:
    python -m agents.promote_cli <command> [options]

Commands:
    review [project_id]              Review pending promotions
    decide <project_id> <wave> [--approve|--reject] [--notes "..."]
                                     Approve or reject a promotion
    rollback <backup_path>           Emergency rollback from backup
    list <wave>                      List files in staging area
    show <wave> <file_path>          Show contents of staged file

Examples:
    python -m agents.promote_cli review
    python -m agents.promote_cli review my_project
    python -m agents.promote_cli decide my_project 003 --approve
    python -m agents.promote_cli decide my_project 003 --reject --notes "Auth logic wrong"
    python -m agents.promote_cli rollback 003
    python -m agents.promote_cli list 003
    python -m agents.promote_cli show 003 app/main.py
""")


def main() -> None:
    """Main CLI entry point."""
    args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help", "help"]:
        print_usage()
        return

    cmd = args[0]

    if cmd == "review":
        project_id = args[1] if len(args) > 1 else None
        cmd_review(project_id)

    elif cmd == "decide":
        if len(args) < 3:
            print("[ERROR] Usage: decide <project_id> <wave> --approve/--reject [--notes '...']")
            return

        project_id = args[1]
        wave = args[2]

        approve = "--approve" in args
        reject = "--reject" in args

        notes = ""
        if "--notes" in args:
            notes_idx = args.index("--notes")
            if notes_idx + 1 < len(args):
                notes = args[notes_idx + 1]

        cmd_decide(project_id, wave, approve, reject, notes)

    elif cmd == "rollback":
        if len(args) < 2:
            print("[ERROR] Usage: rollback <backup_path>")
            return
        cmd_rollback(args[1])

    elif cmd == "list":
        if len(args) < 2:
            print("[ERROR] Usage: list <wave>")
            return
        cmd_list_files(args[1])

    elif cmd == "show":
        if len(args) < 3:
            print("[ERROR] Usage: show <wave> <file_path>")
            return
        cmd_show_file(args[1], args[2])

    else:
        print(f"[ERROR] Unknown command: {cmd}")
        print_usage()


if __name__ == "__main__":
    main()
