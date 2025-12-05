"""Quick test of the swarm on a real task."""
import asyncio
import sys
from pathlib import Path

# Ensure parent is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from agents.orchestrator import run_project


async def main():
    # Test on sandbox - safe to trash
    sandbox_root = Path(__file__).parent / "sandbox"

    project = await run_project(
        goal="Add /api/metrics endpoint to sandbox API",
        tasks=[
            """Add a GET /api/metrics endpoint to app/main.py that returns:
            - total_queries: int (from engine.state.total_queries if available, else 0)
            - total_tokens: int (from engine.state.total_tokens_used if available, else 0)
            - uptime_seconds: float (time since startup)
            - memory_nodes: int (from engine.memory_count if available, else 0)

            Follow existing endpoint patterns in the file.
            Handle cases where engine might not have these attributes yet."""
        ],
        project_root=sandbox_root,
    )

    print("\n" + "="*60)
    print("TEST RESULT")
    print("="*60)

    if project.waves:
        wave = project.waves[0]
        print(f"Needs human: {wave.needs_human}")
        print(f"Files written: {wave.files_written}")

        print(f"\n--- CONFIG OUTPUT ---")
        print(wave.config_output[:800] if wave.config_output else "None")

        print(f"\n--- REVIEWER OUTPUT ---")
        print(wave.reviewer_output[:800] if wave.reviewer_output else "None")

        print(f"\n--- FINAL OUTPUT (first 1500 chars) ---")
        print(wave.final_output[:1500] if wave.final_output else "None")
    else:
        print("No waves completed")


if __name__ == "__main__":
    asyncio.run(main())
