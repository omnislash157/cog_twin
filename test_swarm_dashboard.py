"""
Integration test: Run swarm with dashboard visualization.

This is the self-referential test - the swarm builds its own dashboard.

Usage:
    python test_swarm_dashboard.py

Prerequisites:
    1. Backend running: cd backend && uvicorn app.main:app --reload
    2. Frontend running: cd frontend && npm run dev
    3. Open http://localhost:5173 and switch to Swarm mode
"""
import asyncio
from pathlib import Path
from agents.swarm_orchestrator import SwarmOrchestrator


async def test_dashboard_project():
    """Run the swarm on building its own dashboard."""
    print("=" * 60)
    print("SWARM DASHBOARD INTEGRATION TEST")
    print("=" * 60)
    print()
    print("This test runs the swarm to verify the dashboard visualization")
    print("works correctly. Make sure you have:")
    print("  1. Backend running on port 8000")
    print("  2. Frontend running on port 5173")
    print("  3. Browser open to http://localhost:5173")
    print("  4. SwarmPanel visible (click 🐝 in header if needed)")
    print()
    print("=" * 60)

    # Use agents/sandbox as project root
    project_root = Path(__file__).parent / "agents" / "sandbox"
    project_root.mkdir(exist_ok=True)

    # Create orchestrator without broadcast callback for CLI test
    # The backend will use broadcast_callback when called via API
    orchestrator = SwarmOrchestrator(
        project_root=project_root,
        project_name="swarm_dashboard_test",
        goal="Test the live agent visualization dashboard"
    )

    # Simple test tasks that don't require actual code generation
    tasks = [
        "Add a simple hello world function to test file operations",
    ]

    print("\nStarting swarm with test task...")
    print(f"Project ID: {orchestrator.project.id}")
    print(f"Project Root: {project_root}")
    print()

    try:
        project = await orchestrator.run_project(tasks)

        print()
        print("=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Project ID: {project.id}")
        print(f"Status: {project.status}")
        print(f"Waves completed: {orchestrator.current_wave}")
        print(f"Total tokens in: {orchestrator.wave_tokens_in}")
        print(f"Total tokens out: {orchestrator.wave_tokens_out}")

        # Check if data was persisted
        data_dir = Path(__file__).parent / "agents" / "data" / "projects" / project.id
        if data_dir.exists():
            print(f"\nPersisted data at: {data_dir}")
            for f in data_dir.rglob("*.json"):
                print(f"  - {f.relative_to(data_dir)}")

        return project

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        raise


async def test_websocket_broadcast():
    """Test that WebSocket broadcasts are working."""
    import json

    print("\n" + "=" * 60)
    print("WEBSOCKET BROADCAST TEST")
    print("=" * 60)

    # Track received events
    events_received = []

    async def mock_broadcast(message: str):
        event = json.loads(message)
        events_received.append(event)
        print(f"  [BROADCAST] {event['type']}")

    project_root = Path(__file__).parent / "agents" / "sandbox"
    project_root.mkdir(exist_ok=True)

    orchestrator = SwarmOrchestrator(
        project_root=project_root,
        project_name="broadcast_test",
        goal="Test WebSocket broadcasting",
        broadcast_callback=mock_broadcast
    )

    # Run minimal task
    await orchestrator.run_project([
        "Create a test function"
    ])

    print(f"\nTotal events broadcasted: {len(events_received)}")

    # Verify expected events
    event_types = [e["type"] for e in events_received]
    expected_types = [
        "swarm_project_start",
        "swarm_wave_start",
        "swarm_agent_start",
        "swarm_turn",
    ]

    for expected in expected_types:
        if expected in event_types:
            print(f"  ✓ {expected}")
        else:
            print(f"  ✗ {expected} (MISSING)")

    return events_received


def print_help():
    """Print usage instructions."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           SWARM DASHBOARD INTEGRATION TEST                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  This test verifies the live swarm visualization works.      ║
║                                                              ║
║  SETUP:                                                      ║
║    Terminal 1: cd backend && uvicorn app.main:app --reload   ║
║    Terminal 2: cd frontend && npm run dev                    ║
║    Browser: Open http://localhost:5173                       ║
║                                                              ║
║  TO TEST VIA API:                                            ║
║    POST http://localhost:8000/api/swarm/start                ║
║    {                                                         ║
║      "project_name": "test_project",                         ║
║      "goal": "Test the dashboard",                           ║
║      "tasks": ["Create hello world function"]                ║
║    }                                                         ║
║                                                              ║
║  WATCH DASHBOARD:                                            ║
║    - Open SwarmPanel in the right sidebar                    ║
║    - Switch to 🐝 mode in the 3D view                        ║
║    - See agents pulse as they process                        ║
║    - Watch reasoning steps unfold                            ║
║    - See code preview update                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print_help()
        sys.exit(0)

    print_help()

    # Run both tests
    print("\n[1/2] Running WebSocket broadcast test...")
    asyncio.run(test_websocket_broadcast())

    print("\n[2/2] Running dashboard integration test...")
    asyncio.run(test_dashboard_project())

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
