"""
Quick integration test - verify ChatMemoryStore is wired correctly in CogTwin.
Tests init and record_exchange without running full think() loop.
"""
from pathlib import Path
from cog_twin import CogTwin
import asyncio

async def test_integration():
    """Test that ChatMemoryStore is initialized and functional."""

    print("=== Quick Integration Test ===\n")

    # Test 1: CogTwin initializes ChatMemoryStore
    print("Test 1: CogTwin init with ChatMemoryStore")
    twin = CogTwin(Path("./data"))
    assert hasattr(twin, 'chat_memory'), "❌ chat_memory not found"
    assert hasattr(twin, '_last_exchange_id'), "❌ _last_exchange_id not found"
    print("✓ CogTwin has chat_memory and _last_exchange_id attributes")

    # Test 2: ChatMemoryStore is initialized
    print("\nTest 2: ChatMemoryStore initialization")
    assert twin.chat_memory is not None, "❌ chat_memory is None"
    print(f"✓ ChatMemoryStore loaded: {len(twin.chat_memory.exchanges)} existing exchanges")

    # Test 3: Manual record (simulate what happens in think())
    print("\nTest 3: Manual record_exchange call")
    exchange_id = twin.chat_memory.record_exchange(
        session_id=twin.state.session_id,
        user_query="Test integration query",
        model_response="Test integration response",
        model_trace="retrieve: test; synthesize: test",
        cognitive_phase="focused",
        response_confidence=0.85,
        tokens_used=500,
        retrieval_time_ms=50.0,
        trace_id="test_trace_123",
    )
    print(f"✓ Recorded exchange: {exchange_id}")

    # Test 4: Verify it's in memory and on disk
    print("\nTest 4: Verify persistence")
    assert exchange_id in twin.chat_memory.exchange_map, "❌ Not in exchange_map"
    print("✓ Exchange in memory map")

    exchange_file = Path(f"./data/chat_exchanges/exchange_{exchange_id}.json")
    assert exchange_file.exists(), "❌ JSON file not created"
    print(f"✓ JSON file exists: {exchange_file.name}")

    # Test 5: Query recent
    print("\nTest 5: Query recent")
    recent = twin.chat_memory.query_recent(1)
    assert len(recent) > 0, "❌ No recent exchanges"
    assert recent[0].id == exchange_id, "❌ Most recent is not our exchange"
    print(f"✓ Most recent exchange is ours: {recent[0].user_query}")

    # Test 6: Add rating
    print("\nTest 6: Add rating")
    twin.chat_memory.add_rating(
        exchange_id,
        overall=0.9,
        accuracy=0.95,
        temporal=0.85,
        tone=0.88,
        notes="Test rating integration"
    )
    updated = twin.chat_memory.exchange_map[exchange_id]
    assert updated.rating_overall == 0.9, "❌ Rating not saved"
    print(f"✓ Rating saved: {updated.rating_overall} (notes: {updated.rating_notes})")

    # Cleanup
    print("\nTest 7: Cleanup")
    exchange_file.unlink()
    print(f"✓ Cleaned up test file")

    await twin.stop()
    print("\n" + "=" * 50)
    print("INTEGRATION TEST PASSED ✓")
    print("=" * 50)
    print("\nChatMemoryStore is correctly wired into CogTwin!")
    print("Ready to record exchanges during think() calls.")


if __name__ == "__main__":
    asyncio.run(test_integration())
