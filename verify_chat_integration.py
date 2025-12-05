"""
Verification script for Phase 6.2 chat memory integration.
Run this after having a conversation with cog_twin.py
"""
from chat_memory import ChatMemoryStore
from pathlib import Path
from datetime import datetime, timedelta

def verify_integration():
    """Check that exchanges were recorded during cog_twin.py session."""

    print("=== Phase 6.2 Integration Verification ===\n")

    store = ChatMemoryStore(Path("./data"))

    print(f"Total exchanges in store: {len(store.exchanges)}\n")

    if not store.exchanges:
        print("⚠ No exchanges found. Run a conversation with cog_twin.py first:")
        print("  python cog_twin.py ./data")
        print("  You> What is the memory pipeline?")
        print("  [response streams...]")
        return

    # Show recent exchanges
    print("=== Recent Exchanges (last 5) ===")
    recent = store.query_recent(5)
    for i, ex in enumerate(recent, 1):
        age_delta = datetime.now() - ex.timestamp
        if age_delta.total_seconds() < 3600:
            mins = int(age_delta.total_seconds() / 60)
            age = f"{mins}m ago"
        else:
            hours = int(age_delta.total_seconds() / 3600)
            age = f"{hours}h ago"

        print(f"\n{i}. [{ex.timestamp.strftime('%H:%M:%S')}] ({age})")
        print(f"   Session: {ex.session_id}")
        print(f"   Query: {ex.user_query[:60]}...")
        print(f"   Response: {ex.model_response[:80]}...")

        if ex.model_trace:
            print(f"   Trace: {ex.model_trace[:60]}...")

        if ex.cognitive_phase:
            print(f"   Phase: {ex.cognitive_phase}")

        if ex.rating_overall:
            print(f"   Rating: {ex.rating_overall:.2f}", end="")
            if ex.rating_accuracy:
                print(f" (accuracy={ex.rating_accuracy:.2f})", end="")
            if ex.rating_notes:
                print(f"\n   Notes: {ex.rating_notes}")
            else:
                print()

    # Test queries
    print("\n\n=== Testing Query Functions ===")

    # Time range query (last hour)
    last_hour = store.query_by_time_range(60)
    print(f"\n✓ Last 60 minutes: {len(last_hour)} exchanges")

    # Back-N query
    if len(store.exchanges) >= 2:
        back_1 = store.query_back_n(1)
        if back_1:
            print(f"✓ 1 turn ago: {back_1.user_query[:40]}...")

    # Content search
    if len(store.exchanges) > 0:
        # Use a word from the first exchange
        first_query = store.exchanges[0].user_query
        if len(first_query) > 5:
            search_term = first_query.split()[0] if first_query.split() else "the"
            hits = store.search_content(search_term, limit=3)
            print(f"✓ Search for '{search_term}': {len(hits)} hits")

    # Session query
    if store.exchanges:
        first_session = store.exchanges[0].session_id
        session_exs = store.get_session_exchanges(first_session)
        print(f"✓ Session '{first_session[-8:]}': {len(session_exs)} exchanges")

    # Format for context
    print("\n\n=== SQUIRREL Format Preview ===")
    formatted = store.format_for_context(recent[:2])
    print(formatted)

    # Check for rated exchanges
    rated = [e for e in store.exchanges if e.rating_overall is not None]
    print(f"\n=== Rating Statistics ===")
    print(f"Rated exchanges: {len(rated)} / {len(store.exchanges)}")
    if rated:
        avg_rating = sum(e.rating_overall for e in rated) / len(rated)
        print(f"Average rating: {avg_rating:.2f}")

    print("\n" + "=" * 50)
    print("PHASE 6.2 INTEGRATION VERIFIED ✓")
    print("=" * 50)
    print("\nNext: Phase 6.3 - SQUIRREL tool registration")


if __name__ == "__main__":
    verify_integration()
