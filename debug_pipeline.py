#!/usr/bin/env python
"""
Debug script to trace what the model actually receives.
Run: python debug_pipeline.py "your test query"
"""

import asyncio
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retrieval import DualRetriever
from venom_voice import VenomVoice
from config import cfg


async def debug_retrieval(query: str):
    """Trace retrieval results step by step."""
    print("=" * 70)
    print(f"DEBUG PIPELINE TRACE")
    print(f"Query: {query}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load retriever
    print("\n[1] Loading retriever...")
    retriever = DualRetriever.load("./data")
    print(f"    Nodes: {len(retriever.process.nodes)}")
    print(f"    Episodes: {len(retriever.episodic.episodes)}")

    # Run retrieval
    print("\n[2] Running retrieval...")
    result = await retriever.retrieve(
        query,
        process_top_k=cfg("retrieval.process_top_k", 10),
        episodic_top_k=cfg("retrieval.episodic_top_k", 5),
    )

    # Process memories
    print(f"\n[3] PROCESS MEMORIES: {len(result.process_memories)} returned")
    print(f"    Scores: {[f'{s:.3f}' for s in result.process_scores[:5]]}")
    for i, (mem, score) in enumerate(zip(result.process_memories[:3], result.process_scores[:3])):
        print(f"\n    [{i}] score={score:.3f}")
        print(f"        human_content: {bool(mem.human_content)} ({len(mem.human_content or '')} chars)")
        print(f"        created_at: {mem.created_at}")
        if mem.human_content:
            print(f"        Q: {mem.human_content[:100]}...")

    # Episodic memories
    print(f"\n[4] EPISODIC MEMORIES: {len(result.episodic_memories)} returned")
    print(f"    Scores: {[f'{s:.3f}' for s in result.episodic_scores[:5]]}")
    for i, (ep, score) in enumerate(zip(result.episodic_memories[:3], result.episodic_scores[:3])):
        print(f"\n    [{i}] score={score:.3f}")
        print(f"        title: {ep.title[:60] if ep.title else 'N/A'}")
        print(f"        summary_text: {bool(ep.summary_text)} ({len(ep.summary_text or '')} chars)")
        if ep.summary_text:
            print(f"        Summary: {ep.summary_text[:150]}...")

    # Build memory lists
    print("\n[5] Building memory lists...")
    process_mems = [
        {
            "human_content": m.human_content,
            "assistant_content": m.assistant_content,
            "timestamp": m.created_at,
            "score": s,
        }
        for m, s in zip(result.process_memories[:5], result.process_scores[:5])
    ]

    episodic_mems = [
        {
            "title": e.title,
            "summary": e.summary_text,
            "start_time": e.created_at,
            "score": s,
        }
        for e, s in zip(result.episodic_memories[:3], result.episodic_scores[:3])
    ]

    print(f"    process_memories: {len(process_mems)}")
    print(f"    episodic_memories: {len(episodic_mems)}")

    # Format memories
    print("\n[6] Formatting memories for prompt...")
    voice = VenomVoice()
    formatted = voice._format_memories(process_mems, episodic_mems)

    print(f"    Formatted length: {len(formatted)} chars")
    print("\n" + "=" * 70)
    print("FORMATTED MEMORIES (what model sees):")
    print("=" * 70)
    print(formatted)
    print("=" * 70)

    # Check episodic section
    print("\n[7] Episodic section analysis:")
    if "EPISODIC RETRIEVAL" in formatted:
        print("    OK Episodic section EXISTS")
    else:
        print("    XX Episodic section MISSING!")

    # Save debug output
    debug_output = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "process_count": len(result.process_memories),
        "episodic_count": len(result.episodic_memories),
        "process_scores": result.process_scores[:5],
        "episodic_scores": result.episodic_scores[:5],
        "has_episodic_section": "EPISODIC RETRIEVAL" in formatted,
        "episodic_in_context": [
            {"title": e["title"], "has_summary": bool(e.get("summary"))}
            for e in episodic_mems
        ],
    }

    debug_file = Path("./data/debug_pipeline_output.json")
    with open(debug_file, "w") as f:
        json.dump(debug_output, f, indent=2, default=str)
    print(f"\n[8] Saved to: {debug_file}")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "How do we handle memory retrieval?"
    asyncio.run(debug_retrieval(query))
