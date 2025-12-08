#!/usr/bin/env python
"""
Trace Reader - View reasoning traces and chat exchanges in readable format.

Usage:
    python read_traces.py                    # Show last 5 exchanges
    python read_traces.py -n 10              # Show last 10 exchanges
    python read_traces.py --traces           # Show reasoning traces instead
    python read_traces.py --query "search"   # Filter by query text
    python read_traces.py --full <id>        # Show full trace/exchange by ID
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def load_exchanges(data_dir: Path, limit: int = 10) -> List[Dict]:
    """Load chat exchanges sorted by timestamp."""
    exchanges_dir = data_dir / "chat_exchanges"
    if not exchanges_dir.exists():
        return []

    exchanges = []
    for f in exchanges_dir.glob("exchange_*.json"):
        try:
            with open(f) as fp:
                ex = json.load(fp)
                ex["_file"] = f.name
                exchanges.append(ex)
        except:
            continue

    # Sort by timestamp descending
    exchanges.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return exchanges[:limit]


def load_traces(data_dir: Path, limit: int = 10) -> List[Dict]:
    """Load reasoning traces sorted by timestamp."""
    traces_dir = data_dir / "reasoning_traces"
    if not traces_dir.exists():
        return []

    traces = []
    for f in traces_dir.glob("trace_*.json"):
        try:
            with open(f) as fp:
                tr = json.load(fp)
                tr["_file"] = f.name
                traces.append(tr)
        except:
            continue

    # Sort by timestamp descending
    traces.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return traces[:limit]


def format_exchange(ex: Dict, verbose: bool = False) -> str:
    """Format a single exchange for display."""
    lines = []

    ts = ex.get("timestamp", "unknown")[:19]
    session = ex.get("session_id", "unknown")[-12:]
    phase = ex.get("cognitive_phase", "?")
    tokens = ex.get("tokens_used", 0)
    retrieval_ms = ex.get("retrieval_time_ms", 0)

    lines.append(f"┌─ {ts} │ {session} │ {phase} │ {tokens} tokens │ {retrieval_ms:.0f}ms")
    lines.append(f"│")

    # Query
    query = ex.get("user_query", "")
    if len(query) > 120:
        query = query[:120] + "..."
    lines.append(f"│ Q: {query}")

    # Trace summary
    trace = ex.get("model_trace", "")
    if trace:
        lines.append(f"│ T: {trace[:100]}...")

    # Response
    response = ex.get("model_response", "")
    if verbose:
        # Show full response wrapped
        lines.append(f"│")
        lines.append(f"│ Response:")
        for i in range(0, len(response), 100):
            lines.append(f"│   {response[i:i+100]}")
    else:
        if len(response) > 150:
            response = response[:150] + "..."
        lines.append(f"│ A: {response}")

    lines.append(f"└─ ID: {ex.get('id', 'unknown')}")

    return "\n".join(lines)


def format_trace(tr: Dict, verbose: bool = False) -> str:
    """Format a single reasoning trace for display."""
    lines = []

    ts = tr.get("timestamp", "unknown")[:19]
    trace_id = tr.get("id", "unknown")[:12]
    n_memories = len(tr.get("memories_retrieved", []))
    n_steps = len(tr.get("steps", []))

    lines.append(f"┌─ {ts} │ {trace_id} │ {n_memories} memories │ {n_steps} steps")
    lines.append(f"│")

    # Query
    query = tr.get("query", "")
    if len(query) > 120:
        query = query[:120] + "..."
    lines.append(f"│ Q: {query}")

    # Scores
    scores = tr.get("retrieval_scores", [])[:5]
    if scores:
        score_str = ", ".join([f"{s:.2f}" for s in scores])
        lines.append(f"│ Scores: [{score_str}]")

    # Steps summary
    steps = tr.get("steps", [])
    for step in steps[:5]:
        step_type = step.get("step_type", "?")
        content = step.get("content", "")[:80]
        lines.append(f"│ [{step_type}] {content}")

    if verbose:
        # Show response
        response = tr.get("response", "")
        if response:
            lines.append(f"│")
            lines.append(f"│ Response:")
            for i in range(0, min(len(response), 500), 100):
                lines.append(f"│   {response[i:i+100]}")

    lines.append(f"└─")

    return "\n".join(lines)


def show_full(data_dir: Path, item_id: str):
    """Show full details of a trace or exchange."""
    # Try exchanges first
    ex_file = data_dir / "chat_exchanges" / f"exchange_{item_id}.json"
    if ex_file.exists():
        with open(ex_file) as f:
            data = json.load(f)
        print(f"\n{'='*70}")
        print(f"EXCHANGE: {item_id}")
        print(f"{'='*70}")
        print(json.dumps(data, indent=2, default=str))
        return

    # Try traces
    tr_file = data_dir / "reasoning_traces" / f"trace_{item_id}.json"
    if tr_file.exists():
        with open(tr_file) as f:
            data = json.load(f)
        print(f"\n{'='*70}")
        print(f"TRACE: {item_id}")
        print(f"{'='*70}")
        print(json.dumps(data, indent=2, default=str))
        return

    # Search by partial ID
    for pattern in ["exchange_*", "trace_*"]:
        for folder in ["chat_exchanges", "reasoning_traces"]:
            folder_path = data_dir / folder
            if folder_path.exists():
                for f in folder_path.glob(f"{pattern}.json"):
                    if item_id in f.stem:
                        with open(f) as fp:
                            data = json.load(fp)
                        print(f"\n{'='*70}")
                        print(f"FOUND: {f.name}")
                        print(f"{'='*70}")
                        print(json.dumps(data, indent=2, default=str))
                        return

    print(f"Not found: {item_id}")


def main():
    parser = argparse.ArgumentParser(description="Read reasoning traces and chat exchanges")
    parser.add_argument("-n", "--limit", type=int, default=5, help="Number of items to show")
    parser.add_argument("-t", "--traces", action="store_true", help="Show traces instead of exchanges")
    parser.add_argument("-q", "--query", type=str, help="Filter by query text")
    parser.add_argument("-f", "--full", type=str, help="Show full item by ID")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full responses")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.full:
        show_full(data_dir, args.full)
        return

    if args.traces:
        items = load_traces(data_dir, args.limit * 2)  # Load extra for filtering
        formatter = format_trace
        title = "REASONING TRACES"
    else:
        items = load_exchanges(data_dir, args.limit * 2)
        formatter = format_exchange
        title = "CHAT EXCHANGES"

    # Filter by query if specified
    if args.query:
        query_lower = args.query.lower()
        items = [i for i in items if query_lower in i.get("query", "").lower() or
                 query_lower in i.get("user_query", "").lower()]

    # Limit
    items = items[:args.limit]

    if not items:
        print(f"No {title.lower()} found.")
        return

    print(f"\n{'='*70}")
    print(f" {title} (last {len(items)})")
    print(f"{'='*70}\n")

    for item in items:
        print(formatter(item, verbose=args.verbose))
        print()


if __name__ == "__main__":
    main()
