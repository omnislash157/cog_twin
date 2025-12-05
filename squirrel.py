"""
squirrel.py - Temporal recall tool for ADHD-style navigation.

"Wait, what was that thing about an hour ago?"

The SQUIRREL tool gives the model temporal vision into chat history.
Unlike GREP (semantic/keyword), SQUIRREL queries by TIME.

Usage by model:
    [SQUIRREL timeframe="-60min"]              # Last 60 minutes
    [SQUIRREL timeframe="-2h:-30min"]          # 2 hours to 30 min ago
    [SQUIRREL back=5]                          # 5 exchanges ago
    [SQUIRREL search="clustering" timeframe="-1h"]  # Keyword in timeframe

This is Lane 4 retrieval - pure temporal, no embeddings.
"""

import re
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from pathlib import Path

from chat_memory import ChatMemoryStore, ChatExchange

logger = logging.getLogger(__name__)


@dataclass
class SquirrelQuery:
    """Parsed squirrel query parameters."""
    timeframe: Optional[str] = None      # "-60min", "-2h", "-90min:-30min"
    back_n: Optional[int] = None         # Integer for back=N
    search_term: Optional[str] = None    # Keyword filter

    @classmethod
    def parse(cls, raw: str) -> "SquirrelQuery":
        """
        Parse [SQUIRREL ...] tag content.

        Examples:
            'timeframe="-60min"' -> SquirrelQuery(timeframe="-60min")
            'back=5' -> SquirrelQuery(back_n=5)
            'search="clustering" timeframe="-1h"' -> SquirrelQuery(search_term="clustering", timeframe="-1h")
        """
        query = cls()

        # Extract timeframe
        tf_match = re.search(r'timeframe="([^"]+)"', raw)
        if tf_match:
            query.timeframe = tf_match.group(1)

        # Extract back=N
        back_match = re.search(r'back=(\d+)', raw)
        if back_match:
            query.back_n = int(back_match.group(1))

        # Extract search term
        search_match = re.search(r'search="([^"]+)"', raw)
        if search_match:
            query.search_term = search_match.group(1)

        return query


def parse_timeframe(timeframe: str) -> Tuple[int, int]:
    """
    Parse timeframe string into (minutes_ago_start, minutes_ago_end).

    Formats:
        "-60min" -> (60, 0)       # Last 60 minutes
        "-2h" -> (120, 0)         # Last 2 hours
        "-90min:-30min" -> (90, 30)  # Window from 90 to 30 min ago
        "-3h:-1h" -> (180, 60)    # Window from 3h to 1h ago

    Returns (start, end) where start > end (further back first).
    """
    # Handle range format: "-90min:-30min"
    if ":" in timeframe and timeframe.count(":") == 1:
        # Split but handle the case of "-90min:-30min"
        # Find the second minus sign that indicates the end
        parts = timeframe.split(":-")
        if len(parts) == 2:
            start_str = parts[0]  # "-90min"
            end_str = "-" + parts[1]  # "-30min"
            start = _parse_single_timeframe(start_str)
            end = _parse_single_timeframe(end_str)
            return (start, end)

    # Single timeframe: "-60min" means (60, 0)
    minutes = _parse_single_timeframe(timeframe)
    return (minutes, 0)


def _parse_single_timeframe(tf: str) -> int:
    """Parse single timeframe like '-60min' or '-2h' to minutes."""
    tf = tf.strip().lstrip("-")

    if tf.endswith("min"):
        return int(tf[:-3])
    elif tf.endswith("m") and not tf.endswith("min"):
        return int(tf[:-1])
    elif tf.endswith("h"):
        return int(tf[:-1]) * 60
    elif tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    else:
        # Assume minutes if no suffix
        try:
            return int(tf)
        except ValueError:
            return 60  # Default to 1 hour


class SquirrelTool:
    """
    Temporal recall tool for chat history.

    The model invokes this via [SQUIRREL ...] tags.
    Returns formatted context for injection into followup turn.
    """

    def __init__(self, chat_memory: ChatMemoryStore):
        self.memory = chat_memory

    def execute(self, query: SquirrelQuery, limit: int = 10) -> str:
        """
        Execute a squirrel query and return formatted results.

        This is the main entry point called by cog_twin.py handler.
        """
        exchanges: List[ChatExchange] = []
        query_description = ""

        # Priority: back_n > timeframe+search > timeframe > search
        if query.back_n is not None:
            # Get specific exchange N turns ago
            ex = self.memory.query_back_n(query.back_n)
            if ex:
                exchanges = [ex]
            query_description = f"exchange {query.back_n} turns ago"

        elif query.timeframe:
            start, end = parse_timeframe(query.timeframe)

            if query.search_term:
                # Time-scoped keyword search
                exchanges = self.memory.search_content(
                    term=query.search_term,
                    limit=limit,
                    minutes_ago=start,
                )
                query_description = f'"{query.search_term}" in last {start} minutes'
            else:
                # Pure time range
                exchanges = self.memory.query_by_time_range(
                    minutes_ago_start=start,
                    minutes_ago_end=end,
                    limit=limit,
                )
                if end > 0:
                    query_description = f"{start} to {end} minutes ago"
                else:
                    query_description = f"last {start} minutes"

        elif query.search_term:
            # Search without time constraint
            exchanges = self.memory.search_content(
                term=query.search_term,
                limit=limit,
            )
            query_description = f'"{query.search_term}" in chat history'

        else:
            # No params - return recent
            exchanges = self.memory.query_recent(limit)
            query_description = f"last {limit} exchanges"

        return self._format_results(exchanges, query_description)

    def _format_results(
        self,
        exchanges: List[ChatExchange],
        query_description: str,
    ) -> str:
        """Format exchanges for context injection."""
        lines = [
            "",
            "=" * 60,
            "SQUIRREL RECALL (temporal query)",
            "=" * 60,
            f"Query: {query_description}",
            f"Found: {len(exchanges)} exchanges",
            "",
        ]

        if not exchanges:
            lines.append("No exchanges found in the specified range.")
            lines.append("")
            return "\n".join(lines)

        now = datetime.now()

        for ex in exchanges:
            # Calculate age
            age = self._format_age(ex.timestamp, now)

            lines.append(f"─── {age} ({ex.timestamp.strftime('%H:%M')}) ───")

            # User query (truncate long ones)
            query_preview = ex.user_query[:200]
            if len(ex.user_query) > 200:
                query_preview += "..."
            lines.append(f"You asked: {query_preview}")

            # Model trace (if present)
            if ex.model_trace:
                trace_preview = ex.model_trace[:150]
                if len(ex.model_trace) > 150:
                    trace_preview += "..."
                lines.append(f"I thought: {trace_preview}")

            # Model response (truncate)
            response_preview = ex.model_response[:300]
            if len(ex.model_response) > 300:
                response_preview += "..."
            lines.append(f"I said: {response_preview}")

            # Rating if present
            if ex.rating_overall is not None:
                lines.append(f"Rating: {ex.rating_overall:.1f}/1.0")

            lines.append("")

        lines.append("─" * 40)
        lines.append("Use this temporal context to inform your response.")
        lines.append("")

        return "\n".join(lines)

    def _format_age(self, timestamp: datetime, now: datetime) -> str:
        """Format age as human-readable string."""
        delta = now - timestamp
        minutes = delta.total_seconds() / 60

        if minutes < 1:
            return "just now"
        elif minutes < 60:
            return f"{int(minutes)} min ago"
        elif minutes < 1440:  # 24 hours
            hours = minutes / 60
            return f"{hours:.1f} hours ago"
        else:
            days = minutes / 1440
            return f"{days:.1f} days ago"
