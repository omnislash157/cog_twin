"""
chat_memory.py - JSON-backed chat exchange storage with temporal queries.

Pattern: Matches CognitiveTracer in reasoning_trace.py
Purpose: Lane 4 retrieval - temporal access to chat history
NOT auto-injected into context window. Accessed via SQUIRREL tool only.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import hashlib
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChatExchange:
    """Single chat exchange (query + trace + response)."""
    id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    timestamp_unix: float = field(default_factory=time.time)

    # The triplet
    user_query: str = ""
    model_trace: Optional[str] = None
    model_response: str = ""

    # Rating (nullable until scored)
    rating_overall: Optional[float] = None
    rating_accuracy: Optional[float] = None
    rating_temporal: Optional[float] = None
    rating_tone: Optional[float] = None
    rating_notes: Optional[str] = None

    # Metadata
    cognitive_phase: Optional[str] = None
    response_confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    retrieval_time_ms: Optional[float] = None
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "timestamp_unix": self.timestamp_unix,
            "user_query": self.user_query,
            "model_trace": self.model_trace,
            "model_response": self.model_response,
            "rating_overall": self.rating_overall,
            "rating_accuracy": self.rating_accuracy,
            "rating_temporal": self.rating_temporal,
            "rating_tone": self.rating_tone,
            "rating_notes": self.rating_notes,
            "cognitive_phase": self.cognitive_phase,
            "response_confidence": self.response_confidence,
            "tokens_used": self.tokens_used,
            "retrieval_time_ms": self.retrieval_time_ms,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChatExchange":
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            timestamp_unix=data.get("timestamp_unix", time.time()),
            user_query=data.get("user_query", ""),
            model_trace=data.get("model_trace"),
            model_response=data.get("model_response", ""),
            rating_overall=data.get("rating_overall"),
            rating_accuracy=data.get("rating_accuracy"),
            rating_temporal=data.get("rating_temporal"),
            rating_tone=data.get("rating_tone"),
            rating_notes=data.get("rating_notes"),
            cognitive_phase=data.get("cognitive_phase"),
            response_confidence=data.get("response_confidence"),
            tokens_used=data.get("tokens_used"),
            retrieval_time_ms=data.get("retrieval_time_ms"),
            trace_id=data.get("trace_id"),
        )


class ChatMemoryStore:
    """
    JSON-backed chat exchange storage with temporal queries.

    Pattern: Match CognitiveTracer in reasoning_trace.py
    Purpose: Lane 4 retrieval - temporal access to chat history

    NOT auto-injected into context window.
    Accessed via SQUIRREL tool only.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.exchanges_dir = self.data_dir / "chat_exchanges"
        self.exchanges_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage (sorted newest-first)
        self.exchanges: List[ChatExchange] = []
        self.exchange_map: Dict[str, ChatExchange] = {}

        self._load_exchanges()

    def _load_exchanges(self):
        """Load existing exchanges from disk."""
        for exchange_file in self.exchanges_dir.glob("exchange_*.json"):
            try:
                with open(exchange_file) as f:
                    data = json.load(f)
                    exchange = ChatExchange.from_dict(data)
                    self.exchanges.append(exchange)
                    self.exchange_map[exchange.id] = exchange
            except Exception as e:
                logger.warning(f"Failed to load exchange {exchange_file}: {e}")

        # Sort by timestamp descending (newest first)
        self.exchanges.sort(key=lambda e: e.timestamp, reverse=True)
        logger.info(f"ChatMemoryStore: loaded {len(self.exchanges)} existing exchanges")

    def record_exchange(
        self,
        session_id: str,
        user_query: str,
        model_response: str,
        model_trace: Optional[str] = None,
        cognitive_phase: Optional[str] = None,
        response_confidence: Optional[float] = None,
        tokens_used: Optional[int] = None,
        retrieval_time_ms: Optional[float] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """
        Record a chat exchange. Returns exchange ID.

        - Generate ID from timestamp + query hash
        - Create ChatExchange dataclass
        - Insert at front of self.exchanges
        - Add to self.exchange_map
        - Save to JSON file
        """
        # Generate ID matching trace pattern
        exchange_id = hashlib.sha256(
            f"{time.time()}_{user_query[:50]}".encode()
        ).hexdigest()[:16]

        exchange = ChatExchange(
            id=exchange_id,
            session_id=session_id,
            timestamp=datetime.now(),
            timestamp_unix=time.time(),
            user_query=user_query,
            model_trace=model_trace,
            model_response=model_response,
            cognitive_phase=cognitive_phase,
            response_confidence=response_confidence,
            tokens_used=tokens_used,
            retrieval_time_ms=retrieval_time_ms,
            trace_id=trace_id,
        )

        # Save to disk
        exchange_file = self.exchanges_dir / f"exchange_{exchange.id}.json"
        with open(exchange_file, "w") as f:
            json.dump(exchange.to_dict(), f, indent=2)

        # Add to in-memory index (insert at front for newest-first)
        self.exchanges.insert(0, exchange)
        self.exchange_map[exchange.id] = exchange

        logger.info(f"Recorded exchange {exchange_id[:8]} in session {session_id}")

        return exchange_id

    def add_rating(
        self,
        exchange_id: str,
        overall: float,
        accuracy: Optional[float] = None,
        temporal: Optional[float] = None,
        tone: Optional[float] = None,
        notes: Optional[str] = None,
    ):
        """
        Add rating to existing exchange.

        - Update in-memory object
        - Re-save JSON file
        """
        if exchange_id not in self.exchange_map:
            logger.warning(f"Exchange {exchange_id} not found for rating")
            return

        exchange = self.exchange_map[exchange_id]
        exchange.rating_overall = overall
        exchange.rating_accuracy = accuracy
        exchange.rating_temporal = temporal
        exchange.rating_tone = tone
        exchange.rating_notes = notes

        # Re-save to disk
        exchange_file = self.exchanges_dir / f"exchange_{exchange.id}.json"
        with open(exchange_file, "w") as f:
            json.dump(exchange.to_dict(), f, indent=2)

        logger.info(f"Added rating to exchange {exchange_id[:8]}: overall={overall:.2f}")

    def query_by_time_range(
        self,
        minutes_ago_start: int,
        minutes_ago_end: int = 0,
        limit: int = 20,
    ) -> List[ChatExchange]:
        """
        Query exchanges within time window.

        Example: query_by_time_range(90, 30) = 90 to 30 minutes ago
        Example: query_by_time_range(60) = last 60 minutes

        Pure Python filter on self.exchanges.
        """
        now = datetime.now()
        start_time = now - timedelta(minutes=minutes_ago_start)
        end_time = now - timedelta(minutes=minutes_ago_end)

        results = [
            e for e in self.exchanges
            if start_time <= e.timestamp <= end_time
        ]

        return results[:limit]

    def query_recent(self, n: int = 20) -> List[ChatExchange]:
        """Get N most recent exchanges."""
        return self.exchanges[:n]

    def query_back_n(self, n: int) -> Optional[ChatExchange]:
        """Get the exchange N turns ago. (0 = most recent)"""
        if n < len(self.exchanges):
            return self.exchanges[n]
        return None

    def search_content(
        self,
        term: str,
        limit: int = 20,
        minutes_ago: Optional[int] = None,
    ) -> List[ChatExchange]:
        """
        Simple keyword search (case-insensitive).

        NOT BM25. Just: if term.lower() in (query + response).lower()
        Optional: filter by time range first.
        """
        term_lower = term.lower()

        # Optional time filter
        candidates = self.exchanges
        if minutes_ago is not None:
            cutoff = datetime.now() - timedelta(minutes=minutes_ago)
            candidates = [e for e in candidates if e.timestamp >= cutoff]

        # Simple substring search
        results = [
            e for e in candidates
            if term_lower in e.user_query.lower() or term_lower in e.model_response.lower()
        ]

        return results[:limit]

    def get_session_exchanges(self, session_id: str) -> List[ChatExchange]:
        """Get all exchanges from a specific session."""
        return [e for e in self.exchanges if e.session_id == session_id]

    def format_for_context(self, exchanges: List[ChatExchange]) -> str:
        """
        Format exchanges for injection into model context.

        Header: "=== SQUIRREL RECALL (temporal query) ==="
        Include: timestamp, age label, query preview, response preview
        If rated: show rating
        """
        if not exchanges:
            return "=== SQUIRREL RECALL (temporal query) ===\n\nNo exchanges found."

        lines = ["=== SQUIRREL RECALL (temporal query) ===\n"]

        now = datetime.now()

        for exchange in exchanges:
            # Calculate age label
            age_delta = now - exchange.timestamp
            if age_delta.total_seconds() < 60:
                age_label = "just now"
            elif age_delta.total_seconds() < 3600:
                mins = int(age_delta.total_seconds() / 60)
                age_label = f"{mins} min ago"
            elif age_delta.total_seconds() < 86400:
                hours = int(age_delta.total_seconds() / 3600)
                age_label = f"{hours}h ago"
            else:
                days = int(age_delta.total_seconds() / 86400)
                age_label = f"{days}d ago"

            # Format timestamp
            ts_str = exchange.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Preview text
            query_preview = exchange.user_query[:100]
            if len(exchange.user_query) > 100:
                query_preview += "..."

            response_preview = exchange.model_response[:200]
            if len(exchange.model_response) > 200:
                response_preview += "..."

            lines.append(f"[{ts_str}] ({age_label})")
            lines.append(f"  Query: {query_preview}")
            lines.append(f"  Response: {response_preview}")

            # Show rating if available
            if exchange.rating_overall is not None:
                rating_parts = [f"overall={exchange.rating_overall:.2f}"]
                if exchange.rating_accuracy is not None:
                    rating_parts.append(f"accuracy={exchange.rating_accuracy:.2f}")
                if exchange.rating_temporal is not None:
                    rating_parts.append(f"temporal={exchange.rating_temporal:.2f}")
                if exchange.rating_tone is not None:
                    rating_parts.append(f"tone={exchange.rating_tone:.2f}")
                lines.append(f"  Rating: {', '.join(rating_parts)}")
                if exchange.rating_notes:
                    lines.append(f"  Notes: {exchange.rating_notes}")

            # Show cognitive phase if available
            if exchange.cognitive_phase:
                lines.append(f"  Phase: {exchange.cognitive_phase}")

            lines.append("")  # Blank line between exchanges

        return "\n".join(lines)
