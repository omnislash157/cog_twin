"""
Cognitive Profiler - Offline analysis of episodic memories to build user profile.

Scans conversation exports to extract:
- Temporal patterns (when you work, session lengths, circadian rhythm)
- Domain focus over time (what topics get most attention)
- Value signals from conversation content
- Decision-making patterns
- Communication style indicators

Outputs a UserProfile JSON to seed the CognitiveTwin.

Usage:
    python cognitive_profiler.py ./chat_exports/

Version: 1.0.0 (cog_twin)
"""

import json
import logging
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import statistics

logger = logging.getLogger(__name__)


@dataclass
class TemporalPattern:
    """Temporal patterns extracted from conversation history."""
    # Hour of day distribution (0-23 -> count)
    hour_distribution: Dict[int, int] = field(default_factory=dict)

    # Day of week distribution (0=Monday, 6=Sunday -> count)
    dow_distribution: Dict[int, int] = field(default_factory=dict)

    # Session durations in minutes
    session_durations: List[float] = field(default_factory=list)

    # Average messages per session
    avg_messages_per_session: float = 0.0

    # Peak activity hours
    peak_hours: List[int] = field(default_factory=list)

    # Night owl score (0=early bird, 1=night owl)
    night_owl_score: float = 0.5

    # Weekend vs weekday activity ratio
    weekend_ratio: float = 0.5


@dataclass
class DomainFocus:
    """Domain focus patterns over time."""
    # Domain -> count
    domain_counts: Dict[str, int] = field(default_factory=dict)

    # Domain -> list of timestamps (for temporal analysis)
    domain_timeline: Dict[str, List[datetime]] = field(default_factory=dict)

    # Top domains in order
    top_domains: List[str] = field(default_factory=list)

    # Domain transitions (what topics follow what)
    domain_transitions: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class ValueSignal:
    """Value and priority signals extracted from content."""
    # Value word -> frequency
    value_frequencies: Dict[str, int] = field(default_factory=dict)

    # Extracted constraints/requirements
    constraints: List[str] = field(default_factory=list)

    # Risk indicators
    risk_seeking_signals: int = 0
    risk_averse_signals: int = 0

    # Time horizon indicators
    short_term_focus: int = 0
    long_term_focus: int = 0


@dataclass
class CommunicationStyle:
    """Communication style indicators."""
    # Average query length (words)
    avg_query_length: float = 0.0

    # Question vs statement ratio
    question_ratio: float = 0.5

    # Uses technical jargon
    technical_density: float = 0.0

    # Emotional language density
    emotional_density: float = 0.0

    # Directness score (0=indirect, 1=direct)
    directness: float = 0.5


@dataclass
class CognitiveProfile:
    """Complete cognitive profile built from analysis."""
    user_id: str
    generated_at: datetime

    # Analyzed conversations
    total_conversations: int = 0
    total_messages: int = 0
    date_range: Tuple[Optional[datetime], Optional[datetime]] = (None, None)

    # Sub-profiles
    temporal: TemporalPattern = field(default_factory=TemporalPattern)
    domain: DomainFocus = field(default_factory=DomainFocus)
    values: ValueSignal = field(default_factory=ValueSignal)
    communication: CommunicationStyle = field(default_factory=CommunicationStyle)

    # Inferred settings for UserProfile
    inferred_risk_tolerance: float = 0.5
    inferred_time_horizon: str = "medium"
    inferred_decision_speed: str = "deliberate"
    inferred_preferred_depth: str = "detailed"
    inferred_preferred_tone: str = "direct"

    def to_user_profile_dict(self) -> Dict[str, Any]:
        """Convert to UserProfile-compatible dict for seeding CognitiveTwin."""
        # Calculate value scores from frequencies
        total_value_mentions = sum(self.values.value_frequencies.values()) or 1
        values = {
            k: min(1.0, v / total_value_mentions * 10)  # Normalize to 0-1
            for k, v in self.values.value_frequencies.items()
        }

        # Calculate framework preferences from domain focus
        framework_scores = {}
        if "PSYCHOLOGY" in self.domain.domain_counts or "RELATIONSHIP" in self.domain.domain_counts:
            framework_scores["UTILITARIAN"] = 0.6
            framework_scores["BOY_SCOUT"] = 0.7
        if "FINANCE" in self.domain.domain_counts or "BUSINESS" in self.domain.domain_counts:
            framework_scores["MACHIAVELLIAN"] = 0.6
            framework_scores["PRAGMATIC_HYBRID"] = 0.8
        if "PHILOSOPHY" in self.domain.domain_counts:
            framework_scores["DEONTOLOGICAL"] = 0.7

        return {
            "user_id": self.user_id,
            "created_at": self.generated_at.isoformat(),
            "last_updated": self.generated_at.isoformat(),
            "values": values,
            "priorities": self.domain.top_domains[:5],
            "constraints": self.values.constraints[:10],
            "risk_tolerance": self.inferred_risk_tolerance,
            "time_horizon": self.inferred_time_horizon,
            "decision_speed": self.inferred_decision_speed,
            "framework_scores": framework_scores,
            "framework_usage_count": {},
            "preferred_depth": self.inferred_preferred_depth,
            "preferred_tone": self.inferred_preferred_tone,
            "likes_probing_questions": True,
            "crisis_escalation_threshold": 0.8,
            "crisis_communication_style": "grounding",
            "total_interactions": self.total_messages,
            "feedback_received": 0,
            "profile_confidence": min(0.7, 0.3 + (self.total_conversations / 1000))  # More data = more confidence
        }


class CognitiveProfiler:
    """
    Analyzes conversation exports to build cognitive profile.

    Design: Pattern extraction without ML - uses heuristics, keyword matching,
    and statistical analysis to infer cognitive patterns from raw conversations.
    """

    # Domain detection keywords (expanded from ingest.py)
    DOMAIN_KEYWORDS = {
        "CODING": ["code", "python", "javascript", "function", "debug", "error", "api", "class", "variable", "loop", "async", "await"],
        "WRITING": ["write", "story", "essay", "blog", "article", "draft", "edit", "prose", "narrative"],
        "CREATIVE": ["creative", "idea", "brainstorm", "design", "art", "music", "create"],
        "BUSINESS": ["business", "startup", "company", "revenue", "profit", "customer", "market", "strategy"],
        "FINANCE": ["money", "invest", "budget", "finance", "stock", "crypto", "savings", "expense"],
        "PSYCHOLOGY": ["feeling", "emotion", "therapy", "mental", "anxiety", "depression", "mindset", "behavior"],
        "RELATIONSHIP": ["relationship", "partner", "friend", "family", "dating", "marriage", "conflict"],
        "HEALTH": ["health", "diet", "exercise", "sleep", "weight", "medical", "symptom", "doctor"],
        "PHILOSOPHY": ["meaning", "purpose", "existence", "ethics", "moral", "philosophy", "consciousness"],
        "LEARNING": ["learn", "study", "course", "book", "research", "understand", "explain"],
        "PRODUCTIVITY": ["productivity", "time", "focus", "habit", "routine", "organize", "todo", "schedule"],
        "CAREER": ["job", "career", "interview", "resume", "promotion", "salary", "work", "manager"],
    }

    # Value indicators
    VALUE_KEYWORDS = {
        "integrity": ["honest", "integrity", "truth", "authentic", "genuine"],
        "growth": ["grow", "improve", "learn", "develop", "progress", "evolve"],
        "efficiency": ["efficient", "fast", "quick", "optimize", "streamline", "automate"],
        "quality": ["quality", "excellent", "best", "thorough", "careful", "precise"],
        "creativity": ["creative", "innovative", "original", "unique", "novel"],
        "connection": ["connect", "relationship", "together", "collaborate", "community"],
        "autonomy": ["independent", "freedom", "control", "self", "own"],
        "security": ["safe", "secure", "stable", "reliable", "certain"],
        "impact": ["impact", "change", "matter", "difference", "meaningful"],
    }

    # Risk indicators
    RISK_SEEKING_PATTERNS = [
        "try", "experiment", "risk", "bold", "aggressive", "bet", "chance",
        "what if", "let's see", "push the limits", "go for it"
    ]
    RISK_AVERSE_PATTERNS = [
        "safe", "careful", "cautious", "conservative", "sure", "certain",
        "what could go wrong", "downside", "worst case", "backup plan"
    ]

    # Time horizon indicators
    SHORT_TERM_PATTERNS = ["today", "now", "immediately", "asap", "urgent", "quick", "this week"]
    LONG_TERM_PATTERNS = ["future", "eventually", "long term", "years", "sustainable", "legacy", "vision"]

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.conversations: List[Dict[str, Any]] = []
        self.profile = CognitiveProfile(
            user_id=user_id,
            generated_at=datetime.now()
        )

    def load_anthropic_export(self, file_path: Path) -> int:
        """Load and parse Anthropic/Claude export."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = []
        for conv in data:
            # Parse Anthropic format
            messages = []
            created_at = None

            if "chat_messages" in conv:
                for msg in conv["chat_messages"]:
                    role = "human" if msg.get("sender") == "human" else "assistant"
                    content = msg.get("text", "")
                    timestamp = msg.get("created_at")

                    if timestamp:
                        try:
                            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            if created_at is None:
                                created_at = ts
                        except:
                            pass

                    messages.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })

            if messages:
                conversations.append({
                    "title": conv.get("name", "Untitled"),
                    "created_at": created_at or datetime.now(),
                    "messages": messages,
                    "source": "anthropic"
                })

        self.conversations.extend(conversations)
        logger.info(f"Loaded {len(conversations)} conversations from Anthropic export")
        return len(conversations)

    def load_openai_export(self, file_path: Path) -> int:
        """Load and parse OpenAI/ChatGPT export."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = []
        for conv in data:
            messages = []
            created_at = None

            # OpenAI format uses "mapping" with nested structure
            if "mapping" in conv:
                # Extract messages from mapping
                for node_id, node in conv["mapping"].items():
                    msg = node.get("message")
                    if msg and msg.get("content"):
                        role = msg.get("author", {}).get("role", "unknown")
                        parts = msg["content"].get("parts", [])
                        content = " ".join(str(p) for p in parts if isinstance(p, str))

                        timestamp = msg.get("create_time")
                        if timestamp:
                            try:
                                ts = datetime.fromtimestamp(timestamp)
                                if created_at is None or ts < created_at:
                                    created_at = ts
                            except:
                                pass

                        if content and role in ["user", "assistant"]:
                            messages.append({
                                "role": "human" if role == "user" else "assistant",
                                "content": content,
                                "timestamp": timestamp
                            })

            if messages:
                conversations.append({
                    "title": conv.get("title", "Untitled"),
                    "created_at": created_at or datetime.now(),
                    "messages": messages,
                    "source": "openai"
                })

        self.conversations.extend(conversations)
        logger.info(f"Loaded {len(conversations)} conversations from OpenAI export")
        return len(conversations)

    def load_directory(self, dir_path: Path) -> int:
        """Load all conversation exports from directory."""
        total = 0

        for json_file in dir_path.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Detect format
                if isinstance(data, list) and data:
                    if "chat_messages" in data[0]:
                        total += self.load_anthropic_export(json_file)
                    elif "mapping" in data[0]:
                        total += self.load_openai_export(json_file)
                    else:
                        logger.warning(f"Unknown format in {json_file}")
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return total

    def analyze(self) -> CognitiveProfile:
        """Run full analysis on loaded conversations."""
        if not self.conversations:
            logger.warning("No conversations loaded. Call load_* methods first.")
            return self.profile

        print(f"Analyzing {len(self.conversations)} conversations...")

        # Run all analyzers
        self._analyze_temporal()
        self._analyze_domains()
        self._analyze_values()
        self._analyze_communication()
        self._infer_settings()

        # Update metadata
        self.profile.total_conversations = len(self.conversations)
        self.profile.total_messages = sum(len(c["messages"]) for c in self.conversations)

        # Normalize dates to naive UTC for comparison
        dates = []
        for c in self.conversations:
            dt = c["created_at"]
            if dt:
                # Convert to naive datetime for comparison
                if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                dates.append(dt)

        if dates:
            self.profile.date_range = (min(dates), max(dates))

        return self.profile

    def _analyze_temporal(self) -> None:
        """Analyze temporal patterns."""
        print("  Analyzing temporal patterns...")

        hour_counts = defaultdict(int)
        dow_counts = defaultdict(int)
        session_durations = []
        messages_per_session = []

        for conv in self.conversations:
            created_at = conv["created_at"]
            if not created_at:
                continue

            # Handle timezone-aware vs naive datetimes
            if hasattr(created_at, 'hour'):
                hour_counts[created_at.hour] += 1
                dow_counts[created_at.weekday()] += 1

            # Estimate session duration from messages
            messages = conv["messages"]
            messages_per_session.append(len(messages))

            # Parse timestamps if available
            timestamps = []
            for msg in messages:
                ts = msg.get("timestamp")
                if ts:
                    try:
                        if isinstance(ts, (int, float)):
                            timestamps.append(datetime.fromtimestamp(ts))
                        elif isinstance(ts, str):
                            timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                    except:
                        pass

            if len(timestamps) >= 2:
                # Make all timestamps naive for comparison
                timestamps = [t.replace(tzinfo=None) if t.tzinfo else t for t in timestamps]
                duration = (max(timestamps) - min(timestamps)).total_seconds() / 60
                if 0 < duration < 480:  # Filter unrealistic durations (< 8 hours)
                    session_durations.append(duration)

        # Calculate patterns
        self.profile.temporal.hour_distribution = dict(hour_counts)
        self.profile.temporal.dow_distribution = dict(dow_counts)
        self.profile.temporal.session_durations = session_durations

        if messages_per_session:
            self.profile.temporal.avg_messages_per_session = statistics.mean(messages_per_session)

        # Find peak hours (top 3)
        if hour_counts:
            sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
            self.profile.temporal.peak_hours = [h for h, _ in sorted_hours[:3]]

        # Calculate night owl score (activity 22:00-04:00 vs 06:00-12:00)
        night_hours = sum(hour_counts.get(h, 0) for h in [22, 23, 0, 1, 2, 3, 4])
        morning_hours = sum(hour_counts.get(h, 0) for h in [6, 7, 8, 9, 10, 11, 12])
        total_relevant = night_hours + morning_hours
        if total_relevant > 0:
            self.profile.temporal.night_owl_score = night_hours / total_relevant

        # Weekend vs weekday ratio
        weekend = dow_counts.get(5, 0) + dow_counts.get(6, 0)  # Sat + Sun
        weekday = sum(dow_counts.get(d, 0) for d in range(5))
        total_days = weekend + weekday
        if total_days > 0:
            self.profile.temporal.weekend_ratio = weekend / total_days

    def _analyze_domains(self) -> None:
        """Analyze domain focus patterns."""
        print("  Analyzing domain focus...")

        domain_counts = defaultdict(int)
        domain_timeline = defaultdict(list)
        domain_transitions = defaultdict(lambda: defaultdict(int))

        last_domain = None

        for conv in self.conversations:
            created_at = conv["created_at"]

            # Combine all human messages for domain detection
            human_text = " ".join(
                msg["content"].lower()
                for msg in conv["messages"]
                if msg["role"] == "human"
            )

            # Detect domain
            domain_scores = defaultdict(int)
            for domain, keywords in self.DOMAIN_KEYWORDS.items():
                for kw in keywords:
                    domain_scores[domain] += human_text.count(kw)

            if domain_scores:
                detected_domain = max(domain_scores.items(), key=lambda x: x[1])
                if detected_domain[1] > 0:
                    domain = detected_domain[0]
                    domain_counts[domain] += 1

                    if created_at:
                        domain_timeline[domain].append(created_at)

                    # Track transitions
                    if last_domain and last_domain != domain:
                        domain_transitions[last_domain][domain] += 1
                    last_domain = domain

        self.profile.domain.domain_counts = dict(domain_counts)
        self.profile.domain.domain_timeline = {k: v for k, v in domain_timeline.items()}
        self.profile.domain.domain_transitions = {k: dict(v) for k, v in domain_transitions.items()}

        # Top domains
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        self.profile.domain.top_domains = [d for d, _ in sorted_domains]

    def _analyze_values(self) -> None:
        """Extract value signals from content."""
        print("  Analyzing value signals...")

        value_counts = defaultdict(int)
        constraints = []
        risk_seeking = 0
        risk_averse = 0
        short_term = 0
        long_term = 0

        for conv in self.conversations:
            human_text = " ".join(
                msg["content"].lower()
                for msg in conv["messages"]
                if msg["role"] == "human"
            )

            # Count value keywords
            for value, keywords in self.VALUE_KEYWORDS.items():
                for kw in keywords:
                    count = human_text.count(kw)
                    value_counts[value] += count

            # Detect risk orientation
            for pattern in self.RISK_SEEKING_PATTERNS:
                risk_seeking += human_text.count(pattern)
            for pattern in self.RISK_AVERSE_PATTERNS:
                risk_averse += human_text.count(pattern)

            # Detect time horizon
            for pattern in self.SHORT_TERM_PATTERNS:
                short_term += human_text.count(pattern)
            for pattern in self.LONG_TERM_PATTERNS:
                long_term += human_text.count(pattern)

            # Extract explicit constraints (simple heuristic)
            constraint_patterns = [
                r"i can't\s+[^.]{10,50}",
                r"i must\s+[^.]{10,50}",
                r"i need to\s+[^.]{10,50}",
                r"constraint[s]?\s*[:is]+\s*[^.]{10,100}",
                r"requirement[s]?\s*[:is]+\s*[^.]{10,100}",
            ]
            for pattern in constraint_patterns:
                matches = re.findall(pattern, human_text)
                constraints.extend(matches[:3])  # Limit per conversation

        self.profile.values.value_frequencies = dict(value_counts)
        self.profile.values.constraints = list(set(constraints))[:20]  # Dedupe and limit
        self.profile.values.risk_seeking_signals = risk_seeking
        self.profile.values.risk_averse_signals = risk_averse
        self.profile.values.short_term_focus = short_term
        self.profile.values.long_term_focus = long_term

    def _analyze_communication(self) -> None:
        """Analyze communication style."""
        print("  Analyzing communication style...")

        query_lengths = []
        questions = 0
        statements = 0
        technical_terms = 0
        emotional_terms = 0
        total_words = 0
        direct_phrases = 0
        indirect_phrases = 0

        technical_keywords = [
            "api", "function", "class", "algorithm", "database", "server",
            "async", "await", "parameter", "variable", "implementation"
        ]
        emotional_keywords = [
            "feel", "frustrated", "excited", "worried", "happy", "sad",
            "anxious", "stressed", "overwhelmed", "grateful"
        ]
        direct_patterns = ["i want", "i need", "give me", "do this", "tell me"]
        indirect_patterns = ["could you", "would it be", "i was wondering", "maybe", "perhaps"]

        for conv in self.conversations:
            for msg in conv["messages"]:
                if msg["role"] == "human":
                    content = msg["content"]
                    words = content.split()
                    query_lengths.append(len(words))
                    total_words += len(words)

                    # Questions vs statements
                    if "?" in content:
                        questions += 1
                    else:
                        statements += 1

                    content_lower = content.lower()

                    # Technical density
                    for kw in technical_keywords:
                        technical_terms += content_lower.count(kw)

                    # Emotional density
                    for kw in emotional_keywords:
                        emotional_terms += content_lower.count(kw)

                    # Directness
                    for pattern in direct_patterns:
                        direct_phrases += content_lower.count(pattern)
                    for pattern in indirect_patterns:
                        indirect_phrases += content_lower.count(pattern)

        if query_lengths:
            self.profile.communication.avg_query_length = statistics.mean(query_lengths)

        total_messages = questions + statements
        if total_messages > 0:
            self.profile.communication.question_ratio = questions / total_messages

        if total_words > 0:
            self.profile.communication.technical_density = technical_terms / total_words * 100
            self.profile.communication.emotional_density = emotional_terms / total_words * 100

        total_direction = direct_phrases + indirect_phrases
        if total_direction > 0:
            self.profile.communication.directness = direct_phrases / total_direction

    def _infer_settings(self) -> None:
        """Infer UserProfile settings from analysis."""
        print("  Inferring profile settings...")

        # Risk tolerance from risk signals
        total_risk = self.profile.values.risk_seeking_signals + self.profile.values.risk_averse_signals
        if total_risk > 0:
            self.profile.inferred_risk_tolerance = self.profile.values.risk_seeking_signals / total_risk

        # Time horizon
        total_horizon = self.profile.values.short_term_focus + self.profile.values.long_term_focus
        if total_horizon > 0:
            ratio = self.profile.values.long_term_focus / total_horizon
            if ratio > 0.6:
                self.profile.inferred_time_horizon = "long"
            elif ratio < 0.4:
                self.profile.inferred_time_horizon = "short"
            else:
                self.profile.inferred_time_horizon = "medium"

        # Decision speed from average query length and session duration
        avg_length = self.profile.communication.avg_query_length
        if avg_length > 100:
            self.profile.inferred_decision_speed = "slow"  # Thorough, detailed queries
        elif avg_length < 30:
            self.profile.inferred_decision_speed = "quick"  # Short, direct queries
        else:
            self.profile.inferred_decision_speed = "deliberate"

        # Preferred depth from query length
        if avg_length > 80:
            self.profile.inferred_preferred_depth = "detailed"
        elif avg_length < 40:
            self.profile.inferred_preferred_depth = "concise"
        else:
            self.profile.inferred_preferred_depth = "balanced"

        # Preferred tone from directness
        if self.profile.communication.directness > 0.7:
            self.profile.inferred_preferred_tone = "direct"
        elif self.profile.communication.directness < 0.3:
            self.profile.inferred_preferred_tone = "diplomatic"
        else:
            self.profile.inferred_preferred_tone = "casual"

    def print_summary(self) -> None:
        """Print analysis summary to console."""
        p = self.profile

        print("\n" + "=" * 60)
        print("  COGNITIVE PROFILE ANALYSIS")
        print("=" * 60)

        print(f"\n{p.total_conversations} conversations | {p.total_messages} messages")
        if p.date_range[0] and p.date_range[1]:
            print(f"Date range: {p.date_range[0].strftime('%Y-%m-%d')} to {p.date_range[1].strftime('%Y-%m-%d')}")

        print("\n--- TEMPORAL PATTERNS ---")
        print(f"Peak hours: {p.temporal.peak_hours}")
        print(f"Night owl score: {p.temporal.night_owl_score:.2f} (0=early bird, 1=night owl)")
        print(f"Weekend activity: {p.temporal.weekend_ratio:.1%}")
        if p.temporal.session_durations:
            print(f"Avg session: {statistics.mean(p.temporal.session_durations):.1f} min")
        print(f"Avg messages/session: {p.temporal.avg_messages_per_session:.1f}")

        print("\n--- DOMAIN FOCUS ---")
        print(f"Top domains: {', '.join(p.domain.top_domains[:5])}")
        for domain in p.domain.top_domains[:5]:
            count = p.domain.domain_counts.get(domain, 0)
            pct = count / len(self.conversations) * 100 if self.conversations else 0
            print(f"  {domain}: {count} ({pct:.1f}%)")

        print("\n--- VALUE SIGNALS ---")
        top_values = sorted(p.values.value_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
        for value, count in top_values:
            print(f"  {value}: {count} mentions")
        print(f"Risk orientation: {p.inferred_risk_tolerance:.2f} (0=averse, 1=seeking)")
        print(f"Time horizon: {p.inferred_time_horizon}")

        print("\n--- COMMUNICATION STYLE ---")
        print(f"Avg query length: {p.communication.avg_query_length:.1f} words")
        print(f"Question ratio: {p.communication.question_ratio:.1%}")
        print(f"Technical density: {p.communication.technical_density:.2f}")
        print(f"Directness: {p.communication.directness:.2f} (0=indirect, 1=direct)")

        print("\n--- INFERRED SETTINGS ---")
        print(f"Decision speed: {p.inferred_decision_speed}")
        print(f"Preferred depth: {p.inferred_preferred_depth}")
        print(f"Preferred tone: {p.inferred_preferred_tone}")

        print("\n" + "=" * 60)

    def save_profile(self, output_path: Path) -> None:
        """Save UserProfile-compatible JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile_dict = self.profile.to_user_profile_dict()

        with open(output_path, 'w') as f:
            json.dump(profile_dict, f, indent=2)

        print(f"\nProfile saved to: {output_path}")

    def save_full_analysis(self, output_path: Path) -> None:
        """Save complete analysis (not just UserProfile subset)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        p = self.profile

        analysis = {
            "generated_at": p.generated_at.isoformat(),
            "total_conversations": p.total_conversations,
            "total_messages": p.total_messages,
            "date_range": [
                p.date_range[0].isoformat() if p.date_range[0] else None,
                p.date_range[1].isoformat() if p.date_range[1] else None
            ],
            "temporal": {
                "hour_distribution": p.temporal.hour_distribution,
                "dow_distribution": p.temporal.dow_distribution,
                "peak_hours": p.temporal.peak_hours,
                "night_owl_score": p.temporal.night_owl_score,
                "weekend_ratio": p.temporal.weekend_ratio,
                "avg_messages_per_session": p.temporal.avg_messages_per_session,
                "avg_session_duration_min": statistics.mean(p.temporal.session_durations) if p.temporal.session_durations else None
            },
            "domain": {
                "domain_counts": p.domain.domain_counts,
                "top_domains": p.domain.top_domains,
                "domain_transitions": p.domain.domain_transitions
            },
            "values": {
                "value_frequencies": p.values.value_frequencies,
                "constraints": p.values.constraints,
                "risk_tolerance": p.inferred_risk_tolerance,
                "time_horizon": p.inferred_time_horizon
            },
            "communication": {
                "avg_query_length": p.communication.avg_query_length,
                "question_ratio": p.communication.question_ratio,
                "technical_density": p.communication.technical_density,
                "emotional_density": p.communication.emotional_density,
                "directness": p.communication.directness
            },
            "inferred_settings": {
                "risk_tolerance": p.inferred_risk_tolerance,
                "time_horizon": p.inferred_time_horizon,
                "decision_speed": p.inferred_decision_speed,
                "preferred_depth": p.inferred_preferred_depth,
                "preferred_tone": p.inferred_preferred_tone
            }
        }

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"Full analysis saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    import sys

    print("=" * 60)
    print("  COGNITIVE PROFILER - Analyze Your Thought Patterns")
    print("=" * 60)

    # Get input directory
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    else:
        input_dir = Path("./chat_exports")

    if not input_dir.exists():
        print(f"\nError: Directory not found: {input_dir}")
        print("Usage: python cognitive_profiler.py <chat_exports_dir>")
        return

    # Create profiler
    profiler = CognitiveProfiler(user_id="default")

    # Load conversations
    print(f"\nLoading conversations from {input_dir}...")
    total = profiler.load_directory(input_dir)
    print(f"Loaded {total} conversations total")

    if total == 0:
        print("No conversations found. Check your export files.")
        return

    # Run analysis
    profiler.analyze()

    # Print summary
    profiler.print_summary()

    # Save outputs
    output_dir = Path("./data/user_profiles")
    profiler.save_profile(output_dir / "default_profile.json")
    profiler.save_full_analysis(output_dir / "full_analysis.json")

    print("\nDone! Profile is ready to seed CognitiveTwin.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
