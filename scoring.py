"""
scoring.py - Multi-dimensional response scoring.
ML-style feedback that attaches to reasoning traces.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class ResponseScore:
    """
    Multi-dimensional score for a response.
    All fields 0-1 scale.
    """
    # Core metrics
    accuracy: float = 1.0       # Was information correct?
    personality: float = 1.0    # Did it sound like MY twin?
    usefulness: float = 1.0     # Did it actually help?
    depth: float = 1.0          # Right level of detail?

    # Style metrics
    tone_match: float = 1.0     # Right formality level?
    clarity: float = 1.0        # Easy to understand?

    # Cognitive metrics
    gap_handling: float = 1.0   # Did it identify what was missing?
    context_use: float = 1.0    # Good use of retrieved memories?

    # Meta metrics
    would_share: float = 1.0    # Would I send this to someone?
    saved_time: float = 1.0     # Did this shortcut my thinking?

    # Conditional feedback (only if score < 0.5)
    accuracy_note: Optional[str] = None
    personality_note: Optional[str] = None
    usefulness_note: Optional[str] = None
    depth_note: Optional[str] = None
    general_note: Optional[str] = None

    # Meta
    timestamp: datetime = field(default_factory=datetime.now)

    # Weights for overall calculation
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.15,
        "personality": 0.25,  # Highest - it's YOUR twin
        "usefulness": 0.20,
        "depth": 0.10,
        "tone_match": 0.10,
        "clarity": 0.05,
        "gap_handling": 0.05,
        "context_use": 0.05,
        "would_share": 0.025,
        "saved_time": 0.025,
    })

    @property
    def overall(self) -> float:
        """Weighted average of all scores."""
        total = 0.0
        for field_name, weight in self.WEIGHTS.items():
            total += getattr(self, field_name) * weight
        return total

    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "personality": self.personality,
            "usefulness": self.usefulness,
            "depth": self.depth,
            "tone_match": self.tone_match,
            "clarity": self.clarity,
            "gap_handling": self.gap_handling,
            "context_use": self.context_use,
            "would_share": self.would_share,
            "saved_time": self.saved_time,
            "overall": self.overall,
            "accuracy_note": self.accuracy_note,
            "personality_note": self.personality_note,
            "usefulness_note": self.usefulness_note,
            "depth_note": self.depth_note,
            "general_note": self.general_note,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ResponseScore":
        return cls(
            accuracy=data.get("accuracy", 1.0),
            personality=data.get("personality", 1.0),
            usefulness=data.get("usefulness", 1.0),
            depth=data.get("depth", 1.0),
            tone_match=data.get("tone_match", 1.0),
            clarity=data.get("clarity", 1.0),
            gap_handling=data.get("gap_handling", 1.0),
            context_use=data.get("context_use", 1.0),
            would_share=data.get("would_share", 1.0),
            saved_time=data.get("saved_time", 1.0),
            accuracy_note=data.get("accuracy_note"),
            personality_note=data.get("personality_note"),
            usefulness_note=data.get("usefulness_note"),
            depth_note=data.get("depth_note"),
            general_note=data.get("general_note"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
        )

    def get_feedback_notes(self) -> Dict[str, str]:
        """Get all non-null feedback notes."""
        notes = {}
        if self.accuracy_note:
            notes["accuracy"] = self.accuracy_note
        if self.personality_note:
            notes["personality"] = self.personality_note
        if self.usefulness_note:
            notes["usefulness"] = self.usefulness_note
        if self.depth_note:
            notes["depth"] = self.depth_note
        if self.general_note:
            notes["general"] = self.general_note
        return notes

    def format_for_context(self) -> str:
        """Format score for injection into LLM context."""
        lines = [
            f"Score: overall={self.overall:.2f}",
            f"  accuracy={self.accuracy:.1f}, personality={self.personality:.1f}, usefulness={self.usefulness:.1f}",
        ]

        notes = self.get_feedback_notes()
        if notes:
            lines.append("  Feedback:")
            for field_name, note in notes.items():
                lines.append(f"    {field_name}: {note}")

        return "\n".join(lines)


class TrainingModeUI:
    """
    CLI interface for scoring responses.
    Prompts for multi-field scores after each response.
    """

    def __init__(self, enabled: bool = True, quick_mode: bool = False):
        """
        Args:
            enabled: Whether training mode is active
            quick_mode: If True, only ask for overall score
        """
        self.enabled = enabled
        self.quick_mode = quick_mode

    def prompt_for_score(self) -> Optional[ResponseScore]:
        """
        Prompt user for response score.
        Returns None if user skips.
        """
        if not self.enabled:
            return None

        print("\n" + "=" * 50)
        print("TRAINING MODE - Score this response (1-10 scale)")
        print("=" * 50)

        if self.quick_mode:
            return self._quick_score()
        else:
            return self._full_score()

    def _quick_score(self) -> Optional[ResponseScore]:
        """Three-dimension 1-10 scoring."""
        try:
            print("\nRate 1-10 (enter=skip, 0=skip dimension):\n")

            # Accuracy - was the information correct?
            accuracy_input = input("Accuracy [1-10]: ").strip()
            if not accuracy_input:
                return None
            accuracy = max(1, min(10, int(accuracy_input))) / 10.0 if accuracy_input != "0" else 1.0

            # Temporal accuracy - did it use recent/relevant memories?
            temporal_input = input("Temporal accuracy [1-10]: ").strip()
            temporal = max(1, min(10, int(temporal_input))) / 10.0 if temporal_input and temporal_input != "0" else 1.0

            # Tone - did it sound right?
            tone_input = input("Tone [1-10]: ").strip()
            tone = max(1, min(10, int(tone_input))) / 10.0 if tone_input and tone_input != "0" else 1.0

            # Feedback on low scores
            note = None
            if accuracy < 0.5 or temporal < 0.5 or tone < 0.5:
                note = input("What should improve? ").strip() or None

            # Map to ResponseScore fields
            return ResponseScore(
                accuracy=accuracy,
                personality=tone,          # tone maps to personality
                usefulness=accuracy,       # derive from accuracy
                depth=accuracy,
                tone_match=tone,
                clarity=accuracy,
                gap_handling=temporal,     # temporal awareness = gap handling
                context_use=temporal,      # temporal = context use
                would_share=(accuracy + tone) / 2,
                saved_time=accuracy,
                general_note=note,
            )
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\n[Scoring skipped]")
            return None

    def _full_score(self) -> Optional[ResponseScore]:
        """Full multi-field scoring."""
        try:
            score = ResponseScore()

            # Core metrics
            score.accuracy = self._get_score("Accuracy", default=1.0)
            score.personality = self._get_score("Personality", default=1.0)
            score.usefulness = self._get_score("Usefulness", default=1.0)
            score.depth = self._get_score("Depth", default=1.0)

            # Conditional notes
            if score.accuracy < 0.5:
                score.accuracy_note = input("  What was wrong? ").strip() or None

            if score.personality < 0.5:
                score.personality_note = input("  How should it sound? ").strip() or None

            if score.usefulness < 0.5:
                score.usefulness_note = input("  What was missing? ").strip() or None

            if score.depth < 0.5:
                score.depth_note = input("  Too shallow or too deep? ").strip() or None

            # Set remaining to defaults or infer
            score.tone_match = score.personality
            score.clarity = score.usefulness
            score.gap_handling = score.accuracy
            score.context_use = score.accuracy
            score.would_share = (score.usefulness + score.personality) / 2
            score.saved_time = score.usefulness

            print(f"\nOverall score: {score.overall:.2f}")

            return score

        except (ValueError, EOFError, KeyboardInterrupt):
            print("\n[Scoring skipped]")
            return None

    def _get_score(self, name: str, default: float = 1.0) -> float:
        """Get a single score field."""
        prompt = f"{name} [0-1, enter={default}]: "
        val = input(prompt).strip()
        if not val:
            return default
        return max(0.0, min(1.0, float(val)))
