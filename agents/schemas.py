"""
Unified schema for all swarm communications.
Every exchange gets a UUID, timestamp, and is saved to disk.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
import uuid
import json
from pathlib import Path


def generate_id() -> str:
    return str(uuid.uuid4())[:8]

def now_iso() -> str:
    return datetime.now().isoformat()


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    CONFIG = "config"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    ADVERSARIAL = "adversarial"
    QUALITY_GATE = "quality_gate"
    BUDGET = "budget"
    CONVO_HOLDER = "convo_holder"
    CODE_HOLDER = "code_holder"
    USER_HOLDER = "user_holder"
    MEMORY_ORACLE = "memory_oracle"
    HUMAN = "human"


class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    HUMAN_REVIEW = "human_review"


class ConfigMode(Enum):
    """CONFIG agent operating modes."""
    PLANNING = "planning"      # Normal flow: analyze task, emit modification plan
    DIAGNOSTIC = "diagnostic"  # On failure: query holders, emit Diagnosis


@dataclass
class ReasoningStep:
    """Single step in model's reasoning trace."""
    step: int
    content: str
    observation: Optional[str] = None
    decision: Optional[str] = None


@dataclass
class OutboundTurn:
    """Prompt sent to an agent."""
    id: str = field(default_factory=generate_id)
    timestamp: str = field(default_factory=now_iso)

    wave: str = "001"
    sequence: int = 1

    direction: Literal["outbound"] = "outbound"
    from_agent: str = "orchestrator"
    to_agent: str = "config"

    model: str = ""
    temperature: float = 0.7

    system_prompt: str = ""
    user_prompt: str = ""

    tokens_in: int = 0

    reasoning_before: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def filename(self) -> str:
        return f"{self.sequence:02d}_{self.from_agent}_to_{self.to_agent}.json"


@dataclass
class InboundTurn:
    """Response received from an agent."""
    id: str = ""  # Same as outbound
    timestamp: str = field(default_factory=now_iso)

    wave: str = "001"
    sequence: int = 2

    direction: Literal["inbound"] = "inbound"
    from_agent: str = "config"
    to_agent: str = "orchestrator"

    model: str = ""

    raw_response: str = ""
    parsed: Dict[str, Any] = field(default_factory=dict)

    reasoning_trace: List[ReasoningStep] = field(default_factory=list)

    tokens_out: int = 0
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['reasoning_trace'] = [asdict(r) for r in self.reasoning_trace]
        return d

    def filename(self) -> str:
        return f"{self.sequence:02d}_{self.from_agent}_response.json"


@dataclass
class FileOperation:
    """Record of file changes."""
    file_path: str
    operation: Literal["create", "modify", "append", "delete"]
    lines_added: int = 0
    lines_removed: int = 0
    content_hash: str = ""
    preview: str = ""  # First 500 chars


@dataclass
class FilesWritten:
    """Record of all file operations in a wave."""
    id: str = field(default_factory=generate_id)
    timestamp: str = field(default_factory=now_iso)

    wave: str = "001"
    sequence: int = 99

    direction: Literal["internal"] = "internal"
    from_agent: str = "code_holder"
    to_agent: str = "filesystem"

    operations: List[FileOperation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['operations'] = [asdict(op) for op in self.operations]
        return d

    def filename(self) -> str:
        return f"{self.sequence:02d}_files_written.json"


@dataclass
class WaveSummary:
    """Summary of a completed wave."""
    wave: str = "001"
    started_at: str = ""
    completed_at: str = field(default_factory=now_iso)

    task: str = ""

    agents_invoked: List[str] = field(default_factory=list)
    turns_count: int = 0

    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0

    verdict: str = "pass"
    failure_ref: Optional[str] = None  # Path to failure file if failed

    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)

    summary: str = ""
    orchestrator_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Failure:
    """Record of a wave failure."""
    id: str = field(default_factory=generate_id)
    timestamp: str = field(default_factory=now_iso)

    wave: str = "001"

    failure_source: str = "quality_gate"
    failure_type: str = "full_rewrite"
    agent_blamed: str = "executor"

    verdict_raw: str = ""
    root_cause: str = ""
    recommendation: str = ""

    context_snapshot: Dict[str, str] = field(default_factory=dict)

    learning: str = ""
    retry_wave: str = "001.1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_learning_prompt(self) -> str:
        """Format for injection into planning prompts."""
        return f"""PAST FAILURE (Wave {self.wave}):
- Type: {self.failure_type}
- Agent: {self.agent_blamed}
- Cause: {self.root_cause}
- Learning: {self.learning}
AVOID THIS PATTERN.
"""


@dataclass
class Project:
    """Full project state."""
    id: str = field(default_factory=generate_id)
    name: str = ""
    created_at: str = field(default_factory=now_iso)

    goal: str = ""
    spec_file: Optional[str] = None

    status: Literal["planning", "executing", "paused", "review", "complete", "failed"] = "planning"
    current_wave: str = "001"

    budget_limit_usd: float = 10.0
    budget_spent_usd: float = 0.0

    acceptance_criteria: List[str] = field(default_factory=list)

    outcome: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Resilience Loop Schemas (Phase 10e)
# =============================================================================

@dataclass
class ToolFailure:
    """
    Failure information from SandboxExecutor → ORCHESTRATOR.

    Note on consulted_already: This is set by _on_tool_failure looking up
    orchestrator state, NOT by handle_tool_failure. It means "we consulted
    on a PREVIOUS failure for this wave", not "we just consulted in THIS call".
    """
    wave: str
    tool_name: str
    tool_args: Dict[str, Any]
    error_type: str           # Exception class name
    error_message: str
    stack_trace: str
    attempt: int              # 1, 2, or 3
    executor_context: str     # What EXECUTOR was trying to accomplish
    timestamp: str = field(default_factory=now_iso)
    consulted_already: bool = False  # True if we consulted on PREVIOUS failure

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsultationOutcome:
    """Result of consulting an agent during diagnostic flow."""
    agent: str
    risk: str
    confidence: float
    recommended_action: str
    notes: str = ""
    helpful: Optional[bool] = None  # Populated post-retry to track effectiveness

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Diagnosis:
    """
    CONFIG DIAGNOSTIC mode output → ORCHESTRATOR.

    All fields are open strings with suggestions, not strict enums.
    This allows CONFIG to express novel failure types and strategies.
    """
    failure_type: str         # Suggestions: import_missing, path_wrong, syntax_error, dep_conflict, etc.
    root_cause: str           # Open field, blank if unknown
    fix_strategy: str         # Suggestions: retry_executor, retry_tool, add_dep, modify_config, human_review, abort
    consult_with: List[str]   # Agent names or empty
    context_for_retry: str    # Injected into next EXECUTOR prompt
    next_step_hint: str       # Neutral hint for any agent (REVIEWER, QUALITY_GATE, etc.)
    confidence: float         # 0.0 - 1.0
    holder_queries_made: List[Dict[str, Any]] = field(default_factory=list)  # Audit trail
    consultation_outcomes: List[ConsultationOutcome] = field(default_factory=list)
    reasoning: str = ""
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['consultation_outcomes'] = [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.consultation_outcomes]
        return d


@dataclass
class DiagnosticResult:
    """
    Result of handle_tool_failure() decision.
    """
    action: Literal["retry", "human_review", "abort"]
    diagnosis: Optional[Diagnosis] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {"action": self.action, "message": self.message}
        if self.diagnosis:
            d["diagnosis"] = self.diagnosis.to_dict()
        return d
