"""Multi-agent wave loop for coding swarm."""
from .registry import AgentRole, spawn_agent
from .schemas import Project, WaveSummary, Failure, Verdict
from .persistence import SwarmPersistence
from .holders import CodeHolder, ConvoHolder, HolderQuery
from .sandbox_executor import (
    SandboxExecutor, ToolCall, ToolResult, ExecutionResult, FailureInfo,
    PromotionRequest, PromotionResult, HITLRequest, HITLType,
    parse_tool_calls, format_results, KNOWN_PACKAGES
)
from .schemas import (
    ToolFailure, Diagnosis, DiagnosticResult, ConsultationOutcome, ConfigMode
)
from .holder_daemon import (
    DaemonManager, CodeHolderDaemon, ConvoHolderDaemon, UserHolderDaemon
)
from .claude_orchestrator import ClaudeOrchestrator, run_swarm

__all__ = [
    # Core
    "AgentRole", "spawn_agent",
    "SwarmPersistence",
    "Project", "WaveSummary", "Failure", "Verdict",
    "CodeHolder", "ConvoHolder", "HolderQuery",
    # Sandbox Executor
    "SandboxExecutor", "ToolCall", "ToolResult", "ExecutionResult", "FailureInfo",
    "PromotionRequest", "PromotionResult", "HITLRequest", "HITLType",
    "parse_tool_calls", "format_results", "KNOWN_PACKAGES",
    # Schemas
    "ToolFailure", "Diagnosis", "DiagnosticResult", "ConsultationOutcome", "ConfigMode",
    # Holder Daemons
    "DaemonManager", "CodeHolderDaemon", "ConvoHolderDaemon", "UserHolderDaemon",
    # Claude Orchestrator
    "ClaudeOrchestrator", "run_swarm",
]
