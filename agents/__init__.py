"""Multi-agent wave loop for coding swarm."""
from .registry import AgentRole, spawn_agent
from .orchestrator import run_wave, run_project
from .swarm_orchestrator import SwarmOrchestrator
from .schemas import Project, WaveSummary, Failure, Verdict
from .persistence import SwarmPersistence
from .holders import CodeHolder, ConvoHolder, HolderQuery
from .sandbox_executor import (
    SandboxExecutor, ToolCall, ToolResult, ExecutionResult, FailureInfo,
    PromotionRequest, PromotionResult, HITLRequest, HITLType,
    parse_tool_calls, format_results, KNOWN_PACKAGES
)
from .diagnostic import (
    build_agent_awareness, build_diagnostic_prompt,
    parse_diagnosis_response, build_retry_context
)
from .consultation import (
    build_consultation_prompt, parse_consultation_response,
    create_consultation_outcome, format_consultation_insights,
    determine_agents_to_consult, merge_context_for_retry,
    aggregate_consultation_confidence, should_abort_based_on_consultations,
    get_consultation_warnings
)
from .schemas import (
    ToolFailure, Diagnosis, DiagnosticResult, ConsultationOutcome, ConfigMode
)

__all__ = [
    # Core
    "AgentRole", "spawn_agent", "run_wave", "run_project",
    "SwarmOrchestrator", "SwarmPersistence",
    "Project", "WaveSummary", "Failure", "Verdict",
    "CodeHolder", "ConvoHolder", "HolderQuery",
    # Sandbox Executor
    "SandboxExecutor", "ToolCall", "ToolResult", "ExecutionResult", "FailureInfo",
    "PromotionRequest", "PromotionResult", "HITLRequest", "HITLType",
    "parse_tool_calls", "format_results", "KNOWN_PACKAGES",
    # Diagnostic (Phase 10e)
    "build_agent_awareness", "build_diagnostic_prompt",
    "parse_diagnosis_response", "build_retry_context",
    # Consultation (Phase 10e)
    "build_consultation_prompt", "parse_consultation_response",
    "create_consultation_outcome", "format_consultation_insights",
    "determine_agents_to_consult", "merge_context_for_retry",
    "aggregate_consultation_confidence", "should_abort_based_on_consultations",
    "get_consultation_warnings",
    # Schemas (Phase 10e)
    "ToolFailure", "Diagnosis", "DiagnosticResult", "ConsultationOutcome", "ConfigMode",
]
