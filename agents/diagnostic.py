"""
Diagnostic mode logic for CONFIG agent.

When tool execution fails, CONFIG switches to DIAGNOSTIC mode to:
1. Query holders (CodeHolder, ConvoHolder) to understand context
2. Analyze the failure
3. Suggest fixes and agents to consult
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .schemas import (
    ToolFailure, Diagnosis, ConfigMode, Project, AgentRole
)


# =============================================================================
# Agent Awareness Block - Injected into all agent prompts
# =============================================================================

AGENT_AWARENESS_TEMPLATE = """=== SWARM CONTEXT ===
PROJECT: {project_name}
GOAL: {project_goal}
CURRENT_WAVE: {wave_number}

YOU ARE: {agent_role}
YOUR SPECIALTY: {role_description}

OTHER AGENTS IN THIS SWARM:
┌─────────────┬────────────────────────────────────────────────────────┐
│ AGENT       │ SPECIALTY                                              │
├─────────────┼────────────────────────────────────────────────────────┤
│ CONFIG      │ Codebase analysis, modification planning, diagnostics  │
│ EXECUTOR    │ Implementation, writing complete files, tool emission  │
│ REVIEWER    │ Bug detection (P0/P1/P2), code quality audit           │
│ QUALITY_GATE│ Architectural fit, final verdict, scope check          │
│ ORCHESTRATOR│ Coordination, decision-making, human escalation        │
└─────────────┴────────────────────────────────────────────────────────┘

CONSULT_WITH GUIDANCE:
- CONFIG: "What files are affected?" / "What are the dependencies?"
- EXECUTOR: "How would you implement X?" / "Is this approach feasible?"
- REVIEWER: "Is this pattern safe?" / "What could break here?"
- QUALITY_GATE: "Does this fit the architecture?" / "Is this scope creep?"
- ORCHESTRATOR: Escalate when stuck, confidence < 0.5, or human needed

KNOWLEDGE RESOURCES:
- CodeHolder: get_file, search, get_structure, summarize
- ConvoHolder: get_wave, get_failures, summarize_all, search

OPERATING PRINCIPLE:
Never guess. Query holders. Suggest consultations. Admit uncertainty.
If you cannot solve it, say so and recommend who can help.
=== END SWARM CONTEXT ===
"""

ROLE_DESCRIPTIONS = {
    AgentRole.CONFIG: "Codebase analysis, modification planning, diagnostics",
    AgentRole.EXECUTOR: "Implementation, writing complete files, tool emission",
    AgentRole.REVIEWER: "Bug detection (P0/P1/P2), code quality audit",
    AgentRole.QUALITY_GATE: "Architectural fit, final verdict, scope check",
    AgentRole.ORCHESTRATOR: "Coordination, decision-making, human escalation",
}


def build_agent_awareness(
    project: Project,
    wave: str,
    agent_role: AgentRole
) -> str:
    """Build the agent awareness block for injection into prompts."""
    return AGENT_AWARENESS_TEMPLATE.format(
        project_name=project.name,
        project_goal=project.goal,
        wave_number=wave,
        agent_role=agent_role.value.upper(),
        role_description=ROLE_DESCRIPTIONS.get(agent_role, "Unknown specialty")
    )


# =============================================================================
# DIAGNOSTIC Mode Prompt
# =============================================================================

DIAGNOSTIC_PROMPT_TEMPLATE = """{agent_awareness}

MODE: DIAGNOSTIC

TOOL FAILURE REPORT:
- Wave: {wave}
- Tool: {tool_name}
- Arguments: {tool_args}
- Error Type: {error_type}
- Error Message: {error_message}
- Stack Trace:
{stack_trace}
- Attempt: {attempt}/3

EXECUTOR WAS TRYING TO:
{executor_context}

---

YOUR TASK:
1. Query holders to understand the failure context
2. Identify the root cause
3. Suggest a fix strategy
4. Recommend agents to consult if needed

HOLDER QUERY SYNTAX:
<holder_query holder="code" action="get_file" target="requirements.txt"/>
<holder_query holder="code" action="search" target="import {{module}}"/>
<holder_query holder="code" action="get_structure"/>
<holder_query holder="convo" action="get_wave" target="{{wave}}"/>
<holder_query holder="convo" action="get_failures"/>

Query as many times as needed. Then respond with:

---

DIAGNOSIS:

FAILURE_TYPE: (suggestions: import_missing, path_wrong, syntax_error, dep_conflict, env_missing, logic_error | other | cannot categorize | new category - describe)

ROOT_CAUSE: (describe in your own words, leave blank if truly unknown)

FIX_STRATEGY: (suggestions: retry_executor, retry_tool, add_dep, modify_config, human_review, abort | other - describe your approach)

CONSULT_WITH: (suggestions: EXECUTOR, REVIEWER, QUALITY_GATE, none | other - explain why)

CONTEXT_FOR_RETRY: (specific instructions to inject into next EXECUTOR prompt, or "N/A" if not retrying)

NEXT_STEP_HINT: (neutral one-liner any agent can use, e.g., "Refactor import placement before retry")

CONFIDENCE: (0.0 - 1.0, be honest)

REASONING:
(brief explanation of your diagnosis)
"""


def build_diagnostic_prompt(
    project: Project,
    failure: ToolFailure
) -> str:
    """Build the DIAGNOSTIC mode prompt for CONFIG."""
    agent_awareness = build_agent_awareness(project, failure.wave, AgentRole.CONFIG)

    return DIAGNOSTIC_PROMPT_TEMPLATE.format(
        agent_awareness=agent_awareness,
        wave=failure.wave,
        tool_name=failure.tool_name,
        tool_args=failure.tool_args,
        error_type=failure.error_type,
        error_message=failure.error_message,
        stack_trace=failure.stack_trace,
        attempt=failure.attempt,
        executor_context=failure.executor_context
    )


# =============================================================================
# Diagnosis Parsing
# =============================================================================

def parse_diagnosis_response(response: str) -> Diagnosis:
    """
    Parse CONFIG's DIAGNOSTIC mode response into a Diagnosis object.

    Uses flexible parsing - extracts what it can, provides defaults for missing fields.
    """
    # Extract holder queries made (for audit trail)
    holder_queries = []
    query_pattern = r'<holder_query\s+holder="(\w+)"\s+action="(\w+)"\s+target="([^"]*)"'
    for match in re.finditer(query_pattern, response):
        holder_queries.append({
            "holder": match.group(1),
            "action": match.group(2),
            "target": match.group(3)
        })

    # Parse structured fields
    def extract_field(name: str, default: str = "") -> str:
        pattern = rf'{name}:\s*(.+?)(?:\n[A-Z_]+:|$)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return default

    failure_type = extract_field("FAILURE_TYPE", "unknown")
    root_cause = extract_field("ROOT_CAUSE", "")
    fix_strategy = extract_field("FIX_STRATEGY", "human_review")
    consult_with_str = extract_field("CONSULT_WITH", "none")
    context_for_retry = extract_field("CONTEXT_FOR_RETRY", "N/A")
    next_step_hint = extract_field("NEXT_STEP_HINT", "")
    confidence_str = extract_field("CONFIDENCE", "0.5")
    reasoning = extract_field("REASONING", "")

    # Parse consult_with into list
    consult_with: List[str] = []
    if consult_with_str.lower() not in ["none", "n/a", ""]:
        # Extract agent names (EXECUTOR, REVIEWER, QUALITY_GATE)
        for agent in ["EXECUTOR", "REVIEWER", "QUALITY_GATE", "CONFIG"]:
            if agent in consult_with_str.upper():
                consult_with.append(agent)

    # Parse confidence
    try:
        confidence = float(confidence_str.split()[0])  # Handle "0.75 - based on..."
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, IndexError):
        confidence = 0.5

    return Diagnosis(
        failure_type=failure_type,
        root_cause=root_cause,
        fix_strategy=fix_strategy,
        consult_with=consult_with,
        context_for_retry=context_for_retry,
        next_step_hint=next_step_hint,
        confidence=confidence,
        holder_queries_made=holder_queries,
        reasoning=reasoning
    )


# =============================================================================
# Holder Query Execution
# =============================================================================

def extract_holder_queries(response: str) -> List[Dict[str, str]]:
    """Extract holder queries from CONFIG response for execution."""
    queries = []
    # Match <holder_query holder="code" action="get_file" target="path"/>
    pattern = r'<holder_query\s+holder="(\w+)"\s+action="(\w+)"\s+target="([^"]*)"'
    for match in re.finditer(pattern, response):
        queries.append({
            "holder": match.group(1),
            "action": match.group(2),
            "target": match.group(3)
        })
    return queries


def format_holder_results(results: List[Dict[str, Any]]) -> str:
    """Format holder query results for injection back into CONFIG."""
    if not results:
        return "No holder queries executed."

    lines = ["HOLDER QUERY RESULTS:\n"]
    for r in results:
        lines.append(f"[{r['holder'].upper()}:{r['action']}] target={r['target']}")
        if r.get('success'):
            content = r.get('content', '')
            if len(content) > 1000:
                content = content[:1000] + f"\n... ({len(content) - 1000} more chars)"
            lines.append(content)
        else:
            lines.append(f"ERROR: {r.get('error', 'unknown')}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Retry Context Builder
# =============================================================================

def build_retry_context(
    diagnosis: Diagnosis,
    original_task: str,
    consultation_insights: str = ""
) -> str:
    """
    Build the context to inject into EXECUTOR for retry.

    Combines diagnosis context with consultation insights.
    """
    parts = []

    parts.append("=== RETRY CONTEXT ===")
    parts.append(f"PREVIOUS ATTEMPT FAILED.")
    parts.append(f"FAILURE TYPE: {diagnosis.failure_type}")

    if diagnosis.root_cause:
        parts.append(f"ROOT CAUSE: {diagnosis.root_cause}")

    parts.append(f"\nFIX STRATEGY: {diagnosis.fix_strategy}")

    if diagnosis.context_for_retry and diagnosis.context_for_retry.upper() != "N/A":
        parts.append(f"\nSPECIFIC INSTRUCTIONS:")
        parts.append(diagnosis.context_for_retry)

    if diagnosis.next_step_hint:
        parts.append(f"\nHINT: {diagnosis.next_step_hint}")

    if consultation_insights:
        parts.append(f"\n{consultation_insights}")

    parts.append("=== END RETRY CONTEXT ===")
    parts.append(f"\nORIGINAL TASK: {original_task}")

    return "\n".join(parts)
