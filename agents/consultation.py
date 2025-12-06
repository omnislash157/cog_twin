"""
Cross-agent consultation flow.

When CONFIG suggests CONSULT_WITH agents, this module handles:
1. Building consultation prompts per agent specialty
2. Parsing structured consultation responses
3. Aggregating insights for retry context
"""

import re
from typing import Dict, Any, List, Optional

from .schemas import (
    ToolFailure, Diagnosis, ConsultationOutcome, Project, AgentRole
)
from .diagnostic import build_agent_awareness


# =============================================================================
# Consultation Prompt Templates
# =============================================================================

CONSULTATION_TEMPLATE = """{agent_awareness}

CONFIG diagnosed a tool failure and wants your input.

FAILURE: {tool_name} failed with {error_type}
ERROR MESSAGE: {error_message}
ROOT CAUSE (per CONFIG): {root_cause}
PROPOSED FIX: {fix_strategy}

{agent_specific_question}

RESPOND IN THIS EXACT FORMAT:
RISK: (what could go wrong with this fix, or "none identified")
CONFIDENCE: (0.0 - 1.0 in the proposed fix)
RECOMMENDED_ACTION: (one sentence: proceed | modify | abort | escalate)
NOTES: (optional, 1-2 sentences of additional context)
"""

AGENT_QUESTIONS = {
    "EXECUTOR": "QUESTION: Is this fix feasible to implement? Any technical blockers?",
    "REVIEWER": "QUESTION: Could this fix introduce new bugs? Is the pattern safe?",
    "QUALITY_GATE": "QUESTION: Does this fix align with the project architecture? Is it scope creep?",
    "CONFIG": "QUESTION: Is there additional context from the codebase that could help?",
}


def build_consultation_prompt(
    project: Project,
    wave: str,
    agent_name: str,
    failure: ToolFailure,
    diagnosis: Diagnosis
) -> Optional[str]:
    """
    Build consultation prompt for a specific agent.

    Returns None if agent_name is not recognized.
    """
    if agent_name not in AGENT_QUESTIONS:
        return None

    try:
        agent_role = AgentRole[agent_name]
    except KeyError:
        return None

    agent_awareness = build_agent_awareness(project, wave, agent_role)

    return CONSULTATION_TEMPLATE.format(
        agent_awareness=agent_awareness,
        tool_name=failure.tool_name,
        error_type=failure.error_type,
        error_message=failure.error_message,
        root_cause=diagnosis.root_cause or "Unknown",
        fix_strategy=diagnosis.fix_strategy,
        agent_specific_question=AGENT_QUESTIONS[agent_name]
    )


# =============================================================================
# Consultation Response Parsing
# =============================================================================

def parse_consultation_response(response: str) -> Dict[str, Any]:
    """
    Extract structured fields from consultation response.

    Handles malformed responses gracefully with sensible defaults.
    """
    result: Dict[str, Any] = {
        "risk": "unknown",
        "confidence": 0.5,
        "recommended_action": "proceed",
        "notes": ""
    }

    for line in response.strip().split("\n"):
        line = line.strip()

        if line.upper().startswith("RISK:"):
            result["risk"] = line.split(":", 1)[1].strip()

        elif line.upper().startswith("CONFIDENCE:"):
            try:
                conf_str = line.split(":", 1)[1].strip()
                # Handle formats like "0.8", "0.8 - high", "80%"
                conf_str = conf_str.split()[0].replace("%", "")
                conf = float(conf_str)
                if conf > 1.0:  # Assume percentage
                    conf = conf / 100.0
                result["confidence"] = max(0.0, min(1.0, conf))
            except (ValueError, IndexError):
                result["confidence"] = 0.5

        elif line.upper().startswith("RECOMMENDED_ACTION:"):
            result["recommended_action"] = line.split(":", 1)[1].strip()

        elif line.upper().startswith("NOTES:"):
            result["notes"] = line.split(":", 1)[1].strip()

    return result


def create_consultation_outcome(
    agent: str,
    parsed_response: Dict[str, Any]
) -> ConsultationOutcome:
    """Create a ConsultationOutcome from parsed response."""
    return ConsultationOutcome(
        agent=agent,
        risk=parsed_response.get("risk", "unknown"),
        confidence=parsed_response.get("confidence", 0.5),
        recommended_action=parsed_response.get("recommended_action", "proceed"),
        notes=parsed_response.get("notes", ""),
        helpful=None  # Set after retry success/fail
    )


# =============================================================================
# Consultation Aggregation
# =============================================================================

def format_consultation_insights(outcomes: List[ConsultationOutcome]) -> str:
    """
    Format consultation outcomes for injection into EXECUTOR retry prompt.
    """
    if not outcomes:
        return ""

    lines = ["CONSULTATION INSIGHTS:"]

    for outcome in outcomes:
        lines.append(f"\n[{outcome.agent}]")
        lines.append(f"  Risk: {outcome.risk}")
        lines.append(f"  Confidence: {outcome.confidence:.1%}")
        lines.append(f"  Action: {outcome.recommended_action}")
        if outcome.notes:
            lines.append(f"  Notes: {outcome.notes}")

    return "\n".join(lines)


def aggregate_consultation_confidence(outcomes: List[ConsultationOutcome]) -> float:
    """
    Aggregate confidence from multiple consultations.

    Uses weighted average where lower confidence has more weight (pessimistic).
    """
    if not outcomes:
        return 0.5

    # Weight lower confidence more heavily (be conservative)
    total_weight = 0.0
    weighted_sum = 0.0

    for outcome in outcomes:
        weight = 2.0 - outcome.confidence  # Lower confidence = higher weight
        weighted_sum += outcome.confidence * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5

    return weighted_sum / total_weight


def should_abort_based_on_consultations(outcomes: List[ConsultationOutcome]) -> bool:
    """
    Check if any consultation recommends abort.

    Returns True if any agent explicitly recommends abort or escalate.
    """
    for outcome in outcomes:
        action = outcome.recommended_action.lower()
        if "abort" in action or "escalate" in action:
            return True
    return False


def get_consultation_warnings(outcomes: List[ConsultationOutcome]) -> List[str]:
    """
    Extract warnings from consultations for human review.
    """
    warnings = []

    for outcome in outcomes:
        if outcome.confidence < 0.5:
            warnings.append(f"{outcome.agent} has low confidence ({outcome.confidence:.0%})")

        if outcome.risk.lower() not in ["none", "none identified", "n/a", ""]:
            warnings.append(f"{outcome.agent} identified risk: {outcome.risk}")

    return warnings


# =============================================================================
# Consultation Flow Control
# =============================================================================

def determine_agents_to_consult(diagnosis: Diagnosis) -> List[str]:
    """
    Determine which agents to consult based on diagnosis.

    Filters to valid, consultable agents.
    """
    valid_agents = {"EXECUTOR", "REVIEWER", "QUALITY_GATE", "CONFIG"}

    agents = []
    for agent in diagnosis.consult_with:
        agent_upper = agent.upper()
        if agent_upper in valid_agents:
            agents.append(agent_upper)

    return agents


def merge_context_for_retry(
    diagnosis: Diagnosis,
    consultation_insights: str
) -> str:
    """
    Merge diagnosis context with consultation insights.

    Handles "N/A" and empty context gracefully.
    """
    base_ctx = diagnosis.context_for_retry or ""
    if base_ctx.strip().upper() == "N/A":
        base_ctx = ""

    if not consultation_insights:
        return base_ctx

    if base_ctx:
        return f"{base_ctx}\n\n{consultation_insights}"
    else:
        return consultation_insights
