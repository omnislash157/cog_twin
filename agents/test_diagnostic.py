"""Tests for Phase 10e: Diagnostic and Consultation flows."""

import pytest
from pathlib import Path

from agents.schemas import (
    Project, ToolFailure, Diagnosis, ConsultationOutcome, ConfigMode, AgentRole
)
from agents.diagnostic import (
    build_agent_awareness, build_diagnostic_prompt,
    parse_diagnosis_response, build_retry_context,
    extract_holder_queries, format_holder_results
)
from agents.consultation import (
    build_consultation_prompt, parse_consultation_response,
    create_consultation_outcome, format_consultation_insights,
    determine_agents_to_consult, merge_context_for_retry,
    aggregate_consultation_confidence, should_abort_based_on_consultations,
    get_consultation_warnings
)
from agents.persistence import SwarmPersistence


class TestAgentAwareness:
    """Test agent awareness block generation."""

    def test_build_awareness_for_config(self):
        project = Project(name="test_proj", goal="Build a thing")
        awareness = build_agent_awareness(project, "001", AgentRole.CONFIG)

        assert "PROJECT: test_proj" in awareness
        assert "GOAL: Build a thing" in awareness
        assert "CURRENT_WAVE: 001" in awareness
        assert "YOU ARE: CONFIG" in awareness
        assert "Codebase analysis" in awareness

    def test_build_awareness_for_executor(self):
        project = Project(name="test_proj", goal="Build a thing")
        awareness = build_agent_awareness(project, "002", AgentRole.EXECUTOR)

        assert "YOU ARE: EXECUTOR" in awareness
        assert "Implementation" in awareness


class TestDiagnosticPrompt:
    """Test DIAGNOSTIC mode prompt generation."""

    def test_build_diagnostic_prompt(self):
        project = Project(name="test_proj", goal="Build API")
        failure = ToolFailure(
            wave="001",
            tool_name="write_file",
            tool_args={"path": "app.py", "content": "..."},
            error_type="FileNotFoundError",
            error_message="No such file or directory",
            stack_trace="Traceback...",
            attempt=2,
            executor_context="Writing main app file"
        )

        prompt = build_diagnostic_prompt(project, failure)

        assert "MODE: DIAGNOSTIC" in prompt
        assert "write_file" in prompt
        assert "FileNotFoundError" in prompt
        assert "attempt=2" in prompt or "Attempt: 2/3" in prompt
        assert "HOLDER QUERY SYNTAX" in prompt

    def test_parse_diagnosis_response_complete(self):
        response = """
I analyzed the failure.

<holder_query holder="code" action="get_file" target="requirements.txt"/>

DIAGNOSIS:

FAILURE_TYPE: import_missing

ROOT_CAUSE: The fastapi package is not installed in the environment.

FIX_STRATEGY: add_dep

CONSULT_WITH: EXECUTOR, REVIEWER

CONTEXT_FOR_RETRY: Install fastapi before retrying: pip install fastapi

NEXT_STEP_HINT: Add fastapi to requirements.txt

CONFIDENCE: 0.85

REASONING:
The error clearly shows ModuleNotFoundError for fastapi.
"""
        diagnosis = parse_diagnosis_response(response)

        assert diagnosis.failure_type == "import_missing"
        assert "fastapi" in diagnosis.root_cause
        assert diagnosis.fix_strategy == "add_dep"
        assert "EXECUTOR" in diagnosis.consult_with
        assert "REVIEWER" in diagnosis.consult_with
        assert diagnosis.confidence == 0.85
        assert len(diagnosis.holder_queries_made) == 1

    def test_parse_diagnosis_response_minimal(self):
        response = """
FAILURE_TYPE: unknown

ROOT_CAUSE:

FIX_STRATEGY: human_review

CONSULT_WITH: none

CONFIDENCE: 0.3
"""
        diagnosis = parse_diagnosis_response(response)

        assert diagnosis.failure_type == "unknown"
        assert diagnosis.fix_strategy == "human_review"
        assert len(diagnosis.consult_with) == 0
        assert diagnosis.confidence == 0.3


class TestHolderQueries:
    """Test holder query extraction."""

    def test_extract_holder_queries(self):
        response = """
Let me check the codebase:
<holder_query holder="code" action="get_file" target="requirements.txt"/>
<holder_query holder="code" action="search" target="import fastapi"/>
<holder_query holder="convo" action="get_failures" target=""/>
"""
        queries = extract_holder_queries(response)

        assert len(queries) == 3
        assert queries[0]["holder"] == "code"
        assert queries[0]["action"] == "get_file"
        assert queries[1]["action"] == "search"

    def test_format_holder_results(self):
        results = [
            {"holder": "code", "action": "get_file", "target": "req.txt", "success": True, "content": "fastapi==0.100"},
            {"holder": "code", "action": "search", "target": "import", "success": False, "error": "not found"},
        ]
        formatted = format_holder_results(results)

        assert "[CODE:get_file]" in formatted
        assert "fastapi==0.100" in formatted
        assert "ERROR: not found" in formatted


class TestRetryContext:
    """Test retry context building."""

    def test_build_retry_context_basic(self):
        diagnosis = Diagnosis(
            failure_type="import_missing",
            root_cause="Missing dependency",
            fix_strategy="add_dep",
            consult_with=[],
            context_for_retry="pip install fastapi first",
            next_step_hint="Check requirements.txt",
            confidence=0.8
        )

        ctx = build_retry_context(diagnosis, "Add /api/users endpoint")

        assert "RETRY CONTEXT" in ctx
        assert "import_missing" in ctx
        assert "pip install fastapi" in ctx
        assert "Add /api/users endpoint" in ctx

    def test_build_retry_context_with_consultation(self):
        diagnosis = Diagnosis(
            failure_type="logic_error",
            root_cause="Wrong import",
            fix_strategy="retry_executor",
            consult_with=["REVIEWER"],
            context_for_retry="Use correct import path",
            next_step_hint="Fix import",
            confidence=0.7
        )

        insights = "CONSULTATION INSIGHTS:\n[REVIEWER] Risk: none"
        ctx = build_retry_context(diagnosis, "Fix imports", insights)

        assert "CONSULTATION INSIGHTS" in ctx
        assert "REVIEWER" in ctx


class TestConsultation:
    """Test consultation flow."""

    def test_build_consultation_prompt(self):
        project = Project(name="test", goal="test goal")
        failure = ToolFailure(
            wave="001",
            tool_name="run_command",
            tool_args={"cmd": "pytest"},
            error_type="RuntimeError",
            error_message="Tests failed",
            stack_trace="...",
            attempt=3,
            executor_context="Running tests"
        )
        diagnosis = Diagnosis(
            failure_type="logic_error",
            root_cause="Test assertion failed",
            fix_strategy="retry_executor",
            consult_with=["EXECUTOR"],
            context_for_retry="Fix the logic",
            next_step_hint="Debug test",
            confidence=0.6
        )

        prompt = build_consultation_prompt(project, "001", "EXECUTOR", failure, diagnosis)

        assert prompt is not None
        assert "EXECUTOR" in prompt
        assert "Is this fix feasible" in prompt
        assert "RuntimeError" in prompt

    def test_build_consultation_prompt_invalid_agent(self):
        project = Project(name="test", goal="test goal")
        failure = ToolFailure(
            wave="001", tool_name="x", tool_args={},
            error_type="E", error_message="e", stack_trace="",
            attempt=1, executor_context=""
        )
        diagnosis = Diagnosis(
            failure_type="x", root_cause="", fix_strategy="",
            consult_with=[], context_for_retry="", next_step_hint="",
            confidence=0.5
        )

        prompt = build_consultation_prompt(project, "001", "INVALID_AGENT", failure, diagnosis)
        assert prompt is None

    def test_parse_consultation_response(self):
        response = """
RISK: Could introduce race conditions
CONFIDENCE: 0.75
RECOMMENDED_ACTION: proceed with caution
NOTES: Consider adding locks
"""
        parsed = parse_consultation_response(response)

        assert parsed["risk"] == "Could introduce race conditions"
        assert parsed["confidence"] == 0.75
        assert "proceed" in parsed["recommended_action"]
        assert "locks" in parsed["notes"]

    def test_create_consultation_outcome(self):
        parsed = {
            "risk": "none",
            "confidence": 0.9,
            "recommended_action": "proceed",
            "notes": ""
        }
        outcome = create_consultation_outcome("REVIEWER", parsed)

        assert outcome.agent == "REVIEWER"
        assert outcome.confidence == 0.9
        assert outcome.helpful is None  # Not yet known


class TestConsultationAggregation:
    """Test consultation aggregation and decision logic."""

    def test_format_consultation_insights(self):
        outcomes = [
            ConsultationOutcome("EXECUTOR", "none", 0.9, "proceed", ""),
            ConsultationOutcome("REVIEWER", "possible bug", 0.6, "modify approach", "check edge cases"),
        ]
        formatted = format_consultation_insights(outcomes)

        assert "[EXECUTOR]" in formatted
        assert "[REVIEWER]" in formatted
        assert "90" in formatted  # 90.0% or 90%
        assert "check edge cases" in formatted

    def test_aggregate_confidence_high(self):
        outcomes = [
            ConsultationOutcome("EXECUTOR", "none", 0.9, "proceed", ""),
            ConsultationOutcome("REVIEWER", "none", 0.8, "proceed", ""),
        ]
        conf = aggregate_consultation_confidence(outcomes)
        assert conf > 0.7

    def test_aggregate_confidence_low(self):
        outcomes = [
            ConsultationOutcome("EXECUTOR", "high risk", 0.3, "abort", ""),
            ConsultationOutcome("REVIEWER", "high risk", 0.2, "abort", ""),
        ]
        conf = aggregate_consultation_confidence(outcomes)
        assert conf < 0.5

    def test_should_abort_true(self):
        outcomes = [
            ConsultationOutcome("EXECUTOR", "none", 0.9, "proceed", ""),
            ConsultationOutcome("REVIEWER", "major issue", 0.3, "abort immediately", ""),
        ]
        assert should_abort_based_on_consultations(outcomes) is True

    def test_should_abort_false(self):
        outcomes = [
            ConsultationOutcome("EXECUTOR", "none", 0.9, "proceed", ""),
            ConsultationOutcome("REVIEWER", "minor", 0.7, "proceed with caution", ""),
        ]
        assert should_abort_based_on_consultations(outcomes) is False

    def test_get_warnings(self):
        outcomes = [
            ConsultationOutcome("EXECUTOR", "none", 0.9, "proceed", ""),
            ConsultationOutcome("REVIEWER", "possible regression", 0.4, "modify", ""),
        ]
        warnings = get_consultation_warnings(outcomes)

        assert len(warnings) >= 1
        assert any("REVIEWER" in w for w in warnings)

    def test_determine_agents_to_consult(self):
        diagnosis = Diagnosis(
            failure_type="x",
            root_cause="",
            fix_strategy="",
            consult_with=["EXECUTOR", "reviewer", "INVALID"],
            context_for_retry="",
            next_step_hint="",
            confidence=0.5
        )
        agents = determine_agents_to_consult(diagnosis)

        assert "EXECUTOR" in agents
        assert "REVIEWER" in agents
        assert "INVALID" not in agents


class TestMergeContext:
    """Test context merging for retry."""

    def test_merge_with_both(self):
        diagnosis = Diagnosis(
            failure_type="x",
            root_cause="",
            fix_strategy="",
            consult_with=[],
            context_for_retry="Fix the import",
            next_step_hint="",
            confidence=0.5
        )
        insights = "CONSULTATION: proceed"
        merged = merge_context_for_retry(diagnosis, insights)

        assert "Fix the import" in merged
        assert "CONSULTATION" in merged

    def test_merge_na_context(self):
        diagnosis = Diagnosis(
            failure_type="x",
            root_cause="",
            fix_strategy="",
            consult_with=[],
            context_for_retry="N/A",
            next_step_hint="",
            confidence=0.5
        )
        insights = "CONSULTATION: proceed"
        merged = merge_context_for_retry(diagnosis, insights)

        assert "N/A" not in merged
        assert "CONSULTATION" in merged


class TestDiagnosisPersistence:
    """Test diagnosis persistence."""

    def test_save_and_load_diagnosis(self, tmp_path):
        persistence = SwarmPersistence(tmp_path)
        project = persistence.create_project("test", "test goal")

        diagnosis = Diagnosis(
            failure_type="import_missing",
            root_cause="Missing fastapi",
            fix_strategy="add_dep",
            consult_with=["EXECUTOR"],
            context_for_retry="pip install fastapi",
            next_step_hint="Check requirements",
            confidence=0.8,
            consultation_outcomes=[
                ConsultationOutcome("EXECUTOR", "none", 0.9, "proceed", "looks good")
            ]
        )

        persistence.save_diagnosis(project.id, "001", diagnosis)
        loaded = persistence.load_diagnosis(project.id, "001")

        assert loaded is not None
        assert loaded.failure_type == "import_missing"
        assert loaded.confidence == 0.8
        assert len(loaded.consultation_outcomes) == 1
        assert loaded.consultation_outcomes[0].agent == "EXECUTOR"

    def test_get_diagnostics_empty(self, tmp_path):
        persistence = SwarmPersistence(tmp_path)
        project = persistence.create_project("test", "test goal")

        diagnostics = persistence.get_diagnostics(project.id)
        assert len(diagnostics) == 0

    def test_mark_consultation_helpful(self, tmp_path):
        persistence = SwarmPersistence(tmp_path)
        project = persistence.create_project("test", "test goal")

        diagnosis = Diagnosis(
            failure_type="x",
            root_cause="",
            fix_strategy="",
            consult_with=[],
            context_for_retry="",
            next_step_hint="",
            confidence=0.5,
            consultation_outcomes=[
                ConsultationOutcome("EXECUTOR", "none", 0.9, "proceed", "")
            ]
        )

        persistence.save_diagnosis(project.id, "001", diagnosis)
        persistence.mark_consultation_helpful(project.id, "001", "EXECUTOR", True)

        loaded = persistence.load_diagnosis(project.id, "001")
        assert loaded.consultation_outcomes[0].helpful is True

    def test_get_agent_helpfulness_stats(self, tmp_path):
        persistence = SwarmPersistence(tmp_path)
        project = persistence.create_project("test", "test goal")

        # Save two diagnoses with different outcomes
        d1 = Diagnosis(
            failure_type="x", root_cause="", fix_strategy="", consult_with=[],
            context_for_retry="", next_step_hint="", confidence=0.5,
            consultation_outcomes=[
                ConsultationOutcome("EXECUTOR", "", 0.9, "proceed", "", helpful=True),
                ConsultationOutcome("REVIEWER", "", 0.8, "proceed", "", helpful=True),
            ]
        )
        d2 = Diagnosis(
            failure_type="y", root_cause="", fix_strategy="", consult_with=[],
            context_for_retry="", next_step_hint="", confidence=0.5,
            consultation_outcomes=[
                ConsultationOutcome("EXECUTOR", "", 0.9, "proceed", "", helpful=False),
            ]
        )

        persistence.save_diagnosis(project.id, "001", d1)
        persistence.save_diagnosis(project.id, "002", d2)

        stats = persistence.get_agent_helpfulness_stats(project.id)

        assert "EXECUTOR" in stats
        assert stats["EXECUTOR"]["helpful"] == 1
        assert stats["EXECUTOR"]["unhelpful"] == 1
        assert "REVIEWER" in stats
        assert stats["REVIEWER"]["helpful"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
