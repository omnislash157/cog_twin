"""
SwarmOrchestrator - New orchestrator with full persistence and WebSocket broadcasting.
Import this into orchestrator.py.
"""
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Awaitable

from .registry import AgentRole, spawn_agent, AGENTS
from .schemas import (
    Project, WaveSummary, Failure,
    OutboundTurn, InboundTurn, FilesWritten, FileOperation,
    Verdict, now_iso,
    ToolFailure, Diagnosis, DiagnosticResult, ConsultationOutcome
)
from .persistence import SwarmPersistence
from .reasoning import extract_reasoning_trace
from .holders import CodeHolder, ConvoHolder, HolderQuery
from .sandbox_executor import (
    SandboxExecutor, parse_tool_calls, format_results,
    ExecutionResult, KNOWN_PACKAGES, FailureInfo
)
from .diagnostic import (
    build_diagnostic_prompt, parse_diagnosis_response, build_retry_context
)
from .consultation import (
    build_consultation_prompt, parse_consultation_response,
    create_consultation_outcome, format_consultation_insights,
    determine_agents_to_consult, merge_context_for_retry,
    aggregate_consultation_confidence, should_abort_based_on_consultations,
    get_consultation_warnings
)
from .registry import spawn_agent_with_swarm_context


# Type for WebSocket broadcast callback
BroadcastCallback = Callable[[str], Awaitable[None]]


class SwarmOrchestrator:
    """
    Main orchestrator for multi-agent coding swarm.
    Full persistence of every turn, failure tracking, holder queries.
    Now with WebSocket broadcasting for live dashboard updates.
    """

    def __init__(
        self,
        project_root: Path,
        project_name: str,
        goal: str,
        broadcast_callback: Optional[BroadcastCallback] = None,
        sandbox_root: Optional[Path] = None
    ):
        self.project_root = Path(project_root)
        self.persistence = SwarmPersistence()
        self.project = self.persistence.create_project(project_name, goal)
        self.code_holder = CodeHolder(project_root)
        self.convo_holder = ConvoHolder(self.persistence, self.project.id)
        self.current_wave: str = "001"
        self.sequence: int = 0
        self.wave_tokens_in: int = 0
        self.wave_tokens_out: int = 0
        self.wave_agents: List[str] = []
        self._broadcast = broadcast_callback

        # Initialize sandbox executor
        sandbox_path = sandbox_root or (Path(__file__).parent / "sandbox")
        self.sandbox = SandboxExecutor(sandbox_path, self.project_root)
        self.execution_enabled = True  # Set to False to skip execution

        # Wave consultation state tracking (Phase 10e: Resilience Loop)
        self._wave_consulted: Dict[str, bool] = {}

    # === Wave Consultation State (Phase 10e) ===

    def _mark_wave_consulted(self, wave: str) -> None:
        """Mark that consultation has occurred for this wave."""
        self._wave_consulted[wave] = True

    def _wave_was_consulted(self, wave: str) -> bool:
        """Check if consultation already occurred for this wave."""
        return self._wave_consulted.get(wave, False)

    async def _broadcast_event(self, event: Dict[str, Any]) -> None:
        """Broadcast event to WebSocket clients if callback is set."""
        if self._broadcast:
            try:
                await self._broadcast(json.dumps(event))
            except Exception as e:
                print(f"[BROADCAST] Error: {e}")

    async def _broadcast_turn(self, outbound: OutboundTurn, inbound: InboundTurn) -> None:
        """Broadcast turn completion to WebSocket clients."""
        event = {
            "type": "swarm_turn",
            "project_id": self.project.id,
            "wave": self.current_wave,
            "sequence": self.sequence,
            "agent": inbound.from_agent,
            "tokens_in": outbound.tokens_in,
            "tokens_out": inbound.tokens_out,
            "latency_ms": inbound.latency_ms,
            "reasoning": [
                {"step": r.step, "content": r.content}
                for r in inbound.reasoning_trace
            ],
            "preview": inbound.raw_response[:500],
            "parsed": inbound.parsed,
            "timestamp": now_iso(),
            "wave_progress": {
                "current_agent": inbound.from_agent,
                "agents_completed": self.wave_agents.copy(),
                "total_tokens_in": self.wave_tokens_in,
                "total_tokens_out": self.wave_tokens_out,
            }
        }
        await self._broadcast_event(event)

    async def _broadcast_wave_start(self, task: str) -> None:
        """Broadcast wave start event."""
        event = {
            "type": "swarm_wave_start",
            "project_id": self.project.id,
            "wave": self.current_wave,
            "task": task,
            "timestamp": now_iso(),
        }
        await self._broadcast_event(event)

    async def _broadcast_wave_end(self, summary: WaveSummary) -> None:
        """Broadcast wave completion event."""
        event = {
            "type": "swarm_wave_end",
            "project_id": self.project.id,
            "wave": self.current_wave,
            "verdict": summary.verdict,
            "total_tokens_in": summary.total_tokens_in,
            "total_tokens_out": summary.total_tokens_out,
            "files_modified": summary.files_modified,
            "summary": summary.summary,
            "timestamp": now_iso(),
        }
        await self._broadcast_event(event)

    async def _broadcast_failure(self, failure: Failure) -> None:
        """Broadcast failure event for dashboard alerts."""
        event = {
            "type": "swarm_failure",
            "project_id": self.project.id,
            "wave": self.current_wave,
            "failure_type": failure.failure_type,
            "agent_blamed": failure.agent_blamed,
            "root_cause": failure.root_cause,
            "recommendation": failure.recommendation,
            "timestamp": now_iso(),
        }
        await self._broadcast_event(event)

    async def _broadcast_diagnostic(self, diagnosis: Diagnosis, failure: ToolFailure) -> None:
        """Broadcast diagnostic event for dashboard visibility."""
        event = {
            "type": "swarm_diagnostic",
            "project_id": self.project.id,
            "wave": self.current_wave,
            "failure_type": diagnosis.failure_type,
            "root_cause": diagnosis.root_cause,
            "fix_strategy": diagnosis.fix_strategy,
            "consult_with": diagnosis.consult_with,
            "confidence": diagnosis.confidence,
            "tool_failed": failure.tool_name,
            "attempt": failure.attempt,
            "timestamp": now_iso(),
        }
        await self._broadcast_event(event)

    async def _broadcast_consultation(
        self,
        agent: str,
        outcome: ConsultationOutcome
    ) -> None:
        """Broadcast consultation result for dashboard visibility."""
        event = {
            "type": "swarm_consultation",
            "project_id": self.project.id,
            "wave": self.current_wave,
            "agent": agent,
            "risk": outcome.risk,
            "confidence": outcome.confidence,
            "recommended_action": outcome.recommended_action,
            "notes": outcome.notes,
            "timestamp": now_iso(),
        }
        await self._broadcast_event(event)

    # === Resilience Loop (Phase 10e) ===

    def _build_tool_failure(
        self,
        failure_info: FailureInfo,
        wave: str
    ) -> ToolFailure:
        """Convert sandbox FailureInfo to ToolFailure for diagnostic flow."""
        return ToolFailure(
            wave=wave,
            tool_name=failure_info.tool_name,
            tool_args=failure_info.tool_args,
            error_type=failure_info.error_type,
            error_message=failure_info.error_message,
            stack_trace=failure_info.stack_trace,
            attempt=failure_info.attempt,
            executor_context=failure_info.executor_context,
            consulted_already=self._wave_was_consulted(wave)
        )

    async def _spawn_config_diagnostic(
        self,
        failure: ToolFailure
    ) -> Diagnosis:
        """
        Spawn CONFIG in DIAGNOSTIC mode to analyze the failure.

        Returns a Diagnosis with fix strategy and agents to consult.
        """
        print(f"\n[DIAGNOSTIC] CONFIG analyzing failure: {failure.tool_name}")

        # Build diagnostic prompt
        diagnostic_prompt = build_diagnostic_prompt(self.project, failure)

        # Spawn CONFIG with swarm context
        response = await spawn_agent_with_swarm_context(
            role=AgentRole.CONFIG,
            message=diagnostic_prompt,
            project_name=self.project.name,
            project_goal=self.project.goal,
            wave=failure.wave,
            context=""
        )

        # Parse the diagnosis
        diagnosis = parse_diagnosis_response(response)

        print(f"[DIAGNOSTIC] Failure type: {diagnosis.failure_type}")
        print(f"[DIAGNOSTIC] Fix strategy: {diagnosis.fix_strategy}")
        print(f"[DIAGNOSTIC] Confidence: {diagnosis.confidence:.0%}")
        if diagnosis.consult_with:
            print(f"[DIAGNOSTIC] Consult with: {', '.join(diagnosis.consult_with)}")

        # Broadcast diagnostic event
        await self._broadcast_diagnostic(diagnosis, failure)

        return diagnosis

    async def _run_consultations(
        self,
        failure: ToolFailure,
        diagnosis: Diagnosis
    ) -> List[ConsultationOutcome]:
        """
        Run consultations with agents recommended by CONFIG.

        Returns list of consultation outcomes.
        """
        agents_to_consult = determine_agents_to_consult(diagnosis)

        if not agents_to_consult:
            print("[CONSULTATION] No agents to consult")
            return []

        print(f"\n[CONSULTATION] Consulting: {', '.join(agents_to_consult)}")

        outcomes: List[ConsultationOutcome] = []

        for agent_name in agents_to_consult:
            # Build consultation prompt
            prompt = build_consultation_prompt(
                project=self.project,
                wave=self.current_wave,
                agent_name=agent_name,
                failure=failure,
                diagnosis=diagnosis
            )

            if prompt is None:
                continue

            try:
                # Map agent name to role
                agent_role = AgentRole[agent_name]

                # Spawn agent with swarm context
                response = await spawn_agent_with_swarm_context(
                    role=agent_role,
                    message=prompt,
                    project_name=self.project.name,
                    project_goal=self.project.goal,
                    wave=self.current_wave,
                    context=""
                )

                # Parse consultation response
                parsed = parse_consultation_response(response)
                outcome = create_consultation_outcome(agent_name, parsed)
                outcomes.append(outcome)

                print(f"[CONSULTATION] {agent_name}: {outcome.recommended_action} (conf: {outcome.confidence:.0%})")

                # Broadcast consultation result
                await self._broadcast_consultation(agent_name, outcome)

            except Exception as e:
                print(f"[CONSULTATION] Error consulting {agent_name}: {e}")

        # Mark wave as consulted
        self._mark_wave_consulted(self.current_wave)

        return outcomes

    async def handle_tool_failure(
        self,
        failure_info: FailureInfo,
        original_task: str
    ) -> tuple:
        """
        Handle tool execution failure with diagnostic + consultation flow.

        Decision tree:
        1. If retries exhausted AND not consulted → run diagnostic + consultation
        2. If confidence >= 0.7 → retry with context
        3. If confidence < 0.5 OR abort recommended → human_review
        4. If already consulted AND still failing → abort

        Returns: (action, diagnosis, retry_context)
        Where action is "retry" | "human_review" | "abort"
        """
        failure = self._build_tool_failure(failure_info, self.current_wave)

        # If already consulted and still failing, abort
        if failure.consulted_already:
            print("[DIAGNOSTIC] Already consulted, still failing → abort")
            return ("abort", None, "")

        # Run CONFIG DIAGNOSTIC mode
        diagnosis = await self._spawn_config_diagnostic(failure)

        # Save diagnosis for learning
        self.persistence.save_diagnosis(self.project.id, self.current_wave, diagnosis)

        # Check for human_review or abort strategies
        if diagnosis.fix_strategy.lower() in ["human_review", "abort"]:
            print(f"[DIAGNOSTIC] CONFIG recommends: {diagnosis.fix_strategy}")
            return (diagnosis.fix_strategy.lower(), diagnosis, "")

        # Run consultations if recommended
        consultation_insights = ""
        if diagnosis.consult_with:
            outcomes = await self._run_consultations(failure, diagnosis)
            diagnosis.consultation_outcomes = outcomes

            # Update saved diagnosis with consultation outcomes
            self.persistence.save_diagnosis(self.project.id, self.current_wave, diagnosis)

            # Check if any agent recommends abort
            if should_abort_based_on_consultations(outcomes):
                warnings = get_consultation_warnings(outcomes)
                print(f"[DIAGNOSTIC] Consultation recommends abort: {warnings}")
                return ("human_review", diagnosis, "")

            # Aggregate consultation confidence
            agg_confidence = aggregate_consultation_confidence(outcomes)
            if agg_confidence < 0.5:
                print(f"[DIAGNOSTIC] Aggregate confidence too low: {agg_confidence:.0%}")
                return ("human_review", diagnosis, "")

            # Format insights for retry
            consultation_insights = format_consultation_insights(outcomes)

        # Check diagnosis confidence
        if diagnosis.confidence < 0.5:
            print(f"[DIAGNOSTIC] Confidence too low: {diagnosis.confidence:.0%}")
            return ("human_review", diagnosis, "")

        # Build retry context
        retry_context = build_retry_context(
            diagnosis=diagnosis,
            original_task=original_task,
            consultation_insights=consultation_insights
        )

        print(f"[DIAGNOSTIC] Proceeding with retry")
        return ("retry", diagnosis, retry_context)

    async def run_project(self, tasks: List[str], max_waves: int = 10) -> Project:
        """Run full project through wave loop."""
        for task in tasks[:max_waves]:
            self.current_wave = self.persistence.next_wave_number(self.project.id)
            self.persistence.create_wave_dir(self.project.id, self.current_wave)
            print(f"\n{'='*60}")
            print(f"WAVE {self.current_wave}: {task[:50]}...")
            print('='*60)
            result = await self.run_wave(task)
            if result.verdict == "fail":
                print(f"\n[WAVE {self.current_wave}] FAILED - logged to failures/")
                continue
            if result.verdict == "human_review":
                print(f"\n[WAVE {self.current_wave}] NEEDS HUMAN REVIEW")
                self.project.status = "review"
                break
            print(f"\n[WAVE {self.current_wave}] PASSED")
        self.persistence.save_project(self.project)
        return self.project

    async def run_wave(self, task: str) -> WaveSummary:
        """Execute one complete wave with full persistence."""
        started_at = now_iso()
        self.sequence = 0
        self.wave_tokens_in = 0
        self.wave_tokens_out = 0
        self.wave_agents = []

        failure_context = self.persistence.get_failure_context(self.project.id)
        file_keys = list(self.code_holder.files.keys())[:20]
        file_context = self.code_holder.query(
            HolderQuery(action="get_files", target=file_keys, truncate="none")
        ).content

        context = f"""PROJECT GOAL: {self.project.goal}

LEARNINGS FROM PAST FAILURES:
{failure_context}

EXISTING PROJECT FILES:
{file_context}
"""
        # 1. CONFIG
        config_response = await self._execute_turn(
            "orchestrator", "config", f"Task: {task}", context, {"thought": "Starting wave"}
        )

        # Check for new packages
        if "NEW_PACKAGES:" in config_response.raw_response:
            packages_line = config_response.raw_response.split("NEW_PACKAGES:")[1].split("\n")[0]
            if packages_line.strip().lower() not in ["none", "[]", "", '["none"]', "['none']"]:
                bracket_match = re.search(r'\[([^\]]+)\]', packages_line)
                if bracket_match:
                    inner = bracket_match.group(1)
                    pkg_matches = re.findall(r"['\"](\w+)['\"]", inner)
                    unknown = [p for p in pkg_matches if p.lower() not in KNOWN_PACKAGES]
                    if unknown:
                        summary = WaveSummary(
                            wave=self.current_wave, started_at=started_at, task=task,
                            agents_invoked=self.wave_agents, turns_count=self.sequence,
                            total_tokens_in=self.wave_tokens_in, total_tokens_out=self.wave_tokens_out,
                            verdict="human_review", summary=f"New packages needed: {unknown}"
                        )
                        self.persistence.save_wave_summary(self.project.id, self.current_wave, summary)
                        return summary

        # 2. EXECUTOR
        executor_response = await self._execute_turn(
            "orchestrator", "executor",
            f"Implement based on this scaffold:\n\n{config_response.raw_response}",
            "", {"thought": "CONFIG done"}
        )

        # 2.5 SANDBOX EXECUTION (self-healing loop with diagnostic fallback)
        execution_result: Optional[ExecutionResult] = None
        execution_output = ""

        if self.execution_enabled:
            print(f"\n[SANDBOX] Executing tool calls...")
            execution_result = await self.sandbox.execute_with_retry(
                executor_response.raw_response,
                spawn_agent,
                max_retries=3
            )

            execution_output = format_results(execution_result.results)
            print(f"[SANDBOX] Execution {'SUCCESS' if execution_result.success else 'FAILED'}")
            print(f"[SANDBOX] Retries used: {execution_result.retries_used}")

            # Broadcast execution result
            await self._broadcast_event({
                "type": "swarm_execution",
                "project_id": self.project.id,
                "wave": self.current_wave,
                "success": execution_result.success,
                "results": [r.to_dict() for r in execution_result.results],
                "retries_used": execution_result.retries_used,
                "needs_human": execution_result.needs_human,
                "timestamp": now_iso(),
            })

            # If execution needs human review, return early
            if execution_result.needs_human:
                summary = WaveSummary(
                    wave=self.current_wave, started_at=started_at, task=task,
                    agents_invoked=self.wave_agents, turns_count=self.sequence,
                    total_tokens_in=self.wave_tokens_in, total_tokens_out=self.wave_tokens_out,
                    verdict="human_review",
                    summary=f"Execution requires human review: {execution_result.hitl_request.context if execution_result.hitl_request else 'unknown'}"
                )
                self.persistence.save_wave_summary(self.project.id, self.current_wave, summary)
                return summary

            # Phase 10e: Diagnostic flow on execution failure
            if not execution_result.success and execution_result.last_failure_info:
                print(f"\n[SANDBOX] Execution failed after {execution_result.retries_used} retries")
                print(f"[SANDBOX] Triggering diagnostic flow...")

                action, diagnosis, retry_context = await self.handle_tool_failure(
                    execution_result.last_failure_info,
                    task
                )

                if action == "retry" and retry_context:
                    # Re-run EXECUTOR with diagnostic context
                    print(f"\n[DIAGNOSTIC] Re-running EXECUTOR with retry context")
                    executor_response = await self._execute_turn(
                        "orchestrator", "executor",
                        f"RETRY after failure.\n\n{retry_context}\n\nOriginal scaffold:\n{config_response.raw_response}",
                        "", {"thought": "DIAGNOSTIC retry"}
                    )

                    # Re-execute with one more attempt
                    execution_result = await self.sandbox.execute_with_retry(
                        executor_response.raw_response,
                        spawn_agent,
                        max_retries=1  # Single retry after diagnostic
                    )
                    execution_output = format_results(execution_result.results)

                    if execution_result.success:
                        print(f"[DIAGNOSTIC] Retry succeeded!")
                        # Mark consultation as helpful if used
                        if diagnosis and diagnosis.consultation_outcomes:
                            for outcome in diagnosis.consultation_outcomes:
                                self.persistence.mark_consultation_helpful(
                                    self.project.id, self.current_wave, outcome.agent, True
                                )
                    else:
                        # Diagnostic retry also failed
                        print(f"[DIAGNOSTIC] Retry also failed, escalating to human review")
                        if diagnosis and diagnosis.consultation_outcomes:
                            for outcome in diagnosis.consultation_outcomes:
                                self.persistence.mark_consultation_helpful(
                                    self.project.id, self.current_wave, outcome.agent, False
                                )

                        summary = WaveSummary(
                            wave=self.current_wave, started_at=started_at, task=task,
                            agents_invoked=self.wave_agents, turns_count=self.sequence,
                            total_tokens_in=self.wave_tokens_in, total_tokens_out=self.wave_tokens_out,
                            verdict="human_review",
                            summary=f"Diagnostic retry failed: {diagnosis.failure_type if diagnosis else 'unknown'}"
                        )
                        self.persistence.save_wave_summary(self.project.id, self.current_wave, summary)
                        return summary

                elif action in ["human_review", "abort"]:
                    summary = WaveSummary(
                        wave=self.current_wave, started_at=started_at, task=task,
                        agents_invoked=self.wave_agents, turns_count=self.sequence,
                        total_tokens_in=self.wave_tokens_in, total_tokens_out=self.wave_tokens_out,
                        verdict="human_review",
                        summary=f"Diagnostic recommended {action}: {diagnosis.root_cause if diagnosis else 'unknown'}"
                    )
                    self.persistence.save_wave_summary(self.project.id, self.current_wave, summary)
                    return summary

        # 3. REVIEWER (now sees execution results too)
        reviewer_prompt = f"Review this code:\n\n{executor_response.raw_response}"
        if execution_output:
            reviewer_prompt += f"\n\nEXECUTION RESULTS:\n{execution_output}"

        reviewer_response = await self._execute_turn(
            "orchestrator", "reviewer",
            reviewer_prompt,
            "", {"thought": "EXECUTOR done"}
        )

        # 4. QUALITY GATE
        gate_prompt = f"""Review the complete wave:

TASK: {task}

CONFIG OUTPUT:
{config_response.raw_response}

EXECUTOR OUTPUT:
{executor_response.raw_response}

EXECUTION RESULTS:
{execution_output if execution_output else "Execution disabled or no tool calls"}

REVIEWER OUTPUT:
{reviewer_response.raw_response}

Provide your verdict."""
        gate_response = await self._execute_turn(
            "orchestrator", "quality_gate", gate_prompt, "", {"thought": "Running quality gate"}
        )

        verdict = self._parse_verdict(gate_response.raw_response)

        if verdict == Verdict.FAIL:
            failure = self._create_failure(
                task, gate_response.raw_response,
                config_response.raw_response, executor_response.raw_response
            )
            self.persistence.save_failure(self.project.id, failure)
            await self._broadcast_failure(failure)
            summary = WaveSummary(
                wave=self.current_wave, started_at=started_at, task=task,
                agents_invoked=self.wave_agents, turns_count=self.sequence,
                total_tokens_in=self.wave_tokens_in, total_tokens_out=self.wave_tokens_out,
                verdict="fail", failure_ref=f"failures/wave_{self.current_wave}_failure.json",
                summary="Quality gate failed."
            )
            self.persistence.save_wave_summary(self.project.id, self.current_wave, summary)
            return summary

        files_modified = await self._save_code_files(executor_response.raw_response, config_response.parsed)

        # Stage files for promotion (Skynet Procedure)
        if files_modified and self.execution_enabled:
            print(f"\n[SKYNET] Staging {len(files_modified)} files for promotion...")
            promotion_request = self.sandbox.prepare_promotion(
                files=files_modified,
                wave=self.current_wave,
                project_id=self.project.id,
                test_output=execution_output
            )
            print(f"[SKYNET] Staged at: sandbox/staging/wave_{self.current_wave}/")
            print(f"[SKYNET] Review with: python -m agents.promote_cli review")

            # Broadcast promotion ready
            await self._broadcast_event({
                "type": "swarm_promotion_ready",
                "project_id": self.project.id,
                "wave": self.current_wave,
                "files": files_modified,
                "staging_path": f"sandbox/staging/wave_{self.current_wave}/",
                "timestamp": now_iso(),
            })

        summary = WaveSummary(
            wave=self.current_wave, started_at=started_at, task=task,
            agents_invoked=self.wave_agents, turns_count=self.sequence,
            total_tokens_in=self.wave_tokens_in, total_tokens_out=self.wave_tokens_out,
            verdict="pass" if verdict == Verdict.PASS else "human_review",
            files_modified=files_modified,
            summary=self._generate_summary(task, config_response, executor_response, reviewer_response)
        )
        self.persistence.save_wave_summary(self.project.id, self.current_wave, summary)
        return summary

    async def _execute_turn(
        self, from_agent: str, to_agent: str, user_prompt: str,
        context: str, reasoning_before: Dict[str, str]
    ) -> InboundTurn:
        """Execute a single turn with full persistence."""
        self.sequence += 1
        agent_config = AGENTS[AgentRole(to_agent)]
        full_prompt = f"{context}\n\n{user_prompt}" if context else user_prompt
        outbound = OutboundTurn(
            wave=self.current_wave, sequence=self.sequence,
            from_agent=from_agent, to_agent=to_agent,
            model=agent_config.model, system_prompt=agent_config.system_prompt,
            user_prompt=full_prompt,
            tokens_in=len(agent_config.system_prompt + user_prompt) // 4,
            reasoning_before=reasoning_before
        )
        self.persistence.save_turn(self.project.id, self.current_wave, outbound)
        self.wave_tokens_in += outbound.tokens_in
        print(f"\n[{to_agent.upper()}] Processing...")
        start = time.time()
        raw_response = await spawn_agent(AgentRole(to_agent), outbound.user_prompt, "")
        latency = int((time.time() - start) * 1000)
        reasoning_trace = extract_reasoning_trace(raw_response)
        parsed = self._parse_agent_response(raw_response, to_agent)
        self.sequence += 1
        inbound = InboundTurn(
            id=outbound.id, wave=self.current_wave, sequence=self.sequence,
            from_agent=to_agent, to_agent=from_agent,
            model=agent_config.model, raw_response=raw_response,
            parsed=parsed, reasoning_trace=reasoning_trace,
            tokens_out=len(raw_response) // 4, latency_ms=latency
        )
        self.persistence.save_turn(self.project.id, self.current_wave, inbound)
        self.wave_tokens_out += inbound.tokens_out
        self.wave_agents.append(to_agent)
        print(f"[{to_agent.upper()}] Done. {inbound.tokens_out} tokens, {latency}ms")

        # Broadcast turn completion
        await self._broadcast_turn(outbound, inbound)

        return inbound

    def _parse_agent_response(self, response: str, agent: str) -> Dict[str, Any]:
        """Parse structured fields from agent response."""
        parsed = {}
        fields_map = {
            "config": ["TARGET_FILE", "MODIFICATION_TYPE", "LOCATION", "NEW_IMPORTS_NEEDED", "NEW_PACKAGES"],
            "reviewer": ["P0_ISSUES", "P1_ISSUES", "P2_ISSUES", "VERDICT"],
            "quality_gate": ["VERDICT", "FAILURE_TYPE", "AGENT_BLAMED", "ROOT_CAUSE", "RECOMMENDATION"]
        }
        for fld in fields_map.get(agent, []):
            match = re.search(rf"{fld}:\s*(.+?)(?:\n|$)", response)
            if match:
                parsed[fld.lower()] = match.group(1).strip()
        return parsed

    def _parse_verdict(self, response: str) -> Verdict:
        """Parse verdict from quality gate response."""
        if "VERDICT: PASS" in response.upper():
            return Verdict.PASS
        elif "VERDICT: FAIL" in response.upper():
            return Verdict.FAIL
        return Verdict.HUMAN_REVIEW

    def _create_failure(
        self, task: str, gate_response: str, config_response: str, executor_response: str
    ) -> Failure:
        """Create failure record from quality gate output."""
        ft = re.search(r"FAILURE_TYPE:\s*(.+?)(?:\n|$)", gate_response)
        ab = re.search(r"AGENT_BLAMED:\s*(.+?)(?:\n|$)", gate_response)
        rc = re.search(r"ROOT_CAUSE:\s*(.+?)(?:\n|$)", gate_response)
        rec = re.search(r"RECOMMENDATION:\s*(.+?)(?:\n|$)", gate_response)
        return Failure(
            wave=self.current_wave, failure_source="quality_gate",
            failure_type=ft.group(1).strip() if ft else "unknown",
            agent_blamed=ab.group(1).strip() if ab else "unknown",
            verdict_raw=gate_response,
            root_cause=rc.group(1).strip() if rc else "Unknown",
            recommendation=rec.group(1).strip() if rec else "None",
            context_snapshot={"config_output": config_response[:500], "executor_output": executor_response[:500]},
            learning=f"Avoid {ft.group(1).strip() if ft else 'error'}",
            retry_wave=f"{self.current_wave}.1" if '.' not in self.current_wave else f"{self.current_wave[:-1]}{int(self.current_wave[-1])+1}"
        )

    async def _save_code_files(self, executor_output: str, config_parsed: Dict) -> List[str]:
        """Save files from executor output."""
        files_modified = []
        blocks = re.findall(r'```(?:python|py)?\n(.*?)```', executor_output, re.DOTALL)
        if not blocks:
            return files_modified
        code = blocks[-1]
        target_file = config_parsed.get("target_file", "output.py")
        mod_type = config_parsed.get("modification_type", "new_file")
        if mod_type and mod_type.lower() in ["add_endpoint", "add_function"]:
            existing = self.code_holder.get_file_content(target_file)
            if existing:
                new_content = existing.rstrip() + "\n\n" + code.strip() + "\n"
                self.code_holder.update_file(target_file, new_content)
            else:
                self.code_holder.update_file(target_file, code)
        else:
            self.code_holder.update_file(target_file, code)
        files_modified.append(target_file)
        print(f"[FILE] Saved: {target_file}")
        self.sequence += 1
        files_written = FilesWritten(
            wave=self.current_wave, sequence=self.sequence,
            operations=[FileOperation(
                file_path=target_file,
                operation="append" if mod_type and mod_type.lower() in ["add_endpoint", "add_function"] else "create",
                lines_added=len(code.split('\n')), preview=code[:500]
            )]
        )
        self.persistence.save_files_written(self.project.id, self.current_wave, files_written)

        # Broadcast file write
        await self._broadcast_event({
            "type": "swarm_file_written",
            "project_id": self.project.id,
            "wave": self.current_wave,
            "file_path": target_file,
            "operation": files_written.operations[0].operation,
            "lines_added": files_written.operations[0].lines_added,
            "preview": code[:500],
            "timestamp": now_iso(),
        })

        return files_modified

    def _generate_summary(
        self, task: str, config: InboundTurn, executor: InboundTurn, reviewer: InboundTurn
    ) -> str:
        """Generate wave summary."""
        target = config.parsed.get("target_file", "unknown")
        mod_type = config.parsed.get("modification_type", "unknown")
        verdict = reviewer.parsed.get("verdict", "unknown")
        return f"CONFIG identified {target} for {mod_type}. EXECUTOR implemented. REVIEWER: {verdict}."


async def test_swarm():
    """Test with new SwarmOrchestrator."""
    import asyncio
    orchestrator = SwarmOrchestrator(
        project_root=Path(__file__).parent / "sandbox",
        project_name="test_metrics",
        goal="Add /api/metrics endpoint"
    )
    project = await orchestrator.run_project([
        "Add a GET /api/metrics endpoint that returns uptime, query count, and token usage"
    ])
    print(f"\nProject complete: {project.id}")
    print(f"Status: {project.status}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_swarm())
