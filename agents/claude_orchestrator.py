"""
Claude Opus Orchestrator - Multi-agent wave loop with holder daemons.

This is the main entry point for the Swarm V2 architecture.
Claude Opus directs workers (Grok agents) through waves.
"""

import asyncio
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal, Tuple

from .registry import spawn_agent, AgentRole, AGENTS
from .schemas import (
    Project, WaveSummary, Failure,
    OutboundTurn, InboundTurn, FilesWritten, FileOperation,
    Verdict, generate_id, now_iso,
    ToolFailure, Diagnosis, DiagnosticResult, ConfigMode
)
from .persistence import SwarmPersistence
from .holder_daemon import DaemonManager
from .sandbox_executor import (
    SandboxExecutor, parse_tool_calls, format_results,
    ExecutionResult, ToolResult, HITLRequest, HITLType
)


# =============================================================================
# Action Parsing
# =============================================================================

@dataclass
class OrchestratorAction:
    """Parsed action from ORCHESTRATOR output."""
    action_type: Literal["QUERY", "SPAWN", "SPAWN_PARALLEL", "LOG", "COMPLETE", "ESCALATE"]
    params: Dict[str, Any]
    thinking: str = ""


def parse_orchestrator_output(response: str) -> OrchestratorAction:
    """Parse ORCHESTRATOR response into an action."""
    # Extract thinking
    thinking = ""
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Extract action
    action_match = re.search(r'<action\s+type="(\w+)">(.*?)</action>', response, re.DOTALL)
    if not action_match:
        # Default to LOG if no action found
        return OrchestratorAction(
            action_type="LOG",
            params={"content": response[:500]},
            thinking=thinking
        )

    action_type = action_match.group(1)
    action_body = action_match.group(2).strip()

    try:
        params = json.loads(action_body)
    except json.JSONDecodeError:
        params = {"raw": action_body}

    return OrchestratorAction(
        action_type=action_type,
        params=params,
        thinking=thinking
    )


# =============================================================================
# Wave State
# =============================================================================

@dataclass
class WaveState:
    """State for a single wave execution."""
    wave: str
    task: str
    started_at: str = field(default_factory=now_iso)
    sequence: int = 0
    turns: List[Dict[str, Any]] = field(default_factory=list)
    files_written: List[str] = field(default_factory=list)
    status: Literal["running", "pass", "fail", "human_review", "escalated"] = "running"
    failure_info: Optional[Dict[str, Any]] = None
    consulted_already: bool = False  # Track if we consulted holders for this wave


# =============================================================================
# Claude Orchestrator
# =============================================================================

class ClaudeOrchestrator:
    """
    Main orchestrator using Claude Opus.

    Architecture:
    - Claude Opus (ORCHESTRATOR) directs the swarm
    - Three holder daemons (CODE_HOLDER, CONVO_HOLDER, USER_HOLDER) as views over JSON
    - Worker agents (CONFIG, EXECUTOR, REVIEWER, QUALITY_GATE) do the work
    - SandboxExecutor handles tool execution
    """

    def __init__(
        self,
        project_root: Path,
        sandbox_root: Path,
        persistence: SwarmPersistence,
        project_id: str,
        max_waves: int = 10,
        max_retries_per_wave: int = 3,
        broadcast_callback: Optional[Any] = None,  # async fn(dict) for WebSocket
    ):
        self.project_root = Path(project_root)
        self.sandbox_root = Path(sandbox_root)
        self.persistence = persistence
        self.project_id = project_id
        self.max_waves = max_waves
        self.max_retries = max_retries_per_wave
        self.broadcast = broadcast_callback  # WebSocket broadcast function

        # Initialize components
        self.daemons = DaemonManager()
        self.executor = SandboxExecutor(sandbox_root, project_root)

        # State
        self.current_wave: Optional[WaveState] = None
        self.wave_count = 0
        self.initialized = False
        self.goal: str = ""
        self.tasks: List[str] = []
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    async def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event to WebSocket if broadcast callback is set."""
        if self.broadcast:
            try:
                await self.broadcast({
                    "type": event_type,
                    "timestamp": now_iso(),
                    **data
                })
            except Exception as e:
                print(f"[ORCH] Broadcast error: {e}")

    async def initialize(self) -> Dict[str, str]:
        """Initialize all daemons and prepare for execution."""
        results = await self.daemons.start(
            sandbox_root=self.sandbox_root,
            persistence=self.persistence,
            project_id=self.project_id
        )
        self.initialized = True
        print(f"[ORCH] Initialized for project {self.project_id}")
        return results

    async def run_project(self, goal: str, tasks: List[str]) -> Project:
        """Run a full project through wave loop."""
        if not self.initialized:
            await self.initialize()

        # Store for broadcasting
        self.goal = goal
        self.tasks = tasks

        project = Project(
            name=self.project_id,
            goal=goal,
            status="executing",
            current_wave="001"
        )

        print(f"\n{'#'*60}")
        print(f"# PROJECT: {goal[:50]}...")
        print(f"# Tasks: {len(tasks)}")
        print('#'*60)

        # Broadcast project start
        await self._emit("swarm_project_start", {
            "project_id": self.project_id,
            "project_name": self.project_id,
            "goal": goal,
            "tasks": tasks,
        })

        for i, task in enumerate(tasks[:self.max_waves]):
            wave_num = f"{i + 1:03d}"
            project.current_wave = wave_num

            result = await self.run_wave(task, wave_num)

            if result.status == "escalated":
                project.status = "review"
                print(f"\n[ORCH] Wave {wave_num} escalated to human")
                break
            elif result.status == "human_review":
                project.status = "review"
                print(f"\n[ORCH] Wave {wave_num} needs human review")
                break
            elif result.status == "fail":
                # Try to continue if not escalated
                print(f"\n[ORCH] Wave {wave_num} failed, continuing...")

            print(f"\n[WAVE {wave_num} COMPLETE] Files: {result.files_written}")

        # Final status
        if project.status == "executing":
            project.status = "complete"

        # Broadcast project end
        await self._emit("swarm_project_end", {
            "project_id": self.project_id,
            "status": project.status,
            "waves_completed": self.wave_count,
        })

        return project

    async def run_wave(self, task: str, wave: str) -> WaveState:
        """Execute a single wave with ORCHESTRATOR control loop."""
        print(f"\n{'='*60}")
        print(f"WAVE {wave}: {task[:50]}...")
        print('='*60)

        # Refresh CODE_HOLDER for new wave
        await self.daemons.refresh_code_for_wave(wave)

        # Initialize wave state
        self.current_wave = WaveState(wave=wave, task=task)
        self.wave_count += 1

        # Broadcast wave start
        await self._emit("swarm_wave_start", {
            "wave": wave,
            "task": task,
            "project_id": self.project_id,
        })

        # Build initial context for ORCHESTRATOR
        initial_context = f"""WAVE {wave} STARTING

TASK: {task}

You have three holder daemons ready:
- CODE_HOLDER: Fresh codebase snapshot loaded
- CONVO_HOLDER: Can query previous wave communications
- USER_HOLDER: Can query user intent

Begin by querying CODE_HOLDER about relevant files, then plan with CONFIG."""

        # ORCHESTRATOR control loop
        max_turns = 20
        response = await self._spawn_orchestrator(initial_context)

        for turn in range(max_turns):
            action = parse_orchestrator_output(response)
            self.current_wave.sequence += 1

            # Log turn
            self._log_turn(action)

            # Execute action
            if action.action_type == "COMPLETE":
                result = await self._handle_complete(action)
                # Broadcast wave end
                await self._emit("swarm_wave_end", {
                    "wave": wave,
                    "verdict": result.status,
                    "total_tokens_in": self.total_tokens_in,
                    "total_tokens_out": self.total_tokens_out,
                    "files_modified": result.files_written,
                    "summary": action.params.get("summary", ""),
                })
                return result

            elif action.action_type == "ESCALATE":
                result = await self._handle_escalate(action)
                await self._emit("swarm_wave_end", {
                    "wave": wave,
                    "verdict": "escalated",
                    "total_tokens_in": self.total_tokens_in,
                    "total_tokens_out": self.total_tokens_out,
                    "files_modified": result.files_written,
                    "summary": action.params.get("reason", "Escalated to human"),
                })
                return result

            elif action.action_type == "QUERY":
                result = await self._handle_query(action)
                response = await self._spawn_orchestrator(
                    f"HOLDER RESPONSE:\n{result}\n\nContinue with your plan."
                )

            elif action.action_type == "SPAWN":
                result = await self._handle_spawn(action)
                response = await self._spawn_orchestrator(
                    f"AGENT RESPONSE:\n{result}\n\nEvaluate and continue."
                )

            elif action.action_type == "SPAWN_PARALLEL":
                results = await self._handle_spawn_parallel(action)
                response = await self._spawn_orchestrator(
                    f"PARALLEL RESPONSES:\n{results}\n\nEvaluate and continue."
                )

            elif action.action_type == "LOG":
                # Just logging, continue
                response = await self._spawn_orchestrator("Logged. Continue with your plan.")

            else:
                # Unknown action
                response = await self._spawn_orchestrator(
                    f"Unknown action type: {action.action_type}. Use QUERY, SPAWN, COMPLETE, or ESCALATE."
                )

        # Max turns exceeded
        self.current_wave.status = "fail"
        self.current_wave.failure_info = {"reason": "max_turns_exceeded", "turns": max_turns}

        # Broadcast wave end on failure
        await self._emit("swarm_wave_end", {
            "wave": wave,
            "verdict": "fail",
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "files_modified": self.current_wave.files_written,
            "summary": f"Max turns ({max_turns}) exceeded",
        })

        return self.current_wave

    async def _spawn_orchestrator(self, message: str) -> str:
        """Spawn ORCHESTRATOR with message."""
        context = self._build_orchestrator_context()
        return await spawn_agent(AgentRole.ORCHESTRATOR, message, context=context)

    def _build_orchestrator_context(self) -> str:
        """Build context for ORCHESTRATOR."""
        parts = [f"PROJECT: {self.project_id}"]
        parts.append(f"WAVE: {self.current_wave.wave if self.current_wave else '000'}")

        if self.current_wave:
            parts.append(f"TASK: {self.current_wave.task}")
            parts.append(f"TURN: {self.current_wave.sequence}")
            if self.current_wave.files_written:
                parts.append(f"FILES_WRITTEN: {self.current_wave.files_written}")

        return "\n".join(parts)

    async def _handle_query(self, action: OrchestratorAction) -> str:
        """Handle QUERY action - query a holder daemon."""
        holder = action.params.get("holder", "").upper()
        question = action.params.get("question", "")
        wave = action.params.get("wave", "current")

        print(f"  [QUERY] {holder}: {question[:50]}...")

        if holder == "CODE_HOLDER":
            return await self.daemons.query_code(question)
        elif holder == "CONVO_HOLDER":
            mode = action.params.get("mode", "summary")
            return await self.daemons.query_convo(question, wave=wave, mode=mode)
        elif holder == "USER_HOLDER":
            return await self.daemons.query_user(question)
        else:
            return f"Unknown holder: {holder}. Use CODE_HOLDER, CONVO_HOLDER, or USER_HOLDER."

    async def _handle_spawn(self, action: OrchestratorAction) -> str:
        """Handle SPAWN action - spawn a single worker agent."""
        agent_name = action.params.get("agent", "").upper()
        task = action.params.get("task", "")
        context = action.params.get("context", "")

        print(f"  [SPAWN] {agent_name}: {task[:50]}...")

        # Map string to AgentRole
        agent_role = self._get_agent_role(agent_name)
        if agent_role is None:
            return f"Unknown agent: {agent_name}. Use CONFIG, EXECUTOR, REVIEWER, or QUALITY_GATE."

        # Broadcast agent start
        agent_key = agent_name.lower()
        await self._emit("swarm_agent_start", {
            "agent": agent_key,
            "wave": self.current_wave.wave if self.current_wave else "001",
        })

        # Spawn agent and time it
        start_time = time.time()
        response = await spawn_agent(agent_role, task, context=context)
        latency_ms = int((time.time() - start_time) * 1000)

        # Estimate tokens (rough: 4 chars per token)
        tokens_in = len(task + context) // 4
        tokens_out = len(response) // 4
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out

        # Extract reasoning if present
        reasoning = []
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            for i, step in enumerate(reasoning_text.split('\n'), 1):
                step = step.strip()
                if step:
                    reasoning.append({"step": i, "content": step[:200]})

        # If EXECUTOR, also execute tool calls
        if agent_role == AgentRole.EXECUTOR:
            response = await self._execute_and_append(response, task)

        # Broadcast agent turn complete
        await self._emit("swarm_turn", {
            "agent": agent_key,
            "wave": self.current_wave.wave if self.current_wave else "001",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "reasoning": reasoning,
            "preview": response[:500],
            "parsed": {},
        })

        return response

    async def _handle_spawn_parallel(self, action: OrchestratorAction) -> str:
        """Handle SPAWN_PARALLEL action - spawn multiple agents concurrently."""
        agents = action.params.get("agents", [])
        if not agents:
            return "No agents specified for parallel spawn."

        print(f"  [SPAWN_PARALLEL] {len(agents)} agents...")

        # Create tasks for parallel execution
        tasks = []
        for spec in agents:
            agent_name = spec.get("agent", "").upper()
            task = spec.get("task", "")
            context = spec.get("context", "")

            agent_role = self._get_agent_role(agent_name)
            if agent_role:
                tasks.append(self._spawn_worker(agent_role, task, context, agent_name))

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Format results
        output_parts = []
        for i, result in enumerate(results):
            agent_name = agents[i].get("agent", "?")
            if isinstance(result, Exception):
                output_parts.append(f"=== {agent_name} (ERROR) ===\n{str(result)}")
            else:
                output_parts.append(f"=== {agent_name} ===\n{result}")

        return "\n\n".join(output_parts)

    async def _spawn_worker(
        self,
        role: AgentRole,
        task: str,
        context: str,
        name: str
    ) -> str:
        """Spawn a worker agent and handle tool execution if EXECUTOR."""
        response = await spawn_agent(role, task, context=context)

        if role == AgentRole.EXECUTOR:
            response = await self._execute_and_append(response, task)

        return response

    async def _execute_and_append(self, executor_response: str, task: str) -> str:
        """Execute tool calls from EXECUTOR and append results."""
        # Execute with retry
        exec_result = await self.executor.execute_with_retry(
            executor_response,
            spawn_agent,
            max_retries=self.max_retries,
            executor_context=task
        )

        # Track files written and broadcast
        for result in exec_result.results:
            if result.tool == "write_file" and result.success:
                # Extract path from output
                path_match = re.search(r'to\s+(\S+)', result.output)
                if path_match:
                    file_path = path_match.group(1)
                    self.current_wave.files_written.append(file_path)
                    # Also inject into CODE_HOLDER
                    await self.daemons.inject_file(file_path, "")

                    # Broadcast file write
                    await self._emit("swarm_file_written", {
                        "wave": self.current_wave.wave if self.current_wave else "001",
                        "file_path": file_path,
                        "operation": "create",
                        "lines_added": result.output.count('\n'),
                        "preview": result.output[:300],
                    })

        # Build result summary
        result_summary = format_results(exec_result.results)

        if exec_result.needs_human:
            result_summary += f"\n\nHITL REQUIRED: {exec_result.hitl_request.context if exec_result.hitl_request else 'unknown'}"

        if not exec_result.success and exec_result.last_failure_info:
            # Trigger diagnostic flow
            diagnosis = await self._run_diagnostic(exec_result.last_failure_info)
            result_summary += f"\n\nDIAGNOSIS:\n{json.dumps(diagnosis.to_dict(), indent=2)}"

            # Broadcast diagnostic event
            await self._emit("swarm_diagnostic", {
                "wave": self.current_wave.wave if self.current_wave else "001",
                "failure_type": diagnosis.failure_type,
                "fix_strategy": diagnosis.fix_strategy,
                "confidence": diagnosis.confidence,
            })

        return f"{executor_response}\n\n=== EXECUTION RESULTS ===\n{result_summary}"

    async def _run_diagnostic(self, failure_info) -> Diagnosis:
        """Run CONFIG in DIAGNOSTIC mode to analyze failure."""
        print(f"  [DIAG] Analyzing failure: {failure_info.tool_name}")

        # Query holders for context
        code_context = await self.daemons.query_code(
            f"What files might be related to {failure_info.error_message}?"
        )
        convo_context = await self.daemons.query_convo(
            f"What has been tried so far regarding {failure_info.executor_context}?",
            mode="summary"
        )

        # Run CONFIG in DIAGNOSTIC mode
        diag_prompt = f"""DIAGNOSTIC MODE - Analyze this failure:

TOOL: {failure_info.tool_name}
ARGS: {json.dumps(failure_info.tool_args)}
ERROR: {failure_info.error_message}
ATTEMPT: {failure_info.attempt}
CONTEXT: {failure_info.executor_context}

CODE_HOLDER says: {code_context[:500]}
CONVO_HOLDER says: {convo_context[:500]}

Output a diagnosis with:
- failure_type (import_missing, path_wrong, syntax_error, dep_conflict, etc.)
- root_cause
- fix_strategy (retry_executor, retry_tool, add_dep, modify_config, human_review, abort)
- context_for_retry
- confidence (0.0-1.0)"""

        response = await spawn_agent(AgentRole.CONFIG, diag_prompt, context="DIAGNOSTIC MODE")

        # Parse response into Diagnosis
        diagnosis = Diagnosis(
            failure_type="unknown",
            root_cause=failure_info.error_message,
            fix_strategy="human_review",
            consult_with=[],
            context_for_retry="",
            next_step_hint="",
            confidence=0.5,
            reasoning=response[:1000]
        )

        # Try to extract structured data from response
        if "failure_type:" in response.lower():
            match = re.search(r'failure_type:\s*(\w+)', response, re.I)
            if match:
                diagnosis.failure_type = match.group(1)

        if "fix_strategy:" in response.lower():
            match = re.search(r'fix_strategy:\s*(\w+)', response, re.I)
            if match:
                diagnosis.fix_strategy = match.group(1)

        if "confidence:" in response.lower():
            match = re.search(r'confidence:\s*([\d.]+)', response, re.I)
            if match:
                try:
                    diagnosis.confidence = float(match.group(1))
                except ValueError:
                    pass

        return diagnosis

    def _handle_complete(self, action: OrchestratorAction) -> WaveState:
        """Handle COMPLETE action."""
        status = action.params.get("status", "pass")
        summary = action.params.get("summary", "")

        print(f"  [COMPLETE] Status: {status}")

        self.current_wave.status = "pass" if status == "pass" else "fail"

        # Save wave summary
        wave_summary = WaveSummary(
            wave=self.current_wave.wave,
            started_at=self.current_wave.started_at,
            task=self.current_wave.task,
            turns_count=self.current_wave.sequence,
            verdict=status,
            files_modified=self.current_wave.files_written,
            summary=summary
        )
        self.persistence.save_wave_summary(self.project_id, wave_summary)

        return self.current_wave

    def _handle_escalate(self, action: OrchestratorAction) -> WaveState:
        """Handle ESCALATE action."""
        reason = action.params.get("reason", "Unknown")

        print(f"  [ESCALATE] {reason[:50]}...")

        self.current_wave.status = "escalated"
        self.current_wave.failure_info = {"reason": reason}

        return self.current_wave

    def _get_agent_role(self, name: str) -> Optional[AgentRole]:
        """Map agent name string to AgentRole."""
        mapping = {
            "CONFIG": AgentRole.CONFIG,
            "EXECUTOR": AgentRole.EXECUTOR,
            "REVIEWER": AgentRole.REVIEWER,
            "QUALITY_GATE": AgentRole.QUALITY_GATE,
        }
        return mapping.get(name.upper())

    def _log_turn(self, action: OrchestratorAction) -> None:
        """Log a turn to persistence."""
        turn = {
            "sequence": self.current_wave.sequence,
            "action_type": action.action_type,
            "params": action.params,
            "thinking": action.thinking[:500] if action.thinking else "",
            "timestamp": now_iso()
        }
        self.current_wave.turns.append(turn)

    async def shutdown(self) -> None:
        """Shutdown orchestrator and daemons."""
        await self.daemons.shutdown()
        print(f"[ORCH] Shutdown complete for {self.project_id}")


# =============================================================================
# CLI Entry Point
# =============================================================================

async def run_swarm(
    goal: str,
    tasks: List[str],
    project_root: Path,
    sandbox_root: Optional[Path] = None,
    project_id: str = "default"
) -> Project:
    """Convenience function to run a swarm project."""
    project_root = Path(project_root)
    sandbox_root = sandbox_root or project_root / "agents" / "sandbox"

    # Initialize persistence
    data_dir = project_root / "agents" / "data"
    persistence = SwarmPersistence(data_dir)

    # Create and run orchestrator
    orchestrator = ClaudeOrchestrator(
        project_root=project_root,
        sandbox_root=sandbox_root,
        persistence=persistence,
        project_id=project_id
    )

    try:
        result = await orchestrator.run_project(goal, tasks)
        return result
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    async def test():
        project = await run_swarm(
            goal="Add a /metrics endpoint to the backend",
            tasks=[
                "Create a /metrics endpoint that returns query count and uptime"
            ],
            project_root=Path("."),
            project_id="test-run"
        )
        print(f"\nProject status: {project.status}")

    asyncio.run(test())
