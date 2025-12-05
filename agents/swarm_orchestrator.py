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
    Verdict, now_iso
)
from .persistence import SwarmPersistence
from .reasoning import extract_reasoning_trace
from .holders import CodeHolder, ConvoHolder, HolderQuery


# Type for WebSocket broadcast callback
BroadcastCallback = Callable[[str], Awaitable[None]]


KNOWN_PACKAGES = {
    'fastapi', 'uvicorn', 'pydantic', 'requests', 'numpy', 'faiss',
    'anthropic', 'openai', 'python-dotenv', 'aiofiles', 'httpx',
    'websockets', 'pyyaml', 'scipy', 'sklearn', 'hdbscan', 'river',
}


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
        broadcast_callback: Optional[BroadcastCallback] = None
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

        # 3. REVIEWER
        reviewer_response = await self._execute_turn(
            "orchestrator", "reviewer",
            f"Review this code:\n\n{executor_response.raw_response}",
            "", {"thought": "EXECUTOR done"}
        )

        # 4. QUALITY GATE
        gate_prompt = f"""Review the complete wave:

TASK: {task}

CONFIG OUTPUT:
{config_response.raw_response}

EXECUTOR OUTPUT:
{executor_response.raw_response}

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
