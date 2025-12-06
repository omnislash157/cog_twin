"""
Persistence layer for swarm data.
All data saved as human-readable JSON files.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .schemas import (
    Project, WaveSummary, Failure,
    OutboundTurn, InboundTurn, FilesWritten,
    Diagnosis, ConsultationOutcome
)


class SwarmPersistence:
    """Handles all swarm data storage."""

    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent / "data" / "projects"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_project_dir: Optional[Path] = None

    # === Project Management ===

    def create_project(self, name: str, goal: str, spec_file: Optional[str] = None) -> Project:
        """Create new project with folder structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"{name}_{timestamp}"

        project_dir = self.base_dir / project_id
        project_dir.mkdir(parents=True)
        (project_dir / "waves").mkdir()
        (project_dir / "failures").mkdir()

        project = Project(
            id=project_id,
            name=name,
            goal=goal,
            spec_file=spec_file,
        )

        self.save_project(project)
        self.current_project_dir = project_dir

        return project

    def project_dir(self, project_id: str) -> Path:
        return self.base_dir / project_id

    def save_project(self, project: Project) -> None:
        path = self.project_dir(project.id) / "project.json"
        path.write_text(json.dumps(project.to_dict(), indent=2))

    def load_project(self, project_id: str) -> Project:
        path = self.project_dir(project_id) / "project.json"
        data = json.loads(path.read_text())
        return Project(**data)

    def list_projects(self) -> List[str]:
        """List all project IDs."""
        if not self.base_dir.exists():
            return []
        return sorted([d.name for d in self.base_dir.iterdir() if d.is_dir()])

    # === Wave Management ===

    def create_wave_dir(self, project_id: str, wave: str) -> Path:
        """Create directory for a wave."""
        wave_dir = self.project_dir(project_id) / "waves" / f"wave_{wave}"
        wave_dir.mkdir(parents=True, exist_ok=True)
        return wave_dir

    def get_wave_dir(self, project_id: str, wave: str) -> Path:
        return self.project_dir(project_id) / "waves" / f"wave_{wave}"

    def save_turn(self, project_id: str, wave: str, turn: OutboundTurn | InboundTurn) -> None:
        """Save a turn (prompt or response) to wave folder."""
        wave_dir = self.get_wave_dir(project_id, wave)
        wave_dir.mkdir(parents=True, exist_ok=True)

        filepath = wave_dir / turn.filename()
        filepath.write_text(json.dumps(turn.to_dict(), indent=2))

    def save_files_written(self, project_id: str, wave: str, files: FilesWritten) -> None:
        """Save file operations record."""
        wave_dir = self.get_wave_dir(project_id, wave)
        filepath = wave_dir / files.filename()
        filepath.write_text(json.dumps(files.to_dict(), indent=2))

    def save_wave_summary(self, project_id: str, wave: str, summary: WaveSummary) -> None:
        """Save wave summary."""
        wave_dir = self.get_wave_dir(project_id, wave)
        filepath = wave_dir / "_summary.json"
        filepath.write_text(json.dumps(summary.to_dict(), indent=2))

    def load_wave_summary(self, project_id: str, wave: str) -> Optional[WaveSummary]:
        """Load wave summary if exists."""
        filepath = self.get_wave_dir(project_id, wave) / "_summary.json"
        if not filepath.exists():
            return None
        data = json.loads(filepath.read_text())
        return WaveSummary(**data)

    def list_waves(self, project_id: str) -> List[str]:
        """List all waves for a project."""
        waves_dir = self.project_dir(project_id) / "waves"
        if not waves_dir.exists():
            return []
        return sorted([d.name.replace("wave_", "") for d in waves_dir.iterdir() if d.is_dir()])

    def load_wave_turns(self, project_id: str, wave: str) -> List[Dict[str, Any]]:
        """Load all turns from a wave in sequence order."""
        wave_dir = self.get_wave_dir(project_id, wave)
        if not wave_dir.exists():
            return []

        turns = []
        for f in sorted(wave_dir.glob("*.json")):
            if f.name.startswith("_"):  # Skip _summary.json
                continue
            data = json.loads(f.read_text())
            turns.append(data)

        return turns

    # === Failure Management ===

    def save_failure(self, project_id: str, failure: Failure) -> None:
        """Save failure record."""
        failures_dir = self.project_dir(project_id) / "failures"
        failures_dir.mkdir(exist_ok=True)

        filepath = failures_dir / f"wave_{failure.wave}_failure.json"
        filepath.write_text(json.dumps(failure.to_dict(), indent=2))

    def load_failures(self, project_id: str) -> List[Failure]:
        """Load all failures for a project."""
        failures_dir = self.project_dir(project_id) / "failures"
        if not failures_dir.exists():
            return []

        failures = []
        for f in sorted(failures_dir.glob("*.json")):
            data = json.loads(f.read_text())
            failures.append(Failure(**data))

        return failures

    def get_failure_context(self, project_id: str) -> str:
        """Get formatted failure context for planning prompts."""
        failures = self.load_failures(project_id)
        if not failures:
            return "No previous failures recorded."

        return "\n\n".join([f.to_learning_prompt() for f in failures])

    # === Wave Number Management ===

    def next_wave_number(self, project_id: str) -> str:
        """Get next wave number (handles retries like 001.1)."""
        waves = self.list_waves(project_id)
        if not waves:
            return "001"

        last = waves[-1]

        # Check if last wave passed
        summary = self.load_wave_summary(project_id, last)
        if summary and summary.verdict == "pass":
            # Move to next major wave
            major = int(last.split('.')[0])
            return f"{major + 1:03d}"
        else:
            # Retry - increment minor version
            if '.' in last:
                major, minor = last.split('.')
                return f"{major}.{int(minor) + 1}"
            else:
                return f"{last}.1"

    # === Diagnosis Management (Phase 10e: Resilience Loop) ===

    def save_diagnosis(self, project_id: str, wave: str, diagnosis: Diagnosis) -> None:
        """Store diagnosis for learning."""
        diag_dir = self.project_dir(project_id) / "diagnostics"
        diag_dir.mkdir(exist_ok=True)

        filepath = diag_dir / f"wave_{wave}_diagnosis.json"
        filepath.write_text(json.dumps(diagnosis.to_dict(), indent=2, default=str))

    def load_diagnosis(self, project_id: str, wave: str) -> Optional[Diagnosis]:
        """Load diagnosis for a wave if exists."""
        filepath = self.project_dir(project_id) / "diagnostics" / f"wave_{wave}_diagnosis.json"
        if not filepath.exists():
            return None

        data = json.loads(filepath.read_text())

        # Reconstruct ConsultationOutcome objects
        outcomes = []
        for outcome_data in data.get("consultation_outcomes", []):
            outcomes.append(ConsultationOutcome(**outcome_data))
        data["consultation_outcomes"] = outcomes

        return Diagnosis(**data)

    def get_diagnostics(self, project_id: str) -> List[Diagnosis]:
        """Retrieve all diagnostics for pattern analysis."""
        diag_dir = self.project_dir(project_id) / "diagnostics"
        if not diag_dir.exists():
            return []

        diagnostics = []
        for f in sorted(diag_dir.glob("*.json")):
            data = json.loads(f.read_text())

            # Reconstruct ConsultationOutcome objects
            outcomes = []
            for outcome_data in data.get("consultation_outcomes", []):
                outcomes.append(ConsultationOutcome(**outcome_data))
            data["consultation_outcomes"] = outcomes

            diagnostics.append(Diagnosis(**data))

        return diagnostics

    def mark_consultation_helpful(
        self,
        project_id: str,
        wave: str,
        agent: str,
        helpful: bool
    ) -> None:
        """
        After retry success/fail, mark which consultation was helpful.

        This allows the system to learn which agents give useful advice
        for different failure types.
        """
        filepath = self.project_dir(project_id) / "diagnostics" / f"wave_{wave}_diagnosis.json"
        if not filepath.exists():
            return

        data = json.loads(filepath.read_text())

        for outcome in data.get("consultation_outcomes", []):
            if outcome.get("agent") == agent:
                outcome["helpful"] = helpful

        filepath.write_text(json.dumps(data, indent=2, default=str))

    def get_agent_helpfulness_stats(self, project_id: str) -> Dict[str, Dict[str, int]]:
        """
        Aggregate which agents give helpful advice over time.

        Returns: {agent_name: {"helpful": N, "unhelpful": N, "unknown": N}}
        """
        diagnostics = self.get_diagnostics(project_id)
        stats: Dict[str, Dict[str, int]] = {}

        for d in diagnostics:
            for outcome in d.consultation_outcomes:
                agent = outcome.agent
                if agent not in stats:
                    stats[agent] = {"helpful": 0, "unhelpful": 0, "unknown": 0}

                if outcome.helpful is True:
                    stats[agent]["helpful"] += 1
                elif outcome.helpful is False:
                    stats[agent]["unhelpful"] += 1
                else:
                    stats[agent]["unknown"] += 1

        return stats

    def get_failure_type_stats(self, project_id: str) -> Dict[str, int]:
        """
        Count failure types across all diagnostics.

        Useful for identifying common failure patterns.
        """
        diagnostics = self.get_diagnostics(project_id)
        stats: Dict[str, int] = {}

        for d in diagnostics:
            ft = d.failure_type
            stats[ft] = stats.get(ft, 0) + 1

        return stats
