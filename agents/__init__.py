"""Multi-agent wave loop for coding swarm."""
from .registry import AgentRole, spawn_agent
from .orchestrator import run_wave, run_project
from .swarm_orchestrator import SwarmOrchestrator
from .schemas import Project, WaveSummary, Failure, Verdict
from .persistence import SwarmPersistence
from .holders import CodeHolder, ConvoHolder, HolderQuery

__all__ = [
    "AgentRole", "spawn_agent", "run_wave", "run_project",
    "SwarmOrchestrator", "SwarmPersistence",
    "Project", "WaveSummary", "Failure", "Verdict",
    "CodeHolder", "ConvoHolder", "HolderQuery",
]
