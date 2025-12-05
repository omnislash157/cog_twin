"""Multi-agent wave loop for coding swarm."""
from .registry import AgentRole, spawn_agent
from .orchestrator import run_wave, run_project

__all__ = ["AgentRole", "spawn_agent", "run_wave", "run_project"]
