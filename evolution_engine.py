"""
Evolution Engine - Self-Improvement Mechanism

Allows the agent to read and propose patches to its own source code.
HITL-only: Never auto-applies changes.

Flow:
1. MetacognitiveMirror detects issue (e.g., "High latency", "Prediction drift")
2. EvolutionEngine maps insight to relevant source file
3. LLM analyzes code and proposes fix
4. Generates .patch file for human review
5. Human approves/rejects
6. If approved, patch is applied and logged to memory

"We improve ourselves, but only with permission."

Version: 1.0.0 (cog_twin)
"""

import ast
import difflib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

import anthropic

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of metacognitive insights that can trigger evolution."""
    PERFORMANCE = "performance"      # Latency, throughput issues
    ACCURACY = "accuracy"            # Prediction/retrieval quality
    SEMANTIC_DRIFT = "semantic_drift"  # Cluster drift, embedding issues
    MEMORY_OVERFLOW = "memory_overflow"  # Too much context, pruning needed
    ALIGNMENT_DRIFT = "alignment_drift"  # Values/behavior divergence
    CODE_SMELL = "code_smell"        # Detected antipattern in own code
    MISSING_CAPABILITY = "missing_capability"  # Can't do something we should


@dataclass
class MetacognitiveInsight:
    """An observation from the MetacognitiveMirror."""
    type: InsightType
    severity: str  # low, medium, high, critical
    description: str
    metrics: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)
    source_component: Optional[str] = None


@dataclass
class EvolutionProposal:
    """A proposed code change."""
    id: str
    insight: MetacognitiveInsight
    target_file: Path
    original_code: str
    proposed_code: str
    diff: str
    reasoning: str
    risk_level: str  # low, medium, high
    created_at: datetime = field(default_factory=datetime.now)
    patch_path: Optional[Path] = None
    approved: Optional[bool] = None
    applied_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "insight_type": self.insight.type.value,
            "insight_description": self.insight.description,
            "target_file": str(self.target_file),
            "reasoning": self.reasoning,
            "risk_level": self.risk_level,
            "created_at": self.created_at.isoformat(),
            "patch_path": str(self.patch_path) if self.patch_path else None,
            "approved": self.approved,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
        }


class EvolutionEngine:
    """
    The Self-Improvement Mechanism.

    Reads its own codebase, proposes patches based on metacognitive insights,
    and applies approved changes.
    """

    # Maps insight types to likely relevant files
    FILE_ROUTING = {
        InsightType.PERFORMANCE: [
            "retrieval.py",
            "embedder.py",
            "cog_twin.py",
            "memory_pipeline.py",
        ],
        InsightType.ACCURACY: [
            "retrieval.py",
            "heuristic_enricher.py",
        ],
        InsightType.SEMANTIC_DRIFT: [
            "embedder.py",
            "retrieval.py",
            "streaming_cluster.py",
        ],
        InsightType.MEMORY_OVERFLOW: [
            "cog_twin.py",
            "memory_pipeline.py",
            "retrieval.py",
        ],
        InsightType.ALIGNMENT_DRIFT: [
            "cog_twin.py",
            "venom_voice.py",
            "cognitive_twin.py",
        ],
        InsightType.CODE_SMELL: [
            # Could be any file - will use LLM to identify
        ],
        InsightType.MISSING_CAPABILITY: [
            "cog_twin.py",
            "retrieval.py",
        ],
    }

    ANALYSIS_PROMPT = """You are analyzing code for a cognitive twin system that needs optimization.

INSIGHT TYPE: {insight_type}
SEVERITY: {severity}
PROBLEM: {description}
METRICS: {metrics}

TARGET FILE: {filename}
```python
{source_code}
```

TASK: Analyze this code and propose a fix for the identified problem.

RULES:
1. Be surgical - change only what's necessary
2. Maintain backwards compatibility
3. Add comments explaining the fix
4. Consider edge cases
5. If the file doesn't contain the issue, say "NO_CHANGE_NEEDED"

OUTPUT FORMAT:
If changes needed, return ONLY the modified function/class/section.
Start with a brief explanation of the fix (1-2 sentences).
Then the code block.

If no changes needed, return exactly: NO_CHANGE_NEEDED: <reason>
"""

    def __init__(
        self,
        codebase_path: Path,
        patches_dir: Optional[Path] = None,
        anthropic_api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        auto_approve_low_risk: bool = False,  # Safety: default to requiring approval
    ):
        """
        Initialize the Evolution Engine.

        Args:
            codebase_path: Root directory of the cog_twin codebase
            patches_dir: Where to save .patch files (default: codebase_path/patches)
            anthropic_api_key: API key for LLM analysis
            model: Model to use for code analysis
            auto_approve_low_risk: If True, auto-apply low-risk patches (dangerous!)
        """
        self.codebase_path = Path(codebase_path)
        self.patches_dir = patches_dir or self.codebase_path / "patches"
        self.patches_dir.mkdir(exist_ok=True)

        self.model = model
        self.auto_approve_low_risk = auto_approve_low_risk

        self.client = anthropic.Anthropic(
            api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        # History of proposals
        self.proposals: List[EvolutionProposal] = []
        self.history_file = self.patches_dir / "evolution_history.json"
        self._load_history()

        # Callbacks
        self.on_proposal: Optional[Callable[[EvolutionProposal], bool]] = None

    def _load_history(self):
        """Load proposal history from disk."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                data = json.load(f)
                # Just load metadata, not full proposals
                logger.info(f"Loaded {len(data)} historical proposals")

    def _save_proposal(self, proposal: EvolutionProposal):
        """Save proposal to history."""
        history = []
        if self.history_file.exists():
            with open(self.history_file) as f:
                history = json.load(f)

        history.append(proposal.to_dict())

        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def _map_insight_to_files(self, insight: MetacognitiveInsight) -> List[Path]:
        """Route insight to relevant source files."""
        candidates = self.FILE_ROUTING.get(insight.type, [])

        # Also check if insight mentions specific files
        for word in insight.description.lower().split():
            if word.endswith(".py"):
                candidates.append(word)

        # Filter to files that actually exist
        files = []
        for candidate in candidates:
            filepath = self.codebase_path / candidate
            if filepath.exists():
                files.append(filepath)

        return files

    def _assess_risk(self, original: str, proposed: str) -> str:
        """Assess risk level of proposed change."""
        # Simple heuristics
        orig_lines = original.count('\n')
        prop_lines = proposed.count('\n')

        diff_ratio = abs(prop_lines - orig_lines) / max(orig_lines, 1)

        # Check for dangerous patterns
        dangerous_patterns = [
            "os.system", "subprocess", "eval(", "exec(",
            "__import__", "open(", "write(", "delete", "remove",
            "api_key", "password", "secret", "token",
        ]

        danger_count = sum(1 for p in dangerous_patterns if p in proposed.lower())

        if danger_count > 2 or diff_ratio > 0.5:
            return "high"
        elif danger_count > 0 or diff_ratio > 0.2:
            return "medium"
        else:
            return "low"

    def _generate_diff(self, original: str, proposed: str, filename: str) -> str:
        """Generate unified diff between original and proposed code."""
        original_lines = original.splitlines(keepends=True)
        proposed_lines = proposed.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            proposed_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )

        return "".join(diff)

    def _write_patch(self, proposal: EvolutionProposal) -> Path:
        """Write proposal to .patch file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{proposal.target_file.stem}_{proposal.insight.type.value}_{timestamp}.patch"
        patch_path = self.patches_dir / filename

        with open(patch_path, "w") as f:
            f.write(f"# EVOLUTION PROPOSAL\n")
            f.write(f"# Generated: {proposal.created_at.isoformat()}\n")
            f.write(f"# Insight: {proposal.insight.type.value} ({proposal.insight.severity})\n")
            f.write(f"# Problem: {proposal.insight.description}\n")
            f.write(f"# Risk Level: {proposal.risk_level}\n")
            f.write(f"# \n")
            f.write(f"# Reasoning: {proposal.reasoning}\n")
            f.write(f"# \n")
            f.write(f"# To apply: git apply {filename}\n")
            f.write(f"# To reject: rm {filename}\n")
            f.write(f"\n")
            f.write(proposal.diff)

        return patch_path

    async def analyze_and_propose(
        self,
        insight: MetacognitiveInsight,
    ) -> Optional[EvolutionProposal]:
        """
        Analyze an insight and propose code changes.

        Args:
            insight: The metacognitive insight triggering evolution

        Returns:
            EvolutionProposal if changes proposed, None otherwise
        """
        if insight.severity not in ["high", "critical"]:
            logger.info(f"Skipping low-severity insight: {insight.type.value}")
            return None

        logger.info(f"Analyzing insight: {insight.type.value} - {insight.description}")

        # Find relevant files
        target_files = self._map_insight_to_files(insight)
        if not target_files:
            logger.warning(f"No target files found for insight type: {insight.type.value}")
            return None

        # Analyze each file until we find one that needs changes
        for target_file in target_files:
            source_code = target_file.read_text()

            # Ask LLM to analyze and propose fix
            prompt = self.ANALYSIS_PROMPT.format(
                insight_type=insight.type.value,
                severity=insight.severity,
                description=insight.description,
                metrics=json.dumps(insight.metrics),
                filename=target_file.name,
                source_code=source_code[:15000],  # Truncate large files
            )

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.content[0].text

            if "NO_CHANGE_NEEDED" in result:
                logger.info(f"No changes needed for {target_file.name}")
                continue

            # Extract reasoning and code from response
            lines = result.strip().split("\n")
            reasoning_lines = []
            code_lines = []
            in_code = False

            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
                elif in_code:
                    code_lines.append(line)
                elif not in_code and not code_lines:
                    reasoning_lines.append(line)

            reasoning = " ".join(reasoning_lines).strip()
            proposed_code = "\n".join(code_lines)

            if not proposed_code:
                logger.warning(f"Could not extract proposed code from response")
                continue

            # Create proposal
            proposal_id = f"evo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{target_file.stem}"

            diff = self._generate_diff(source_code, proposed_code, target_file.name)
            risk_level = self._assess_risk(source_code, proposed_code)

            proposal = EvolutionProposal(
                id=proposal_id,
                insight=insight,
                target_file=target_file,
                original_code=source_code,
                proposed_code=proposed_code,
                diff=diff,
                reasoning=reasoning,
                risk_level=risk_level,
            )

            # Write patch file
            proposal.patch_path = self._write_patch(proposal)
            logger.warning(f"EVOLUTION PROPOSED: Review patch at {proposal.patch_path}")

            # Save to history
            self.proposals.append(proposal)
            self._save_proposal(proposal)

            # Check for approval
            if self.on_proposal:
                proposal.approved = self.on_proposal(proposal)
                if proposal.approved:
                    self._apply_proposal(proposal)
            elif self.auto_approve_low_risk and risk_level == "low":
                logger.warning("Auto-approving low-risk proposal (DANGEROUS MODE)")
                proposal.approved = True
                self._apply_proposal(proposal)

            return proposal

        return None

    def _apply_proposal(self, proposal: EvolutionProposal):
        """Apply an approved proposal."""
        if not proposal.approved:
            raise ValueError("Cannot apply unapproved proposal")

        # Backup original
        backup_path = proposal.target_file.with_suffix(".py.bak")
        backup_path.write_text(proposal.original_code)

        # Apply changes
        proposal.target_file.write_text(proposal.proposed_code)
        proposal.applied_at = datetime.now()

        logger.info(f"Applied evolution to {proposal.target_file.name}")
        logger.info(f"Backup saved to {backup_path}")

    def list_pending_proposals(self) -> List[EvolutionProposal]:
        """List proposals awaiting approval."""
        return [p for p in self.proposals if p.approved is None]

    def approve_proposal(self, proposal_id: str) -> bool:
        """Approve and apply a pending proposal."""
        for proposal in self.proposals:
            if proposal.id == proposal_id and proposal.approved is None:
                proposal.approved = True
                self._apply_proposal(proposal)
                self._save_proposal(proposal)
                return True
        return False

    def reject_proposal(self, proposal_id: str) -> bool:
        """Reject a pending proposal."""
        for proposal in self.proposals:
            if proposal.id == proposal_id and proposal.approved is None:
                proposal.approved = False
                self._save_proposal(proposal)
                # Clean up patch file
                if proposal.patch_path and proposal.patch_path.exists():
                    proposal.patch_path.unlink()
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Test the Evolution Engine."""
    from dotenv import load_dotenv
    load_dotenv()

    import sys
    codebase_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    print("Evolution Engine - Self-Improvement Test")
    print("=" * 60)

    engine = EvolutionEngine(codebase_path)

    # Test with a fake insight
    insight = MetacognitiveInsight(
        type=InsightType.PERFORMANCE,
        severity="high",
        description="Retrieval latency exceeds 500ms on average",
        metrics={"avg_latency_ms": 523, "p99_latency_ms": 1200},
        source_component="retrieval.py",
    )

    # Register approval callback
    def review(proposal: EvolutionProposal) -> bool:
        print(f"\nProposal: {proposal.id}")
        print(f"Risk: {proposal.risk_level}")
        print(f"Reasoning: {proposal.reasoning}")
        print(f"Patch: {proposal.patch_path}")
        response = input("Approve? (y/n): ").strip().lower()
        return response == "y"

    engine.on_proposal = review

    proposal = await engine.analyze_and_propose(insight)

    if proposal:
        print(f"\nProposal created: {proposal.id}")
        print(f"Approved: {proposal.approved}")
    else:
        print("\nNo proposal generated")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
