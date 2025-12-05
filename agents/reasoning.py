"""
Extract reasoning traces from model responses.
Models should wrap reasoning in <reasoning> tags.
"""

import re
from typing import List
from .schemas import ReasoningStep


def extract_reasoning_trace(response: str) -> List[ReasoningStep]:
    """
    Extract reasoning steps from model response.

    Expected format in response:
    <reasoning>
    Step 1: Analyzing the existing code structure
    Step 2: Found /health endpoint pattern to follow
    Step 3: Decided to append after existing endpoints
    </reasoning>
    """
    match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    if not match:
        return []

    reasoning_text = match.group(1).strip()
    steps = []

    # Parse "Step N:" format
    step_pattern = re.compile(r'Step\s*(\d+)\s*:\s*(.+?)(?=Step\s*\d+:|$)', re.DOTALL)
    matches = step_pattern.findall(reasoning_text)

    for step_num, content in matches:
        content = content.strip()

        # Try to split into thought/observation/decision
        step = ReasoningStep(step=int(step_num), content=content)

        # Look for sub-labels
        if 'Observation:' in content:
            parts = content.split('Observation:', 1)
            step.content = parts[0].strip()
            step.observation = parts[1].strip()

        if 'Decision:' in content:
            parts = content.split('Decision:', 1)
            if step.observation:
                step.observation = parts[0].strip()
            else:
                step.content = parts[0].strip()
            step.decision = parts[1].strip()

        steps.append(step)

    # Fallback: split by newlines if no Step N: format
    if not steps:
        for i, line in enumerate(reasoning_text.split('\n'), 1):
            line = line.strip()
            if line:
                steps.append(ReasoningStep(step=i, content=line))

    return steps


def strip_reasoning_tags(response: str) -> str:
    """Remove reasoning tags from response for clean output."""
    return re.sub(r'<reasoning>.*?</reasoning>', '', response, flags=re.DOTALL).strip()


def format_reasoning_trace(steps: List[ReasoningStep]) -> str:
    """Format reasoning steps for display."""
    if not steps:
        return "No reasoning trace captured."

    lines = []
    for step in steps:
        line = f"Step {step.step}: {step.content}"
        if step.observation:
            line += f"\n  Observation: {step.observation}"
        if step.decision:
            line += f"\n  Decision: {step.decision}"
        lines.append(line)

    return "\n".join(lines)


REASONING_INSTRUCTION = """
When responding, include your reasoning process in <reasoning> tags:

<reasoning>
Step 1: [What you're analyzing or considering]
Step 2: [What you found or observed]
Step 3: [What you decided and why]
</reasoning>

Then provide your actual response after the reasoning block.
"""
