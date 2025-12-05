"""Parse [ARTIFACT ...] tags from LLM output."""

import re
from typing import List, Tuple
from .actions import ArtifactAction, ArtifactType


# Matches [ARTIFACT type="..." key="value" ...]
ARTIFACT_PATTERN = re.compile(
    r'\[ARTIFACT\s+([^\]]+)\]',
    re.IGNORECASE
)

# Matches key="value" pairs
ATTR_PATTERN = re.compile(
    r'(\w+)="([^"]*)"'
)


def parse_artifact_tag(tag_content: str) -> ArtifactAction | None:
    """Parse a single ARTIFACT tag's attributes into ArtifactAction."""
    attrs = dict(ATTR_PATTERN.findall(tag_content))

    if "type" not in attrs:
        return None

    try:
        artifact_type = ArtifactType(attrs.pop("type"))
    except ValueError:
        return None

    # Parse ids as list if present
    if "ids" in attrs:
        attrs["ids"] = [id.strip() for id in attrs["ids"].split(",")]

    # Parse items as list if present
    if "items" in attrs:
        attrs["items"] = [item.strip() for item in attrs["items"].split(",")]

    return ArtifactAction(type=artifact_type, **attrs)


def extract_artifacts(text: str) -> Tuple[str, List[ArtifactAction]]:
    """
    Extract all [ARTIFACT ...] tags from text.

    Returns:
        (clean_text, list_of_artifacts)
        clean_text has artifact tags stripped out
    """
    artifacts = []

    for match in ARTIFACT_PATTERN.finditer(text):
        artifact = parse_artifact_tag(match.group(1))
        if artifact:
            artifacts.append(artifact)

    # Strip tags from text
    clean_text = ARTIFACT_PATTERN.sub('', text).strip()
    # Clean up any double newlines left behind
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

    return clean_text, artifacts
