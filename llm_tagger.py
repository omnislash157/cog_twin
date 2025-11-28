"""
LLM Tagger - Cheap Episodic Memory Tagging via Grok

Uses Grok API (xAI) for fast, cheap tagging of episodic memories.
~$0.30 for 2200 conversations with grok-2-fast.

Tags extracted:
- summary: One-line summary
- primary_intent: What were we trying to do
- outcome: resolved/abandoned/ongoing/paused
- interruption_reason: Why we stopped (if applicable)
- key_entities: People, projects, concepts
- emotional_arc: start→end emotional journey
- domains: What domains touched
- time_context: "late night session", "quick check", etc.

Usage:
    tagger = GrokTagger(api_key)
    episodes = await tagger.tag_batch(episodes, batch_size=10)

Version: 1.0.0 (cog_twin)
"""

import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

import httpx

from schemas import EpisodicMemory

logger = logging.getLogger(__name__)


# Tagging prompt template
TAGGING_PROMPT = """Analyze this conversation and extract structured tags.

CONVERSATION:
{conversation_text}

Extract the following in JSON format:
{{
    "summary": "One sentence summary of what this conversation was about",
    "primary_intent": "What was the user trying to accomplish? (e.g., debug, learn, decide, create, explore, vent)",
    "outcome": "One of: resolved, abandoned, ongoing, paused, reference",
    "interruption_reason": "If paused/abandoned, why? Otherwise empty string",
    "key_entities": ["List", "of", "key", "people", "projects", "concepts"],
    "emotional_arc": "start_emotion → end_emotion (e.g., frustrated → satisfied)",
    "domains": ["List", "of", "domains", "touched"],
    "time_context": "Context like 'debugging session', 'quick question', 'deep dive', 'late night work'"
}}

Respond with ONLY valid JSON, no markdown or explanation."""


class GrokTagger:
    """
    Tags episodic memories using Grok API.

    Cheap and fast: grok-2-fast is $0.50/M output tokens.
    2200 conversations ≈ $0.30.
    """

    # Grok/xAI API endpoint (OpenAI-compatible)
    API_URL = "https://api.x.ai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-2-fast",  # Cheapest, fastest
        requests_per_minute: int = 60,
    ):
        """
        Initialize Grok tagger.

        Args:
            api_key: xAI/Grok API key (or from XAI_API_KEY env)
            model: Model to use (grok-2-fast recommended)
            requests_per_minute: Rate limit
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY or GROK_API_KEY required")

        self.model = model
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._rate_lock = asyncio.Lock()

        # Stats
        self.stats = {
            "episodes_tagged": 0,
            "api_calls": 0,
            "errors": 0,
            "total_tokens": 0,
        }

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        async with self._rate_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()

    def _format_conversation(self, episode: EpisodicMemory, max_chars: int = 8000) -> str:
        """Format conversation for the prompt."""
        parts = [f"Title: {episode.title}"]

        for msg in episode.messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            parts.append(f"\n{role}: {content}")

        text = "\n".join(parts)

        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[...truncated...]"

        return text

    async def _tag_single(
        self,
        client: httpx.AsyncClient,
        episode: EpisodicMemory,
    ) -> Dict[str, Any]:
        """Tag a single episode."""
        await self._rate_limit()

        conversation_text = self._format_conversation(episode)
        prompt = TAGGING_PROMPT.format(conversation_text=conversation_text)

        try:
            response = await client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # Low temp for consistent extraction
                    "max_tokens": 500,
                },
                timeout=30.0,
            )

            response.raise_for_status()
            data = response.json()

            self.stats["api_calls"] += 1

            # Extract content
            content = data["choices"][0]["message"]["content"]

            # Track tokens
            usage = data.get("usage", {})
            self.stats["total_tokens"] += usage.get("total_tokens", 0)

            # Parse JSON response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            tags = json.loads(content.strip())
            self.stats["episodes_tagged"] += 1

            return tags

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for episode {episode.id}: {e}")
            self.stats["errors"] += 1
            return self._default_tags()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error tagging episode {episode.id}: {e.response.status_code}")
            self.stats["errors"] += 1
            return self._default_tags()

        except Exception as e:
            logger.error(f"Error tagging episode {episode.id}: {e}")
            self.stats["errors"] += 1
            return self._default_tags()

    def _default_tags(self) -> Dict[str, Any]:
        """Return default tags on error."""
        return {
            "summary": "",
            "primary_intent": "unknown",
            "outcome": "unknown",
            "interruption_reason": "",
            "key_entities": [],
            "emotional_arc": "neutral→neutral",
            "domains": [],
            "time_context": "",
        }

    async def tag_batch(
        self,
        episodes: List[EpisodicMemory],
        batch_size: int = 10,
        max_concurrent: int = 5,
        show_progress: bool = True,
    ) -> List[EpisodicMemory]:
        """
        Tag a batch of episodes.

        Args:
            episodes: Episodes to tag
            batch_size: Not used (keeping for API consistency)
            max_concurrent: Max concurrent requests
            show_progress: Print progress

        Returns:
            Episodes with llm_tags populated
        """
        if show_progress:
            print(f"Tagging {len(episodes)} episodes with {self.model}...")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def tag_with_semaphore(client: httpx.AsyncClient, ep: EpisodicMemory, idx: int):
            async with semaphore:
                tags = await self._tag_single(client, ep)
                ep.llm_tags = tags

                if show_progress and (idx + 1) % 50 == 0:
                    print(f"  Progress: {idx + 1}/{len(episodes)}")

                return ep

        start_time = time.time()

        async with httpx.AsyncClient() as client:
            tasks = [
                tag_with_semaphore(client, ep, i)
                for i, ep in enumerate(episodes)
            ]
            results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        if show_progress:
            print(f"Tagged {len(episodes)} episodes in {elapsed:.1f}s")
            print(f"Rate: {len(episodes) / elapsed:.1f} episodes/sec")
            print(f"Total tokens: {self.stats['total_tokens']}")
            # Estimate cost (grok-2-fast: $0.50/M output)
            estimated_cost = (self.stats['total_tokens'] / 1_000_000) * 0.50
            print(f"Estimated cost: ${estimated_cost:.4f}")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get tagging statistics."""
        return self.stats


# ═══════════════════════════════════════════════════════════════════════════
# ALTERNATIVE TAGGERS (for different providers)
# ═══════════════════════════════════════════════════════════════════════════

class OpenAITagger:
    """Fallback tagger using OpenAI (more expensive)."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",  # Cheapest GPT-4 class
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.stats = {"episodes_tagged": 0, "errors": 0}

    # Similar implementation to GrokTagger...
    # (Keeping brief since Grok is preferred)


class AnthropicTagger:
    """Fallback tagger using Anthropic (higher quality, more expensive)."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",  # Cheapest Claude
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.stats = {"episodes_tagged": 0, "errors": 0}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Test the tagger."""
    from dotenv import load_dotenv
    load_dotenv()

    import sys

    print("LLM Tagger Test")
    print("=" * 60)

    # Create test episode
    test_episode = EpisodicMemory(
        id="test-123",
        title="Debugging async handlers",
        messages=[
            {"role": "human", "content": "I'm getting a weird error with my async FastAPI handler. It says TypeError: object NoneType can't be used in await expression"},
            {"role": "assistant", "content": "That error typically means a function that should return a coroutine is returning None instead. Can you show me the handler code?"},
            {"role": "human", "content": "def get_user(id): return db.query(User).filter(User.id == id).first()"},
            {"role": "assistant", "content": "I see the issue - your function isn't async but you're probably awaiting it. Either make it async def and use await with the DB query, or don't await it in the caller."},
            {"role": "human", "content": "Oh that fixed it! Thanks!"},
        ],
        message_count=5,
        source="anthropic",
        created_at="2024-01-15T10:30:00",
        updated_at="2024-01-15T10:35:00",
        duration_minutes=5.0,
    )

    # Fix the Source enum issue
    from schemas import Source
    test_episode.source = Source.ANTHROPIC

    from datetime import datetime
    test_episode.created_at = datetime.fromisoformat("2024-01-15T10:30:00")
    test_episode.updated_at = datetime.fromisoformat("2024-01-15T10:35:00")

    tagger = GrokTagger()

    print(f"\nTagging test episode: {test_episode.title}")
    tagged = await tagger.tag_batch([test_episode], show_progress=True)

    print("\nExtracted tags:")
    for key, value in tagged[0].llm_tags.items():
        print(f"  {key}: {value}")

    print(f"\nStats: {tagger.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
