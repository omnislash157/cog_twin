"""
Enterprise CogTwin - Config-driven wrapper for enterprise deployment.

Conditionally loads memory pipelines or context stuffing based on config.
Adds tenant context support and division-aware doc loading.

Usage:
    from enterprise_twin import EnterpriseTwin

    twin = EnterpriseTwin(config_path="enterprise_config.yaml")
    await twin.start()

    # With tenant context
    async for chunk in twin.think("How do I do X?", tenant=tenant_context):
        print(chunk, end="")

Version: 1.0.0
"""

import asyncio
import logging
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncIterator, Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

# Enterprise config
from config_loader import (
    load_config,
    cfg,
    get_config,
    memory_enabled,
    context_stuffing_enabled,
    is_enterprise_mode,
    get_docs_dir,
    get_max_stuffing_tokens,
    get_division_categories,
)

# Enterprise components
from enterprise_voice import EnterpriseVoice, get_voice_for_division
from enterprise_tenant import TenantContext

# Model adapter
from model_adapter import create_adapter

logger = logging.getLogger(__name__)


class EnterpriseTwin:
    """
    Enterprise-ready CogTwin with config-driven feature flags.

    When memory_pipelines=false:
    - No FAISS, no embeddings, no memory retrieval
    - Just loads docs and stuffs them into context
    - Fast startup, simple operation

    When memory_pipelines=true:
    - Full CogTwin experience with all 5 lanes
    - Delegates to CogTwin class
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize enterprise twin.

        Args:
            config_path: Path to enterprise_config.yaml
            data_dir: Override data directory
        """
        # Load enterprise config first
        if config_path:
            load_config(config_path)
        else:
            load_config()  # Auto-detect

        self.data_dir = Path(data_dir) if data_dir else Path(cfg("paths.data_dir", "./data"))
        self.model = cfg("model.name", "grok-4-fast-reasoning")

        # Track mode
        self._memory_mode = memory_enabled()
        self._context_stuffing_mode = context_stuffing_enabled()

        logger.info(f"Enterprise mode: memory={self._memory_mode}, stuffing={self._context_stuffing_mode}")

        if self._memory_mode:
            # Full memory mode - delegate to CogTwin
            from cog_twin import CogTwin
            self._twin = CogTwin(data_dir=self.data_dir)
            self._doc_builder = None
            self.memory_count = self._twin.memory_count
            logger.info(f"Full memory mode: {self.memory_count} nodes loaded")

        else:
            # Context stuffing mode - lightweight
            self._twin = None
            self.memory_count = 0

            # Initialize LLM client directly
            provider = cfg("model.provider", "xai")
            if provider == "xai":
                api_key = os.getenv("XAI_API_KEY")
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY")

            self.client = create_adapter(
                provider=provider,
                api_key=api_key,
                model=self.model,
            )

            # Initialize doc loader if stuffing enabled
            if self._context_stuffing_mode:
                from doc_loader import DocLoader, DivisionContextBuilder

                docs_dir = Path(get_docs_dir())
                if docs_dir.exists():
                    self._doc_loader = DocLoader(docs_dir)
                    self._doc_builder = DivisionContextBuilder(self._doc_loader)
                    stats = self._doc_loader.get_stats()
                    logger.info(f"DocLoader ready: {stats.total_docs} docs, ~{stats.total_tokens} tokens")
                else:
                    self._doc_loader = None
                    self._doc_builder = None
                    logger.warning(f"Docs directory not found: {docs_dir}")
            else:
                self._doc_loader = None
                self._doc_builder = None

        # Track session
        self.session_id = f"enterprise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.query_count = 0

    async def start(self):
        """Start the twin (only needed for memory mode)."""
        if self._twin:
            await self._twin.start()
        logger.info(f"EnterpriseTwin started: {self.session_id}")

    async def stop(self):
        """Stop the twin."""
        if self._twin:
            await self._twin.stop()
        logger.info(f"EnterpriseTwin stopped. Queries: {self.query_count}")

    async def think(
        self,
        user_input: str,
        tenant: Optional[TenantContext] = None,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Process user input and generate response.

        In memory mode: delegates to full CogTwin.
        In stuffing mode: builds context from docs and calls LLM directly.

        Args:
            user_input: The user's query
            tenant: Optional tenant context for division-aware responses
            stream: Whether to stream response chunks

        Yields:
            Response chunks
        """
        self.query_count += 1

        if self._memory_mode and self._twin:
            # Delegate to full CogTwin
            async for chunk in self._twin.think(user_input, stream=stream):
                yield chunk
            return

        # ========== CONTEXT STUFFING MODE ==========

        start_time = time.time()

        # Determine division from tenant or default
        division = tenant.division if tenant else "default"

        # Build doc context
        doc_context = ""
        if self._doc_builder:
            # Get categories for this division
            categories = get_division_categories(division)
            max_tokens = get_max_stuffing_tokens()

            # Build context for each category
            for category in categories:
                category_context = self._doc_builder.get_context_for_division(
                    category,
                    max_tokens=max_tokens // len(categories),
                )
                doc_context += category_context

            logger.info(f"Stuffed {len(doc_context)} chars of docs for division: {division}")

        # Get voice for this division
        voice = get_voice_for_division(division, config=get_config())

        # Build system prompt
        system_prompt = voice.build_system_prompt(
            memory_count=0,
            user_zone=tenant.zone if tenant else None,
            user_role=tenant.role if tenant else None,
            doc_context=doc_context,
        )

        # Generate response
        full_response = ""

        if stream:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=cfg("model.max_tokens", 8192),
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}],
            ) as stream_response:
                for chunk in stream_response.text_stream:
                    full_response += chunk
                    yield chunk

            response_obj = stream_response.get_final_message()
            tokens_used = response_obj.usage.input_tokens + response_obj.usage.output_tokens
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=cfg("model.max_tokens", 8192),
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}],
            )
            full_response = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            yield full_response

        elapsed = time.time() - start_time
        logger.info(f"Query complete: {elapsed:.2f}s, {tokens_used} tokens")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "query_count": self.query_count,
            "memory_mode": self._memory_mode,
            "context_stuffing_mode": self._context_stuffing_mode,
            "memory_count": self.memory_count,
        }


# =============================================================================
# CLI
# =============================================================================

async def main():
    """Quick test of enterprise twin."""
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("Initializing EnterpriseTwin...")
    twin = EnterpriseTwin(config_path=config_path)
    await twin.start()

    print(f"\nSession: {twin.session_id}")
    print(f"Memory mode: {twin._memory_mode}")
    print(f"Context stuffing: {twin._context_stuffing_mode}")
    print(f"Memory count: {twin.memory_count}")

    # Test query
    print("\n" + "=" * 60)
    print("Test query: 'What are the night shift procedures?'")
    print("=" * 60 + "\n")

    async for chunk in twin.think("What are the night shift procedures?"):
        print(chunk, end="", flush=True)

    print("\n\n" + "=" * 60)
    print(f"Stats: {twin.get_session_stats()}")

    await twin.stop()


if __name__ == "__main__":
    asyncio.run(main())
