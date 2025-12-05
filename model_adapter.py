"""
model_adapter.py - Unified LLM interface for multiple providers.

Provides Anthropic-style interface regardless of backend.
Currently supports: Anthropic (Claude), xAI (Grok)

The adapter normalizes:
- Streaming interface (.messages.stream() context manager)
- Response format (.content[0].text, .usage.input_tokens)
- System prompt handling (Grok uses messages, Claude uses system param)

Usage:
    from model_adapter import create_adapter

    # Create adapter (reads provider from config or uses default)
    adapter = create_adapter(
        provider="xai",  # or "anthropic"
        api_key=os.getenv("XAI_API_KEY"),
        model="grok-4-1-fast-reasoning",
    )

    # Use exactly like Anthropic client
    response = adapter.messages.create(
        model="grok-4-1-fast-reasoning",
        max_tokens=4096,
        system="You are helpful.",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.content[0].text)
"""

import os
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator, Generator
from contextlib import contextmanager

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Response Dataclasses (match Anthropic SDK structure)
# =============================================================================

@dataclass
class Usage:
    """Token usage - matches anthropic.types.Usage"""
    input_tokens: int
    output_tokens: int


@dataclass
class TextBlock:
    """Content block - matches anthropic.types.TextBlock"""
    type: str = "text"
    text: str = ""


@dataclass
class Message:
    """Response message - matches anthropic.types.Message"""
    id: str
    type: str
    role: str
    content: List[TextBlock]
    model: str
    stop_reason: Optional[str]
    usage: Usage


# =============================================================================
# Streaming Support
# =============================================================================

class StreamManager:
    """
    Context manager for streaming responses.
    Matches Anthropic's client.messages.stream() interface.
    """

    def __init__(self, response_iter: Generator, model: str):
        self._response_iter = response_iter
        self._model = model
        self._collected_text = ""
        self._usage = None
        self._stream_exhausted = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exhaust stream if not already done
        if not self._stream_exhausted:
            for _ in self.text_stream:
                pass

    @property
    def text_stream(self) -> Iterator[str]:
        """Iterate over text chunks as they arrive."""
        for chunk_data in self._response_iter:
            if chunk_data.get("choices"):
                delta = chunk_data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    self._collected_text += content
                    yield content

            # Capture usage from final chunk if present
            if chunk_data.get("usage"):
                self._usage = Usage(
                    input_tokens=chunk_data["usage"].get("prompt_tokens", 0),
                    output_tokens=chunk_data["usage"].get("completion_tokens", 0),
                )

        self._stream_exhausted = True

    def get_final_message(self) -> Message:
        """Get complete message after streaming."""
        # Ensure stream is exhausted
        if not self._stream_exhausted:
            for _ in self.text_stream:
                pass

        # Estimate tokens if not provided
        if not self._usage:
            # Rough estimate: 1 token ≈ 4 chars
            self._usage = Usage(
                input_tokens=0,  # Unknown for streamed
                output_tokens=len(self._collected_text) // 4,
            )

        return Message(
            id=f"msg_{int(time.time())}",
            type="message",
            role="assistant",
            content=[TextBlock(text=self._collected_text)],
            model=self._model,
            stop_reason="end_turn",
            usage=self._usage,
        )


# =============================================================================
# Grok (xAI) Adapter
# =============================================================================

class GrokMessages:
    """
    Messages interface for Grok - matches anthropic.Anthropic().messages
    """

    API_BASE = "https://api.x.ai/v1"

    def __init__(self, api_key: str, default_model: str = "grok-4-1-fast-reasoning"):
        self.api_key = api_key
        self.default_model = default_model
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create configured requests session."""
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "CogTwin/2.7.0",
        })
        return session

    def _convert_to_openai_format(
        self,
        system: str,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Convert Anthropic-style (system param + messages) to OpenAI-style (all in messages).
        """
        converted = []

        # System prompt becomes first message
        if system:
            converted.append({"role": "system", "content": system})

        # Add user/assistant messages
        converted.extend(messages)

        return converted

    def create(
        self,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: str = "",
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> Message:
        """
        Create a message (non-streaming).
        Matches anthropic.Anthropic().messages.create()
        """
        model = model or self.default_model
        messages = messages or []

        # Convert to OpenAI format
        openai_messages = self._convert_to_openai_format(system, messages)

        payload = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        url = f"{self.API_BASE}/chat/completions"

        logger.debug(f"Grok API request: {len(openai_messages)} messages")
        start = time.time()

        response = self.session.post(url, json=payload, timeout=(10, 120))
        response.raise_for_status()

        data = response.json()
        elapsed = time.time() - start
        logger.info(f"Grok API response: {elapsed:.2f}s")

        # Extract content
        content = data["choices"][0]["message"]["content"]
        usage_data = data.get("usage", {})

        return Message(
            id=data.get("id", f"msg_{int(time.time())}"),
            type="message",
            role="assistant",
            content=[TextBlock(text=content)],
            model=model,
            stop_reason=data["choices"][0].get("finish_reason", "end_turn"),
            usage=Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
            ),
        )

    @contextmanager
    def stream(
        self,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: str = "",
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> Generator[StreamManager, None, None]:
        """
        Stream a message response.
        Matches anthropic.Anthropic().messages.stream() context manager.

        Usage:
            with adapter.messages.stream(...) as stream:
                for chunk in stream.text_stream:
                    print(chunk, end="")
            response = stream.get_final_message()
        """
        model = model or self.default_model
        messages = messages or []

        # Convert to OpenAI format
        openai_messages = self._convert_to_openai_format(system, messages)

        payload = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},  # Request usage in stream
        }

        url = f"{self.API_BASE}/chat/completions"

        logger.debug(f"Grok API stream request: {len(openai_messages)} messages")

        response = self.session.post(
            url,
            json=payload,
            timeout=(10, 300),  # Longer timeout for streaming
            stream=True,
        )
        response.raise_for_status()

        def chunk_generator():
            """Parse SSE stream into chunk dicts."""
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE chunk: {data_str}")

        yield StreamManager(chunk_generator(), model)


class GrokAdapter:
    """
    Grok adapter with Anthropic-compatible interface.

    Usage:
        adapter = GrokAdapter(api_key="...")
        response = adapter.messages.create(...)
    """

    def __init__(self, api_key: str, model: str = "grok-4-1-fast-reasoning"):
        self.messages = GrokMessages(api_key, default_model=model)


# =============================================================================
# Anthropic Passthrough (for comparison/fallback)
# =============================================================================

class AnthropicAdapter:
    """
    Thin wrapper around native Anthropic client.
    Exists so we can use the same create_adapter() interface.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.messages = self._client.messages
        self.default_model = model


# =============================================================================
# Factory Function
# =============================================================================

def create_adapter(
    provider: str = "xai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Create an LLM adapter for the specified provider.

    Args:
        provider: "xai" (Grok) or "anthropic" (Claude)
        api_key: API key (or reads from env)
        model: Model name (uses provider default if not specified)

    Returns:
        Adapter with .messages.create() and .messages.stream() interface
    """
    if provider == "xai":
        api_key = api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY required for Grok")
        model = model or "grok-4-1-fast-reasoning"
        return GrokAdapter(api_key=api_key, model=model)

    elif provider == "anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY required for Claude")
        model = model or "claude-sonnet-4-20250514"
        return AnthropicAdapter(api_key=api_key, model=model)

    else:
        raise ValueError(f"Unknown provider: {provider}")
