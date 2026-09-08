"""Pluggable LLM client.

The reasoner (`dq_agent.llm.dq_reasoner`) only depends on the
`LLMClient.complete()` method, so the underlying model is swappable —
Claude by default, a mock for offline tests/demos, or a different
vendor by implementing this same tiny interface.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024) -> str:
        """Return the model's text completion for `prompt`."""


class AnthropicClient(LLMClient):
    """Real Claude client. Requires `anthropic` and `ANTHROPIC_API_KEY`."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it, pass api_key=..., or use "
                "MockLLMClient for an offline run."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return "".join(block.text for block in response.content if block.type == "text")


class MockLLMClient(LLMClient):
    """Deterministic, offline stand-in for tests, demos, and CI.

    Doesn't call any network. `dq_reasoner` structures its prompts so
    that a keyword-driven mock like this can still produce plausible,
    schema-valid-looking output — good enough to exercise the whole
    pipeline without an API key, while a real deployment swaps in
    `AnthropicClient` for genuinely reasoned output.
    """

    def __init__(self, canned_responses: Optional[dict[str, str]] = None):
        self.canned_responses = canned_responses or {}
        self.calls: list[str] = []

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024) -> str:
        self.calls.append(prompt)
        # Match keywords against the prompt AND the system message
        # (case-insensitively): the reasoner encodes *what kind* of call
        # this is mostly in the system message ("investigating a data
        # quality failure", "draft a concise ticket"), so a canned
        # response keyed on that phrase needs to see it too.
        haystack = f"{system or ''}\n{prompt}".lower()
        for keyword, response in self.canned_responses.items():
            if keyword.lower() in haystack:
                return response
        return _default_mock_response(haystack)


def _default_mock_response(haystack: str) -> str:
    # Extremely small heuristic responder so the reasoner's JSON-parsing
    # path has something sane to work with even with zero configuration.
    # `haystack` is already the lowercased prompt+system text.
    if "propose data quality rules" in haystack:
        return "[]"
    if "root cause" in haystack:
        return (
            "Unable to determine a specific root cause without a live model; "
            "this is the offline mock response. Recommend investigating the "
            "most recent upstream load for this table."
        )
    if "draft a concise ticket" in haystack:
        return "Data quality check failed; see attached metrics. (mock response, no LLM configured)"
    return "(mock response, no LLM configured)"
