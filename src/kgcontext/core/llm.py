"""LLM client protocol and OpenAI implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol any LLM client must satisfy."""

    def complete(self, prompt: str) -> str:
        """Send a prompt and return the completion text."""
        ...


class OpenAIClient:
    """OpenAI-compatible client. Works with Ollama, vLLM, etc. via base_url."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            from openai import OpenAI

            kwargs: dict = {"model": self._model}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


class MockLLMClient:
    """Mock client for testing — returns deterministic responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or []
        self._call_count = 0
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return '{"facts": [], "evidence": [], "requirements": []}'