"""Configuration for kgcontext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KGConfig:
    max_context_tokens: int = 4000
    llm_provider: str = "openai"
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    vector_store: str | None = "lancedb"
    storage_path: str = "./.kgcontext"
    include_unsupported: bool = False
    include_conflicts: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "llm_provider": self.llm_provider,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_api_key": self.llm_api_key,
            "vector_store": self.vector_store,
            "storage_path": self.storage_path,
            "include_unsupported": self.include_unsupported,
            "include_conflicts": self.include_conflicts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KGConfig:
        return cls(
            max_context_tokens=int(d.get("max_context_tokens", 4000)),
            llm_provider=str(d.get("llm_provider", "openai")),
            llm_base_url=d.get("llm_base_url"),
            llm_model=str(d.get("llm_model", "gpt-4o-mini")),
            llm_api_key=d.get("llm_api_key"),
            vector_store=d.get("vector_store", "lancedb"),
            storage_path=str(d.get("storage_path", "./.kgcontext")),
            include_unsupported=bool(d.get("include_unsupported", False)),
            include_conflicts=bool(d.get("include_conflicts", True)),
        )