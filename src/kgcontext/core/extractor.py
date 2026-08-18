"""LLM-gestützte Fact/Evidence-Extraktion."""

from __future__ import annotations

import json
from typing import Any

from .llm import LLMClient
from ..types import Document, Evidence, Fact, Requirement


class ExtractionError(Exception):
    """Raised when LLM extraction fails."""


EXTRACT_PROMPT = """\
You are a knowledge extraction system. Analyze the following document and extract:
1. Facts — atomic statements that are stated or derivable from the text.
2. Evidence — specific excerpts that support each fact.
3. Requirements — obligations or conditions stated in the text.

Return ONLY valid JSON with this structure:
{{
  "facts": [{{"content": "...", "evidence": ["excerpt1", "excerpt2"]}}],
  "requirements": [{{"content": "..."}}]
}}

Document content:
---
{content}
---
"""


class LLMExtractor:
    """Extracts facts, evidence, and requirements from documents via LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def extract(self, document: Document) -> dict[str, Any]:
        """Extract structured knowledge from a document.

        Returns:
            dict with keys 'facts' (list of {content, evidence}), 'requirements' (list of {content}).

        Raises:
            ExtractionError: If LLM response cannot be parsed.
        """
        # Truncate very long documents to avoid token blowout
        content = document.content[:50000]
        prompt = EXTRACT_PROMPT.format(content=content)
        raw = self._llm.complete(prompt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ExtractionError(f"LLM returned invalid JSON: {e}") from e

        if not isinstance(data, dict):
            raise ExtractionError("LLM response is not a JSON object")

        return data

    def extract_facts(self, document: Document) -> list[Fact]:
        """Extract facts from a document."""
        data = self.extract(document)
        return [Fact(content=f["content"]) for f in data.get("facts", [])]

    def extract_evidence(self, document: Document) -> list[Evidence]:
        """Extract evidence excerpts from a document."""
        data = self.extract(document)
        evidence: list[Evidence] = []
        for f in data.get("facts", []):
            for excerpt in f.get("evidence", []):
                evidence.append(Evidence(content=excerpt))
        return evidence

    def extract_requirements(self, document: Document) -> list[Requirement]:
        """Extract requirements from a document."""
        data = self.extract(document)
        return [Requirement(content=r["content"]) for r in data.get("requirements", [])]