"""Context builder — graph traversal to curated LLM context."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tiktoken

from ..types import Conflict, Document, Evidence, Fact, Requirement

if TYPE_CHECKING:
    from .pipeline import KnowledgeGraph


class Context:
    """Curated context for LLM prompting."""

    def __init__(
        self,
        facts: list[Fact],
        evidence: list[Evidence],
        requirements: list[Requirement],
        conflicts: list[Conflict],
        documents: dict[str, Document],
        provenance: dict[str, dict[str, str]],
        token_count: int = 0,
    ) -> None:
        self.facts = facts
        self.evidence = evidence
        self.requirements = requirements
        self.conflicts = conflicts
        self.documents = documents
        self.provenance = provenance
        self.token_count = token_count

    def to_prompt(self) -> str:
        """Format context as a single prompt string."""
        parts: list[str] = []

        if self.requirements:
            parts.append("## Requirements")
            for r in self.requirements:
                parts.append(f"- [REQ] {r.content}")

        if self.facts:
            parts.append("\n## Verified Facts")
            for f in self.facts:
                prov = self.provenance.get(f.id, {})
                doc = prov.get("document_path", "unknown")
                excerpt = prov.get("evidence_excerpt", "")
                parts.append(f"- [FACT] {f.content}")
                if excerpt:
                    parts.append(f'  Evidence: "{excerpt}"')
                parts.append(f"  Source: {doc}")

        if self.conflicts:
            parts.append("\n## Conflicts")
            for c in self.conflicts:
                parts.append(f"- [CONFLICT] {c.content}")

        if not parts:
            parts.append("No documents have been ingested yet.")

        parts.append(f"\n-- Token count: {self.token_count} --")
        return "\n".join(parts)

    def to_messages(self) -> list[dict[str, str]]:
        """Format context as chat messages."""
        return [
            {"role": "system", "content": "You are a document analysis assistant. Answer based only on the provided context."},
            {"role": "user", "content": self.to_prompt()},
        ]


class ContextBuilder:
    """Builds curated context from graph traversal."""

    def __init__(self, max_tokens: int = 4000) -> None:
        self._max_tokens = max_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def build(
        self,
        kg: KnowledgeGraph,
        query: str = "",
        include_unsupported: bool = False,
        include_conflicts: bool = True,
    ) -> Context:
        """Build curated context from the graph.

        Only verified paths (Fact → Evidence → Document) are included.
        """
        facts: list[Fact] = []
        evidence_list: list[Evidence] = []
        conflicts: list[Conflict] = []
        documents: dict[str, Document] = {}
        provenance: dict[str, dict[str, str]] = {}

        token_budget = self._max_tokens
        # Reserve tokens for the query + overhead
        query_tokens = self._count_tokens(query) if query else 0
        token_budget -= query_tokens + 200  # 200 for formatting overhead

        for fact in kg.facts:
            ev = kg.get_evidence_for_fact(fact.id)
            if not ev and not include_unsupported:
                continue

            fact_text = f"[FACT] {fact.content}"
            fact_tokens = self._count_tokens(fact_text)
            if token_budget - fact_tokens < 0:
                break
            token_budget -= fact_tokens

            facts.append(fact)

            for e in ev:
                ev_text = f'[EVIDENCE] "{e.content}"'
                ev_tokens = self._count_tokens(ev_text)
                if token_budget - ev_tokens < 0:
                    break
                token_budget -= ev_tokens
                evidence_list.append(e)

                doc = kg.get_source_document(e.id)
                if doc:
                    documents[doc.id] = doc
                    provenance[fact.id] = {
                        "document_path": doc.source_path,
                        "evidence_excerpt": e.content,
                    }
                else:
                    provenance[fact.id] = {
                        "document_path": "unknown",
                        "evidence_excerpt": e.content,
                    }

            if include_conflicts:
                fact_conflicts = kg.get_conflicts(fact.id)
                for c in fact_conflicts:
                    conflicts.append(c)

        total_tokens = self._max_tokens - token_budget
        return Context(
            facts=facts,
            evidence=evidence_list,
            requirements=kg.requirements,
            conflicts=conflicts,
            documents=documents,
            provenance=provenance,
            token_count=total_tokens,
        )