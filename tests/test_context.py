"""Tests for context builder."""

from __future__ import annotations

from kgcontext import KnowledgeGraph, Document


class TestContextBuilder:
    def test_empty_graph_context(self, empty_kg: KnowledgeGraph) -> None:
        from kgcontext.core.context import ContextBuilder
        cb = ContextBuilder(max_tokens=4000)
        ctx = cb.build(empty_kg)
        assert "No documents" in ctx.to_prompt() or len(ctx.facts) == 0

    def test_only_verified_facts_included(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)

        # Supported fact
        f1 = empty_kg.add_fact("supported fact", doc_id)
        empty_kg.add_evidence("evidence text", f1, doc_id)

        # Unsupported fact
        empty_kg.add_fact("unsupported fact", doc_id)

        from kgcontext.core.context import ContextBuilder
        cb = ContextBuilder(max_tokens=4000)
        ctx = cb.build(empty_kg, include_unsupported=False)
        assert len(ctx.facts) == 1
        assert ctx.facts[0].content == "supported fact"

    def test_include_unsupported_flag(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)

        f1 = empty_kg.add_fact("supported", doc_id)
        empty_kg.add_evidence("ev", f1, doc_id)
        empty_kg.add_fact("unsupported", doc_id)

        from kgcontext.core.context import ContextBuilder
        cb = ContextBuilder(max_tokens=4000)
        ctx = cb.build(empty_kg, include_unsupported=True)
        assert len(ctx.facts) == 2

    def test_provenance_populated(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/doc.pdf", doc_type="pdf")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("fact with provenance", doc_id)
        empty_kg.add_evidence("exact excerpt", fact_id, doc_id)

        from kgcontext.core.context import ContextBuilder
        cb = ContextBuilder(max_tokens=4000)
        ctx = cb.build(empty_kg)
        assert fact_id in ctx.provenance
        assert ctx.provenance[fact_id]["document_path"] == "/doc.pdf"
        assert ctx.provenance[fact_id]["evidence_excerpt"] == "exact excerpt"

    def test_token_budget_respected(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)

        # Add many facts with long content
        for i in range(50):
            f = empty_kg.add_fact(f"fact number {i} " * 100, doc_id)
            empty_kg.add_evidence(f"evidence {i} " * 50, f, doc_id)

        from kgcontext.core.context import ContextBuilder
        cb = ContextBuilder(max_tokens=500)
        ctx = cb.build(empty_kg)
        assert ctx.token_count <= 500

    def test_conflicts_included(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("fact", doc_id)
        empty_kg.add_evidence("ev", fact_id, doc_id)
        empty_kg.add_conflict("this conflicts", fact_id)

        from kgcontext.core.context import ContextBuilder
        cb = ContextBuilder(max_tokens=4000)
        ctx = cb.build(empty_kg, include_conflicts=True)
        assert len(ctx.conflicts) == 1
        assert "this conflicts" in ctx.conflicts[0].content

    def test_to_messages_format(self, empty_kg: KnowledgeGraph) -> None:
        from kgcontext.core.context import ContextBuilder
        cb = ContextBuilder(max_tokens=4000)
        ctx = cb.build(empty_kg)
        msgs = ctx.to_messages()
        assert isinstance(msgs, list)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_to_prompt_has_facts(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("the sky is blue", doc_id)
        empty_kg.add_evidence("sky appears blue due to scattering", fact_id, doc_id)

        from kgcontext.core.context import ContextBuilder
        cb = ContextBuilder(max_tokens=4000)
        ctx = cb.build(empty_kg)
        prompt = ctx.to_prompt()
        assert "[FACT]" in prompt
        assert "the sky is blue" in prompt
        assert "Evidence:" in prompt
        assert "Source:" in prompt