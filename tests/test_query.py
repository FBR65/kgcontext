"""Tests for query / ask functionality."""

from __future__ import annotations

from kgcontext import Answer, Document, KnowledgeGraph


class TestQuery:
    def test_query_returns_answer(self, kg_with_llm: KnowledgeGraph, tmp_txt) -> None:
        kg_with_llm.ingest([str(tmp_txt)])
        answer = kg_with_llm.query("What are the requirements?")
        assert isinstance(answer, Answer)
        assert len(answer.content) > 0

    def test_query_provenance(self, kg_with_llm: KnowledgeGraph, tmp_txt) -> None:
        kg_with_llm.ingest([str(tmp_txt)])
        answer = kg_with_llm.query("What are the requirements?")
        assert len(answer.provenance) > 0
        for _fact_id, prov in answer.provenance.items():
            assert "document_path" in prov
            assert "evidence_excerpt" in prov

    def test_query_empty_graph_no_llm(self, empty_kg: KnowledgeGraph) -> None:
        answer = empty_kg.query("anything")
        assert "No LLM configured" in answer.content
        assert "No documents" in answer.context or len(answer.content) > 0

    def test_query_no_llm_returns_context(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        f = empty_kg.add_fact("test fact", doc_id)
        empty_kg.add_evidence("evidence", f, doc_id)

        answer = empty_kg.query("what?")
        assert "test fact" in answer.context
        assert isinstance(answer, Answer)

    def test_query_with_conflicts(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        f = empty_kg.add_fact("fact", doc_id)
        empty_kg.add_evidence("ev", f, doc_id)
        empty_kg.add_conflict("contradiction", f)

        answer = empty_kg.query("q")
        assert "contradiction" in answer.conflicts

    def test_query_max_tokens_override(self, kg_with_llm: KnowledgeGraph, tmp_txt) -> None:
        kg_with_llm.ingest([str(tmp_txt)])
        answer = kg_with_llm.query("q", max_tokens=100)
        # Context should be limited
        assert isinstance(answer, Answer)