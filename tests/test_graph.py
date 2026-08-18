"""Tests for KnowledgeGraph node management and traversal."""

from __future__ import annotations

import pytest

from kgcontext import EdgeType, Evidence, KnowledgeGraph


class TestNodeManagement:
    def test_add_document_returns_id(self, empty_kg: KnowledgeGraph, tmp_txt) -> None:
        from kgcontext import parse_file
        doc = parse_file(str(tmp_txt))
        doc_id = empty_kg._internal_graph.add_node(doc)
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_add_fact_with_document_link(self, empty_kg: KnowledgeGraph) -> None:
        from kgcontext import Document
        doc = Document(content="test", source_path="/test.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("test fact", doc_id)
        assert isinstance(fact_id, str)
        # Verify edge
        neighbors = empty_kg._internal_graph.get_neighbors(fact_id, EdgeType.EXTRACTED_FROM)
        assert doc_id in neighbors

    def test_add_evidence_links_fact_and_document(self, empty_kg: KnowledgeGraph) -> None:
        from kgcontext import Document
        doc = Document(content="test", source_path="/test.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("test fact", doc_id)
        ev_id = empty_kg.add_evidence("excerpt", fact_id, doc_id)

        # Evidence supports fact
        supports = empty_kg._internal_graph.get_neighbors(
            fact_id, EdgeType.SUPPORTS, direction="in"
        )
        assert ev_id in supports
        # Evidence extracted from document
        extracted = empty_kg._internal_graph.get_neighbors(ev_id, EdgeType.EXTRACTED_FROM)
        assert doc_id in extracted

    def test_add_requirement_and_link(self, empty_kg: KnowledgeGraph) -> None:
        req_id = empty_kg.add_requirement("Must support 100 users")
        from kgcontext import Document
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("supports 100 users", doc_id)
        empty_kg.link_requirement_fact(req_id, fact_id)

        facts = empty_kg.get_facts_for_requirement(req_id)
        assert len(facts) == 1
        assert facts[0].content == "supports 100 users"

    def test_add_conflict(self, empty_kg: KnowledgeGraph) -> None:
        from kgcontext import Document
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("fact with conflict", doc_id)
        empty_kg.add_conflict("contradicts other source", fact_id)

        conflicts = empty_kg.get_conflicts(fact_id)
        assert len(conflicts) == 1
        assert conflicts[0].content == "contradicts other source"

    def test_ingest_idempotency(self, kg_with_llm: KnowledgeGraph, tmp_txt) -> None:
        ids1 = kg_with_llm.ingest([str(tmp_txt)])
        ids2 = kg_with_llm.ingest([str(tmp_txt)])
        assert ids1 == ids2  # same doc_id returned

    def test_ingest_multiple_files(self, kg_with_llm: KnowledgeGraph, tmp_txt, tmp_md) -> None:
        ids = kg_with_llm.ingest([str(tmp_txt), str(tmp_md)])
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_ingest_no_extract(self, kg_with_llm: KnowledgeGraph, tmp_txt) -> None:
        ids = kg_with_llm.ingest([str(tmp_txt)], extract=False)
        assert len(ids) == 1
        # No facts should exist
        assert len(kg_with_llm.facts) == 0


class TestTraversal:
    def test_get_evidence_for_fact(self, empty_kg: KnowledgeGraph) -> None:
        from kgcontext import Document
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("fact1", doc_id)
        empty_kg.add_evidence("evidence1", fact_id, doc_id)
        empty_kg.add_evidence("evidence2", fact_id, doc_id)

        evidence = empty_kg.get_evidence_for_fact(fact_id)
        assert len(evidence) == 2

    def test_get_source_document(self, empty_kg: KnowledgeGraph) -> None:
        from kgcontext import Document
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("fact1", doc_id)
        ev_id = empty_kg.add_evidence("ev1", fact_id, doc_id)

        result = empty_kg.get_source_document(ev_id)
        assert result is not None
        assert result.source_path == "/t.txt"

    def test_get_source_document_no_document(self, empty_kg: KnowledgeGraph) -> None:
        ev = Evidence(content="orphan evidence")
        ev_id = empty_kg._internal_graph.add_node(ev)
        result = empty_kg.get_source_document(ev_id)
        assert result is None

    def test_get_unsupported_facts(self, empty_kg: KnowledgeGraph) -> None:
        from kgcontext import Document
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)

        fact_with_ev = empty_kg.add_fact("supported fact", doc_id)
        empty_kg.add_evidence("ev", fact_with_ev, doc_id)

        empty_kg.add_fact("unsupported fact", doc_id)

        unsupported = empty_kg.get_unsupported_facts()
        assert len(unsupported) == 1
        assert unsupported[0].content == "unsupported fact"

    def test_get_incomplete_requirements(self, empty_kg: KnowledgeGraph) -> None:
        empty_kg.add_requirement("req1")
        req2 = empty_kg.add_requirement("req2")

        from kgcontext import Document
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("satisfies req2", doc_id)
        empty_kg.link_requirement_fact(req2, fact_id)

        incomplete = empty_kg.get_incomplete_requirements()
        assert len(incomplete) == 1
        assert incomplete[0].content == "req1"

    def test_nonexistent_node_raises(self, empty_kg: KnowledgeGraph) -> None:
        with pytest.raises(KeyError):
            empty_kg._internal_graph.get_node("nonexistent-id")

    def test_multi_edge_one_doc_multiple_facts(self, empty_kg: KnowledgeGraph) -> None:
        from kgcontext import Document
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)

        empty_kg.add_fact("fact1", doc_id)
        empty_kg.add_fact("fact2", doc_id)
        empty_kg.add_fact("fact3", doc_id)

        assert len(empty_kg.facts) == 3