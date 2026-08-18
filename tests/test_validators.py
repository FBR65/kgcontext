"""Tests for deterministic validators."""

from __future__ import annotations

from kgcontext import Document, KnowledgeGraph


class TestValidators:
    def test_fact_without_evidence_warning(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        empty_kg.add_fact("unsupported fact", doc_id)

        report = empty_kg.validate()
        assert report.has_warnings
        assert not report.has_errors
        assert any("plain-language explanation" in i.message for i in report.issues)

    def test_fact_with_evidence_no_warning(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("supported fact", doc_id)
        empty_kg.add_evidence("excerpt", fact_id, doc_id)

        report = empty_kg.validate()
        assert not any("plain-language explanation" in i.message for i in report.issues)

    def test_requirement_without_fact_info(self, empty_kg: KnowledgeGraph) -> None:
        empty_kg.add_requirement("unfulfilled requirement")

        report = empty_kg.validate()
        assert any(i.level == "info" for i in report.issues)
        assert any("not satisfied" in i.message for i in report.issues)

    def test_fact_with_conflict_warning(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("conflicted fact", doc_id)
        empty_kg.add_evidence("ev", fact_id, doc_id)
        empty_kg.add_conflict("contradicts", fact_id)

        report = empty_kg.validate()
        assert any("conflict" in i.message for i in report.issues)

    def test_empty_graph_clean_report(self, empty_kg: KnowledgeGraph) -> None:
        report = empty_kg.validate()
        assert not report.has_errors
        assert not report.has_warnings
        assert "0 errors" in report.summary()

    def test_report_summary_format(self, empty_kg: KnowledgeGraph) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        empty_kg.add_fact("f", doc_id)
        empty_kg.add_requirement("r")

        report = empty_kg.validate()
        summary = report.summary()
        assert "errors" in summary
        assert "warnings" in summary
        assert "info" in summary