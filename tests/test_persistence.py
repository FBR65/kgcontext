"""Tests for persistence (save/load)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kgcontext import Document, KGConfig, KnowledgeGraph, PersistenceError


class TestPersistence:
    def test_save_load_json_roundtrip(self, empty_kg: KnowledgeGraph, tmp_path: Path) -> None:
        doc = Document(content="test content", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        fact_id = empty_kg.add_fact("test fact", doc_id)
        empty_kg.add_evidence("evidence text", fact_id, doc_id)
        empty_kg.add_requirement("req1")

        path = str(tmp_path / "session.json")
        empty_kg.save(path)

        loaded = KnowledgeGraph.load(path)
        assert len(loaded.facts) == 1
        assert loaded.facts[0].content == "test fact"
        assert len(loaded.requirements) == 1
        assert len(loaded.documents) == 1
        # Evidence preserved
        ev = loaded.get_evidence_for_fact(loaded.facts[0].id)
        assert len(ev) == 1
        assert ev[0].content == "evidence text"

    def test_save_creates_parent_dirs(self, empty_kg: KnowledgeGraph, tmp_path: Path) -> None:
        path = str(tmp_path / "subdir" / "nested" / "session.json")
        empty_kg.save(path)
        assert Path(path).exists()

    def test_save_json_structure(self, empty_kg: KnowledgeGraph, tmp_path: Path) -> None:
        path = str(tmp_path / "session.json")
        empty_kg.save(path)
        data = json.loads(Path(path).read_text())
        assert "config" in data
        assert "graph" in data
        assert "nodes" in data["graph"]
        assert "edges" in data["graph"]

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            KnowledgeGraph.load(str(tmp_path / "nonexistent.json"))

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json")
        with pytest.raises(PersistenceError, match="Corrupt"):
            KnowledgeGraph.load(str(path))

    def test_load_missing_graph_key(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"config": {}}))
        with pytest.raises(PersistenceError, match="missing 'graph'"):
            KnowledgeGraph.load(str(path))

    def test_config_persisted(self, tmp_path: Path) -> None:
        from kgcontext import KGConfig
        kg = KnowledgeGraph(
            llm=None,
            config=KGConfig(max_context_tokens=8000, vector_store=None),
        )
        path = str(tmp_path / "session.json")
        kg.save(path)

        loaded = KnowledgeGraph.load(path)
        assert loaded.config.max_context_tokens == 8000

    def test_save_graphml(self, empty_kg: KnowledgeGraph, tmp_path: Path) -> None:
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = empty_kg._internal_graph.add_node(doc)
        empty_kg.add_fact("fact", doc_id)

        path = str(tmp_path / "session.graphml")
        empty_kg.save(path)
        assert Path(path).exists()

        loaded = KnowledgeGraph.load(path)
        assert len(loaded.facts) == 1

    def test_ingest_after_load(self, kg_with_llm: KnowledgeGraph, tmp_txt, tmp_path: Path) -> None:
        """Incremental: load, ingest more, save again."""
        kg_with_llm.ingest([str(tmp_txt)])
        path = str(tmp_path / "session.json")
        kg_with_llm.save(path)

        loaded = KnowledgeGraph.load(path)
        # Should be able to ingest more
        ids = loaded.ingest([str(tmp_txt)])  # same file → idempotent
        assert len(ids) == 1


class TestPersistenceProperties:
    @given(
        n_facts=st.integers(min_value=0, max_value=10),
        n_reqs=st.integers(min_value=0, max_value=5),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_roundtrip_preserves_counts(self, n_facts: int, n_reqs: int, tmp_path: Path) -> None:
        kg = KnowledgeGraph(llm=None, config=KGConfig(vector_store=None))
        doc = Document(content="test", source_path="/t.txt", doc_type="txt")
        doc_id = kg._internal_graph.add_node(doc)

        for i in range(n_facts):
            f = kg.add_fact(f"fact {i}", doc_id)
            kg.add_evidence(f"ev {i}", f, doc_id)

        for i in range(n_reqs):
            kg.add_requirement(f"req {i}")

        path = str(tmp_path / "rt.json")
        kg.save(path)
        loaded = KnowledgeGraph.load(path)

        assert len(loaded.facts) == n_facts
        assert len(loaded.requirements) == n_reqs
        assert len(loaded.documents) == 1