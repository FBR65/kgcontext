"""Tests for LanceDB vector store."""

from __future__ import annotations

from kgcontext import Document, KGConfig, KnowledgeGraph
from kgcontext.core.retrieval import LanceDBStore


class TestLanceDBStore:
    def test_search_empty_store(self, tmp_path) -> None:
        store = LanceDBStore(uri=str(tmp_path / "ldb"))
        results = store.search("anything", k=5)
        assert results == []

    def test_index_and_search(self, tmp_path) -> None:
        store = LanceDBStore(uri=str(tmp_path / "ldb"))
        docs = [
            Document(
                content="The system shall support PostgreSQL database",
                source_path="/a.txt", doc_type="txt",
            ),
            Document(
                content="User authentication via OAuth2 tokens",
                source_path="/b.txt", doc_type="txt",
            ),
        ]
        store.index_documents(docs)

        results = store.search("database PostgreSQL", k=2)
        assert len(results) > 0
        assert results[0].document_id in [d.id for d in docs]

    def test_search_graph_aware(self, tmp_path) -> None:
        store = LanceDBStore(uri=str(tmp_path / "ldb"))
        docs = [
            Document(
                content="SSL certificate required for HTTPS",
                source_path="/a.txt", doc_type="txt",
            ),
        ]
        store.index_documents(docs)

        from kgcontext import KnowledgeGraph
        kg = KnowledgeGraph(llm=None, config=KGConfig(vector_store=None))
        kg.add_requirement("SSL support needed")

        results = store.search_graph_aware("certificates", kg, k=5)
        assert len(results) > 0

    def test_optional_no_vector_store(self) -> None:
        """Graph works without vector store."""
        kg = KnowledgeGraph(llm=None, config=KGConfig(vector_store=None))
        assert kg._vector_store is None