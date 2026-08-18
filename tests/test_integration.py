"""Test that ingest() indexes documents into vector store when configured."""

from __future__ import annotations

import json

from kgcontext import KGConfig, KnowledgeGraph, MockLLMClient


def test_ingest_indexes_vector_store(tmp_path) -> None:
    """Ingest with vector_store enabled should index documents."""
    extraction_response = json.dumps({
        "facts": [{"content": "test fact", "evidence": ["evidence"]}],
        "requirements": [],
    })
    llm = MockLLMClient(responses=[extraction_response] * 5)

    config = KGConfig(
        vector_store="lancedb",
        storage_path=str(tmp_path / ".kgcontext"),
    )
    kg = KnowledgeGraph(llm=llm, config=config)

    # Create a temp file
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("Some test content for vector indexing")

    kg.ingest([str(txt_path)])

    # Verify vector store was populated
    assert kg._vector_store is not None
    results = kg._vector_store.search("test content", k=5)
    assert len(results) > 0
    assert "test content" in results[0].content


def test_ingest_no_vector_store(tmp_path) -> None:
    """Ingest with vector_store=None should not create a store."""
    config = KGConfig(vector_store=None)
    kg = KnowledgeGraph(llm=None, config=config)

    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("Some content")
    kg.ingest([str(txt_path)], extract=False)

    assert kg._vector_store is None