"""Test fixtures and helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgcontext import KnowledgeGraph, MockLLMClient


@pytest.fixture
def tmp_txt(tmp_path: Path) -> Path:
    """Create a temporary TXT file."""
    p = tmp_path / "test.txt"
    p.write_text("The system shall support 500 concurrent users. "
                 "Response time must be under 200ms. "
                 "The database must be PostgreSQL 15 or higher.")
    return p


@pytest.fixture
def tmp_md(tmp_path: Path) -> Path:
    """Create a temporary MD file."""
    p = tmp_path / "test.md"
    p.write_text("# Requirements\n\n- The API shall return JSON.\n- Auth via OAuth2.\n")
    return p


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Mock LLM that returns structured extraction results."""
    extraction_response = json.dumps({
        "facts": [
            {"content": "System supports 500 concurrent users", "evidence": ["The system shall support 500 concurrent users"]},
            {"content": "Response time under 200ms", "evidence": ["Response time must be under 200ms"]},
        ],
        "requirements": [
            {"content": "Support 500 concurrent users"},
            {"content": "Response time under 200ms"},
            {"content": "PostgreSQL 15+"},
        ],
    })
    query_response = "Based on the facts, the system requires 500 concurrent users."
    # Provide enough extraction responses for multiple files + query responses
    return MockLLMClient(responses=[extraction_response] * 10 + [query_response] * 10)


@pytest.fixture
def empty_kg() -> KnowledgeGraph:
    """Empty KnowledgeGraph with no LLM."""
    return KnowledgeGraph(llm=None, config=__import__("kgcontext").KGConfig(vector_store=None))


@pytest.fixture
def kg_with_llm(mock_llm: MockLLMClient) -> KnowledgeGraph:
    """KnowledgeGraph with mock LLM."""
    from kgcontext import KGConfig
    return KnowledgeGraph(llm=mock_llm, config=KGConfig(vector_store=None))