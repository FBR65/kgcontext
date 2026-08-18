"""Tests for LLM extractor."""

from __future__ import annotations

import json

import pytest

from kgcontext import KnowledgeGraph, Document, MockLLMClient, ExtractionError
from kgcontext.core.extractor import LLMExtractor


class TestLLMExtractor:
    def test_extract_facts_from_mock(self) -> None:
        response = json.dumps({
            "facts": [
                {"content": "fact1", "evidence": ["ev1"]},
                {"content": "fact2", "evidence": []},
            ],
            "requirements": [
                {"content": "req1"},
            ],
        })
        llm = MockLLMClient(responses=[response])
        extractor = LLMExtractor(llm=llm)
        doc = Document(content="some text", source_path="/t.txt", doc_type="txt")

        facts = extractor.extract_facts(doc)
        assert len(facts) == 2
        assert facts[0].content == "fact1"
        assert facts[1].content == "fact2"

    def test_extract_requirements_from_mock(self) -> None:
        response = json.dumps({
            "facts": [],
            "requirements": [{"content": "must support SSL"}],
        })
        llm = MockLLMClient(responses=[response])
        extractor = LLMExtractor(llm=llm)
        doc = Document(content="text", source_path="/t.txt", doc_type="txt")

        reqs = extractor.extract_requirements(doc)
        assert len(reqs) == 1
        assert reqs[0].content == "must support SSL"

    def test_extract_evidence_from_mock(self) -> None:
        response = json.dumps({
            "facts": [
                {"content": "fact1", "evidence": ["excerpt A", "excerpt B"]},
            ],
            "requirements": [],
        })
        llm = MockLLMClient(responses=[response])
        extractor = LLMExtractor(llm=llm)
        doc = Document(content="text", source_path="/t.txt", doc_type="txt")

        evidence = extractor.extract_evidence(doc)
        assert len(evidence) == 2
        assert evidence[0].content == "excerpt A"

    def test_malformed_json_raises(self) -> None:
        llm = MockLLMClient(responses=["not valid json"])
        extractor = LLMExtractor(llm=llm)
        doc = Document(content="text", source_path="/t.txt", doc_type="txt")

        with pytest.raises(ExtractionError, match="invalid JSON"):
            extractor.extract(doc)

    def test_non_object_json_raises(self) -> None:
        llm = MockLLMClient(responses=["[]"])  # valid JSON but not object
        extractor = LLMExtractor(llm=llm)
        doc = Document(content="text", source_path="/t.txt", doc_type="txt")

        with pytest.raises(ExtractionError, match="not a JSON object"):
            extractor.extract(doc)

    def test_ingest_with_extraction(self, kg_with_llm: KnowledgeGraph, tmp_txt) -> None:
        """End-to-end: ingest triggers LLM extraction into graph."""
        ids = kg_with_llm.ingest([str(tmp_txt)], extract=True)
        assert len(ids) == 1
        # Facts should have been extracted
        assert len(kg_with_llm.facts) > 0
        # Evidence should exist
        for f in kg_with_llm.facts:
            ev = kg_with_llm.get_evidence_for_fact(f.id)
            assert len(ev) > 0  # each fact has evidence from mock

    def test_llm_protocol_compliance(self) -> None:
        """Any object with complete() method satisfies LLMClient protocol."""
        class CustomClient:
            def complete(self, prompt: str) -> str:
                return '{"facts": [], "requirements": []}'

        from kgcontext.core.llm import LLMClient
        client = CustomClient()
        assert isinstance(client, LLMClient)