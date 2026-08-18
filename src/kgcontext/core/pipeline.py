"""High-level KnowledgeGraph — the single entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import KGConfig
from ..types import (
    Answer,
    Conflict,
    Document,
    EdgeType,
    Evidence,
    Fact,
    NodeType,
    Requirement,
    ValidationReport,
)
from .context import ContextBuilder
from .extractor import LLMExtractor
from .graph import KnowledgeGraphInternal
from .llm import LLMClient
from .parsers import parse_file
from .retrieval import LanceDBStore
from .validator import GraphValidator


class KnowledgeGraph:
    """Knowledge Graph middleware for LLM document analysis.

    Drop-in system: ingest documents, validate deterministically,
    query with curated context.

    Example:
        kg = KnowledgeGraph()
        kg.ingest(["doc.pdf"])
        answer = kg.query("What are the key requirements?")
        kg.save("session.json")
    """

    def __init__(
        self,
        config: KGConfig | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config or KGConfig()
        self._llm = llm
        self._graph = KnowledgeGraphInternal()
        self._validator = GraphValidator()
        self._context_builder = ContextBuilder(max_tokens=self.config.max_context_tokens)
        self._vector_store: LanceDBStore | None = None

        if self.config.vector_store == "lancedb":
            store_path = str(Path(self.config.storage_path) / "lancedb")
            self._vector_store = LanceDBStore(uri=store_path)

        self._extractor: LLMExtractor | None = None
        if self._llm is not None:
            self._extractor = LLMExtractor(llm=self._llm)

    # ── Ingest ──────────────────────────────────────────────

    def ingest(self, files: str | list[str], extract: bool = True) -> list[str]:
        """Parse files, optionally extract facts/evidence via LLM, index vector store.

        Args:
            files: Single path or list of paths.
            extract: If True, run LLM extraction. If False, only parse.

        Returns:
            List of document node IDs added.

        Raises:
            FileNotFoundError: If a file does not exist.
            UnsupportedFormatError: If format is not recognized.
            ExtractionError: If LLM extraction fails.
        """
        if isinstance(files, str):
            files = [files]

        doc_ids: list[str] = []
        documents: list[Document] = []

        for fpath in files:
            doc = parse_file(fpath)
            # Idempotency: skip if already ingested
            existing = self._graph.find_document_by_path(doc.source_path)
            if existing:
                doc_ids.append(existing)
                continue

            doc_id = self._graph.add_node(doc)
            doc_ids.append(doc_id)
            documents.append(doc)

            if extract and self._extractor is not None:
                self._extract_into_graph(doc, doc_id)

        # Index into vector store
        if self._vector_store is not None and documents:
            self._vector_store.index_documents(documents)

        return doc_ids

    def _extract_into_graph(self, document: Document, doc_id: str) -> None:
        """Extract facts, evidence, and requirements from a document into the graph."""
        assert self._extractor is not None
        data = self._extractor.extract(document)

        for fact_data in data.get("facts", []):
            fact = Fact(content=fact_data["content"])
            fact_id = self._graph.add_node(fact)
            self._graph.add_edge(fact_id, doc_id, EdgeType.EXTRACTED_FROM)

            for excerpt in fact_data.get("evidence", []):
                ev = Evidence(content=excerpt)
                ev_id = self._graph.add_node(ev)
                self._graph.add_edge(ev_id, fact_id, EdgeType.SUPPORTS)
                self._graph.add_edge(ev_id, doc_id, EdgeType.EXTRACTED_FROM)

        for req_data in data.get("requirements", []):
            req = Requirement(content=req_data["content"])
            self._graph.add_node(req)

    # ── Validate ────────────────────────────────────────────

    def validate(self) -> ValidationReport:
        """Run deterministic validation checks on the graph."""
        return self._validator.validate(self)

    # ── Query ───────────────────────────────────────────────

    def query(
        self,
        query: str,
        max_tokens: int | None = None,
    ) -> Answer:
        """Query the knowledge graph with curated context.

        Args:
            query: The question to ask.
            max_tokens: Override config max_context_tokens.

        Returns:
            Answer with content, provenance, and conflicts.
        """
        if max_tokens is not None:
            self._context_builder = ContextBuilder(max_tokens=max_tokens)

        ctx = self._context_builder.build(
            self,
            query=query,
            include_unsupported=self.config.include_unsupported,
            include_conflicts=self.config.include_conflicts,
        )

        if self._llm is None:
            return Answer(
                content="No LLM configured. Use Context output directly.",
                context=ctx.to_prompt(),
                provenance=ctx.provenance,
                conflicts=[c.content for c in ctx.conflicts],
            )

        prompt = ctx.to_prompt() + f"\n\n## Question\n{query}\n\n## Answer"
        raw = self._llm.complete(prompt)

        return Answer(
            content=raw,
            context=ctx.to_prompt(),
            provenance=ctx.provenance,
            conflicts=[c.content for c in ctx.conflicts],
        )

    # ── Graph Inspection ────────────────────────────────────

    @property
    def facts(self) -> list[Fact]:
        nodes = self._graph.get_nodes_by_type(NodeType.FACT)
        return [self._to_fact(n) for n in nodes]

    @property
    def requirements(self) -> list[Requirement]:
        nodes = self._graph.get_nodes_by_type(NodeType.REQUIREMENT)
        return [
            Requirement(id=n["id"], content=n.get("content", ""), metadata=n.get("metadata", {}))
            for n in nodes
        ]

    @property
    def documents(self) -> list[Document]:
        nodes = self._graph.get_nodes_by_type(NodeType.DOCUMENT)
        return [
            Document(
                id=n["id"],
                content=n.get("content", ""),
                source_path=n.get("source_path", ""),
                doc_type=n.get("doc_type", ""),
                metadata=n.get("metadata", {}),
            )
            for n in nodes
        ]

    @property
    def conflicts(self) -> list[Conflict]:
        nodes = self._graph.get_nodes_by_type(NodeType.CONFLICT)
        return [
            Conflict(id=n["id"], content=n.get("content", ""), metadata=n.get("metadata", {}))
            for n in nodes
        ]

    def get_evidence_for_fact(self, fact_id: str) -> list[Evidence]:
        neighbor_ids = self._graph.get_neighbors(fact_id, EdgeType.SUPPORTS, direction="in")
        result: list[Evidence] = []
        for nid in neighbor_ids:
            data = self._graph.get_node(nid)
            result.append(
                Evidence(id=nid, content=data.get("content", ""), metadata=data.get("metadata", {}))
            )
        return result

    def get_facts_for_requirement(self, req_id: str) -> list[Fact]:
        neighbor_ids = self._graph.get_neighbors(req_id, EdgeType.SATISFIED_BY, direction="out")
        return [self._to_fact(self._graph.get_node(nid)) for nid in neighbor_ids]

    def get_source_document(self, node_id: str) -> Document | None:
        neighbor_ids = self._graph.get_neighbors(node_id, EdgeType.EXTRACTED_FROM, direction="out")
        for nid in neighbor_ids:
            data = self._graph.get_node(nid)
            if data.get("node_type") == "document":
                return Document(
                    id=nid,
                    content=data.get("content", ""),
                    source_path=data.get("source_path", ""),
                    doc_type=data.get("doc_type", ""),
                    metadata=data.get("metadata", {}),
                )
        return None

    def get_conflicts(self, fact_id: str) -> list[Conflict]:
        neighbor_ids = self._graph.get_neighbors(fact_id, EdgeType.CONFLICTS_WITH, direction="in")
        return [
            Conflict(id=nid, content=self._graph.get_node(nid).get("content", ""))
            for nid in neighbor_ids
        ]

    def get_unsupported_facts(self) -> list[Fact]:
        return [f for f in self.facts if not self.get_evidence_for_fact(f.id)]

    def get_incomplete_requirements(self) -> list[Requirement]:
        return [r for r in self.requirements if not self.get_facts_for_requirement(r.id)]

    # ── Manual Graph Operations ─────────────────────────────

    def add_fact(self, content: str, document_id: str) -> str:
        """Manually add a fact linked to a document."""
        fact = Fact(content=content)
        fact_id = self._graph.add_node(fact)
        self._graph.add_edge(fact_id, document_id, EdgeType.EXTRACTED_FROM)
        return fact_id

    def add_evidence(self, excerpt: str, fact_id: str, document_id: str) -> str:
        """Manually add evidence linked to a fact and document."""
        ev = Evidence(content=excerpt)
        ev_id = self._graph.add_node(ev)
        self._graph.add_edge(ev_id, fact_id, EdgeType.SUPPORTS)
        self._graph.add_edge(ev_id, document_id, EdgeType.EXTRACTED_FROM)
        return ev_id

    def add_requirement(self, text: str) -> str:
        """Manually add a requirement."""
        req = Requirement(content=text)
        return self._graph.add_node(req)

    def link_requirement_fact(self, req_id: str, fact_id: str) -> None:
        """Link a requirement to a satisfying fact."""
        self._graph.add_edge(req_id, fact_id, EdgeType.SATISFIED_BY)

    def add_conflict(self, description: str, fact_id: str) -> str:
        """Manually add a conflict to a fact."""
        conflict = Conflict(content=description)
        cid = self._graph.add_node(conflict)
        self._graph.add_edge(cid, fact_id, EdgeType.CONFLICTS_WITH)
        return cid

    # ── Persistence ─────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save graph + config to file. Auto-detect format by extension."""
        from ..persistence import save_graph

        save_graph(self, path)

    @classmethod
    def load(cls, path: str) -> KnowledgeGraph:
        """Load graph + config from file."""
        from ..persistence import load_graph

        return load_graph(path)

    # ── Internal ────────────────────────────────────────────

    @property
    def _internal_graph(self) -> KnowledgeGraphInternal:
        return self._graph

    def _to_fact(self, data: dict[str, Any]) -> Fact:
        return Fact(
            id=data.get("id", data.get("node_id", "")),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
        )