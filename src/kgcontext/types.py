"""Core type definitions for kgcontext."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    DOCUMENT = "document"
    FACT = "fact"
    EVIDENCE = "evidence"
    REQUIREMENT = "requirement"
    CONFLICT = "conflict"


class EdgeType(str, Enum):
    EXTRACTED_FROM = "extracted_from"
    SUPPORTS = "supports"
    SATISFIED_BY = "satisfied_by"
    CONFLICTS_WITH = "conflicts_with"
    DERIVED_FROM = "derived_from"


class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    node_type: NodeType
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class Document(Node):
    node_type: NodeType = NodeType.DOCUMENT
    source_path: str = ""
    doc_type: str = ""  # pdf, docx, txt, md


class Fact(Node):
    node_type: NodeType = NodeType.FACT


class Evidence(Node):
    node_type: NodeType = NodeType.EVIDENCE


class Requirement(Node):
    node_type: NodeType = NodeType.REQUIREMENT


class Conflict(Node):
    node_type: NodeType = NodeType.CONFLICT


class ValidationIssue(BaseModel):
    level: str  # "error", "warning", "info"
    node_id: str
    message: str


class ValidationReport(BaseModel):
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self.issues)

    def summary(self) -> str:
        errors = sum(1 for i in self.issues if i.level == "error")
        warnings = sum(1 for i in self.issues if i.level == "warning")
        infos = sum(1 for i in self.issues if i.level == "info")
        return f"Validation: {errors} errors, {warnings} warnings, {infos} info"


class Provenance(BaseModel):
    fact_id: str
    document_path: str
    evidence_excerpt: str


class Answer(BaseModel):
    content: str
    context: str = ""
    provenance: dict[str, dict[str, str]] = Field(default_factory=dict)
    conflicts: list[str] = Field(default_factory=list)