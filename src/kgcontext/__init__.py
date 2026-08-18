"""kgcontext — Knowledge Graph middleware for LLM document analysis."""

from .config import KGConfig
from .core.llm import LLMClient, MockLLMClient, OpenAIClient
from .core.parsers import UnsupportedFormatError, parse_file
from .core.extractor import ExtractionError
from .core.pipeline import KnowledgeGraph
from .persistence import PersistenceError
from .types import (
    Answer,
    Conflict,
    Document,
    EdgeType,
    Evidence,
    Fact,
    Node,
    NodeType,
    Requirement,
    ValidationIssue,
    ValidationReport,
)

__all__ = [
    "KnowledgeGraph",
    "KGConfig",
    "Answer",
    "Document",
    "Fact",
    "Evidence",
    "Requirement",
    "Conflict",
    "Node",
    "NodeType",
    "EdgeType",
    "ValidationReport",
    "ValidationIssue",
    "LLMClient",
    "OpenAIClient",
    "MockLLMClient",
    "parse_file",
    "UnsupportedFormatError",
    "ExtractionError",
    "PersistenceError",
]