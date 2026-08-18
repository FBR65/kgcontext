"""Persistence — save/load KnowledgeGraph to JSON or GraphML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import KGConfig
from .core.graph import KnowledgeGraphInternal
from .core.pipeline import KnowledgeGraph


class PersistenceError(Exception):
    """Raised when save/load fails."""


def save_graph(kg: KnowledgeGraph, path: str) -> None:
    """Save graph + config to file.

    Auto-detects format by extension:
    - .json → JSON (includes config)
    - .graphml → GraphML (graph only, no config)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix == ".graphml":
        import networkx as nx

        g = kg._internal_graph._graph
        # GraphML doesn't support dict values — serialize them to JSON strings
        for nid, ndata in g.nodes(data=True):
            if isinstance(ndata.get("metadata"), dict):
                ndata["metadata"] = json.dumps(ndata["metadata"])
        nx.write_graphml(g, str(p))
        # Restore dict metadata in the in-memory graph
        for nid, ndata in g.nodes(data=True):
            if isinstance(ndata.get("metadata"), str):
                try:
                    ndata["metadata"] = json.loads(ndata["metadata"])
                except (json.JSONDecodeError, ValueError):
                    pass
        return

    # JSON format
    data: dict[str, Any] = {
        "config": kg.config.to_dict(),
        "graph": kg._internal_graph.to_dict(),
    }
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_graph(path: str) -> KnowledgeGraph:
    """Load graph + config from file.

    Raises:
        PersistenceError: If file is corrupt or cannot be parsed.
        FileNotFoundError: If file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix == ".graphml":
        import networkx as nx

        kg = KnowledgeGraph(config=KGConfig(vector_store=None))
        g = nx.read_graphml(str(p))
        # Convert to MultiDiGraph
        mdg = nx.MultiDiGraph(g)
        # Deserialize metadata from JSON strings back to dicts
        for nid, ndata in mdg.nodes(data=True):
            if isinstance(ndata.get("metadata"), str):
                try:
                    ndata["metadata"] = json.loads(ndata["metadata"])
                except (json.JSONDecodeError, ValueError):
                    ndata["metadata"] = {}
            else:
                ndata.setdefault("metadata", {})
        kg._internal_graph._graph = mdg
        return kg

    # JSON format
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise PersistenceError(f"Corrupt JSON file: {e}") from e

    if not isinstance(data, dict) or "graph" not in data:
        raise PersistenceError("Invalid file structure: missing 'graph' key")

    config = KGConfig.from_dict(data.get("config", {}))
    kg = KnowledgeGraph(config=config)
    kg._internal_graph.from_dict(data["graph"])
    return kg