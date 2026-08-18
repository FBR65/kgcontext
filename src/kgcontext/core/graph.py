"""Internal knowledge graph — networkx MultiDiGraph with typed nodes and edges."""

from __future__ import annotations

from typing import Any

import networkx as nx

from ..types import (
    Conflict,
    Document,
    EdgeType,
    Evidence,
    Fact,
    Node,
    NodeType,
    Requirement,
)


class KnowledgeGraphInternal:
    """Thin wrapper around networkx.MultiDiGraph with typed nodes/edges."""

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_node(self, node: Node) -> str:
        self._graph.add_node(
            node.id,
            node_type=node.node_type.value,
            content=node.content,
            metadata=node.metadata,
            created_at=node.created_at,
            source_path=getattr(node, "source_path", ""),
            doc_type=getattr(node, "doc_type", ""),
        )
        return node.id

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, **attrs: Any) -> None:
        self._graph.add_edge(source_id, target_id, edge_type=edge_type.value, **attrs)

    def get_node(self, node_id: str) -> dict[str, Any]:
        if node_id not in self._graph:
            raise KeyError(f"Node not found: {node_id}")
        return dict(self._graph.nodes[node_id])

    def get_nodes_by_type(self, node_type: NodeType) -> list[dict[str, Any]]:
        return [
            {"id": nid, **data}
            for nid, data in self._graph.nodes(data=True)
            if data.get("node_type") == node_type.value
        ]

    def get_neighbors(
        self, node_id: str, edge_type: EdgeType, direction: str = "out"
    ) -> list[str]:
        if direction == "out":
            return [
                target
                for _, target, edata in self._graph.out_edges(node_id, data=True)
                if edata.get("edge_type") == edge_type.value
            ]
        else:
            return [
                source
                for source, _, edata in self._graph.in_edges(node_id, data=True)
                if edata.get("edge_type") == edge_type.value
            ]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def find_document_by_path(self, path: str) -> str | None:
        for nid, data in self._graph.nodes(data=True):
            if data.get("node_type") == "document" and data.get("source_path") == path:
                return str(nid)
        return None

    def all_edges(self) -> list[dict[str, Any]]:
        return [
            {"source": s, "target": t, **data}
            for s, t, data in self._graph.edges(data=True)
        ]

    def node_count(self) -> int:
        return int(self._graph.number_of_nodes())

    def edge_count(self) -> int:
        return int(self._graph.number_of_edges())

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"id": nid, **data}
                for nid, data in self._graph.nodes(data=True)
            ],
            "edges": [
                {"source": s, "target": t, **data}
                for s, t, data in self._graph.edges(data=True)
            ],
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        self._graph = nx.MultiDiGraph()
        for node in data.get("nodes", []):
            nid = node.pop("id")
            self._graph.add_node(nid, **node)
        for edge in data.get("edges", []):
            s = edge.pop("source")
            t = edge.pop("target")
            self._graph.add_edge(s, t, **edge)

    def _node_to_typed(self, nid: str, data: dict[str, Any]) -> Node:
        nt = NodeType(data.get("node_type", "fact"))
        common = {
            "id": nid,
            "content": data.get("content", ""),
            "metadata": data.get("metadata", {}),
            "created_at": data.get("created_at", ""),
        }
        if nt == NodeType.DOCUMENT:
            return Document(
                **common,
                source_path=data.get("source_path", ""),
                doc_type=data.get("doc_type", ""),
            )
        elif nt == NodeType.FACT:
            return Fact(**common)
        elif nt == NodeType.EVIDENCE:
            return Evidence(**common)
        elif nt == NodeType.REQUIREMENT:
            return Requirement(**common)
        elif nt == NodeType.CONFLICT:
            return Conflict(**common)
        return Node(**common, node_type=nt)