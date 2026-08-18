"""Deterministische Graph-Validatoren."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import ValidationIssue, ValidationReport

if TYPE_CHECKING:
    from .pipeline import KnowledgeGraph


class GraphValidator:
    """Runs deterministic checks on the knowledge graph."""

    def validate(self, kg: KnowledgeGraph) -> ValidationReport:
        issues: list[ValidationIssue] = []

        # Fact without Evidence → WARNING
        for fact in kg.facts:
            evidence = kg.get_evidence_for_fact(fact.id)
            if not evidence:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        node_id=fact.id,
                        message=f"Fact has no supporting evidence: {fact.content[:80]}",
                    )
                )

            # Fact with Conflict → WARNING
            conflicts = kg.get_conflicts(fact.id)
            if conflicts:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        node_id=fact.id,
                        message=f"Fact has {len(conflicts)} conflict(s)",
                    )
                )

        # Evidence without Document → ERROR
        internal = kg._internal_graph
        for node_id, node_data in internal._graph.nodes(data=True):
            if node_data.get("node_type") == "evidence":
                has_doc = any(
                    edge_data.get("edge_type") == "extracted_from"
                    for _, _, edge_data in internal._graph.out_edges(node_id, data=True)
                )
                if not has_doc:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            node_id=node_id,
                            message="Evidence not linked to any document",
                        )
                    )

        # Requirement without satisfied_by → INFO
        for req in kg.requirements:
            facts = kg.get_facts_for_requirement(req.id)
            if not facts:
                issues.append(
                    ValidationIssue(
                        level="info",
                        node_id=req.id,
                        message=f"Requirement not satisfied: {req.content[:80]}",
                    )
                )

        return ValidationReport(issues=issues)