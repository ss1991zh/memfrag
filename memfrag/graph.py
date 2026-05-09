"""Relationship graph layer.

Maintains directed edges between fragments using NetworkX.
Supports the four relationship types: co-topic, temporal, causal, override.
Persists as edge list in SQLite (managed by the store layer).
"""

from __future__ import annotations

import logging
from typing import Iterator

import networkx as nx

from memfrag.models import RelationType, Relationship

logger = logging.getLogger(__name__)


class RelationshipGraph:
    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    # ── mutation ──────────────────────────────────────────────────────────────

    def add_fragment(self, fragment_id: str) -> None:
        self._g.add_node(fragment_id)

    def remove_fragment(self, fragment_id: str) -> None:
        self._g.remove_node(fragment_id)

    def add_relationship(self, rel: Relationship) -> None:
        self._g.add_edge(
            rel.source_id,
            rel.target_id,
            relation_type=rel.relation_type.value,
            weight=rel.weight,
            created_at=rel.created_at,
        )
        logger.debug(
            "Edge: %s -[%s]-> %s", rel.source_id, rel.relation_type.value, rel.target_id
        )

    def remove_relationship(self, source_id: str, target_id: str) -> None:
        if self._g.has_edge(source_id, target_id):
            self._g.remove_edge(source_id, target_id)

    # ── query ─────────────────────────────────────────────────────────────────

    def expand(
        self,
        seed_ids: list[str],
        hops: int = 2,
        relation_filter: set[RelationType] | None = None,
    ) -> set[str]:
        """BFS expansion from seed nodes up to `hops` edges away.

        Returns all reachable fragment IDs (seeds included).
        """
        visited: set[str] = set(seed_ids)
        frontier: set[str] = set(seed_ids)

        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for _, neighbor, data in self._g.edges(node, data=True):
                    if relation_filter:
                        if data.get("relation_type") not in {r.value for r in relation_filter}:
                            continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return visited

    def overrides_of(self, fragment_id: str) -> list[str]:
        """Return fragment IDs that override (supersede) the given fragment."""
        return [
            src
            for src, _, data in self._g.in_edges(fragment_id, data=True)
            if data.get("relation_type") == RelationType.OVERRIDE.value
        ]

    def is_overridden(self, fragment_id: str) -> bool:
        return len(self.overrides_of(fragment_id)) > 0

    def relationships(self) -> Iterator[Relationship]:
        for src, tgt, data in self._g.edges(data=True):
            yield Relationship(
                source_id=src,
                target_id=tgt,
                relation_type=RelationType(data["relation_type"]),
                weight=data.get("weight", 1.0),
                created_at=data.get("created_at", 0.0),
            )

    def stats(self) -> dict:
        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
        }

    # ── auto-relationship inference ───────────────────────────────────────────

    def infer_override(self, old_id: str, new_id: str) -> Relationship:
        """Create an override relationship: new fragment supersedes old."""
        rel = Relationship(
            source_id=new_id,
            target_id=old_id,
            relation_type=RelationType.OVERRIDE,
            weight=1.0,
        )
        self.add_relationship(rel)
        return rel

    def infer_co_topic(self, id_a: str, id_b: str) -> Relationship:
        rel = Relationship(
            source_id=id_a,
            target_id=id_b,
            relation_type=RelationType.CO_TOPIC,
            weight=0.8,
        )
        self.add_relationship(rel)
        return rel
