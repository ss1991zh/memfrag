"""Storage layer — SQLite-backed persistence for fragments, relationships,
sub-memory archives, and the relationship graph.

Schema:
  fragments   — one row per fragment (JSON blob for embedding)
  edges       — relationship graph edge list
  sub_memories — raw text archives with back-links
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

from memfrag.graph import RelationshipGraph
from memfrag.models import Fragment, FragmentType, Relationship, RelationType, SubMemory

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS fragments (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    fragment_type   TEXT NOT NULL,
    source_id       TEXT DEFAULT '',
    strength        REAL NOT NULL DEFAULT 1.0,
    created_at      REAL NOT NULL,
    last_recalled_at REAL NOT NULL,
    recall_count    INTEGER NOT NULL DEFAULT 0,
    embedding       TEXT,           -- JSON array
    metadata        TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edges (
    source_id       TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    relation_type   TEXT NOT NULL,
    weight          REAL NOT NULL DEFAULT 1.0,
    created_at      REAL NOT NULL,
    PRIMARY KEY (source_id, target_id)
);

CREATE TABLE IF NOT EXISTS sub_memories (
    id              TEXT PRIMARY KEY,
    raw_text        TEXT NOT NULL,
    turn_index      INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    fragment_ids    TEXT DEFAULT '[]'   -- JSON array
);
"""


class FragmentStore:
    def __init__(self, db_path: str | Path = ":memory:"):
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()
        self.graph = RelationshipGraph()
        self._load_graph()

    # ── fragment CRUD ─────────────────────────────────────────────────────────

    def save_fragment(self, frag: Fragment) -> None:
        self._conn.execute(
            """
            INSERT INTO fragments
              (id, text, fragment_type, source_id, strength, created_at,
               last_recalled_at, recall_count, embedding, metadata)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              text=excluded.text,
              strength=excluded.strength,
              last_recalled_at=excluded.last_recalled_at,
              recall_count=excluded.recall_count,
              embedding=excluded.embedding,
              metadata=excluded.metadata
            """,
            (
                frag.id, frag.text, frag.fragment_type.value, frag.source_id,
                frag.strength, frag.created_at, frag.last_recalled_at,
                frag.recall_count,
                json.dumps(frag.embedding) if frag.embedding else None,
                json.dumps(frag.metadata),
            ),
        )
        self._conn.commit()
        self.graph.add_fragment(frag.id)

    def get_fragment(self, fragment_id: str) -> Optional[Fragment]:
        row = self._conn.execute(
            "SELECT * FROM fragments WHERE id=?", (fragment_id,)
        ).fetchone()
        return self._row_to_fragment(row) if row else None

    def delete_fragment(self, fragment_id: str) -> None:
        self._conn.execute("DELETE FROM fragments WHERE id=?", (fragment_id,))
        self._conn.execute(
            "DELETE FROM edges WHERE source_id=? OR target_id=?",
            (fragment_id, fragment_id),
        )
        self._conn.commit()
        if fragment_id in self.graph._g:
            self.graph.remove_fragment(fragment_id)

    def all_fragments(self, include_cold: bool = False) -> list[Fragment]:
        rows = self._conn.execute("SELECT * FROM fragments ORDER BY strength DESC").fetchall()
        frags = [self._row_to_fragment(r) for r in rows]
        if not include_cold:
            frags = [f for f in frags if not f.is_cold]
        return frags

    def embeddings_index(self) -> list[tuple[str, list[float]]]:
        """Return (id, embedding) pairs for all fragments that have embeddings."""
        rows = self._conn.execute(
            "SELECT id, embedding FROM fragments WHERE embedding IS NOT NULL"
        ).fetchall()
        return [(r["id"], json.loads(r["embedding"])) for r in rows]

    # ── relationship CRUD ─────────────────────────────────────────────────────

    def save_relationship(self, rel: Relationship) -> None:
        self._conn.execute(
            """
            INSERT INTO edges (source_id, target_id, relation_type, weight, created_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(source_id, target_id) DO UPDATE SET
              relation_type=excluded.relation_type,
              weight=excluded.weight
            """,
            (rel.source_id, rel.target_id, rel.relation_type.value, rel.weight, rel.created_at),
        )
        self._conn.commit()
        self.graph.add_relationship(rel)

    # ── sub-memory archive ────────────────────────────────────────────────────

    def save_sub_memory(self, sm: SubMemory) -> None:
        self._conn.execute(
            """
            INSERT INTO sub_memories (id, raw_text, turn_index, created_at, fragment_ids)
            VALUES (?,?,?,?,?)
            ON CONFLICT(id) DO NOTHING
            """,
            (sm.id, sm.raw_text, sm.turn_index, sm.created_at, json.dumps(sm.fragment_ids)),
        )
        self._conn.commit()

    def get_sub_memory(self, sm_id: str) -> Optional[SubMemory]:
        row = self._conn.execute(
            "SELECT * FROM sub_memories WHERE id=?", (sm_id,)
        ).fetchone()
        if not row:
            return None
        return SubMemory(
            id=row["id"],
            raw_text=row["raw_text"],
            turn_index=row["turn_index"],
            created_at=row["created_at"],
            fragment_ids=json.loads(row["fragment_ids"]),
        )

    # ── decay helpers ─────────────────────────────────────────────────────────

    def apply_decay(self, now: Optional[float] = None) -> tuple[int, int]:
        """Apply forgetting-curve decay to all fragments.

        Returns (n_cold_moved, n_stale_deleted).
        """
        now = now or time.time()
        rows = self._conn.execute("SELECT id, strength, last_recalled_at FROM fragments").fetchall()
        cold_count = stale_count = 0

        for row in rows:
            days = (now - row["last_recalled_at"]) / 86400
            new_strength = row["strength"] * (0.85 ** days)
            if new_strength < 0.1:
                self.delete_fragment(row["id"])
                stale_count += 1
            else:
                self._conn.execute(
                    "UPDATE fragments SET strength=? WHERE id=?",
                    (new_strength, row["id"]),
                )
                if new_strength < 0.3:
                    cold_count += 1

        self._conn.commit()
        logger.info("Decay applied: %d cold, %d deleted", cold_count, stale_count)
        return cold_count, stale_count

    # ── internal helpers ──────────────────────────────────────────────────────

    def _row_to_fragment(self, row: sqlite3.Row) -> Fragment:
        return Fragment(
            id=row["id"],
            text=row["text"],
            fragment_type=FragmentType(row["fragment_type"]),
            source_id=row["source_id"] or "",
            strength=row["strength"],
            created_at=row["created_at"],
            last_recalled_at=row["last_recalled_at"],
            recall_count=row["recall_count"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _load_graph(self) -> None:
        rows = self._conn.execute("SELECT * FROM edges").fetchall()
        for row in rows:
            self.graph.add_relationship(
                Relationship(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relation_type=RelationType(row["relation_type"]),
                    weight=row["weight"],
                    created_at=row["created_at"],
                )
            )
        frags = self._conn.execute("SELECT id FROM fragments").fetchall()
        for f in frags:
            self.graph.add_fragment(f["id"])

    def stats(self) -> dict:
        n_frags = self._conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
        n_edges = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        n_sm = self._conn.execute("SELECT COUNT(*) FROM sub_memories").fetchone()[0]
        return {"fragments": n_frags, "edges": n_edges, "sub_memories": n_sm}
