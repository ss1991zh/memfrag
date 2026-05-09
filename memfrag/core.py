"""MemFrag — public facade.

Typical usage:
    mf = MemFrag(api_key="sk-...")
    mf.ingest(turns)        # after a conversation turn
    result = mf.recall(query)  # before generating a response
    print(result.context_prefix)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import anthropic

from memfrag.decay import DecayScheduler
from memfrag.extractor import FragmentExtractor
from memfrag.fingerprint import FingerprintEngine
from memfrag.models import (
    ConversationTurn,
    Fragment,
    RecallResult,
    RelationType,
    Relationship,
    SubMemory,
)
from memfrag.recall import RecallEngine
from memfrag.store import FragmentStore

logger = logging.getLogger(__name__)


class MemFrag:
    def __init__(
        self,
        api_key: str,
        db_path: str | Path = "memfrag.db",
        llm_model: str = "claude-haiku-4-5-20251001",
        embed_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 8,
        graph_hops: int = 2,
        token_budget: int = 800,
        similarity_threshold: float = 0.45,
        duplicate_threshold: float = 0.92,
        cold_start_turns: int = 5,
    ):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._store = FragmentStore(db_path)
        self._fp = FingerprintEngine(embed_model)
        self._extractor = FragmentExtractor(self._client, llm_model)
        self._recall_engine = RecallEngine(
            self._store, self._fp, self._client, llm_model,
            top_k=top_k, graph_hops=graph_hops,
            token_budget=token_budget, similarity_threshold=similarity_threshold,
        )
        self._decay = DecayScheduler(self._store)
        self._dup_threshold = duplicate_threshold
        self._cold_start_turns = cold_start_turns
        self._turn_count: int = 0

    # ── write path ─────────────────────────────────────────────────────────

    def ingest(self, turns: list[ConversationTurn]) -> list[Fragment]:
        """Extract fragments from conversation turns and store them."""
        self._turn_count += 1
        raw_text = "\n".join(f"{t.role.upper()}: {t.content}" for t in turns)

        # archive the raw exchange first
        sm = SubMemory(raw_text=raw_text, turn_index=self._turn_count)

        # extract fragments
        fragments = self._extractor.extract(turns)
        if not fragments:
            self._store.save_sub_memory(sm)
            return []

        # embed all at once for efficiency
        texts = [f.text for f in fragments]
        embeddings = self._fp.embed(texts)
        index = self._store.embeddings_index()

        saved: list[Fragment] = []
        for frag, emb in zip(fragments, embeddings):
            frag.embedding = emb
            frag.source_id = sm.id

            # deduplication check
            dup_id = self._fp.find_duplicate(emb, index, threshold=self._dup_threshold)
            if dup_id:
                existing = self._store.get_fragment(dup_id)
                if existing:
                    # check if this is an update (different text) or true duplicate
                    if existing.text.lower() != frag.text.lower():
                        # new fragment overrides the old one
                        self._store.save_fragment(frag)
                        rel = self._store.graph.infer_override(dup_id, frag.id)
                        self._store.save_relationship(rel)
                        logger.info("Override: %s → %s", frag.id, dup_id)
                    else:
                        # true duplicate — just bump the existing one
                        existing.bump()
                        self._store.save_fragment(existing)
                    saved.append(frag)
                    continue

            # new fragment — look for co-topic relationships with recent fragments
            self._store.save_fragment(frag)
            self._link_co_topics(frag, index)
            saved.append(frag)

            # update local index for next iteration
            index.append((frag.id, emb))

        sm.fragment_ids = [f.id for f in saved]
        self._store.save_sub_memory(sm)

        # run decay check every 24h
        self._decay.run_if_due(interval_hours=24.0)

        logger.info("Ingested %d fragments (turn %d)", len(saved), self._turn_count)
        return saved

    # ── read path ──────────────────────────────────────────────────────────

    def recall(self, query: str) -> RecallResult:
        """Recall relevant fragments for a query and return a context prefix."""
        if self._turn_count < self._cold_start_turns:
            logger.info(
                "Cold-start mode (turn %d/%d) — returning empty recall",
                self._turn_count, self._cold_start_turns,
            )
            return RecallResult(fragments=[], context_prefix="", source_ids=[], token_estimate=0)

        return self._recall_engine.recall(query)

    # ── maintenance ────────────────────────────────────────────────────────

    def run_decay(self):
        return self._decay.run(force=True)

    def stats(self) -> dict:
        s = self._store.stats()
        s["graph"] = self._store.graph.stats()
        s["turns_ingested"] = self._turn_count
        return s

    # ── internal ───────────────────────────────────────────────────────────

    def _link_co_topics(
        self, new_frag: Fragment, index: list[tuple[str, list[float]]]
    ) -> None:
        if not new_frag.embedding or not index:
            return
        co_topic_threshold = 0.75
        hits = self._fp.top_k(
            new_frag.embedding, index, k=3, threshold=co_topic_threshold
        )
        for related_id, _ in hits:
            if related_id == new_frag.id:
                continue
            rel = Relationship(
                source_id=new_frag.id,
                target_id=related_id,
                relation_type=RelationType.CO_TOPIC,
                weight=0.8,
            )
            self._store.save_relationship(rel)
