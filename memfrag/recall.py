"""Recall engine — vector search + graph expansion + recomposition.

Recall path:
  1. Embed the current query
  2. Vector top-K against all stored fragments
  3. Graph expand 1-2 hops for related fragments
  4. Rank by strength × similarity, truncate at token budget
  5. If fragments below minimum, fallback to sub-memory archive
  6. Recompose into a natural-language context prefix
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import anthropic

from memfrag.fingerprint import FingerprintEngine
from memfrag.models import Fragment, RecallResult, RelationType

if TYPE_CHECKING:
    from memfrag.store import FragmentStore

logger = logging.getLogger(__name__)

_RECOMPOSE_SYSTEM = """\
You receive a list of memory fragments tagged with their source IDs.
Rewrite them as a concise, natural-language memory block (≤120 words) that will
be prepended to an AI assistant's context.

Rules:
- Prefix each sentence with [MEMORY:<source_id>]
- Keep wording close to the original fragments — do not invent details
- If two fragments contradict, prefer the one tagged (override)
- End with nothing extra
"""

_TOKEN_CHARS = 4  # rough chars-per-token estimate


class RecallEngine:
    def __init__(
        self,
        store: FragmentStore,
        fp_engine: FingerprintEngine,
        llm_client: anthropic.Anthropic,
        model: str = "claude-haiku-4-5-20251001",
        top_k: int = 8,
        graph_hops: int = 2,
        token_budget: int = 800,
        similarity_threshold: float = 0.45,
    ):
        self._store = store
        self._fp = fp_engine
        self._llm = llm_client
        self._model = model
        self._top_k = top_k
        self._graph_hops = graph_hops
        self._token_budget = token_budget
        self._sim_threshold = similarity_threshold

    def recall(self, query: str) -> RecallResult:
        query_vec = self._fp.embed_one(query)
        index = self._store.embeddings_index()

        # 1. vector search
        vector_hits = self._fp.top_k(
            query_vec, index, k=self._top_k, threshold=self._sim_threshold
        )
        seed_ids = [fid for fid, _ in vector_hits]
        sim_map = {fid: score for fid, score in vector_hits}

        # 2. graph expansion
        expanded_ids = self._store.graph.expand(
            seed_ids,
            hops=self._graph_hops,
            relation_filter={RelationType.CO_TOPIC, RelationType.CAUSAL, RelationType.OVERRIDE},
        )

        # 3. fetch & rank by strength × similarity
        fragments: list[Fragment] = []
        for fid in expanded_ids:
            frag = self._store.get_fragment(fid)
            if frag and not frag.is_cold:
                # skip overridden fragments
                if self._store.graph.is_overridden(frag.id):
                    continue
                fragments.append(frag)

        def score(f: Fragment) -> float:
            return f.strength * sim_map.get(f.id, 0.3)

        fragments.sort(key=score, reverse=True)

        # 4. token budget trim
        chosen: list[Fragment] = []
        used_chars = 0
        for frag in fragments:
            chars = len(frag.text)
            if used_chars + chars > self._token_budget * _TOKEN_CHARS:
                break
            chosen.append(frag)
            used_chars += chars

        # 5. fallback to sub-memory if too few fragments
        source_ids: list[str] = []
        if len(chosen) < 2:
            for frag in fragments[:3]:
                if frag.source_id:
                    sm = self._store.get_sub_memory(frag.source_id)
                    if sm:
                        source_ids.append(sm.id)
                        logger.info("Fallback: loaded sub-memory %s", sm.id)

        # 6. bump strength of recalled fragments
        for frag in chosen:
            frag.bump()
            self._store.save_fragment(frag)

        # 7. recompose
        context_prefix = self._recompose(chosen)

        return RecallResult(
            fragments=chosen,
            context_prefix=context_prefix,
            source_ids=source_ids,
            token_estimate=len(context_prefix) // _TOKEN_CHARS,
        )

    def _recompose(self, fragments: list[Fragment]) -> str:
        if not fragments:
            return ""

        frag_lines = "\n".join(
            f"[{frag.id}] ({frag.fragment_type.value}) {frag.text}"
            for frag in fragments
        )

        try:
            resp = self._llm.messages.create(
                model=self._model,
                max_tokens=300,
                system=_RECOMPOSE_SYSTEM,
                messages=[{"role": "user", "content": frag_lines}],
            )
            return resp.content[0].text.strip()
        except anthropic.APIError as exc:
            logger.warning("Recompose LLM call failed: %s", exc)
            # fallback: plain list
            return "\n".join(
                f"[MEMORY:{f.id}] {f.text}" for f in fragments
            )
