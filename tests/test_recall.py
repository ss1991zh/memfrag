"""Mock tests for RecallEngine."""

import json
from unittest.mock import MagicMock

import pytest

from memfrag.fingerprint import FingerprintEngine
from memfrag.models import Fragment, FragmentType, Relationship, RelationType
from memfrag.recall import RecallEngine
from memfrag.store import FragmentStore


# ── fixtures ──────────────────────────────────────────────────────────────────

def _mock_fp(embed_vec: list[float] | None = None) -> FingerprintEngine:
    """FingerprintEngine that returns a fixed vector without loading any model."""
    vec = embed_vec or [1.0, 0.0, 0.0]
    fp = FingerprintEngine.__new__(FingerprintEngine)
    fp.embed_one = MagicMock(return_value=vec)
    fp.embed = MagicMock(return_value=[vec])
    return fp


def _mock_llm(recompose_text: str = "[MEMORY:abc] user prefers Python") -> MagicMock:
    content_block = MagicMock()
    content_block.text = recompose_text
    msg = MagicMock()
    msg.content = [content_block]
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


def _store_with_fragments(*texts: str) -> FragmentStore:
    store = FragmentStore(":memory:")
    for text in texts:
        frag = Fragment(
            text=text,
            fragment_type=FragmentType.PREFERENCE,
            embedding=[1.0, 0.0, 0.0],
            strength=2.0,
        )
        store.save_fragment(frag)
    return store


# ── tests ─────────────────────────────────────────────────────────────────────

def test_recall_returns_context_prefix():
    store = _store_with_fragments("user prefers Python", "deadline June 30")
    engine = RecallEngine(store, _mock_fp(), _mock_llm(), top_k=5)

    result = engine.recall("what language should I use?")

    assert result.context_prefix != ""
    assert len(result.fragments) > 0


def test_recall_bumps_fragment_strength():
    store = _store_with_fragments("user prefers Python")
    frag_id = store.all_fragments()[0].id
    original_strength = store.get_fragment(frag_id).strength

    engine = RecallEngine(store, _mock_fp(), _mock_llm(), top_k=5)
    engine.recall("language preference?")

    updated = store.get_fragment(frag_id)
    assert updated.strength > original_strength


def test_recall_skips_overridden_fragments():
    store = FragmentStore(":memory:")

    old = Fragment(text="user likes JS", embedding=[1.0, 0.0, 0.0], strength=2.0)
    new = Fragment(text="user prefers Python", embedding=[1.0, 0.0, 0.0], strength=2.0)
    store.save_fragment(old)
    store.save_fragment(new)

    override_rel = Relationship(
        source_id=new.id, target_id=old.id, relation_type=RelationType.OVERRIDE
    )
    store.save_relationship(override_rel)

    engine = RecallEngine(store, _mock_fp(), _mock_llm(), top_k=5)
    result = engine.recall("language?")

    recalled_ids = [f.id for f in result.fragments]
    assert old.id not in recalled_ids
    assert new.id in recalled_ids


def test_recall_empty_store_returns_empty():
    store = FragmentStore(":memory:")
    engine = RecallEngine(store, _mock_fp(), _mock_llm(), top_k=5)

    result = engine.recall("anything")

    assert result.fragments == []
    assert result.context_prefix == ""


def test_recall_graph_expansion_includes_co_topic():
    store = FragmentStore(":memory:")

    seed = Fragment(text="user prefers Python", embedding=[1.0, 0.0, 0.0], strength=2.0)
    related = Fragment(text="Python project Alpha", embedding=[0.0, 1.0, 0.0], strength=2.0)
    store.save_fragment(seed)
    store.save_fragment(related)

    rel = Relationship(
        source_id=seed.id, target_id=related.id, relation_type=RelationType.CO_TOPIC
    )
    store.save_relationship(rel)

    engine = RecallEngine(store, _mock_fp([1.0, 0.0, 0.0]), _mock_llm(), top_k=5, graph_hops=1)
    result = engine.recall("Python?")

    recalled_ids = [f.id for f in result.fragments]
    assert related.id in recalled_ids


def test_recall_recompose_llm_error_falls_back_to_plain_list():
    import anthropic

    store = _store_with_fragments("user prefers Python")
    llm = MagicMock()
    llm.messages.create.side_effect = anthropic.APIConnectionError(request=MagicMock())

    engine = RecallEngine(store, _mock_fp(), llm, top_k=5)
    result = engine.recall("language?")

    # fallback: plain [MEMORY:id] text lines
    assert "[MEMORY:" in result.context_prefix


def test_recall_token_budget_truncates():
    store = _store_with_fragments(
        *[f"fragment number {i} with some extra words" for i in range(20)]
    )
    engine = RecallEngine(store, _mock_fp(), _mock_llm(), top_k=20, token_budget=10)
    result = engine.recall("anything")

    # with a tiny budget only a few fragments should make it through
    total_chars = sum(len(f.text) for f in result.fragments)
    assert total_chars <= 10 * 4 + 50  # small budget + small slack
