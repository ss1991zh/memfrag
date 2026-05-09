"""Mock tests for MemFrag core facade — full write→recall pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest

from memfrag.core import MemFrag
from memfrag.models import ConversationTurn, FragmentType, RelationType


# ── shared mock helpers ───────────────────────────────────────────────────────

_DUMMY_VEC = [0.1] * 384  # BGE-small output dim


def _make_extractor_response(*texts: str) -> MagicMock:
    payload = json.dumps([
        {"text": t, "fragment_type": "preference"} for t in texts
    ])
    block = MagicMock()
    block.text = payload
    msg = MagicMock()
    msg.content = [block]
    return msg


def _make_recompose_response(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


def _build_memfrag(extract_texts: list[str], recompose_text: str) -> MemFrag:
    """Build a MemFrag instance with all external calls mocked."""
    with patch("memfrag.core.anthropic.Anthropic") as MockAnthropic, \
         patch("memfrag.core.FingerprintEngine") as MockFP:

        # LLM: first call = extraction, subsequent calls = recomposition
        client = MagicMock()
        client.messages.create.side_effect = [
            _make_extractor_response(*extract_texts),
            _make_recompose_response(recompose_text),
        ]
        MockAnthropic.return_value = client

        # Embedding: always return the same dummy vector
        fp_instance = MagicMock()
        fp_instance.embed.return_value = [_DUMMY_VEC] * max(len(extract_texts), 1)
        fp_instance.embed_one.return_value = _DUMMY_VEC
        fp_instance.find_duplicate.return_value = None
        fp_instance.top_k.return_value = []
        MockFP.return_value = fp_instance

        mf = MemFrag(api_key="sk-test", db_path=":memory:", cold_start_turns=0)

    # store the mocks for assertions
    mf._test_client = client
    mf._test_fp = fp_instance
    return mf


# ── tests ─────────────────────────────────────────────────────────────────────

TURNS = [
    ConversationTurn(role="user", content="I prefer Python for all backend work."),
    ConversationTurn(role="assistant", content="Understood, I'll use Python."),
]


def test_ingest_saves_fragments():
    mf = _build_memfrag(["user prefers Python", "use Python for backend"], "")

    saved = mf.ingest(TURNS)

    assert len(saved) == 2
    assert mf._store.stats()["fragments"] == 2


def test_ingest_creates_sub_memory():
    mf = _build_memfrag(["user prefers Python"], "")
    mf.ingest(TURNS)

    assert mf._store.stats()["sub_memories"] == 1


def test_ingest_empty_extraction_still_saves_sub_memory():
    mf = _build_memfrag([], "")
    # Override side_effect for empty extraction
    mf._test_client.messages.create.side_effect = [
        _make_extractor_response(),  # returns []
    ]

    saved = mf.ingest(TURNS)

    assert saved == []
    assert mf._store.stats()["sub_memories"] == 1


def test_recall_returns_context_prefix():
    recompose = "[MEMORY:abc] User prefers Python for backend work."
    mf = _build_memfrag(["user prefers Python"], recompose)

    # First ingest so fragments exist
    mf.ingest(TURNS)

    # Patch top_k to return the stored fragment's id
    stored_id = mf._store.all_fragments()[0].id
    mf._fp.top_k.return_value = [(stored_id, 0.95)]
    mf._recall_engine._fp = mf._fp

    # Reset LLM side_effect for the recompose call
    mf._test_client.messages.create.side_effect = [
        _make_recompose_response(recompose)
    ]

    result = mf.recall("what language should I use?")

    assert result.context_prefix == recompose


def test_cold_start_suppresses_recall():
    mf = _build_memfrag(["user prefers Python"], "")
    mf._cold_start_turns = 10  # force cold-start

    result = mf.recall("anything")

    assert result.fragments == []
    assert result.context_prefix == ""


def test_ingest_deduplication_bumps_existing():
    mf = _build_memfrag(["user prefers Python"], "")

    # Make find_duplicate return the id of the first ingested fragment on second ingest
    mf.ingest(TURNS)
    first_id = mf._store.all_fragments()[0].id
    original_strength = mf._store.get_fragment(first_id).strength

    # Second ingest — same text → duplicate found
    mf._fp.find_duplicate.return_value = first_id
    mf._test_client.messages.create.side_effect = [
        _make_extractor_response("user prefers Python"),  # same text
        _make_recompose_response(""),
    ]
    mf.ingest(TURNS)

    updated = mf._store.get_fragment(first_id)
    assert updated.strength > original_strength


def test_ingest_override_creates_relationship():
    mf = _build_memfrag(["user prefers Python"], "")

    mf.ingest(TURNS)
    first_id = mf._store.all_fragments()[0].id

    # Second ingest — similar but different text → override
    mf._fp.find_duplicate.return_value = first_id
    mf._test_client.messages.create.side_effect = [
        _make_extractor_response("user now prefers Go"),  # different text
        _make_recompose_response(""),
    ]
    mf.ingest(TURNS)

    edges = list(mf._store.graph.relationships())
    override_edges = [e for e in edges if e.relation_type == RelationType.OVERRIDE]
    assert len(override_edges) == 1
    assert override_edges[0].target_id == first_id


def test_stats_returns_expected_keys():
    mf = _build_memfrag([], "")
    s = mf.stats()

    assert "fragments" in s
    assert "edges" in s
    assert "sub_memories" in s
    assert "graph" in s
    assert "turns_ingested" in s


def test_run_decay_removes_stale_fragments():
    from memfrag.models import Fragment

    mf = _build_memfrag([], "")
    stale = Fragment(text="very old fact", strength=0.05)
    mf._store.save_fragment(stale)

    mf.run_decay()

    assert mf._store.get_fragment(stale.id) is None
