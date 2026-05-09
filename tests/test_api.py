"""Mock tests for the FastAPI REST layer."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from memfrag.api import app, get_memfrag
from memfrag.decay import DecayReport
from memfrag.models import Fragment, FragmentType, RecallResult


# ── shared fixture ────────────────────────────────────────────────────────────

def _make_mock_mf() -> MagicMock:
    mf = MagicMock()

    mf.ingest.return_value = [
        Fragment(id="frag01", text="user prefers Python", fragment_type=FragmentType.PREFERENCE)
    ]

    mf.recall.return_value = RecallResult(
        fragments=[
            Fragment(id="frag01", text="user prefers Python", fragment_type=FragmentType.PREFERENCE)
        ],
        context_prefix="[MEMORY:frag01] User prefers Python.",
        source_ids=[],
        token_estimate=10,
    )

    mf.run_decay.return_value = DecayReport(
        fragments_checked=5, cold_count=1, deleted_count=0, elapsed_ms=2.3
    )

    mf.stats.return_value = {
        "fragments": 3, "edges": 1, "sub_memories": 2,
        "graph": {"nodes": 3, "edges": 1}, "turns_ingested": 4,
    }

    mf._store.all_fragments.return_value = [
        Fragment(id="frag01", text="user prefers Python", fragment_type=FragmentType.PREFERENCE)
    ]
    mf._store.get_fragment.return_value = Fragment(
        id="frag01", text="user prefers Python", fragment_type=FragmentType.PREFERENCE
    )

    return mf


@pytest.fixture
def client():
    mock_mf = _make_mock_mf()
    app.dependency_overrides[get_memfrag] = lambda: mock_mf
    with TestClient(app) as c:
        yield c, mock_mf
    app.dependency_overrides.clear()


# ── /ingest ───────────────────────────────────────────────────────────────────

def test_ingest_success(client):
    tc, mf = client
    resp = tc.post("/ingest", json={
        "turns": [
            {"role": "user", "content": "I prefer Python."},
            {"role": "assistant", "content": "Got it."},
        ]
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["fragments_saved"] == 1
    assert body["fragment_ids"] == ["frag01"]
    mf.ingest.assert_called_once()


def test_ingest_empty_turns(client):
    tc, _ = client
    resp = tc.post("/ingest", json={"turns": []})
    assert resp.status_code == 200


def test_ingest_missing_field(client):
    tc, _ = client
    resp = tc.post("/ingest", json={"wrong_key": []})
    assert resp.status_code == 422


# ── /recall ───────────────────────────────────────────────────────────────────

def test_recall_success(client):
    tc, mf = client
    resp = tc.post("/recall", json={"query": "what language should I use?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["context_prefix"] == "[MEMORY:frag01] User prefers Python."
    assert body["token_estimate"] == 10
    mf.recall.assert_called_once_with("what language should I use?")


def test_recall_missing_query(client):
    tc, _ = client
    resp = tc.post("/recall", json={})
    assert resp.status_code == 422


# ── /decay ────────────────────────────────────────────────────────────────────

def test_decay_success(client):
    tc, mf = client
    resp = tc.post("/decay")
    assert resp.status_code == 200
    body = resp.json()
    assert body["fragments_checked"] == 5
    assert body["cold_count"] == 1
    assert body["deleted_count"] == 0
    mf.run_decay.assert_called_once()


# ── /stats ────────────────────────────────────────────────────────────────────

def test_stats_success(client):
    tc, _ = client
    resp = tc.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["fragments"] == 3
    assert body["turns_ingested"] == 4


# ── /fragments ────────────────────────────────────────────────────────────────

def test_list_fragments(client):
    tc, _ = client
    resp = tc.get("/fragments")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert body[0]["id"] == "frag01"


def test_list_fragments_include_cold(client):
    tc, mf = client
    tc.get("/fragments?include_cold=true")
    mf._store.all_fragments.assert_called_with(include_cold=True)


def test_delete_fragment_success(client):
    tc, mf = client
    resp = tc.delete("/fragments/frag01")
    assert resp.status_code == 204
    mf._store.delete_fragment.assert_called_once_with("frag01")


def test_delete_fragment_not_found(client):
    tc, mf = client
    mf._store.get_fragment.return_value = None
    resp = tc.delete("/fragments/nonexistent")
    assert resp.status_code == 404
