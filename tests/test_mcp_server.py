"""Tests for the MCP server tool dispatch layer.

We test the tool handler logic directly — no stdio transport needed.
The _get_mf() initialisation is mocked so no API key is required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memfrag.decay import DecayReport
from memfrag.models import Fragment, FragmentType, RecallResult


# ── shared fixture ────────────────────────────────────────────────────────────

def _make_mock_mf() -> MagicMock:
    mf = MagicMock()

    mf.ingest.return_value = [
        Fragment(id="frag01", text="user prefers Python", fragment_type=FragmentType.PREFERENCE, strength=1.0)
    ]
    mf.recall.return_value = RecallResult(
        fragments=[Fragment(id="frag01", text="user prefers Python", fragment_type=FragmentType.PREFERENCE, strength=1.0)],
        context_prefix="[MEMORY:frag01] User prefers Python.",
        source_ids=[],
        token_estimate=8,
    )
    mf.run_decay.return_value = DecayReport(
        fragments_checked=4, cold_count=1, deleted_count=0, elapsed_ms=1.5
    )
    mf.stats.return_value = {"fragments": 2, "edges": 1, "sub_memories": 1, "graph": {}, "turns_ingested": 3}

    mf._store.all_fragments.return_value = [
        Fragment(id="frag01", text="user prefers Python", fragment_type=FragmentType.PREFERENCE, strength=1.0)
    ]
    mf._store.get_fragment.return_value = Fragment(
        id="frag01", text="user prefers Python", fragment_type=FragmentType.PREFERENCE, strength=1.0
    )
    return mf


async def _call(tool_name: str, args: dict) -> dict:
    """Call a tool through the server's call_tool handler and return parsed JSON."""
    import memfrag.mcp_server as srv

    mock_mf = _make_mock_mf()
    with patch.object(srv, "_get_mf", return_value=mock_mf):
        contents = await srv.call_tool(tool_name, args)

    assert len(contents) == 1
    return json.loads(contents[0].text), mock_mf


# ── memfrag_ingest ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_returns_fragment_ids():
    result, mf = await _call("memfrag_ingest", {
        "turns": [
            {"role": "user", "content": "I prefer Python."},
            {"role": "assistant", "content": "Got it."},
        ]
    })
    assert result["fragments_saved"] == 1
    assert result["fragment_ids"] == ["frag01"]
    assert result["fragments"][0]["text"] == "user prefers Python"
    mf.ingest.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_empty_turns():
    result, mf = await _call("memfrag_ingest", {"turns": []})
    mf.ingest.assert_called_once()


# ── memfrag_recall ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recall_returns_context_prefix():
    result, mf = await _call("memfrag_recall", {"query": "what language?"})
    assert result["context_prefix"] == "[MEMORY:frag01] User prefers Python."
    assert result["fragments_recalled"] == 1
    assert result["token_estimate"] == 8
    mf.recall.assert_called_once_with("what language?")


# ── memfrag_list_fragments ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_fragments_default():
    result, mf = await _call("memfrag_list_fragments", {})
    assert result["count"] == 1
    assert result["fragments"][0]["id"] == "frag01"
    mf._store.all_fragments.assert_called_with(include_cold=False)


@pytest.mark.asyncio
async def test_list_fragments_include_cold():
    result, mf = await _call("memfrag_list_fragments", {"include_cold": True})
    mf._store.all_fragments.assert_called_with(include_cold=True)


# ── memfrag_delete_fragment ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_fragment_success():
    result, mf = await _call("memfrag_delete_fragment", {"fragment_id": "frag01"})
    assert result["deleted"] == "frag01"
    mf._store.delete_fragment.assert_called_once_with("frag01")


@pytest.mark.asyncio
async def test_delete_fragment_not_found():
    import memfrag.mcp_server as srv
    mock_mf = _make_mock_mf()
    mock_mf._store.get_fragment.return_value = None

    with patch.object(srv, "_get_mf", return_value=mock_mf):
        contents = await srv.call_tool("memfrag_delete_fragment", {"fragment_id": "nope"})

    result = json.loads(contents[0].text)
    assert "error" in result
    mock_mf._store.delete_fragment.assert_not_called()


# ── memfrag_run_decay ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_decay():
    result, mf = await _call("memfrag_run_decay", {})
    assert result["fragments_checked"] == 4
    assert result["cold_count"] == 1
    assert result["deleted_count"] == 0
    mf.run_decay.assert_called_once()


# ── memfrag_stats ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats():
    result, mf = await _call("memfrag_stats", {})
    assert result["fragments"] == 2
    assert result["turns_ingested"] == 3


# ── unknown tool ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    result, _ = await _call("memfrag_nonexistent", {})
    assert "error" in result


# ── list_tools ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_tools_returns_all_six():
    import memfrag.mcp_server as srv
    tools = await srv.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "memfrag_ingest",
        "memfrag_recall",
        "memfrag_list_fragments",
        "memfrag_delete_fragment",
        "memfrag_run_decay",
        "memfrag_stats",
    }
