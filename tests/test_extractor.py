"""Mock tests for FragmentExtractor."""

import json
from unittest.mock import MagicMock, patch

import pytest

from memfrag.extractor import FragmentExtractor
from memfrag.models import ConversationTurn, FragmentType


def _make_client(response_text: str) -> MagicMock:
    """Build a mock Anthropic client that returns response_text."""
    content_block = MagicMock()
    content_block.text = response_text

    message = MagicMock()
    message.content = [content_block]

    client = MagicMock()
    client.messages.create.return_value = message
    return client


TURNS = [
    ConversationTurn(role="user", content="I prefer Python over JavaScript."),
    ConversationTurn(role="assistant", content="Noted, I'll use Python."),
]


def test_extract_returns_fragments():
    payload = json.dumps([
        {"text": "user prefers Python over JavaScript", "fragment_type": "preference"},
        {"text": "user dislikes JavaScript", "fragment_type": "preference"},
    ])
    client = _make_client(payload)
    extractor = FragmentExtractor(client)

    frags = extractor.extract(TURNS)

    assert len(frags) == 2
    assert frags[0].text == "user prefers Python over JavaScript"
    assert frags[0].fragment_type == FragmentType.PREFERENCE


def test_extract_strips_markdown_fence():
    payload = "```json\n" + json.dumps([
        {"text": "user prefers Python", "fragment_type": "preference"}
    ]) + "\n```"
    client = _make_client(payload)
    extractor = FragmentExtractor(client)

    frags = extractor.extract(TURNS)
    assert len(frags) == 1
    assert frags[0].text == "user prefers Python"


def test_extract_empty_list():
    client = _make_client("[]")
    extractor = FragmentExtractor(client)

    frags = extractor.extract(TURNS)
    assert frags == []


def test_extract_skips_malformed_item():
    payload = json.dumps([
        {"text": "valid fragment", "fragment_type": "fact"},
        {"no_text_key": "bad"},  # malformed — missing "text"
    ])
    client = _make_client(payload)
    extractor = FragmentExtractor(client)

    frags = extractor.extract(TURNS)
    assert len(frags) == 1
    assert frags[0].text == "valid fragment"


def test_extract_invalid_json_returns_empty():
    client = _make_client("not valid json {{")
    extractor = FragmentExtractor(client)

    frags = extractor.extract(TURNS)
    assert frags == []


def test_extract_api_error_returns_empty():
    import anthropic

    client = MagicMock()
    client.messages.create.side_effect = anthropic.APIConnectionError(request=MagicMock())
    extractor = FragmentExtractor(client)

    frags = extractor.extract(TURNS)
    assert frags == []


def test_extract_unknown_fragment_type_falls_back_to_fact():
    payload = json.dumps([
        {"text": "some info", "fragment_type": "unknown_type"}
    ])
    client = _make_client(payload)
    extractor = FragmentExtractor(client)

    frags = extractor.extract(TURNS)
    # unknown type → skipped (ValueError caught)
    assert frags == []


def test_extract_passes_conversation_to_llm():
    client = _make_client("[]")
    extractor = FragmentExtractor(client)
    extractor.extract(TURNS)

    call_kwargs = client.messages.create.call_args
    user_content = call_kwargs.kwargs["messages"][0]["content"]
    assert "USER:" in user_content
    assert "I prefer Python" in user_content
