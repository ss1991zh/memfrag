"""MemFrag MCP Server.

Exposes MemFrag as an MCP server over stdio so Claude Code and
Claude Desktop can call it as a native tool.

Run directly:
    python -m memfrag.mcp_server

Or via the CLI:
    memfrag serve
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from memfrag.core import MemFrag
from memfrag.models import ConversationTurn

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_server = Server("memfrag")
_mf: MemFrag | None = None


def _get_mf() -> MemFrag:
    global _mf
    if _mf is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        db_path = os.environ.get("MEMFRAG_DB", "memfrag.db")
        cold_start = int(os.environ.get("MEMFRAG_COLD_START_TURNS", "5"))
        _mf = MemFrag(api_key=api_key, db_path=db_path, cold_start_turns=cold_start)
        logger.info("MemFrag initialised (db=%s)", db_path)
    return _mf


# ── tool definitions ──────────────────────────────────────────────────────────

@_server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="memfrag_ingest",
            description=(
                "Extract memory fragments from conversation turns and store them. "
                "Call this after each meaningful exchange so Claude can remember it later."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "description": "List of conversation turns to ingest.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    }
                },
                "required": ["turns"],
            },
        ),
        types.Tool(
            name="memfrag_recall",
            description=(
                "Recall relevant memory fragments for a query and return a context block. "
                "Call this before answering questions that may depend on past conversations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The current question or topic to recall memory for.",
                    }
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="memfrag_list_fragments",
            description="List all stored memory fragments.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_cold": {
                        "type": "boolean",
                        "description": "Include low-strength (cold) fragments. Default false.",
                        "default": False,
                    }
                },
            },
        ),
        types.Tool(
            name="memfrag_delete_fragment",
            description="Delete a specific memory fragment by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fragment_id": {
                        "type": "string",
                        "description": "The ID of the fragment to delete.",
                    }
                },
                "required": ["fragment_id"],
            },
        ),
        types.Tool(
            name="memfrag_run_decay",
            description=(
                "Run the forgetting-curve pass manually. "
                "Weakens unused fragments and deletes stale ones."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="memfrag_stats",
            description="Return memory store statistics (fragment count, edge count, etc.).",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ── tool dispatch ─────────────────────────────────────────────────────────────

@_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    mf = _get_mf()

    if name == "memfrag_ingest":
        turns = [ConversationTurn(**t) for t in arguments["turns"]]
        saved = mf.ingest(turns)
        result = {
            "fragments_saved": len(saved),
            "fragment_ids": [f.id for f in saved],
            "fragments": [{"id": f.id, "text": f.text, "type": f.fragment_type.value} for f in saved],
        }

    elif name == "memfrag_recall":
        recall = mf.recall(arguments["query"])
        result = {
            "context_prefix": recall.context_prefix,
            "fragments_recalled": len(recall.fragments),
            "token_estimate": recall.token_estimate,
            "fragments": [
                {"id": f.id, "text": f.text, "strength": round(f.strength, 3)}
                for f in recall.fragments
            ],
        }

    elif name == "memfrag_list_fragments":
        include_cold = arguments.get("include_cold", False)
        frags = mf._store.all_fragments(include_cold=include_cold)
        result = {
            "count": len(frags),
            "fragments": [
                {
                    "id": f.id,
                    "text": f.text,
                    "type": f.fragment_type.value,
                    "strength": round(f.strength, 3),
                    "recall_count": f.recall_count,
                }
                for f in frags
            ],
        }

    elif name == "memfrag_delete_fragment":
        fid = arguments["fragment_id"]
        frag = mf._store.get_fragment(fid)
        if not frag:
            result = {"error": f"Fragment '{fid}' not found"}
        else:
            mf._store.delete_fragment(fid)
            result = {"deleted": fid}

    elif name == "memfrag_run_decay":
        report = mf.run_decay()
        result = {
            "fragments_checked": report.fragments_checked,
            "cold_count": report.cold_count,
            "deleted_count": report.deleted_count,
            "elapsed_ms": report.elapsed_ms,
        }

    elif name == "memfrag_stats":
        result = mf.stats()

    else:
        result = {"error": f"Unknown tool: {name}"}

    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


# ── entry point ───────────────────────────────────────────────────────────────

async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await _server.run(
            read_stream,
            write_stream,
            _server.create_initialization_options(),
        )


def main() -> None:
    import asyncio
    asyncio.run(_run())


if __name__ == "__main__":
    main()
