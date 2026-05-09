"""MemFrag CLI entry point.

Commands:
    memfrag serve   — start the MCP server (stdio)
    memfrag stats   — print store statistics
    memfrag decay   — run forgetting-curve pass
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"

    if cmd == "serve":
        from memfrag.mcp_server import main as serve
        serve()

    elif cmd == "stats":
        from memfrag.core import MemFrag
        import json
        mf = MemFrag(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            db_path=os.environ.get("MEMFRAG_DB", "memfrag.db"),
        )
        print(json.dumps(mf.stats(), indent=2))

    elif cmd == "decay":
        from memfrag.core import MemFrag
        import json
        mf = MemFrag(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            db_path=os.environ.get("MEMFRAG_DB", "memfrag.db"),
        )
        report = mf.run_decay()
        print(json.dumps({
            "fragments_checked": report.fragments_checked,
            "cold_count": report.cold_count,
            "deleted_count": report.deleted_count,
            "elapsed_ms": report.elapsed_ms,
        }, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: memfrag [serve|stats|decay]")
        sys.exit(1)
