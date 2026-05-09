# MemFrag — Fragmented Memory for Claude
### A persistent, graph-aware memory layer that runs as an MCP server inside Claude Code and Claude Desktop

[中文文档](README.zh.md)

---

## What Is MemFrag?

MemFrag gives Claude a long-term memory that works like the human brain — storing information as small, interconnected fragments rather than raw conversation history.

It runs as an **MCP (Model Context Protocol) server**, so Claude Code and Claude Desktop can call it as a native tool. No external platform needed.

> In one line: **Claude remembers you across sessions, without stuffing your entire history into the context window.**

---

## How It Works

```
┌──────────────────────────────────────┐
│     Claude Code / Claude Desktop     │
│                                      │
│  "Remember I prefer Python"          │
│  "What was my project deadline?"     │
└────────────┬─────────────────────────┘
             │ MCP (stdio)
             ▼
┌──────────────────────────────────────┐
│         MemFrag MCP Server           │
│                                      │
│  Tools exposed to Claude:            │
│  ├── memfrag_ingest(turns)           │
│  ├── memfrag_recall(query)           │
│  ├── memfrag_list_fragments()        │
│  ├── memfrag_delete_fragment(id)     │
│  ├── memfrag_run_decay()             │
│  └── memfrag_stats()                 │
│                                      │
│  Internals:                          │
│  ├── Fragment Extractor  (Claude LLM)│
│  ├── Fingerprint Engine  (Embeddings)│
│  ├── Relationship Graph  (NetworkX)  │
│  ├── Storage Layer       (SQLite)    │
│  ├── Recall Engine       (vec+graph) │
│  └── Decay Scheduler                 │
└──────────────────────────────────────┘
```

### Write path (after each conversation turn)
```
Claude calls memfrag_ingest(turns)
  → LLM extracts key fragments (entities, preferences, constraints…)
  → Each fragment gets a semantic fingerprint (embedding)
  → Duplicates are detected and merged; updates create override edges
  → Fragments stored in SQLite; raw text archived with a back-link
  → Co-topic relationships built automatically via graph
```

### Read path (before generating a response)
```
Claude calls memfrag_recall(query)
  → Query embedded → vector top-K search
  → Graph expansion: follow co-topic / causal / override edges (1-2 hops)
  → Rank by strength × similarity; trim to token budget
  → Recompose fragments into a natural-language context block
  → Claude uses that block as grounded memory
```

---

## Three-Tier Memory Model

| Layer | Stores | Lifetime |
|---|---|---|
| **Fragment layer** | Keywords, short phrases, entities, preferences | Long-term (rarely deleted) |
| **Relationship layer** | Links between fragments (co-topic / temporal / causal / override) | Medium-term (strength evolves) |
| **Sub-memory archive** | Full raw conversation text | On-demand fallback |

### Forgetting curve
```
Initial strength    = 1.0
Each recall         → strength × 1.2  (cap: 10)
Every 7 days unused → strength × 0.85
strength < 0.3      → cold storage (excluded from active recall)
strength < 0.1      → deleted
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/ss1991zh/memfrag.git
cd memfrag
pip install -e .
```

### 2. Configure Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "memfrag": {
      "command": "python",
      "args": ["-m", "memfrag.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "MEMFRAG_DB": "/path/to/memfrag.db"
      }
    }
  }
}
```

### 3. Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memfrag": {
      "command": "python",
      "args": ["-m", "memfrag.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "MEMFRAG_DB": "/Users/you/memfrag.db"
      }
    }
  }
}
```

### 4. Use in Claude

Once configured, Claude can use memory naturally:

```
You:    Remember that I'm building a fragmented memory system in Python,
        targeting Claude integration via MCP.

Claude: [calls memfrag_ingest] ✓ Stored 3 fragments.

You:    What was the project I was working on?

Claude: [calls memfrag_recall] Based on memory:
        You're building MemFrag — a fragmented memory layer for Claude,
        implemented in Python and integrated via MCP.
```

---

## MCP Tools Reference

| Tool | Arguments | Description |
|---|---|---|
| `memfrag_ingest` | `turns: [{role, content}]` | Extract and store fragments from conversation turns |
| `memfrag_recall` | `query: str` | Recall relevant fragments and return a context block |
| `memfrag_list_fragments` | `include_cold?: bool` | List all active fragments |
| `memfrag_delete_fragment` | `fragment_id: str` | Delete a specific fragment |
| `memfrag_run_decay` | — | Run the forgetting-curve pass manually |
| `memfrag_stats` | — | Return store statistics |

---

## Tech Stack

| Module | Technology |
|---|---|
| MCP server | `mcp` Python SDK (stdio transport) |
| LLM (extraction + recomposition) | Claude Haiku via Anthropic SDK |
| Embeddings | `sentence-transformers` (BAAI/bge-small-en-v1.5, local) |
| Vector search | NumPy cosine similarity |
| Relationship graph | NetworkX |
| Storage | SQLite (built-in, zero config) |

---

## vs. Existing Approaches

| | MemFrag | Mem0 | Letta | GraphRAG |
|---|---|---|---|---|
| Claude-native (MCP) | ✅ | ❌ | ❌ | ❌ |
| Granularity | Word/phrase | Short sentence | Paragraph | Paragraph |
| Relationship graph | ✅ | ❌ | ❌ | ✅ |
| Forgetting curve | ✅ | ❌ | ❌ | ❌ |
| Zero-config storage | ✅ SQLite | ❌ | ❌ | ❌ |

---

## REST API (optional)

MemFrag also ships a FastAPI server for non-MCP integrations:

```bash
ANTHROPIC_API_KEY=sk-ant-... uvicorn memfrag.api:app --port 8765
```

Endpoints: `POST /ingest`, `POST /recall`, `POST /decay`, `GET /stats`, `GET /fragments`, `DELETE /fragments/{id}`

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/          # 61+ tests, no API key needed
```

---

## Roadmap

- [x] Core fragment extraction, graph, storage, recall
- [x] Forgetting curve
- [x] REST API
- [x] MCP server
- [ ] End-to-end test with real API key
- [ ] Neo4j graph backend (production scale)
- [ ] Multi-user support

---

*Version 0.2.0 · 2026-05-09*
