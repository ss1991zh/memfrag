# MemFrag — Fragmented Memory System for AI Assistants
### A next-generation AI memory layer built on OpenClaw + Mem0 + Knowledge Graph

[中文文档](README.zh.md)

---

## 1. What Is MemFrag?

**MemFrag** is a pluggable AI memory enhancement layer that runs as an OpenClaw Skill. It enables your personal AI assistant to work like the human brain — operating on highly compressed memory fragments during normal use, and falling back to raw content only when needed — instead of stuffing the entire conversation history into the context window on every turn.

> In one line: **Turn AI "memory" from "rewinding a tape" into "reconstructive recall."**

---

## 2. Problems It Solves

| Pain Point with Current Approaches | MemFrag's Solution |
|---|---|
| Conversation history grows endlessly, token costs scale linearly | Only relevant fragments are recalled — context stays lean |
| RAG chunks are coarse (hundreds of words), full of semantic noise | Fragment granularity down to keyword / short-phrase level |
| Memory silos: each memory item is unaware of others | A relationship graph layer explicitly links fragments |
| Model "hallucinations" can't be traced back to a source | Every fragment is bound to a source ID — answers are traceable |
| Memory store bloats over time, retrieval slows down | Forgetting-curve mechanism auto-archives low-frequency fragments |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────┐
│                  OpenClaw Platform                  │
│   WhatsApp / Telegram / WeChat / iMessage / etc.    │
└────────────────────┬────────────────────────────────┘
                     │ User message
                     ▼
┌─────────────────────────────────────────────────────┐
│             MemFrag Skill (Core Layer)               │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────┐ │
│  │Extraction │──▶│Fingerprint──▶│  Relationship   │ │
│  │ Engine   │   │ Engine   │   │  Graph Engine   │ │
│  │(LLM-based)│  │(Embedding)│  │(NetworkX/Neo4j) │ │
│  └──────────┘   └──────────┘   └─────────────────┘ │
│                                        │            │
│  ┌──────────────────────────────────────────────┐   │
│  │              Storage Layer                   │   │
│  │  Fragment Store (Mem0 + Vector DB)  |  Archive│  │
│  └──────────────────────────────────────────────┘   │
│                                        │            │
│  ┌──────────┐   ┌──────────┐           │            │
│  │  Recall  │◀──│ Recompose│◀──────────┘            │
│  │  Engine  │   │  Engine  │                        │
│  │(vec+graph)│  │(frags→ctx)│                       │
│  └──────────┘   └──────────┘                        │
└─────────────────────────────────────────────────────┘
                     │ Enriched context
                     ▼
              LLM generates response
```

---

## 4. Three-Tier Memory Model

### 4.1 Core Fragment Layer (Long-term, rarely deleted)
Stores minimal semantic units: entities, preferences, constraints, key actions.

```
Example fragments:
- [user_pref_001]      User prefers Python, dislikes JavaScript
- [proj_alpha_deadline] Project Alpha deadline: 2026-06-30
- [user_habit_002]     User prefers deep work in the evenings
```

### 4.2 Relationship Layer (Medium-term, strength evolves)
Directed semantic connections between fragments.

```
[user_pref_001] --co-topic--> [proj_alpha_deadline]
[proj_alpha_deadline] --causal--> [user_habit_002]
[user_pref_001_v2] --override--> [user_pref_001]
```

Four supported relationship types:
- **co-topic** — fragments share the same subject
- **temporal** — A happened before B
- **causal** — A caused B
- **override** — B is an update that supersedes A

### 4.3 Sub-Memory Archive Layer (On-demand)
Full original conversations or document excerpts. Not kept in context by default — the system fetches them only when fragment recall is insufficient.

---

## 5. How It Works

### Write Path (runs automatically after each conversation)
```
1. LLM extracts fragment candidates from the conversation
2. Filter: is this fragment reusable / stable / self-contained?
3. Embedding model generates a semantic fingerprint
4. Similarity check against existing fragments:
   - New content   → create new fragment
   - Update        → add override edge, down-weight old fragment
   - Supplementary → add co-topic edge
5. Fragment saved to store; raw text archived with a back-link
```

### Recall Path (runs automatically before each conversation)
```
1. Generate a query vector from the current message
2. Vector search → Top-K fragments
3. Graph expansion: traverse relationship edges (1–2 hops)
4. Rank by strength score; truncate at token budget
5. If fragments are insufficient, fetch sub-memory archive via back-link
6. Recompose into a natural-language context prefix
```

---

## 6. Key Mechanisms

### 6.1 Forgetting Curve
```
Initial strength       = 1.0
Each recall            → strength × 1.2 (cap: 10)
Every 7 days unused    → strength × 0.85
Strength < 0.3         → moved to cold storage
Strength < 0.1         → marked for cleanup
```

### 6.2 Anti-Hallucination
- Every recomposed sentence retains its source fragment ID
- LLM prompt explicitly separates `[MEMORY FACT]` from `[INFERENCE]`
- Uncertain relationships are dropped rather than fabricated

### 6.3 Cold-Start Strategy
- First 5 conversations: use full history (traditional mode)
- After 5 conversations: fragment store is populated, switch to fragment mode
- Hybrid mode available: fragments + last N turns of raw history

---

## 7. Tech Stack

| Module | Technology | Notes |
|---|---|---|
| Platform | OpenClaw | TypeScript, multi-channel |
| Fragment extraction | Mem0 | Python, native LLM-based extraction |
| Embedding | BGE-M3 / text-embedding-3-small | BGE-M3 preferred for Chinese+English |
| Vector store | Qdrant | Self-hosted, supports metadata filtering |
| Relationship graph | NetworkX (dev) / Neo4j (prod) | |
| Archive store | SQLite / local files | Upgrade to S3 later |
| LLM | Claude Sonnet / GPT-4o | Configurable |
| OpenClaw ↔ Mem0 | REST API / gRPC | MemFrag runs as an independent service |

---

## 8. Comparison with Existing Approaches

| Approach | Granularity | Relationship Layer | Channel Integration | Decay Mechanism |
|---|---|---|---|---|
| **MemFrag** | Word / phrase | ✅ Explicit graph | ✅ via OpenClaw | ✅ |
| Mem0 | Short sentence | ❌ | ❌ | ❌ |
| Letta | Paragraph | ❌ | ❌ | ❌ |
| GraphRAG | Paragraph | ✅ | ❌ | ❌ |
| Traditional RAG | Paragraph | ❌ | ❌ | ❌ |

---

## 9. MVP Roadmap

### Phase 1 — Single-scene validation (2 weeks)
**Goal**: Validate fragment mode vs. full-history on "remembering user writing preferences"

- [ ] Set up Mem0 base memory layer
- [ ] Implement basic extraction + vector recall
- [ ] A/B comparison: token cost and answer accuracy

### Phase 2 — Relationship graph (2 weeks)
**Goal**: Add graph layer, measure multi-hop recall quality improvement

- [ ] NetworkX implementation of four relationship types
- [ ] Graph-expansion recall logic
- [ ] Hallucination rate comparison test

### Phase 3 — OpenClaw integration (1 week)
**Goal**: Package as an OpenClaw Skill, real-user test on Telegram

- [ ] MemFrag service layer (REST API)
- [ ] OpenClaw Skill wrapper
- [ ] Telegram end-to-end test

### Phase 4 — Decay + cold storage (1 week)
**Goal**: Confirm memory store doesn't bloat over long-term use

- [ ] Forgetting-curve implementation
- [ ] Cold storage + auto-cleanup
- [ ] 30-day stress test

---

## 10. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Over-fragmentation loses context | Medium | Dynamic granularity: key content stays as phrases, secondary content compressed to keywords |
| Recall accuracy too low | Medium | Dual-path: vector + graph, each compensating the other |
| Model hallucination during recomposition | High | Mandatory source annotation + prompt constraints |
| OpenClaw ↔ MemFrag latency | Low | Async write, sync recall; pre-cache hot fragments |
| Python (Mem0) / TypeScript (OpenClaw) cross-language overhead | Low | MemFrag as an independent microservice over HTTP |

---

## 11. Name & Identity

- **MemFrag**: Memory Fragments
- Internal alias: **Memento** (a nod to the film *Memento*)
- OpenClaw Skill ID: `memfrag-core`

---

## Contributing

This project is in early design phase. Discussion and ideas are welcome via Issues.

---

*Document version: v0.1 | 2026-05-09*
