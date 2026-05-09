"""REST API — FastAPI service layer.

Exposes MemFrag to OpenClaw Skills and other consumers via HTTP.

Endpoints:
  POST /ingest          — store fragments from a conversation turn
  POST /recall          — retrieve context prefix for a query
  POST /decay           — force-run forgetting curve
  GET  /stats           — store statistics
  GET  /fragments       — list all active fragments
  DELETE /fragments/{id} — delete a fragment
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from memfrag.core import MemFrag
from memfrag.models import ConversationTurn, Fragment, RecallResult


# ── request / response schemas ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    turns: list[ConversationTurn]

class IngestResponse(BaseModel):
    fragments_saved: int
    fragment_ids: list[str]

class RecallRequest(BaseModel):
    query: str

class DecayResponse(BaseModel):
    fragments_checked: int
    cold_count: int
    deleted_count: int
    elapsed_ms: float


# ── app lifecycle ─────────────────────────────────────────────────────────────

_memfrag: MemFrag | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memfrag
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    db_path = os.environ.get("MEMFRAG_DB", "memfrag.db")
    _memfrag = MemFrag(api_key=api_key, db_path=db_path)
    yield
    _memfrag = None


app = FastAPI(title="MemFrag API", version="0.1.0", lifespan=lifespan)


def get_memfrag() -> MemFrag:
    if _memfrag is None:
        raise HTTPException(status_code=503, detail="MemFrag not initialised")
    return _memfrag


# ── routes ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
def ingest(
    req: IngestRequest,
    mf: Annotated[MemFrag, Depends(get_memfrag)],
):
    saved = mf.ingest(req.turns)
    return IngestResponse(
        fragments_saved=len(saved),
        fragment_ids=[f.id for f in saved],
    )


@app.post("/recall", response_model=RecallResult)
def recall(
    req: RecallRequest,
    mf: Annotated[MemFrag, Depends(get_memfrag)],
):
    return mf.recall(req.query)


@app.post("/decay", response_model=DecayResponse)
def decay(mf: Annotated[MemFrag, Depends(get_memfrag)]):
    report = mf.run_decay()
    return DecayResponse(
        fragments_checked=report.fragments_checked,
        cold_count=report.cold_count,
        deleted_count=report.deleted_count,
        elapsed_ms=report.elapsed_ms,
    )


@app.get("/stats")
def stats(mf: Annotated[MemFrag, Depends(get_memfrag)]):
    return mf.stats()


@app.get("/fragments", response_model=list[Fragment])
def list_fragments(
    mf: Annotated[MemFrag, Depends(get_memfrag)],
    include_cold: bool = False,
):
    return mf._store.all_fragments(include_cold=include_cold)


@app.delete("/fragments/{fragment_id}", status_code=204)
def delete_fragment(
    fragment_id: str,
    mf: Annotated[MemFrag, Depends(get_memfrag)],
):
    frag = mf._store.get_fragment(fragment_id)
    if not frag:
        raise HTTPException(status_code=404, detail="Fragment not found")
    mf._store.delete_fragment(fragment_id)
