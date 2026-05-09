from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RelationType(str, Enum):
    CO_TOPIC = "co-topic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    OVERRIDE = "override"


class FragmentType(str, Enum):
    PREFERENCE = "preference"
    ENTITY = "entity"
    CONSTRAINT = "constraint"
    ACTION = "action"
    HABIT = "habit"
    FACT = "fact"


class Fragment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    text: str
    fragment_type: FragmentType = FragmentType.FACT
    source_id: str = ""           # back-link to sub-memory archive
    strength: float = 1.0         # forgetting-curve weight
    created_at: float = Field(default_factory=time.time)
    last_recalled_at: float = Field(default_factory=time.time)
    recall_count: int = 0
    embedding: Optional[list[float]] = None
    metadata: dict = Field(default_factory=dict)

    def bump(self) -> None:
        self.strength = min(self.strength * 1.2, 10.0)
        self.last_recalled_at = time.time()
        self.recall_count += 1

    def decay(self, days_elapsed: float) -> None:
        self.strength *= 0.85 ** days_elapsed

    @property
    def is_cold(self) -> bool:
        return self.strength < 0.3

    @property
    def is_stale(self) -> bool:
        return self.strength < 0.1


class Relationship(BaseModel):
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    created_at: float = Field(default_factory=time.time)


class SubMemory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    raw_text: str
    turn_index: int = 0
    created_at: float = Field(default_factory=time.time)
    fragment_ids: list[str] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class RecallResult(BaseModel):
    fragments: list[Fragment]
    context_prefix: str
    source_ids: list[str]
    token_estimate: int
