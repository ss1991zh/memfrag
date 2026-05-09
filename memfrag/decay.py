"""Forgetting-curve scheduler.

Runs periodically to decay fragment strength values according to:
  strength_new = strength_old × 0.85^days_since_last_recall

Thresholds:
  strength < 0.3 → cold storage (excluded from active recall by default)
  strength < 0.1 → deleted

The scheduler can be called manually or triggered by the API on a cron.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memfrag.store import FragmentStore

logger = logging.getLogger(__name__)

DECAY_RATE = 0.85          # per 7-day period
DECAY_PERIOD_DAYS = 7.0
COLD_THRESHOLD = 0.3
STALE_THRESHOLD = 0.1


@dataclass
class DecayReport:
    fragments_checked: int
    cold_count: int
    deleted_count: int
    elapsed_ms: float


class DecayScheduler:
    def __init__(self, store: FragmentStore):
        self._store = store
        self._last_run: float = 0.0

    def run(self, force: bool = False) -> DecayReport:
        now = time.time()
        t0 = now

        cold, deleted = self._store.apply_decay(now=now)
        total = self._store.stats()["fragments"] + deleted

        self._last_run = now
        elapsed = (time.time() - t0) * 1000

        report = DecayReport(
            fragments_checked=total,
            cold_count=cold,
            deleted_count=deleted,
            elapsed_ms=round(elapsed, 2),
        )
        logger.info(
            "Decay run complete — checked=%d cold=%d deleted=%d (%.1fms)",
            total, cold, deleted, elapsed,
        )
        return report

    def run_if_due(self, interval_hours: float = 24.0) -> DecayReport | None:
        due_at = self._last_run + interval_hours * 3600
        if time.time() >= due_at:
            return self.run()
        return None
