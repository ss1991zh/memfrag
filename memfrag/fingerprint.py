"""Semantic fingerprint engine.

Generates embeddings for fragments and computes similarity scores
for deduplication and recall matching.
Uses sentence-transformers locally (no external API needed).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # lightweight multilingual-compatible model


class FingerprintEngine:
    def __init__(self, model_name: str = _MODEL_NAME):
        self._model_name = model_name
        self._model: Optional[object] = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def top_k(
        self,
        query_vec: list[float],
        candidates: list[tuple[str, list[float]]],  # (id, embedding)
        k: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Return up to k (id, score) pairs above threshold, sorted by score desc."""
        scored = [
            (fid, self.cosine_similarity(query_vec, emb))
            for fid, emb in candidates
        ]
        scored = [(fid, score) for fid, score in scored if score >= threshold]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def find_duplicate(
        self,
        query_vec: list[float],
        candidates: list[tuple[str, list[float]]],
        threshold: float = 0.92,
    ) -> Optional[str]:
        """Return the id of the most similar existing fragment if above threshold."""
        results = self.top_k(query_vec, candidates, k=1, threshold=threshold)
        return results[0][0] if results else None
