"""LLM-based fragment extractor.

Pulls minimal semantic units out of a conversation turn:
entities, preferences, constraints, actions, habits, facts.
Only content that is reusable, stable, and self-contained passes the filter.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import anthropic

from memfrag.models import Fragment, FragmentType

if TYPE_CHECKING:
    from memfrag.models import ConversationTurn

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a memory fragment extractor for a personal AI assistant.

Given a conversation turn, extract ONLY content that is:
1. REUSABLE — applies to future conversations, not just this moment
2. STABLE — preferences/identity/long-term goals, not temporary emotions
3. SELF-CONTAINED — understandable without context not being stored

Fragment types:
- preference: user likes/dislikes, workflow style, tool choices
- entity: important people, places, projects, organisations
- constraint: deadlines, budgets, rules, hard requirements
- action: committed future actions or recurring tasks
- habit: recurring behaviour patterns
- fact: stable factual information about the user's world

Return a JSON array. Each item:
{
  "text": "<concise fragment, max 20 words>",
  "fragment_type": "<one of the types above>"
}

Return [] if nothing worth storing is found.
Do NOT extract: one-off questions, temporary states, this-session-only context.
"""


class FragmentExtractor:
    def __init__(self, client: anthropic.Anthropic, model: str = "claude-haiku-4-5-20251001"):
        self._client = client
        self._model = model

    def extract(self, turns: list[ConversationTurn]) -> list[Fragment]:
        conversation_text = "\n".join(
            f"{t.role.upper()}: {t.content}" for t in turns
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": conversation_text}],
            )
            raw = response.content[0].text.strip()

            # strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            candidates: list[dict] = json.loads(raw)
        except (json.JSONDecodeError, IndexError, anthropic.APIError) as exc:
            logger.warning("Extraction failed: %s", exc)
            return []

        fragments = []
        for c in candidates:
            try:
                frag = Fragment(
                    text=c["text"].strip(),
                    fragment_type=FragmentType(c.get("fragment_type", "fact")),
                )
                fragments.append(frag)
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed candidate %s: %s", c, exc)

        logger.info("Extracted %d fragments from %d turns", len(fragments), len(turns))
        return fragments
