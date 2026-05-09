"""Basic usage example — no real API key required for structure demo."""

import os
from memfrag import MemFrag
from memfrag.models import ConversationTurn

mf = MemFrag(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    db_path="demo.db",
    cold_start_turns=0,   # disable cold-start for demo
)

# Simulate a conversation turn
turns = [
    ConversationTurn(role="user", content="I prefer Python over JavaScript for backend work."),
    ConversationTurn(role="assistant", content="Noted! I'll default to Python for your projects."),
    ConversationTurn(role="user", content="Also, our project Alpha must ship before June 30."),
]

print("=== Ingesting conversation ===")
saved = mf.ingest(turns)
for f in saved:
    print(f"  [{f.id}] ({f.fragment_type.value}) {f.text}")

print("\n=== Recall for next query ===")
result = mf.recall("What language should I use for the Alpha backend?")
print(result.context_prefix)

print("\n=== Store stats ===")
print(mf.stats())
