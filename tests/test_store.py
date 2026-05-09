import pytest
from memfrag.store import FragmentStore
from memfrag.models import Fragment, FragmentType, Relationship, RelationType, SubMemory


@pytest.fixture
def store():
    return FragmentStore(":memory:")


def test_save_and_get_fragment(store):
    f = Fragment(text="user prefers Python", fragment_type=FragmentType.PREFERENCE)
    store.save_fragment(f)
    retrieved = store.get_fragment(f.id)
    assert retrieved is not None
    assert retrieved.text == f.text
    assert retrieved.fragment_type == FragmentType.PREFERENCE


def test_delete_fragment(store):
    f = Fragment(text="test fact")
    store.save_fragment(f)
    store.delete_fragment(f.id)
    assert store.get_fragment(f.id) is None


def test_all_fragments_excludes_cold(store):
    warm = Fragment(text="warm", strength=1.0)
    cold = Fragment(text="cold", strength=0.2)
    store.save_fragment(warm)
    store.save_fragment(cold)

    active = store.all_fragments(include_cold=False)
    ids = [f.id for f in active]
    assert warm.id in ids
    assert cold.id not in ids


def test_save_relationship_persists(store):
    a = Fragment(text="a")
    b = Fragment(text="b")
    store.save_fragment(a)
    store.save_fragment(b)

    rel = Relationship(source_id=a.id, target_id=b.id, relation_type=RelationType.CO_TOPIC)
    store.save_relationship(rel)

    assert store.graph._g.has_edge(a.id, b.id)


def test_sub_memory_roundtrip(store):
    sm = SubMemory(raw_text="USER: hello\nASSISTANT: hi", turn_index=1, fragment_ids=["abc"])
    store.save_sub_memory(sm)
    retrieved = store.get_sub_memory(sm.id)
    assert retrieved is not None
    assert retrieved.raw_text == sm.raw_text
    assert retrieved.fragment_ids == ["abc"]


def test_apply_decay_deletes_stale(store):
    stale = Fragment(text="stale", strength=0.05)
    store.save_fragment(stale)

    cold, deleted = store.apply_decay()
    assert deleted >= 1
    assert store.get_fragment(stale.id) is None


def test_embeddings_index(store):
    f = Fragment(text="test", embedding=[0.1, 0.2, 0.3])
    store.save_fragment(f)
    index = store.embeddings_index()
    assert any(fid == f.id for fid, _ in index)


def test_stats(store):
    f = Fragment(text="x")
    store.save_fragment(f)
    s = store.stats()
    assert s["fragments"] == 1
    assert s["edges"] == 0
