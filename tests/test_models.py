import time
from memfrag.models import Fragment, FragmentType, Relationship, RelationType


def test_fragment_defaults():
    f = Fragment(text="user prefers Python")
    assert f.strength == 1.0
    assert f.recall_count == 0
    assert not f.is_cold
    assert not f.is_stale


def test_fragment_bump():
    f = Fragment(text="user prefers Python")
    f.bump()
    assert f.strength == 1.2
    assert f.recall_count == 1


def test_fragment_bump_cap():
    f = Fragment(text="test", strength=9.5)
    f.bump()
    assert f.strength == 10.0


def test_fragment_decay():
    f = Fragment(text="test", strength=1.0)
    f.decay(days_elapsed=7.0)
    # 0.85^7 ≈ 0.3206
    assert abs(f.strength - 0.85**7) < 0.001


def test_fragment_cold_threshold():
    f = Fragment(text="test", strength=0.25)
    assert f.is_cold
    assert not f.is_stale


def test_fragment_stale_threshold():
    f = Fragment(text="test", strength=0.05)
    assert f.is_stale


def test_relationship_creation():
    r = Relationship(
        source_id="a", target_id="b", relation_type=RelationType.CO_TOPIC
    )
    assert r.weight == 1.0
