from memfrag.graph import RelationshipGraph
from memfrag.models import Relationship, RelationType


def make_graph():
    g = RelationshipGraph()
    for fid in ["a", "b", "c", "d"]:
        g.add_fragment(fid)
    return g


def test_add_and_expand_one_hop():
    g = make_graph()
    g.add_relationship(Relationship(source_id="a", target_id="b", relation_type=RelationType.CO_TOPIC))
    g.add_relationship(Relationship(source_id="b", target_id="c", relation_type=RelationType.CAUSAL))

    expanded = g.expand(["a"], hops=1)
    assert "a" in expanded
    assert "b" in expanded
    assert "c" not in expanded  # 2 hops away


def test_expand_two_hops():
    g = make_graph()
    g.add_relationship(Relationship(source_id="a", target_id="b", relation_type=RelationType.CO_TOPIC))
    g.add_relationship(Relationship(source_id="b", target_id="c", relation_type=RelationType.CAUSAL))

    expanded = g.expand(["a"], hops=2)
    assert "c" in expanded


def test_override_detection():
    g = make_graph()
    rel = g.infer_override(old_id="a", new_id="b")
    assert g.is_overridden("a")
    assert not g.is_overridden("b")


def test_remove_fragment_cleans_edges():
    g = make_graph()
    g.add_relationship(Relationship(source_id="a", target_id="b", relation_type=RelationType.CO_TOPIC))
    g.remove_fragment("a")
    assert "a" not in g._g


def test_stats():
    g = make_graph()
    g.add_relationship(Relationship(source_id="a", target_id="b", relation_type=RelationType.CO_TOPIC))
    s = g.stats()
    assert s["nodes"] == 4
    assert s["edges"] == 1
