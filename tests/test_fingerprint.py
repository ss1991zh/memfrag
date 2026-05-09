from memfrag.fingerprint import FingerprintEngine


def test_cosine_similarity_identical():
    score = FingerprintEngine.cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert abs(score - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    score = FingerprintEngine.cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert abs(score) < 1e-6


def test_cosine_similarity_zero_vector():
    score = FingerprintEngine.cosine_similarity([0.0, 0.0], [1.0, 0.0])
    assert score == 0.0


def test_top_k_returns_sorted():
    fp = FingerprintEngine.__new__(FingerprintEngine)
    candidates = [
        ("a", [1.0, 0.0]),
        ("b", [0.9, 0.1]),
        ("c", [0.0, 1.0]),
    ]
    query = [1.0, 0.0]
    results = fp.top_k(query, candidates, k=2, threshold=0.0)
    ids = [r[0] for r in results]
    assert ids[0] == "a"
    assert ids[1] == "b"
    assert len(results) == 2


def test_find_duplicate_above_threshold():
    fp = FingerprintEngine.__new__(FingerprintEngine)
    candidates = [("x", [1.0, 0.0])]
    dup = fp.find_duplicate([1.0, 0.0], candidates, threshold=0.9)
    assert dup == "x"


def test_find_duplicate_below_threshold():
    fp = FingerprintEngine.__new__(FingerprintEngine)
    candidates = [("x", [1.0, 0.0])]
    dup = fp.find_duplicate([0.0, 1.0], candidates, threshold=0.9)
    assert dup is None
