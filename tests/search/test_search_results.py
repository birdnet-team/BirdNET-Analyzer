import numpy as np
import pytest

from birdnet_analyzer.embeddings.core import get_or_create_database
from birdnet_analyzer.search import utils as su

DIM = 8


@pytest.fixture
def db(tmp_path):
    database = get_or_create_database(str(tmp_path / "db"), embedding_dim=DIM)
    yield database
    database.db.close()


def _random_embeddings(n: int, seed: int = 0) -> np.ndarray:
    # The index stores float16, so these survive the round trip unchanged.
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, DIM)).astype(np.float16).astype(np.float32)


def _brute_force(embeddings, queries, score_function):
    """Ranks every window by its mean score over all query embeddings."""
    if score_function == "cosine":
        unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        scores = [unit @ (q / np.linalg.norm(q)) for q in queries]
    elif score_function == "dot":
        scores = [embeddings @ q for q in queries]
    else:
        scores = [np.linalg.norm(embeddings - q, axis=1) for q in queries]

    mean = np.mean(scores, axis=0)
    order = np.argsort(mean if score_function == "euclidean" else -mean)
    return list(order), mean


def _insert(db, embeddings: np.ndarray) -> list[int]:
    deployment_id = db.insert_deployment(name="deployment", project="project")
    recording_id = db.insert_recording(filename="rec.wav", deployment_id=deployment_id)
    ids = [
        db.insert_window(recording_id, [3.0 * i, 3.0 * i + 3.0], embedding)
        for i, embedding in enumerate(embeddings)
    ]
    db.commit()
    return ids


def _search(db, monkeypatch, queries, n_results, score_function):
    monkeypatch.setattr(su, "get_query_embedding", lambda *a, **k: queries)
    return su.get_search_results(
        "query.wav", db, n_results=n_results, score_function=score_function
    )


def test_dot_search_ranks_like_the_dot_product(db, monkeypatch):
    embeddings = _random_embeddings(50)
    embeddings[7] *= 4
    ids = _insert(db, embeddings)
    query = embeddings[7]
    assert su._get_usearch_metric_name(db) == "IP"

    results = _search(db, monkeypatch, [query], 4, "dot")

    expected = np.argsort(-(embeddings @ query))[:4]
    assert [r.window_id for r in results] == [ids[i] for i in expected]
    assert results[0].window_id == ids[7]
    assert results[0].sort_score == pytest.approx(np.dot(query, query), rel=1e-3)


def test_a_multi_query_dot_search_ranks_like_the_mean_dot_product(db, monkeypatch):
    embeddings = _random_embeddings(50)
    embeddings[[6, 31]] *= 4
    ids = _insert(db, embeddings)
    queries = [embeddings[6], embeddings[31]]

    results = _search(db, monkeypatch, queries, 3, "dot")

    order, mean = _brute_force(embeddings, queries, "dot")
    assert [r.window_id for r in results] == [ids[i] for i in order[:3]]
    assert [r.sort_score for r in results] == pytest.approx(mean[order[:3]], rel=1e-3)


@pytest.mark.parametrize("score_function", ["cosine", "dot", "euclidean"])
def test_a_window_found_by_one_query_embedding_is_scored_against_all_of_them(
    db, monkeypatch, score_function
):
    embeddings = _random_embeddings(50)
    ids = _insert(db, embeddings)
    exact, n_results = 5, 3
    queries = [embeddings[exact], _random_embeddings(1, seed=1)[0]]
    order, mean = _brute_force(embeddings, queries, score_function)
    rank = order.index(exact)
    assert rank < n_results
    other_top = _brute_force(embeddings, queries[1:], score_function)[0][:n_results]
    assert exact not in other_top

    results = _search(db, monkeypatch, queries, n_results, score_function)

    top = order[:n_results]
    assert [r.window_id for r in results] == [ids[i] for i in top]
    assert [r.sort_score for r in results] == pytest.approx(mean[top], rel=1e-3)
    assert results[rank].window_id == ids[exact]


@pytest.mark.parametrize("n_windows", [1, 3])
@pytest.mark.parametrize("score_function", ["cosine", "dot", "euclidean"])
def test_a_small_database_returns_all_its_windows(
    db, monkeypatch, score_function, n_windows
):
    embeddings = _random_embeddings(n_windows)
    ids = _insert(db, embeddings)

    results = _search(db, monkeypatch, [embeddings[0]], 10, score_function)

    assert sorted(r.window_id for r in results) == ids
