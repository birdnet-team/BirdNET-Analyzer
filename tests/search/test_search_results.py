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


def _random_embeddings(n: int) -> np.ndarray:
    # The index stores float16, so these survive the round trip unchanged.
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, DIM)).astype(np.float16).astype(np.float32)


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


def test_dot_search_keeps_the_exact_match_of_every_query_embedding(db, monkeypatch):
    embeddings = _random_embeddings(50)
    embeddings[[6, 31]] *= 4
    ids = _insert(db, embeddings)

    results = _search(db, monkeypatch, [embeddings[6], embeddings[31]], 3, "dot")

    assert {r.window_id for r in results[:2]} == {ids[6], ids[31]}


@pytest.mark.parametrize("n_windows", [1, 3])
@pytest.mark.parametrize("score_function", ["cosine", "dot", "euclidean"])
def test_a_small_database_returns_all_its_windows(
    db, monkeypatch, score_function, n_windows
):
    embeddings = _random_embeddings(n_windows)
    ids = _insert(db, embeddings)

    results = _search(db, monkeypatch, [embeddings[0]], 10, score_function)

    assert sorted(r.window_id for r in results) == ids
