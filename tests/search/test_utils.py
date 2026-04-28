"""Correctness and benchmark tests for the vectorized cosine_sim and
euclidean_scoring functions in birdnet_analyzer.search.utils.

The reference implementations below are the exact originals that were
replaced, kept here so the tests remain self-contained and do not depend
on git history.
"""

import timeit

import numpy as np
import pytest
from scipy.spatial.distance import euclidean as scipy_euclidean

from birdnet_analyzer.search.utils import cosine_sim, euclidean_scoring


# ---------------------------------------------------------------------------
# Reference implementations (original loop-based code)
# ---------------------------------------------------------------------------

def _cosine_sim_ref(data: np.ndarray, query: np.ndarray):
    if data.ndim == 2:
        return np.array([_cosine_sim_ref(data[i], query) for i in range(data.shape[0])])
    return np.dot(data, query) / (np.linalg.norm(data) * np.linalg.norm(query))


def _euclidean_scoring_ref(data: np.ndarray, query: np.ndarray):
    if data.ndim == 2:
        return np.array(
            [_euclidean_scoring_ref(data[i], query) for i in range(data.shape[0])]
        )
    return scipy_euclidean(data, query)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


@pytest.fixture(params=[16, 128, 1024])
def dim(request):
    return request.param


@pytest.fixture(params=[1, 10, 500])
def n_rows(request):
    return request.param


# ---------------------------------------------------------------------------
# Correctness: cosine_sim
# ---------------------------------------------------------------------------

class TestCosineSim:
    def test_1d_matches_reference(self, dim):
        a = RNG.random(dim)
        q = RNG.random(dim)
        assert cosine_sim(a, q) == pytest.approx(_cosine_sim_ref(a, q), rel=1e-6)

    def test_2d_matches_reference(self, n_rows, dim):
        data = RNG.random((n_rows, dim))
        query = RNG.random(dim)
        result = cosine_sim(data, query)
        expected = _cosine_sim_ref(data, query)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_1d_range(self, dim):
        """Cosine similarity must be in [-1, 1]."""
        a = RNG.random(dim)
        q = RNG.random(dim)
        val = cosine_sim(a, q)
        assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9

    def test_2d_range(self, dim):
        data = RNG.random((50, dim))
        query = RNG.random(dim)
        vals = cosine_sim(data, query)
        assert np.all(vals >= -1.0 - 1e-9)
        assert np.all(vals <= 1.0 + 1e-9)

    def test_identical_vectors_is_1(self):
        v = RNG.random(64)
        assert cosine_sim(v, v) == pytest.approx(1.0, rel=1e-6)

    def test_orthogonal_vectors_is_0(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_sim(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_single_row_matrix_matches_vector(self, dim):
        v = RNG.random(dim)
        q = RNG.random(dim)
        result_2d = cosine_sim(v[np.newaxis, :], q)
        result_1d = cosine_sim(v, q)
        np.testing.assert_allclose(result_2d, [result_1d], rtol=1e-6)

    def test_output_shape_2d(self, n_rows, dim):
        data = RNG.random((n_rows, dim))
        query = RNG.random(dim)
        result = cosine_sim(data, query)
        assert result.shape == (n_rows,)


# ---------------------------------------------------------------------------
# Correctness: euclidean_scoring
# ---------------------------------------------------------------------------

class TestEuclideanScoring:
    def test_1d_matches_reference(self, dim):
        a = RNG.random(dim)
        q = RNG.random(dim)
        assert euclidean_scoring(a, q) == pytest.approx(_euclidean_scoring_ref(a, q), rel=1e-6)

    def test_2d_matches_reference(self, n_rows, dim):
        data = RNG.random((n_rows, dim))
        query = RNG.random(dim)
        result = euclidean_scoring(data, query)
        expected = _euclidean_scoring_ref(data, query)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_identical_vectors_is_0(self):
        v = RNG.random(64)
        assert euclidean_scoring(v, v) == pytest.approx(0.0, abs=1e-9)

    def test_non_negative(self, dim):
        a = RNG.random(dim)
        q = RNG.random(dim)
        assert euclidean_scoring(a, q) >= 0.0

    def test_2d_non_negative(self, dim):
        data = RNG.random((50, dim))
        query = RNG.random(dim)
        assert np.all(euclidean_scoring(data, query) >= 0.0)

    def test_output_shape_2d(self, n_rows, dim):
        data = RNG.random((n_rows, dim))
        query = RNG.random(dim)
        result = euclidean_scoring(data, query)
        assert result.shape == (n_rows,)

    def test_single_row_matrix_matches_vector(self, dim):
        v = RNG.random(dim)
        q = RNG.random(dim)
        result_2d = euclidean_scoring(v[np.newaxis, :], q)
        result_1d = euclidean_scoring(v, q)
        np.testing.assert_allclose(result_2d, [result_1d], rtol=1e-6)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

BENCH_ROWS = 5000
BENCH_DIM = 1024
BENCH_REPEATS = 5
BENCH_NUMBER = 20

_bench_data = RNG.random((BENCH_ROWS, BENCH_DIM)).astype(np.float32)
_bench_query = RNG.random(BENCH_DIM).astype(np.float32)


def _time(fn, *args, number=BENCH_NUMBER, repeat=BENCH_REPEATS) -> float:
    """Return best-of-repeat wall time in seconds for `number` calls."""
    return min(
        timeit.repeat(lambda: fn(*args), number=number, repeat=repeat)
    ) / number


def test_benchmark_cosine_sim(capsys):
    t_new = _time(cosine_sim, _bench_data, _bench_query)
    t_ref = _time(_cosine_sim_ref, _bench_data, _bench_query)

    with capsys.disabled():
        print(
            f"\ncosine_sim ({BENCH_ROWS}×{BENCH_DIM}): "
            f"new={t_new*1000:.2f}ms  ref={t_ref*1000:.2f}ms  "
            f"speedup={t_ref/t_new:.1f}×"
        )

    assert t_new < t_ref, (
        f"Vectorized cosine_sim ({t_new*1000:.2f}ms) should be faster than "
        f"loop-based reference ({t_ref*1000:.2f}ms)"
    )


def test_benchmark_euclidean_scoring(capsys):
    t_new = _time(euclidean_scoring, _bench_data, _bench_query)
    t_ref = _time(_euclidean_scoring_ref, _bench_data, _bench_query)

    with capsys.disabled():
        print(
            f"\neuclidean_scoring ({BENCH_ROWS}×{BENCH_DIM}): "
            f"new={t_new*1000:.2f}ms  ref={t_ref*1000:.2f}ms  "
            f"speedup={t_ref/t_new:.1f}×"
        )

    assert t_new < t_ref, (
        f"Vectorized euclidean_scoring ({t_new*1000:.2f}ms) should be faster than "
        f"loop-based reference ({t_ref*1000:.2f}ms)"
    )
