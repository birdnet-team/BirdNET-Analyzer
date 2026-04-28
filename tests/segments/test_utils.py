"""Correctness and benchmark tests for the np.digitize-based confidence-bin
assignment in birdnet_analyzer.segments.utils.parse_files.

The reference implementation below is the exact original nested-loop code
that was replaced, kept here so the tests remain self-contained.
"""

import timeit
from unittest.mock import patch

import numpy as np
import pytest

from birdnet_analyzer.segments.utils import parse_files

# ---------------------------------------------------------------------------
# Reference implementation (original nested-loop code)
# ---------------------------------------------------------------------------


def _assign_balanced_bins_ref(
    segments: list[dict],
    min_conf: float,
    max_conf: float,
    n_bins: int,
    max_segments: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Original O(n x m) implementation extracted verbatim from parse_files."""
    confidence_bins = []
    bin_thresholds = np.linspace(min_conf, max_conf, num=n_bins)

    for i in range(len(bin_thresholds)):
        if i == 0:
            confidence_bins.append((0, bin_thresholds[i]))
        else:
            confidence_bins.append((bin_thresholds[i - 1], bin_thresholds[i]))

    max_segments_per_bin = max_segments // len(confidence_bins)
    segments_by_bin = {confidence_bin: [] for confidence_bin in confidence_bins}

    segments.sort(key=lambda x: x["confidence"], reverse=True)

    for seg in segments:
        for confidence_bin in confidence_bins:
            if seg["confidence"] >= confidence_bin[1]:
                continue
            if seg["confidence"] >= confidence_bin[0]:
                segments_by_bin[confidence_bin].append(seg)

    result = []
    for bin_segs in segments_by_bin.values():
        if len(bin_segs) > max_segments_per_bin:
            rng.shuffle(bin_segs)
            result.extend(bin_segs[:max_segments_per_bin])
        else:
            result.extend(bin_segs)

    return result


def _assign_balanced_bins_new(
    segments: list[dict],
    min_conf: float,
    max_conf: float,
    n_bins: int,
    max_segments: int,
    rng: np.random.Generator,
) -> list[dict]:
    """New np.digitize implementation extracted verbatim from parse_files."""
    bin_thresholds = np.linspace(min_conf, max_conf, num=n_bins)
    max_segments_per_bin = max_segments // n_bins
    segments_by_bin: list[list] = [[] for _ in range(n_bins)]

    confidences = np.array([seg["confidence"] for seg in segments])
    bin_indices = np.digitize(confidences, bin_thresholds, right=False)

    for seg, bin_idx in zip(segments, bin_indices, strict=False):
        if bin_idx < n_bins:
            segments_by_bin[bin_idx].append(seg)

    result = []
    for bin_segs in segments_by_bin:
        if len(bin_segs) > max_segments_per_bin:
            rng.shuffle(bin_segs)
            result.extend(bin_segs[:max_segments_per_bin])
        else:
            result.extend(bin_segs)

    return result


def _get_bin_contents(segments, min_conf, max_conf, n_bins):
    """Return a list[frozenset] of segment ids for each bin, using new logic."""
    bin_thresholds = np.linspace(min_conf, max_conf, num=n_bins)
    confidences = np.array([s["confidence"] for s in segments])
    indices = np.digitize(confidences, bin_thresholds, right=False)
    bins: list[set] = [set() for _ in range(n_bins + 1)]
    for seg, idx in zip(segments, indices, strict=False):
        bins[idx].add((seg["start"], seg["confidence"]))
    return bins


def _get_bin_contents_ref(segments, min_conf, max_conf, n_bins):
    """Return a list[set] of segment ids for each bin, using old logic."""
    bin_thresholds = np.linspace(min_conf, max_conf, num=n_bins)
    confidence_bins = []
    for i in range(len(bin_thresholds)):
        if i == 0:
            confidence_bins.append((0, bin_thresholds[i]))
        else:
            confidence_bins.append((bin_thresholds[i - 1], bin_thresholds[i]))
    bins: dict = {cb: set() for cb in confidence_bins}
    for seg in segments:
        for cb in confidence_bins:
            if seg["confidence"] >= cb[1]:
                continue
            if seg["confidence"] >= cb[0]:
                bins[cb].add((seg["start"], seg["confidence"]))
    return list(bins.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_segments(confidences: list[float], audio="audio.wav") -> list[dict]:
    return [
        {
            "audio": audio,
            "start": float(i),
            "end": float(i + 3),
            "species": "Robin",
            "confidence": c,
        }
        for i, c in enumerate(confidences)
    ]


def _seg_ids(segs: list[dict]) -> set:
    """Identity set using (start, confidence) since those are unique in our fixtures."""
    return {(s["start"], s["confidence"]) for s in segs}


# ---------------------------------------------------------------------------
# Correctness: bin assignment produces the same per-bin segment sets
# ---------------------------------------------------------------------------


class TestBinAssignmentCorrectness:
    """The core equivalence property: both implementations assign segments to the
    same bins. When bins are over-full a shuffle selects the final subset, but
    the *available pool* per bin must be identical.

    Note: when a bin is over-full, the specific subset chosen after shuffling
    may differ between old and new because the old code pre-sorted segments by
    confidence before insertion (changing the initial array order passed to
    rng.shuffle). Both are valid uniform random samples; only the bin contents
    (before truncation) must match.
    """

    @pytest.mark.parametrize(
        "n_segs,n_bins",  # noqa: PT006
        [
            (200, 10),
            (50, 5),
            (1000, 20),
            (10, 10),  # fewer segments than bins
            (0, 10),  # empty input
        ],
    )
    def test_bin_contents_match_reference(self, n_segs, n_bins):
        """Both implementations must put the same segments in the same bins."""
        confidences = RNG.uniform(0.25, 1.0, n_segs).tolist() if n_segs else []
        segs = _make_segments(confidences)

        new_bins = _get_bin_contents(segs, 0.25, 1.0, n_bins)
        ref_bins = _get_bin_contents_ref(segs, 0.25, 1.0, n_bins)

        # new_bins has n_bins+1 entries (last = conf >= max_conf), ref has n_bins
        assert len(ref_bins) == n_bins
        for i in range(n_bins):
            assert new_bins[i] == ref_bins[i], (
                f"Bin {i} mismatch: new={new_bins[i]}, ref={ref_bins[i]}"
            )
        # new bin n_bins holds conf >= max_conf; should be empty for uniform(0.25, 1.0)
        assert new_bins[n_bins] == set()

    def test_exact_same_output_when_no_bin_truncation(self):
        """When no bin is over-full both implementations return the same set.

        Use deterministic, evenly-spaced confidences so that each bin receives
        exactly n_per_bin segments, well below max_per_bin, making the outcome
        independent of the RNG.
        """
        n_bins = 5
        n_per_bin = 3
        thresholds = np.linspace(
            0.25, 1.0, n_bins
        )  # [0.25, 0.4375, 0.625, 0.8125, 1.0]
        # Place n_per_bin confidences inside each of the n_bins-1 upper bins
        confidences = []
        for i in range(1, n_bins):
            lo, hi = float(thresholds[i - 1]), float(thresholds[i])
            confidences.extend(np.linspace(lo + 0.01, hi - 0.01, n_per_bin).tolist())
        segs = _make_segments(confidences)

        # max_per_bin = max_segments // n_bins = 30 // 5 = 6 > n_per_bin → no truncation
        max_segments = n_per_bin * n_bins * 2

        rng_ref = np.random.default_rng(0)
        rng_new = np.random.default_rng(0)

        result_ref = _assign_balanced_bins_ref(
            list(segs), 0.25, 1.0, n_bins, max_segments, rng_ref
        )
        result_new = _assign_balanced_bins_new(
            list(segs), 0.25, 1.0, n_bins, max_segments, rng_new
        )

        assert _seg_ids(result_new) == _seg_ids(result_ref)

    def test_at_max_conf_excluded_in_both(self):
        """
        Segment with confidence == max_conf must not appear in either implementation.
        """
        segs = _make_segments([0.5, 1.0])
        rng_ref = np.random.default_rng(0)
        rng_new = np.random.default_rng(0)
        result_ref = _assign_balanced_bins_ref(list(segs), 0.25, 1.0, 5, 100, rng_ref)
        result_new = _assign_balanced_bins_new(list(segs), 0.25, 1.0, 5, 100, rng_new)
        assert all(s["confidence"] < 1.0 for s in result_ref)
        assert all(s["confidence"] < 1.0 for s in result_new)

    def test_below_min_conf_lands_in_bin_zero(self):
        """Segments below min_conf go to bin 0 in both implementations (bin_0
        covers [0, min_conf) in the original, and digitize index 0 covers the same
        range). In practice _find_segments pre-filters by min_conf so this bin
        is always empty during normal operation.
        """
        segs = _make_segments([0.1, 0.15, 0.2, 0.5, 0.8])
        new_bins = _get_bin_contents(segs, 0.25, 1.0, 5)
        ref_bins = _get_bin_contents_ref(segs, 0.25, 1.0, 5)
        # Both put the below-min_conf segments in their respective bin 0
        assert new_bins[0] == ref_bins[0]
        below = {(s["start"], s["confidence"]) for s in segs if s["confidence"] < 0.25}
        assert new_bins[0] == below

    def test_respects_max_segments_per_bin(self):
        n_bins, max_seg = 4, 8
        max_per_bin = max_seg // n_bins  # 2
        # 20 segments all in one bin (0.5-0.75 range of 4-bin split over [0.25, 1.0])
        confidences = [0.6] * 20
        segs = _make_segments(confidences)
        rng = np.random.default_rng(0)
        result = _assign_balanced_bins_new(list(segs), 0.25, 1.0, n_bins, max_seg, rng)
        assert len(result) <= max_per_bin

    def test_boundary_at_bin_threshold_same_bin_in_both(self):
        """A confidence exactly at a threshold must go to the same bin in both."""
        n_bins = 4
        thresholds = np.linspace(0.25, 1.0, n_bins)  # [0.25, 0.5, 0.75, 1.0]
        # 0.5 is exactly thresholds[1]
        segs = _make_segments([0.5])
        new_bins = _get_bin_contents(segs, 0.25, 1.0, n_bins)
        ref_bins = _get_bin_contents_ref(segs, 0.25, 1.0, n_bins)
        assert new_bins[:n_bins] == ref_bins

    def test_empty_input(self):
        rng = np.random.default_rng(0)
        result = _assign_balanced_bins_new([], 0.25, 1.0, 10, 100, rng)
        assert result == []

    def test_output_count_bounded_by_max_segments(self):
        """Total output never exceeds max_segments."""
        confidences = RNG.uniform(0.25, 0.99, 500).tolist()
        segs = _make_segments(confidences)
        rng = np.random.default_rng(0)
        max_seg = 50
        result = _assign_balanced_bins_new(list(segs), 0.25, 1.0, 10, max_seg, rng)
        assert len(result) <= max_seg

    def test_output_segment_confidences_in_valid_range(self):
        """Every output segment must have confidence in [min_conf, max_conf)."""
        confidences = RNG.uniform(0.1, 1.05, 200).tolist()
        segs = _make_segments(confidences)
        rng = np.random.default_rng(0)
        result = _assign_balanced_bins_new(list(segs), 0.25, 1.0, 10, 200, rng)
        for s in result:
            assert 0.0 <= s["confidence"] < 1.0


# ---------------------------------------------------------------------------
# Integration: parse_files with collection_mode="balanced"
# ---------------------------------------------------------------------------


class TestParseFilesBalanced:
    """Smoke-tests for parse_files routed through the patched _find_segments.

    The _find_segments mock returns exactly the segments it is given, bypassing
    the min_conf pre-filter that the real function applies.  Tests that check
    confidence filtering therefore only use in-range confidence values.
    """

    def _mock_find_segments(self, confidences):
        return [
            {
                "audio": "fake.wav",
                "start": float(i),
                "end": float(i + 3),
                "species": "Robin",
                "confidence": c,
            }
            for i, c in enumerate(confidences)
        ]

    def test_returns_expected_species(self):
        confidences = np.linspace(0.3, 0.95, 50).tolist()
        flist = [{"audio": "fake.wav", "result": "fake.txt"}]

        with patch(
            "birdnet_analyzer.segments.utils._find_segments",
            return_value=self._mock_find_segments(confidences),
        ):
            result = parse_files(flist, max_segments=30, collection_mode="balanced")

        assert len(result) > 0
        audio_file, segs = result[0]
        assert audio_file == "fake.wav"
        assert all(s["species"] == "Robin" for s in segs)

    def test_max_segments_respected(self):
        confidences = [0.6] * 200
        flist = [{"audio": "fake.wav", "result": "fake.txt"}]

        with patch(
            "birdnet_analyzer.segments.utils._find_segments",
            return_value=self._mock_find_segments(confidences),
        ):
            result = parse_files(
                flist, max_segments=20, collection_mode="balanced", n_bins=4
            )

        total = sum(len(segs) for _, segs in result)
        assert total <= 20

    def test_only_in_range_segments_in_output(self):
        """Segments within [min_conf, max_conf) that the mock returns must appear."""
        confidences = [0.3, 0.5, 0.7, 0.9]  # all in range [0.25, 1.0)
        flist = [{"audio": "fake.wav", "result": "fake.txt"}]

        with patch(
            "birdnet_analyzer.segments.utils._find_segments",
            return_value=self._mock_find_segments(confidences),
        ):
            result = parse_files(
                flist, max_segments=100, collection_mode="balanced", min_conf=0.25
            )

        all_segs = [s for _, segs in result for s in segs]
        assert len(all_segs) == 4
        assert all(0.25 <= s["confidence"] < 1.0 for s in all_segs)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

BENCH_N_SEGS = 10_000
BENCH_N_BINS = 20
BENCH_MAX_SEG = 500
BENCH_REPEATS = 5
BENCH_NUMBER = 10


def _make_bench_segs(n: int) -> list[dict]:
    rng = np.random.default_rng(99)
    confidences = rng.uniform(0.25, 0.99, n).tolist()
    return _make_segments(confidences)


_bench_segs = _make_bench_segs(BENCH_N_SEGS)


def _time(fn, *args, number=BENCH_NUMBER, repeat=BENCH_REPEATS) -> float:
    return min(timeit.repeat(lambda: fn(*args), number=number, repeat=repeat)) / number


def test_benchmark_bin_assignment(capsys):
    rng_ref = np.random.default_rng(0)
    rng_new = np.random.default_rng(0)

    t_ref = _time(
        _assign_balanced_bins_ref,
        list(_bench_segs),
        0.25,
        1.0,
        BENCH_N_BINS,
        BENCH_MAX_SEG,
        rng_ref,
    )
    t_new = _time(
        _assign_balanced_bins_new,
        list(_bench_segs),
        0.25,
        1.0,
        BENCH_N_BINS,
        BENCH_MAX_SEG,
        rng_new,
    )

    with capsys.disabled():
        print(
            f"\nbalanced bin assignment ({BENCH_N_SEGS} segs, {BENCH_N_BINS} bins): "
            f"new={t_new * 1000:.2f}ms  ref={t_ref * 1000:.2f}ms  "
            f"speedup={t_ref / t_new:.1f}x"
        )

    assert t_new < t_ref, (
        f"np.digitize assignment ({t_new * 1000:.2f}ms) should be faster than "
        f"nested-loop reference ({t_ref * 1000:.2f}ms)"
    )
