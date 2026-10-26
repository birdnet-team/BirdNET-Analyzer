from math import isclose

import numpy as np
import pandas as pd
import pytest

from birdnet_analyzer.analyze.core import _merge_consecutive_segments


def _merge_reference(
    df: pd.DataFrame, merge_consecutive: int, hop_size: float = 3.0
) -> pd.DataFrame:
    if merge_consecutive <= 1:
        return df

    if df.empty:
        return df

    df = df.copy()
    df["start_time"] = df["start_time"].astype("float32", copy=False)
    df["end_time"] = df["end_time"].astype("float32", copy=False)

    df_sorted = df.sort_values(
        by=["input", "species_name", "start_time", "end_time"]
    ).reset_index(drop=True)
    merged_records = []
    i = 0

    while i < len(df_sorted):
        window = [df_sorted.iloc[i]]

        while len(window) < merge_consecutive:
            next_idx = i + len(window)
            if next_idx >= len(df_sorted):
                break

            prev_row = window[-1]
            candidate_row = df_sorted.iloc[next_idx]

            if (
                candidate_row["input"] == prev_row["input"]
                and candidate_row["species_name"] == prev_row["species_name"]
                and isclose(
                    float(candidate_row["start_time"]),
                    float(prev_row["end_time"]),
                    abs_tol=hop_size,
                )
            ):
                window.append(candidate_row)
            else:
                break

        if len(window) > 1:
            merged_row = window[0].copy()
            merged_row["end_time"] = window[-1]["end_time"]

            if "confidence" in df.columns:
                confidences = [float(row["confidence"]) for row in window]
                avg_confidence = sum(confidences) / len(confidences)
                merged_row["confidence"] = type(window[0]["confidence"])(avg_confidence)

            merged_records.append(merged_row.to_dict())
        else:
            merged_records.append(window[0].to_dict())

        i += len(window)

    merged_df = pd.DataFrame.from_records(merged_records, columns=df.columns)
    return merged_df.sort_values(by=["input", "start_time", "end_time", "species_name"])


def _random_detections(seed: int, dtype: str, hop_size: float = 3.0) -> pd.DataFrame:
    rng = np.random.default_rng([seed, round(hop_size * 10)])
    gaps = [-1.5, 0.0, hop_size - 0.5, hop_size, hop_size + 0.5, 2 * hop_size]
    rows = []

    for input_file in ("b.wav", "a.wav", "c.wav"):
        species_names = ["sp_c", "sp_a", "sp_b", "sp_d"]
        n_species = rng.integers(1, 5)

        for species in rng.choice(species_names, n_species, replace=False).tolist():
            start = 1.5 * rng.integers(0, 4)

            for _ in range(rng.integers(1, 25)):
                rows.append((input_file, species, start, start + 3.0, rng.random()))
                start += 3.0 + rng.choice(gaps, p=[0.1, 0.3, 0.1, 0.3, 0.1, 0.1])

    df = pd.DataFrame(
        rows, columns=["input", "species_name", "start_time", "end_time", "confidence"]
    )
    duplicates = df.sample(frac=0.1, random_state=rng)
    duplicates["confidence"] = rng.random(len(duplicates))
    df = pd.concat([df, duplicates], ignore_index=True)
    df["row"] = np.arange(len(df))

    ulp_type = np.float16 if dtype == "float16" else np.float32
    starts = df["start_time"].to_numpy(dtype=ulp_type)
    nudges = rng.choice([-np.inf, 0.0, 0.0, np.inf], size=len(df)).astype(ulp_type)
    df["start_time"] = np.where(nudges == 0, starts, np.nextafter(starts, nudges))

    for column in ("start_time", "end_time", "confidence"):
        df[column] = df[column].astype(dtype)

    return df.sample(frac=1, random_state=rng)


def _detections(rows):
    return pd.DataFrame(
        rows, columns=["input", "species_name", "start_time", "end_time", "confidence"]
    )


@pytest.mark.parametrize(
    ("merge_consecutive", "dtype", "hop_size"),
    [
        *(
            (merge, dtype, 3.0)
            for merge in (1, 3)
            for dtype in ("float16", "float32", "float64")
        ),
        (3, "float32", 1.5),
        (3, "float32", 2.9),
    ],
)
def test_merge_consecutive_segments_matches_reference(
    merge_consecutive, dtype, hop_size
):
    df = _random_detections(seed=merge_consecutive, dtype=dtype, hop_size=hop_size)

    pd.testing.assert_frame_equal(
        _merge_consecutive_segments(df, merge_consecutive, hop_size=hop_size),
        _merge_reference(df, merge_consecutive, hop_size=hop_size),
        check_exact=True,
    )


def test_merge_consecutive_segments_matches_reference_without_confidence():
    df = _random_detections(seed=3, dtype="float32").drop(columns="confidence")

    pd.testing.assert_frame_equal(
        _merge_consecutive_segments(df, merge_consecutive=3, hop_size=3.0),
        _merge_reference(df, 3, hop_size=3.0),
        check_exact=True,
    )


@pytest.mark.parametrize(
    ("hop_size", "ulps", "expected_rows"),
    [(3.0, -1, 1), (3.0, 0, 1), (3.0, 1, 2), (2.9, -1, 1), (2.9, 0, 2), (2.9, 1, 2)],
)
def test_merge_consecutive_segments_gap_boundary(hop_size, ulps, expected_rows):
    start = np.float32(3.0) + np.float32(hop_size)

    for _ in range(abs(ulps)):
        start = np.nextafter(start, np.float32(np.sign(ulps) * np.inf))

    df = pd.DataFrame(
        {
            "input": ["file.wav", "file.wav"],
            "species_name": ["species", "species"],
            "start_time": np.array([0.0, start], dtype="float32"),
            "end_time": np.array([3.0, start + 3], dtype="float32"),
            "confidence": np.array([0.25, 0.75], dtype="float32"),
        }
    )

    merged = _merge_consecutive_segments(df, merge_consecutive=2, hop_size=hop_size)

    assert len(merged) == expected_rows
    pd.testing.assert_frame_equal(
        merged, _merge_reference(df, 2, hop_size=hop_size), check_exact=True
    )


def test_merge_consecutive_segments_keeps_inputs_apart():
    df = pd.DataFrame(
        {
            "input": ["a.wav", "b.wav"],
            "species_name": ["species", "species"],
            "start_time": [0.0, 3.0],
            "end_time": [3.0, 6.0],
            "confidence": [0.5, 0.7],
        }
    )

    merged = _merge_consecutive_segments(df, merge_consecutive=2, hop_size=3.0)

    assert merged["input"].tolist() == ["a.wav", "b.wav"]


def test_merge_consecutive_segments_matches_reference_for_cancelling_confidences():
    df = pd.DataFrame(
        {
            "input": ["file.wav"] * 4,
            "species_name": ["species"] * 4,
            "start_time": [0.0, 3.0, 6.0, 9.0],
            "end_time": [3.0, 6.0, 9.0, 12.0],
            "confidence": [1.0, 1e100, 1.0, -1e100],
        }
    )

    pd.testing.assert_frame_equal(
        _merge_consecutive_segments(df, merge_consecutive=4, hop_size=3.0),
        _merge_reference(df, 4, hop_size=3.0),
        check_exact=True,
    )


def test_merge_consecutive_segments_sorts_by_input_and_time_even_when_nothing_merges():
    df = pd.DataFrame(
        {
            "input": ["b.wav", "a.wav", "a.wav"],
            "species_name": ["species_a", "species_a", "species_b"],
            "start_time": [0.0, 30.0, 0.0],
            "end_time": [3.0, 33.0, 3.0],
            "confidence": [0.5, 0.6, 0.7],
        }
    )

    merged = _merge_consecutive_segments(df, merge_consecutive=2, hop_size=3.0)
    unmerged = _merge_consecutive_segments(df, merge_consecutive=1, hop_size=3.0)

    assert merged[["input", "start_time"]].to_numpy().tolist() == [
        ["a.wav", 0.0],
        ["a.wav", 30.0],
        ["b.wav", 0.0],
    ]
    assert unmerged["input"].tolist() == ["b.wav", "a.wav", "a.wav"]


def test_merge_consecutive_segments_merges_expected_rows():
    df = _detections(
        [
            ("file_a.wav", "species_alpha", 0.0, 2.0, 0.4),
            ("file_a.wav", "species_alpha", 2.0, 4.0, 0.6),
            ("file_a.wav", "species_alpha", 5.0, 6.0, 0.8),
            ("file_b.wav", "species_beta", 0.0, 1.0, 0.5),
            ("file_b.wav", "species_beta", 1.0, 2.0, 0.7),
        ]
    )

    merged = _merge_consecutive_segments(df, merge_consecutive=2, hop_size=2.0)

    assert merged[["input", "start_time", "end_time"]].to_numpy().tolist() == [
        ["file_a.wav", 0.0, 4.0],
        ["file_a.wav", 5.0, 6.0],
        ["file_b.wav", 0.0, 2.0],
    ]
    assert merged["confidence"].tolist() == pytest.approx([0.5, 0.8, 0.6])


def test_merge_consecutive_segments_merges_up_to_the_window_size():
    # Five consecutive segments at merge_consecutive=3: a merged 3, then a merged 2.
    df = _detections(
        [
            ("file.wav", "species", float(i), float(i + 1), 0.1 * (i + 1))
            for i in range(5)
        ]
    )

    merged = _merge_consecutive_segments(df, merge_consecutive=3, hop_size=1.0)

    assert merged[["start_time", "end_time"]].to_numpy().tolist() == [
        [0.0, 3.0],
        [3.0, 5.0],
    ]
    assert merged["confidence"].tolist() == pytest.approx([0.2, 0.45])


def test_merge_consecutive_segments_merges_partial_run_shorter_than_window():
    df = _detections(
        [
            ("file.wav", "species", 0.0, 1.0, 0.25),
            ("file.wav", "species", 1.0, 2.0, 0.75),
        ]
    )

    merged = _merge_consecutive_segments(df, merge_consecutive=3, hop_size=1.0)

    assert merged[["start_time", "end_time"]].to_numpy().tolist() == [[0.0, 2.0]]
    assert merged["confidence"].tolist() == pytest.approx([0.5])


def test_merge_consecutive_segments_no_merge_when_merge_value_is_one():
    df = _detections(
        [("file.wav", "species", 0.0, 1.0, 0.2), ("file.wav", "species", 1.0, 2.0, 0.3)]
    )

    assert _merge_consecutive_segments(df, merge_consecutive=1).equals(df)


def test_merge_consecutive_segments_returns_empty_for_empty_dataframe():
    df = pd.DataFrame(
        columns=["input", "species_name", "start_time", "end_time", "confidence"]
    )

    merged = _merge_consecutive_segments(df, merge_consecutive=3)

    assert merged.empty


def test_merge_consecutive_segments_raises_when_columns_missing():
    df = pd.DataFrame([{"foo": "bar"}])

    with pytest.raises(ValueError, match="missing required columns"):
        _merge_consecutive_segments(df, merge_consecutive=2)


def test_merge_consecutive_segments_raises_for_non_numeric_time_values():
    df = _detections(
        [("file.wav", "species", "a", "b", 0.1), ("file.wav", "species", "c", "d", 0.2)]
    )

    with pytest.raises(ValueError, match="Time columns must be numeric"):
        _merge_consecutive_segments(df, merge_consecutive=2)
