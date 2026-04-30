import pandas as pd
import pytest

from birdnet_analyzer.analyze.core import _merge_consecutive_segments


def test_merge_consecutive_segments_merges_expected_rows():
    df = pd.DataFrame(
        [
            {
                "input": "file_a.wav",
                "species_name": "species_alpha",
                "start_time": 0.0,
                "end_time": 2.0,
                "confidence": 0.4,
            },
            {
                "input": "file_a.wav",
                "species_name": "species_alpha",
                "start_time": 2.0,
                "end_time": 4.0,
                "confidence": 0.6,
            },
            {
                "input": "file_a.wav",
                "species_name": "species_alpha",
                "start_time": 5.0,
                "end_time": 6.0,
                "confidence": 0.8,
            },
            {
                "input": "file_b.wav",
                "species_name": "species_beta",
                "start_time": 0.0,
                "end_time": 1.0,
                "confidence": 0.5,
            },
            {
                "input": "file_b.wav",
                "species_name": "species_beta",
                "start_time": 1.0,
                "end_time": 2.0,
                "confidence": 0.7,
            },
        ]
    )

    merged = _merge_consecutive_segments(df, merge_consecutive=2, hop_size=2.0)

    assert len(merged) == 3

    merged_records = merged.to_dict("records")

    assert merged_records[0]["input"] == "file_a.wav"
    assert merged_records[0]["start_time"] == 0.0
    assert merged_records[0]["end_time"] == 4.0
    assert merged_records[0]["confidence"] == pytest.approx(0.5)

    assert merged_records[1]["input"] == "file_a.wav"
    assert merged_records[1]["start_time"] == 5.0
    assert merged_records[1]["end_time"] == 6.0
    assert merged_records[1]["confidence"] == pytest.approx(0.8)

    assert merged_records[2]["input"] == "file_b.wav"
    assert merged_records[2]["start_time"] == 0.0
    assert merged_records[2]["end_time"] == 2.0
    assert merged_records[2]["confidence"] == pytest.approx(0.6)


def test_merge_consecutive_segments_requires_full_window():
    df = pd.DataFrame(
        [
            {
                "input": "file.wav",
                "species_name": "species",
                "start_time": 0.0,
                "end_time": 1.0,
                "confidence": 0.25,
            },
            {
                "input": "file.wav",
                "species_name": "species",
                "start_time": 1.0,
                "end_time": 2.0,
                "confidence": 0.75,
            },
            {
                "input": "file.wav",
                "species_name": "species",
                "start_time": 2.0,
                "end_time": 3.0,
                "confidence": 0.5,
            },
        ]
    )
    merged = _merge_consecutive_segments(df, merge_consecutive=3, hop_size=1.0)

    assert len(merged) == 1

    record = merged.to_dict("records")[0]
    assert record["start_time"] == 0.0
    assert record["end_time"] == 3.0
    assert record["confidence"] == pytest.approx((0.25 + 0.75 + 0.5) / 3)


def test_merge_consecutive_segments_handles_float16_time_columns():
    df = pd.DataFrame(
        [
            {
                "input": "file.wav",
                "species_name": "species",
                "start_time": 0.0,
                "end_time": 1.0,
                "confidence": 0.1,
            },
            {
                "input": "file.wav",
                "species_name": "species",
                "start_time": 1.0,
                "end_time": 2.0,
                "confidence": 0.3,
            },
        ]
    )

    df["start_time"] = df["start_time"].astype("float16")
    df["end_time"] = df["end_time"].astype("float16")

    merged = _merge_consecutive_segments(df, merge_consecutive=2, hop_size=1.0)

    assert len(merged) == 1
    record = merged.to_dict("records")[0]
    assert record["start_time"] == 0.0
    assert record["end_time"] == 2.0
    assert record["confidence"] == pytest.approx(0.2)


def test_merge_consecutive_segments_no_merge_when_merge_value_is_one():
    df = pd.DataFrame(
        [
            {
                "input": "file.wav",
                "species_name": "species",
                "start_time": 0.0,
                "end_time": 1.0,
                "confidence": 0.2,
            },
            {
                "input": "file.wav",
                "species_name": "species",
                "start_time": 1.0,
                "end_time": 2.0,
                "confidence": 0.3,
            },
        ]
    )

    merged = _merge_consecutive_segments(df, merge_consecutive=1)

    assert merged.equals(df)


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
    df = pd.DataFrame(
        [
            {
                "input": "file.wav",
                "species_name": "species",
                "start_time": "a",
                "end_time": "b",
                "confidence": 0.1,
            },
            {
                "input": "file.wav",
                "species_name": "species",
                "start_time": "c",
                "end_time": "d",
                "confidence": 0.2,
            },
        ]
    )

    with pytest.raises(ValueError, match="Time columns must be numeric"):
        _merge_consecutive_segments(df, merge_consecutive=2)
