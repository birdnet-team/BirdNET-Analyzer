import ntpath
import unicodedata

import numpy as np
import pytest

from birdnet_analyzer.evaluation.preprocessing.utils import (
    detect_columns,
    recording_key,
    table_columns,
)


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("/data/site_a/rec.wav", "rec"),
        ("rec.txt", "rec"),
        ("rec.BirdNET.selection.table.txt", "rec"),
        ("rec.BirdNET.results.csv", "rec"),
        ("rec.BirdNET.results.kaleidoscope.csv", "rec"),
        ("rec.BirdNET.txt", "rec"),
        ("rec.birdnet.selection.table.txt", "rec"),
        ("rec.wav.BirdNET.selection.table.txt", "rec"),
        ("2023.05.01_dawn.txt", "2023.05.01_dawn"),
        ("S19_20180214.Table.1.selections.txt", "S19_20180214"),
        ("rec.Table.12.selections.txt", "rec"),
        ("rec.wav.Table.1.selections.txt", "rec"),
        ("rec.WAV.Table.1.selections.txt", "rec"),
        ("rec.selections.txt", "rec"),
        ("rec.aiff.txt", "rec"),
        ("REC.Wav", "REC"),
        ("rec", "rec"),
        (".hidden", ".hidden"),
    ],
)
def test_a_result_table_and_its_recording_share_one_key(name, key):
    assert recording_key(name) == key


def test_a_recording_path_written_on_windows_gives_the_same_key_everywhere():
    assert recording_key(ntpath.join("field", "site_a", "rec.wav")) == "rec"


def test_a_name_composed_or_decomposed_gives_the_same_key():
    composed = unicodedata.normalize("NFC", "K\u00fcste M\u00e4rz.wav")
    decomposed = unicodedata.normalize("NFD", composed)

    assert decomposed != composed
    assert recording_key(decomposed) == recording_key(composed)


def test_missing_names_pass_through_the_key_extraction():
    assert recording_key(None) is None
    assert np.isnan(recording_key(np.nan))


def test_the_column_choices_are_the_columns_of_every_table_in_first_seen_order(
    tmp_path,
):
    (tmp_path / "site").mkdir()
    (tmp_path / ".cache").mkdir()
    (tmp_path / "a.txt").write_text("Class\tBegin Time (s)\nX\t0\n", encoding="utf-8")
    (tmp_path / "site" / "b.csv").write_text("Begin Time (s),Notes\n", encoding="utf-8")
    (tmp_path / "README.txt").write_text("Annotated by J. Doe\n", encoding="utf-8")
    (tmp_path / "empty.txt").write_bytes(b"")
    (tmp_path / "binary.txt").write_bytes(b"PK\x03\x04\x00\x00junk")
    (tmp_path / "birdnet.analyze-params.csv").write_text("Param,Value\n")
    (tmp_path / ".cache" / "c.txt").write_text("Hidden\n", encoding="utf-8")

    assert table_columns(str(tmp_path)) == [
        "Annotated by J. Doe",
        "Class",
        "Begin Time (s)",
        "Notes",
    ]
    assert table_columns(str(tmp_path / "a.txt")) == ["Class", "Begin Time (s)"]


@pytest.mark.parametrize(
    ("kind", "available", "detected"),
    [
        pytest.param(
            "prediction",
            [
                "Selection",
                "Begin Time (s)",
                "End Time (s)",
                "Common Name",
                "Scientific Name",
                "Species Code",
                "Confidence",
                "File Offset (s)",
                "Begin Path",
            ],
            {
                "Start Time": "Begin Time (s)",
                "End Time": "End Time (s)",
                "Class": "Common Name",
                "Confidence": "Confidence",
                "Recording": "Begin Path",
                "Duration": None,
            },
            id="birdnet selection table",
        ),
        pytest.param(
            "prediction",
            [
                "Start (s)",
                "End (s)",
                "Scientific name",
                "Common name",
                "Confidence",
                "File",
            ],
            {
                "Start Time": "Start (s)",
                "End Time": "End (s)",
                "Class": "Common name",
                "Confidence": "Confidence",
                "Recording": "File",
                "Duration": None,
            },
            id="birdnet csv",
        ),
        pytest.param(
            "annotation",
            [
                "Selection",
                "Begin Time (s)",
                "End Time (s)",
                "Species",
                "Begin File",
                "Begin Path",
                "File Duration (s)",
            ],
            {
                "Start Time": "Begin Time (s)",
                "End Time": "End Time (s)",
                "Class": "Species",
                "Recording": "Begin File",
                "Duration": "File Duration (s)",
            },
            id="raven",
        ),
        pytest.param(
            "annotation",
            ["begin time (s)", "END TIME (S)", "class", "Species"],
            {
                "Start Time": "begin time (s)",
                "End Time": "END TIME (S)",
                "Class": "Species",
                "Recording": None,
                "Duration": None,
            },
            id="exact matches before case-insensitive ones",
        ),
        pytest.param(
            "annotation",
            ["INDIR", "FOLDER", "IN FILE", "DURATION", "OFFSET", "MANUAL ID"],
            dict.fromkeys(["Start Time", "End Time", "Class", "Recording", "Duration"]),
            id="kaleidoscope",
        ),
    ],
)
def test_columns_are_detected_from_the_headers_of_common_tables(
    kind, available, detected
):
    assert list(detect_columns(kind, available).items()) == list(detected.items())
