from pathlib import Path

import pandas as pd
import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.analyze.core import (
    save_as_audacity,
    save_as_csv,
    save_as_kaleidoscope,
    save_as_parquet,
    save_as_rtable,
)
from birdnet_analyzer.segments.utils import (
    _find_segments,
    _find_segments_from_combined,
    parse_files,
    parse_folders,
)

SOUNDSCAPE = str(
    Path(__file__).parents[2] / "birdnet_analyzer" / "example" / "soundscape.wav"
)


def _detections(rows=()) -> pd.DataFrame:
    return pd.DataFrame(
        list(rows),
        columns=["input", "start_time", "end_time", "species_name", "confidence"],
    ).astype(
        {
            "input": object,
            "start_time": "float32",
            "end_time": "float32",
            "species_name": object,
            "confidence": "float32",
        }
    )


def _save_as_rtable(df: pd.DataFrame, output: Path):
    save_as_rtable(df, 0, 15000, 0, 15000, 1.0, output)


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(path, sep="\t" if path.suffix == ".txt" else ",")


@pytest.mark.parametrize(
    ("write", "filename"),
    [
        (_save_as_rtable, cfg.OUTPUT_RAVEN_FILENAME),
        (save_as_csv, cfg.OUTPUT_CSV_FILENAME),
        (save_as_kaleidoscope, cfg.OUTPUT_KALEIDOSCOPE_FILENAME),
        (save_as_parquet, cfg.OUTPUT_PARQUET_FILENAME),
    ],
)
def test_without_detections_a_result_file_holds_only_its_header(
    tmp_path, write, filename
):
    with_detection = tmp_path / "one" / filename
    without_detections = tmp_path / "none" / filename

    write(
        _detections([(SOUNDSCAPE, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5)]),
        with_detection,
    )
    write(_detections(), without_detections)

    table = _read(without_detections)
    assert table.empty
    assert list(table.columns) == list(_read(with_detection).columns)


def test_without_detections_the_audacity_labels_are_empty(tmp_path):
    output = tmp_path / cfg.OUTPUT_AUDACITY_FILENAME

    save_as_audacity(_detections(), output)

    assert output.read_text() == ""


def test_kaleidoscope_results_are_read_back_by_segments(tmp_path):
    first = str(tmp_path / "site_a" / "x.wav")
    second = str(tmp_path / "site_b" / "y.wav")
    rows = [
        (first, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5),
        (first, 3.0, 6.0, "Cardinalis cardinalis_Northern Cardinal", 0.75),
        (second, 6.0, 9.0, "Turdus migratorius_American Robin", 0.875),
    ]
    expected = [
        (afile, start, end, species.split("_", 1)[1], confidence)
        for afile, start, end, species, confidence in rows
    ]
    result = tmp_path / cfg.OUTPUT_KALEIDOSCOPE_FILENAME

    save_as_kaleidoscope(_detections(rows), result)

    assert (
        result.read_text().splitlines()[0]
        == "INDIR,FOLDER,IN FILE,OFFSET,DURATION,TOP1MATCH,TOP1DIST"
    )

    combined = parse_files(
        parse_folders(str(tmp_path), str(tmp_path)), collection_mode="confidence"
    )
    assert sorted(
        (s["audio"], s["start"], s["end"], s["species"], s["confidence"])
        for _, segments in combined
        for s in segments
    ) == sorted(expected)

    per_file = _find_segments(first, str(result))
    assert [
        (s["start"], s["end"], s["species"], s["confidence"]) for s in per_file
    ] == [row[1:] for row in expected]


def test_a_label_without_a_common_name_is_kept_whole(tmp_path):
    detections = _detections([(SOUNDSCAPE, 0.0, 3.0, "Turdus merula", 0.5)])

    save_as_csv(detections, tmp_path / "results.csv")
    save_as_kaleidoscope(detections, tmp_path / "kaleidoscope.csv")

    row = pd.read_csv(tmp_path / "results.csv").iloc[0]
    assert (row["Scientific name"], row["Common name"]) == (
        "Turdus merula",
        "Turdus merula",
    )
    assert pd.read_csv(tmp_path / "kaleidoscope.csv")["TOP1MATCH"].tolist() == [
        "Turdus merula"
    ]


@pytest.mark.parametrize(
    ("write", "filename"),
    [
        (save_as_kaleidoscope, cfg.OUTPUT_KALEIDOSCOPE_FILENAME),
        (save_as_csv, cfg.OUTPUT_CSV_FILENAME),
    ],
)
def test_segments_read_fields_containing_commas(tmp_path, write, filename):
    afile = str(tmp_path / "site, north" / "x.wav")
    result = tmp_path / filename

    write(
        _detections([(afile, 0.0, 3.0, "Cetacea_Whales, Dolphins and Porpoises", 0.5)]),
        result,
    )

    assert [
        (s["audio"], s["species"]) for s in _find_segments_from_combined(str(result))
    ] == [(afile, "Whales, Dolphins and Porpoises")]
    assert [s["species"] for s in _find_segments(afile, str(result))] == [
        "Whales, Dolphins and Porpoises"
    ]
