from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

import birdnet_analyzer.config as cfg
from birdnet_analyzer.analyze import POSSIBLE_ADDITIONAL_COLUMNS
from birdnet_analyzer.analyze.core import (
    save_as_audacity,
    save_as_csv,
    save_as_kaleidoscope,
    save_as_parquet,
    save_as_rtable,
)
from birdnet_analyzer.segments.utils import (
    _find_segments,
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


def _wav(path: Path, seconds: float) -> str:
    sf.write(path, np.zeros(int(seconds * 48000), dtype="float32"), 48000)
    return str(path)


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


COLUMN_VALUES = {
    "lat": 50.83,
    "lon": 12.92,
    "week": 20,
    "overlap": 0.5,
    "min_conf": 0.25,
    "sensitivity": 1.0,
    "species_list_file": "species_list.txt",
    "model_path": "models/BirdNET_v3.0.tflite",
}


def test_csv_and_parquet_results_have_the_same_columns(tmp_path):
    detections = _detections(
        [(SOUNDSCAPE, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5)]
    )
    additional_columns = list(reversed(POSSIBLE_ADDITIONAL_COLUMNS))

    save_as_csv(detections, tmp_path / "r.csv", additional_columns, **COLUMN_VALUES)
    save_as_parquet(
        detections, tmp_path / "r.parquet", additional_columns, **COLUMN_VALUES
    )

    columns = list(_read(tmp_path / "r.csv").columns)
    assert columns == [
        "Start (s)",
        "End (s)",
        "Scientific name",
        "Common name",
        "Confidence",
        "File",
        *additional_columns,
    ]
    assert list(_read(tmp_path / "r.parquet").columns) == columns


@pytest.mark.parametrize(
    "column_values",
    [COLUMN_VALUES, dict.fromkeys(COLUMN_VALUES)],
    ids=["values", "no-values"],
)
def test_an_empty_parquet_result_has_the_schema_of_a_non_empty_one(
    tmp_path, column_values
):
    detections = _detections(
        [(SOUNDSCAPE, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5)]
    )

    for name, df in (("one", detections), ("none", detections.iloc[:0])):
        save_as_parquet(
            df,
            tmp_path / f"{name}.parquet",
            POSSIBLE_ADDITIONAL_COLUMNS,
            **column_values,
        )

    one, none = (pq.read_table(tmp_path / f"{n}.parquet") for n in ("one", "none"))
    assert none.schema.remove_metadata() == one.schema.remove_metadata()
    assert pa.concat_tables([one, none]).num_rows == 1


def test_a_species_list_path_is_written_as_text(tmp_path):
    detections = _detections(
        [(SOUNDSCAPE, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5)]
    )
    species_list = Path("lists") / "species_list.txt"

    for save, name in ((save_as_csv, "r.csv"), (save_as_parquet, "r.parquet")):
        save(
            detections,
            tmp_path / name,
            ["species_list"],
            species_list_file=species_list,
        )

        assert _read(tmp_path / name)["species_list"].tolist() == [str(species_list)]


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

    assert [(s["audio"], s["species"]) for s in _find_segments(None, str(result))] == [
        (afile, "Whales, Dolphins and Porpoises")
    ]
    assert [s["species"] for s in _find_segments(afile, str(result))] == [
        "Whales, Dolphins and Porpoises"
    ]


def test_raven_times_are_offset_by_the_durations_of_the_preceding_files(tmp_path):
    b, a, c = (
        _wav(tmp_path / f"{name}.wav", seconds)
        for name, seconds in (("b", 6), ("a", 9.1), ("c", 12))
    )
    rows = [
        (b, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5),
        (b, 3.0, 6.0, "Turdus migratorius_American Robin", 0.5),
        (a, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5),
        (c, 3.0, 6.0, "Turdus migratorius_American Robin", 0.5),
    ]
    output = tmp_path / cfg.OUTPUT_RAVEN_FILENAME

    _save_as_rtable(_detections(rows), output)

    table = _read(output)
    assert table["Begin Time (s)"].tolist() == [0.0, 3.0, 6.0, 18.1]
    assert table["End Time (s)"].tolist() == [3.0, 6.0, 9.0, 21.1]
    assert table["File Offset (s)"].tolist() == [0.0, 3.0, 0.0, 3.0]


def test_a_combined_raven_table_is_read_back_by_segments(tmp_path):
    a = _wav(tmp_path / "a.wav", 9)
    b = _wav(tmp_path / "b.wav", 6)
    rows = [
        (a, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5),
        (b, 3.0, 6.0, "Cardinalis cardinalis_Northern Cardinal", 0.75),
    ]
    output = tmp_path / cfg.OUTPUT_RAVEN_FILENAME

    _save_as_rtable(_detections(rows), output)

    segments = parse_files(
        [{"isCombinedFile": True, "result": str(output)}],
        collection_mode="confidence",
    )
    assert sorted(
        (s["audio"], s["start"], s["end"]) for _, segs in segments for s in segs
    ) == [(a, 0.0, 3.0), (b, 3.0, 6.0)]


def test_raven_table_accepts_a_categorical_input_column(tmp_path):
    a = _wav(tmp_path / "a.wav", 6)
    b = _wav(tmp_path / "b.wav", 9)
    rows = [
        (a, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5),
        (a, 3.0, 6.0, "Turdus migratorius_American Robin", 0.5),
        (b, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5),
    ]
    output = tmp_path / cfg.OUTPUT_RAVEN_FILENAME

    _save_as_rtable(_detections(rows).astype({"input": "category"}), output)

    table = _read(output)
    assert table["Begin Path"].tolist() == [a, a, b]
    assert table["Begin Time (s)"].tolist() == [0.0, 3.0, 6.0]


def test_raven_times_of_a_file_share_one_offset_wherever_its_rows_are(tmp_path):
    a = _wav(tmp_path / "a.wav", 6)
    b = _wav(tmp_path / "b.wav", 9)
    rows = [
        (a, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5),
        (b, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5),
        (a, 3.0, 6.0, "Turdus migratorius_American Robin", 0.5),
    ]
    output = tmp_path / cfg.OUTPUT_RAVEN_FILENAME

    _save_as_rtable(_detections(rows), output)

    table = _read(output)
    assert table["Begin Time (s)"].tolist() == [0.0, 6.0, 3.0]
    assert table["End Time (s)"].tolist() == [3.0, 9.0, 6.0]


def test_kaleidoscope_indir_of_a_file_below_the_root_is_the_root(tmp_path):
    afile = str(Path(tmp_path.anchor, "audio", "x.wav"))
    result = tmp_path / cfg.OUTPUT_KALEIDOSCOPE_FILENAME

    save_as_kaleidoscope(
        _detections([(afile, 0.0, 3.0, "Turdus migratorius_American Robin", 0.5)]),
        result,
    )

    assert pd.read_csv(result, keep_default_na=False)["INDIR"].tolist() == [
        tmp_path.anchor
    ]
    assert [s["audio"] for s in _find_segments(None, str(result))] == [afile]
