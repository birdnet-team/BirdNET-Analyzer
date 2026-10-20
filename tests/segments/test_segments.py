import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from birdnet_analyzer.cli import segments_parser
from birdnet_analyzer.segments.core import segments
from birdnet_analyzer.segments.utils import (
    _find_segments,
    _find_segments_from_combined,
    parse_files,
    parse_folders,
)

RAVEN_HEADER = (
    "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\t"
    "High Freq (Hz)\tCommon Name\tScientific Name\tSpecies Code\tConfidence\t"
    "Begin Path\tFile Offset (s)"
)


@pytest.fixture
def setup_test_environment():
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")
    results_dir = os.path.join(test_dir, "results")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    file_list = [
        {
            "audio": os.path.join(input_dir, "audio1.wav"),
            "result": os.path.join(results_dir, "result1.csv"),
        },
        {
            "audio": os.path.join(input_dir, "audio2.wav"),
            "result": os.path.join(results_dir, "result2.csv"),
        },
    ]

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "results_dir": results_dir,
        "file_list": file_list,
    }

    shutil.rmtree(test_dir)


@patch("birdnet_analyzer.segments.utils.extract_segments")
@patch("birdnet_analyzer.segments.utils.parse_files")
@patch("birdnet_analyzer.segments.utils.parse_folders")
def test_segments_cli(
    mock_parse_folders: MagicMock,
    mock_parse_files: MagicMock,
    mock_extract_segments: MagicMock,
    setup_test_environment,
):
    env = setup_test_environment

    parser = segments_parser()
    args = parser.parse_args(
        [
            env["input_dir"],
            "--results",
            env["results_dir"],
            "--output",
            env["output_dir"],
            "--threads",
            "1",
        ]
    )

    mock_parse_files.return_value = [
        (
            env["file_list"][0]["audio"],
            [{"start": 0, "end": 3, "species": "sp1", "confidence": 0.9}],
        ),
        (
            env["file_list"][1]["audio"],
            [{"start": 0, "end": 3, "species": "sp2", "confidence": 0.8}],
        ),
    ]

    segments(**vars(args))

    mock_parse_folders.assert_called_once()
    mock_parse_files.assert_called_once()
    mock_extract_segments.assert_called()


@patch("birdnet_analyzer.segments.utils.extract_segments")
@patch("birdnet_analyzer.segments.utils.parse_files")
@patch("birdnet_analyzer.segments.utils.parse_folders")
def test_segments_cli_accepts_full_parser_surface(
    mock_parse_folders: MagicMock,
    mock_parse_files: MagicMock,
    mock_extract_segments: MagicMock,
    setup_test_environment,
):
    env = setup_test_environment

    parser = segments_parser()
    args = parser.parse_args(
        [
            env["input_dir"],
            "--results",
            env["results_dir"],
            "--output",
            env["output_dir"],
            "--max_segments",
            "5",
            "--seg_length",
            "4.5",
            "--max_conf",
            "0.9",
            "--collection_mode",
            "confidence",
            "--n_bins",
            "12",
            "--audio_speed",
            "1.25",
            "--threads",
            "1",
            "--min_conf",
            "0.15",
        ]
    )

    mock_parse_files.return_value = [
        (
            env["file_list"][0]["audio"],
            [{"start": 0, "end": 3, "species": "sp1", "confidence": 0.9}],
        ),
    ]

    segments(**vars(args))

    mock_parse_folders.assert_called_once_with(env["input_dir"], env["results_dir"])
    mock_parse_files.assert_called_once()
    parse_files_kwargs = mock_parse_files.call_args.kwargs
    assert parse_files_kwargs["max_segments"] == 5
    assert parse_files_kwargs["collection_mode"] == "confidence"
    assert parse_files_kwargs["n_bins"] == 12
    assert parse_files_kwargs["min_conf"] == 0.15
    assert parse_files_kwargs["max_conf"] == 0.9
    mock_extract_segments.assert_called_once()


def test_kaleidoscope_results_of_2x_releases_are_still_read(tmp_path):
    rfile = tmp_path / "BirdNET_Kaleidoscope.csv"
    rfile.write_text(
        "INDIR,FOLDER,IN FILE,OFFSET,DURATION,scientific_name,common_name,"
        "confidence,lat,lon,week,overlap,sensitivity\n"
        f"{tmp_path},site_a,x.wav,3.0,3.0,Turdus migratorius,American Robin,"
        "0.8000,-1.0000,-1.0000,-1,0.0,1.0\n",
        encoding="utf-8",
    )
    audio = os.path.join(str(tmp_path), "site_a", "x.wav")
    expected = {
        "audio": audio,
        "start": 3.0,
        "end": 6.0,
        "species": "Turdus migratorius",
        "confidence": 0.8,
    }

    assert _find_segments_from_combined(str(rfile)) == [expected]
    assert _find_segments(audio, str(rfile)) == [expected]


@pytest.mark.parametrize(
    ("audio_dir", "results_dir"),
    [
        (os.path.join("data", "audio"), os.path.join("data", "results")),
        (os.path.join("data", "audio", ""), os.path.join("data", "results")),
        (os.path.join("data", "audio"), os.path.join("data", "audio")),
    ],
    ids=["separate", "trailing-separator", "same-folder"],
)
@pytest.mark.parametrize("folder", ["site1", "tree"])
def test_parse_folders_pairs_every_audio_file_with_its_result(
    tmp_path, monkeypatch, audio_dir, results_dir, folder
):
    monkeypatch.chdir(tmp_path)
    expected = []

    for sub in ("", folder):
        audio = os.path.join(audio_dir, sub, "rec.wav")
        result = os.path.join(results_dir, sub, "rec.BirdNET.selection.table.txt")
        for path in (audio, result):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
        expected.append((audio, result))

    flist = parse_folders(audio_dir, results_dir)

    assert sorted((f["audio"], f["result"]) for f in flist) == sorted(expected)


@pytest.mark.parametrize(
    ("extension", "allowed"),
    [("TXT", ("txt", "csv")), ("TXT", ("TXT",)), ("txt", ("TXT",))],
)
def test_parse_folders_matches_result_extensions_case_insensitively(
    tmp_path, extension, allowed
):
    audio = tmp_path / "rec.wav"
    result = tmp_path / f"rec.BirdNET.selection.table.{extension}"
    audio.touch()
    result.touch()

    flist = parse_folders(str(tmp_path), str(tmp_path), allowed)

    assert flist == [{"audio": str(audio), "result": str(result)}]


@pytest.mark.parametrize("combined", [True, False])
def test_parse_files_drops_detections_above_max_conf(tmp_path, combined):
    audio = str(tmp_path / "rec.wav")
    rfile = tmp_path / "rec.BirdNET.selection.table.txt"
    rows = [
        f"{i}\tSpectrogram 1\t1\t{3 * i}\t{3 * i + 3}\t0\t15000\tAmerican Robin\t"
        f"Turdus migratorius\tamerob\t{confidence}\t{audio}\t{3 * i}"
        for i, confidence in enumerate([0.3, 0.6, 0.9])
    ]
    rfile.write_text("\n".join([RAVEN_HEADER, *rows]) + "\n", encoding="utf-8")
    flist = (
        [{"isCombinedFile": True, "result": str(rfile)}]
        if combined
        else [{"audio": audio, "result": str(rfile)}]
    )

    result = parse_files(flist, min_conf=0.25, max_conf=0.7)

    assert sorted(s["confidence"] for _, segs in result for s in segs) == [0.3, 0.6]
