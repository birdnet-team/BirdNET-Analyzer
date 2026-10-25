import logging
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from birdnet_analyzer.cli import segments_parser
from birdnet_analyzer.segments.core import segments
from birdnet_analyzer.segments.utils import (
    _find_segments,
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

    assert _find_segments(None, str(rfile)) == [expected]
    assert _find_segments(audio, str(rfile)) == [expected]


def test_kaleidoscope_rows_with_an_empty_indir_are_anchored_at_the_root(tmp_path):
    rfile = tmp_path / "BirdNET_Kaleidoscope.csv"
    rfile.write_text(
        "INDIR,FOLDER,IN FILE,OFFSET,DURATION,TOP1MATCH,TOP1DIST\n"
        ",audio,x.wav,3.0,3.0,American Robin,0.8\n",
        encoding="utf-8",
    )

    assert [s["audio"] for s in _find_segments(None, str(rfile))] == [
        os.path.join(os.sep, "audio", "x.wav")
    ]


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


def test_parse_folders_matches_the_birdnet_infix_case_insensitively(tmp_path):
    audio = tmp_path / "X.wav"
    result = tmp_path / "X.birdnet.results.csv"
    audio.touch()
    result.touch()

    assert parse_folders(str(tmp_path), str(tmp_path)) == [
        {"audio": str(audio), "result": str(result)}
    ]


def test_parse_folders_ignores_the_case_of_stems_on_windows_only(tmp_path):
    audio = tmp_path / "rec.wav"
    result = tmp_path / "REC.BirdNET.results.csv"
    audio.touch()
    result.touch()
    pairs = [{"audio": str(audio), "result": str(result)}]

    assert parse_folders(str(tmp_path), str(tmp_path)) == (
        pairs if os.name == "nt" else []
    )


@pytest.mark.parametrize(
    ("name", "text"),
    [
        (
            "rec.BirdNET.selection.table.txt",
            RAVEN_HEADER + "\n1\tSpectrogram 1\t1\t0\t3\t0\t15000\tAmerican Robin\t"
            "Turdus migratorius\tamerob\t0.5\t{audio}\t0\n",
        ),
        (
            "rec.BirdNET.results.csv",
            "Start (s),End (s),Scientific name,Common name,Confidence,File\n"
            "0.0,3.0,Turdus migratorius,American Robin,0.5,{audio}\n",
        ),
        (
            "rec.BirdNET.results.kaleidoscope.csv",
            "INDIR,FOLDER,IN FILE,OFFSET,DURATION,TOP1MATCH,TOP1DIST\n"
            "{indir},{folder},rec.wav,0.0,3.0,American Robin,0.5\n",
        ),
        (
            "rec.BirdNET.results.txt",
            "0.0\t3.0\tTurdus migratorius_American Robin\t0.5\n",
        ),
    ],
    ids=["table", "csv", "kaleidoscope", "audacity"],
)
def test_blank_lines_in_a_result_file_are_skipped(tmp_path, name, text):
    audio = tmp_path / "site" / "rec.wav"
    rfile = tmp_path / name
    first, _, rest = text.partition("\n")
    rfile.write_text(
        f"{first}\n\n{rest}\n  \n".format(
            audio=audio, indir=tmp_path, folder=audio.parent.name
        ),
        encoding="utf-8",
    )

    assert [
        (s["start"], s["end"], s["species"])
        for _, segs in parse_files([{"audio": str(audio), "result": str(rfile)}])
        for s in segs
    ] == [(0.0, 3.0, "American Robin")]


def _flist(rfile, audio, combined):
    if combined:
        return [{"isCombinedFile": True, "result": str(rfile)}]

    return [{"audio": audio, "result": str(rfile)}]


@pytest.mark.parametrize("combined", [True, False])
def test_a_header_only_table_holds_no_segment(tmp_path, combined):
    rfile = tmp_path / "rec.BirdNET.selection.table.txt"
    rfile.write_text(RAVEN_HEADER + "\n", encoding="utf-8")

    flist = _flist(rfile, str(tmp_path / "rec.wav"), combined)

    assert parse_files(flist, min_conf=0) == []


def _empty_file_summaries(caplog):
    return [
        r.getMessage().split()
        for r in caplog.records
        if r.levelno == logging.INFO and "empty" in r.getMessage()
    ]


def test_an_empty_combined_result_file_is_skipped_with_a_warning(tmp_path, caplog):
    rfile = tmp_path / "BirdNET_CombinedTable.csv"
    rfile.touch()
    caplog.set_level(logging.INFO)

    assert parse_files([{"isCombinedFile": True, "result": str(rfile)}]) == []
    assert [r.levelno for r in caplog.records if str(rfile) in r.getMessage()] == [
        logging.WARNING
    ]
    assert _empty_file_summaries(caplog) == []


def test_an_empty_per_file_result_holds_no_detections(tmp_path, caplog):
    rfile = tmp_path / "rec.BirdNET.results.txt"
    rfile.touch()
    flist = [{"audio": str(tmp_path / "rec.wav"), "result": str(rfile)}]
    caplog.set_level(logging.INFO)

    assert parse_files(flist) == []
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert [("1" in words) for words in _empty_file_summaries(caplog)] == [True]


@pytest.mark.parametrize("n_empty", [0, 2])
def test_empty_per_file_results_are_counted_in_one_line(tmp_path, caplog, n_empty):
    flist = []

    for i in range(n_empty + 1):
        rfile = tmp_path / f"rec{i}.BirdNET.results.txt"
        rfile.write_text(
            "" if i else "0.0\t3.0\tTurdus migratorius_American Robin\t0.5\n",
            encoding="utf-8",
        )
        flist.append({"audio": str(tmp_path / f"rec{i}.wav"), "result": str(rfile)})

    caplog.set_level(logging.INFO)

    result = parse_files(flist)

    assert [(audio, len(segs)) for audio, segs in result] == [(flist[0]["audio"], 1)]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert [(str(n_empty) in words) for words in _empty_file_summaries(caplog)] == (
        [True] if n_empty else []
    )


@pytest.mark.parametrize(
    ("label", "species"),
    [
        ("Turdus migratorius_American Robin", "American Robin"),
        ("Turdus migratorius, American Robin", "American Robin"),
        ("Cetacea_Whales, Dolphins and Porpoises", "Whales, Dolphins and Porpoises"),
        ("Turdus merula", "Turdus merula"),
    ],
    ids=["current", "legacy", "comma-in-common-name", "no-common-name"],
)
def test_audacity_labels_are_read(tmp_path, label, species):
    audio = str(tmp_path / "rec.wav")
    rfile = tmp_path / "rec.BirdNET.results.txt"
    rfile.write_text(f"3.0\t6.0\t{label}\t0.5\n", encoding="utf-8")

    assert parse_files([{"audio": audio, "result": str(rfile)}]) == [
        (
            audio,
            [
                {
                    "audio": audio,
                    "start": 3.0,
                    "end": 6.0,
                    "species": species,
                    "confidence": 0.5,
                }
            ],
        )
    ]


@pytest.mark.parametrize("combined", [True, False])
def test_the_nocall_rows_of_2_4_tables_are_no_segments(tmp_path, combined):
    audio = str(tmp_path / "rec.wav")
    rfile = tmp_path / "rec.BirdNET.selection.table.txt"
    rfile.write_text(
        f"{RAVEN_HEADER}\n"
        "1\tSpectrogram 1\t1\t0\t3\t0\t15000\tnocall\tnocall\tnocall\t1.0\t"
        f"{audio}\t0\n"
        f"2\tSpectrogram 1\t1\t3\t6\t0\t15000\tAmerican Robin\tTurdus migratorius\t"
        f"amerob\t0.5\t{audio}\t3\n",
        encoding="utf-8",
    )

    result = parse_files(_flist(rfile, audio, combined))

    assert [s["species"] for _, segs in result for s in segs] == ["American Robin"]


def test_rows_of_a_combined_table_without_an_audio_file_are_no_segments(tmp_path):
    audio = str(tmp_path / "rec.wav")
    rfile = tmp_path / "BirdNET_CombinedTable.csv"
    rfile.write_text(
        "Start (s),End (s),Scientific name,Common name,Confidence,File\n"
        "0.0,3.0,Turdus migratorius,American Robin,0.5,\n"
        f"3.0,6.0,Turdus migratorius,American Robin,0.5,{audio}\n",
        encoding="utf-8",
    )

    result = parse_files([{"isCombinedFile": True, "result": str(rfile)}])

    assert [(a, [s["start"] for s in segs]) for a, segs in result] == [(audio, [3.0])]


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
