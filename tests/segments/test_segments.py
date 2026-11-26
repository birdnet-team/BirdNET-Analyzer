import logging
import os
from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from birdnet_analyzer.cli import segments_parser
from birdnet_analyzer.segments import utils as segments_utils
from birdnet_analyzer.segments.core import segments
from birdnet_analyzer.segments.utils import (
    _find_segments,
    _species_folder,
    parse_files,
    parse_folders,
    relocate_audio,
)

RAVEN_HEADER = (
    "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\t"
    "High Freq (Hz)\tCommon Name\tScientific Name\tSpecies Code\tConfidence\t"
    "Begin Path\tFile Offset (s)"
)


def _raven_row(selection, start, end, common, scientific, code, confidence, audio):
    return (
        f"{selection}\tSpectrogram 1\t1\t{start}\t{end}\t0\t15000\t{common}\t"
        f"{scientific}\t{code}\t{confidence}\t{audio}\t{start}"
    )


@patch("birdnet_analyzer.segments.utils.extract_segments")
@patch("birdnet_analyzer.segments.utils.parse_files")
@patch("birdnet_analyzer.segments.utils.parse_folders")
def test_segments_cli_accepts_full_parser_surface(
    mock_parse_folders, mock_parse_files, mock_extract_segments, tmp_path
):
    audio, results, output = (
        str(tmp_path / name) for name in ("audio", "results", "output")
    )
    args = segments_parser().parse_args(
        [
            audio,
            "--results",
            results,
            "--output",
            output,
            "--max_segments",
            "5",
            "--max_percent",
            "12.5",
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
            os.path.join(audio, "rec.wav"),
            [{"start": 0, "end": 3, "species": "sp1", "confidence": 0.9}],
        ),
    ]

    segments(**vars(args))

    mock_parse_folders.assert_called_once_with(audio, results)
    mock_parse_files.assert_called_once()
    parse_files_kwargs = mock_parse_files.call_args.kwargs
    assert parse_files_kwargs["max_segments"] == 5
    assert parse_files_kwargs["max_percent"] == 12.5
    assert parse_files_kwargs["collection_mode"] == "confidence"
    assert parse_files_kwargs["n_bins"] == 12
    assert parse_files_kwargs["min_conf"] == 0.15
    assert parse_files_kwargs["max_conf"] == 0.9
    mock_extract_segments.assert_called_once()


def test_segments_are_cut_from_a_result_table_of_the_example_soundscape(tmp_path):
    audio_dir, results_dir, out = (
        tmp_path / name for name in ("audio", "results", "segments")
    )
    audio_dir.mkdir()
    results_dir.mkdir()
    audio = audio_dir / "soundscape.wav"
    sig, rate = sf.read("birdnet_analyzer/example/soundscape.wav")
    sf.write(audio, sig[: 9 * rate], rate)
    rows = [
        _raven_row(
            1, 0, 3, "American Robin", "Turdus migratorius", "amerob", 0.9, audio
        ),
        _raven_row(
            2, 3, 6, "Northern Cardinal", "Cardinalis cardinalis", "norcar", 0.8, audio
        ),
    ]
    (results_dir / "soundscape.BirdNET.selection.table.txt").write_text(
        "\n".join([RAVEN_HEADER, *rows]) + "\n", encoding="utf-8"
    )

    done = segments(
        str(audio_dir), str(out), str(results_dir), collection_mode="confidence"
    )

    assert done == [(str(audio), True)]
    clips = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.wav"))
    assert clips == [
        "American Robin/0.900_1_soundscape_0.00s_3.00s.wav",
        "Northern Cardinal/0.800_2_soundscape_3.00s_6.00s.wav",
    ]
    assert [sf.info(out / clip).duration for clip in clips] == [3.0, 3.0]


def test_a_species_name_no_folder_can_carry_still_gets_one_folder(tmp_path):
    audio_dir, results_dir, out = (
        tmp_path / name for name in ("audio", "results", "segments")
    )
    audio_dir.mkdir()
    results_dir.mkdir()
    audio = audio_dir / "soundscape.wav"
    sig, rate = sf.read("birdnet_analyzer/example/soundscape.wav")
    sf.write(audio, sig[: 9 * rate], rate)
    rows = [
        _raven_row(1, 0, 3, "Bontebok/Blesbok", "Damaliscus pygargus", "x", 0.9, audio),
        _raven_row(
            2, 3, 6, "vireó espetec?", "Pteruthius intermedius", "y", 0.8, audio
        ),
        _raven_row(3, 6, 9, "American Robin", "Turdus migratorius", "z", 0.7, audio),
    ]
    (results_dir / "soundscape.BirdNET.selection.table.txt").write_text(
        "\n".join([RAVEN_HEADER, *rows]) + "\n", encoding="utf-8"
    )

    done = segments(
        str(audio_dir), str(out), str(results_dir), collection_mode="confidence"
    )

    assert done == [(str(audio), True)]
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.wav")) == [
        "American Robin/0.700_3_soundscape_6.00s_9.00s.wav",
        "Bontebok-Blesbok/0.900_1_soundscape_0.00s_3.00s.wav",
        "vireó espetec-/0.800_2_soundscape_3.00s_6.00s.wav",
    ]


@pytest.mark.parametrize("char", [*'<>:"/\\|?*', "\x00", "\x1f"])
def test_no_character_a_folder_name_cannot_carry_is_kept(char):
    assert _species_folder(f"Aaa{char}bbb") == "Aaa-bbb"


@pytest.mark.parametrize(
    ("species", "folder"),
    [("Rævespurv sp.", "Rævespurv sp"), ("Name ", "Name"), ("Name. .", "Name")],
)
def test_a_folder_name_does_not_end_in_a_dot_or_a_space(species, folder):
    assert _species_folder(species) == folder


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


@pytest.mark.parametrize(
    "rfile", ["soundscape.BirdNET.results.r.csv", "BirdNET_RTable.csv"]
)
def test_segments_are_cut_from_an_r_table_of_1x_releases(tmp_path, rfile):
    audio_dir, results_dir, out = (
        tmp_path / name for name in ("audio", "results", "segments")
    )
    audio_dir.mkdir()
    results_dir.mkdir()
    audio = audio_dir / "soundscape.wav"
    sig, rate = sf.read("birdnet_analyzer/example/soundscape.wav")
    sf.write(audio, sig[: 9 * rate], rate)
    columns = "-1.0000,-1.0000,-1,0.0,1.0,0.1,None,BirdNET_GLOBAL_6K_V2.4_Model.tflite"
    (results_dir / rfile).write_text(
        "filepath,start,end,scientific_name,common_name,confidence,lat,lon,week,"
        "overlap,sensitivity,min_conf,species_list,model\n"
        f"{audio},0,3.0,Turdus migratorius,American Robin,0.9000,{columns}\n"
        f"{audio},3.0,6.0,Cardinalis cardinalis,Northern Cardinal,0.8000,{columns}\n",
        encoding="utf-8",
    )

    segments(str(audio_dir), str(out), str(results_dir), collection_mode="confidence")

    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.wav")) == [
        "American Robin/0.900_1_soundscape_0.00s_3.00s.wav",
        "Northern Cardinal/0.800_2_soundscape_3.00s_6.00s.wav",
    ]


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


def _species_table(tmp_path, detections):
    audio = str(tmp_path / "rec.wav")
    rfile = tmp_path / "rec.BirdNET.selection.table.txt"
    rows = [
        _raven_row(i, 3 * i, 3 * i + 3, species, species, "x", confidence, audio)
        for i, (species, confidence) in enumerate(detections, start=1)
    ]
    rfile.write_text("\n".join([RAVEN_HEADER, *rows]) + "\n", encoding="utf-8")

    return [{"audio": audio, "result": str(rfile)}]


def _per_species(result):
    return Counter(s["species"] for _, segs in result for s in segs)


@pytest.mark.parametrize("mode", ["random", "confidence"])
def test_a_percentage_keeps_that_share_of_each_species_rounded_up(tmp_path, mode):
    flist = _species_table(
        tmp_path, [("Robin", 0.5)] * 10 + [("Jay", 0.5)] * 3 + [("Wren", 0.5)]
    )

    result = parse_files(flist, max_segments=2, max_percent=25, collection_mode=mode)

    assert _per_species(result) == {"Robin": 3, "Jay": 1, "Wren": 1}


def test_a_percentage_that_gives_a_whole_number_is_not_rounded_up(tmp_path):
    flist = _species_table(tmp_path, [("Robin", 0.5)] * 375)

    result = parse_files(flist, max_percent=8.8)

    assert _per_species(result) == {"Robin": 33}


def test_a_percentage_counts_only_detections_in_the_confidence_range(tmp_path):
    flist = _species_table(tmp_path, [("Robin", 0.1)] * 6 + [("Robin", 0.5)] * 4)

    result = parse_files(flist, min_conf=0.25, max_percent=50)

    assert _per_species(result) == {"Robin": 2}


def test_a_balanced_percentage_takes_its_share_of_every_bin(tmp_path):
    flist = _species_table(tmp_path, [("Robin", 0.3)] * 8 + [("Robin", 0.95)] * 2)

    result = parse_files(flist, max_percent=25, collection_mode="balanced")

    assert sorted(s["confidence"] for _, segs in result for s in segs) == [
        0.3,
        0.3,
        0.95,
    ]


def test_a_random_percentage_is_not_only_the_top_scores(tmp_path):
    confidences = [round(0.26 + 0.007 * i, 3) for i in range(100)]
    flist = _species_table(tmp_path, [("Robin", c) for c in confidences])

    result = parse_files(flist, max_percent=10)

    picked = sorted(s["confidence"] for _, segs in result for s in segs)
    assert len(picked) == 10
    assert picked != confidences[-10:]


RAMP_PER_SECOND = 0.025


def _ramp(seconds, rate=48000):
    """A signal that rises steadily, so a sample's value tells its time."""
    return np.arange(seconds * rate, dtype=np.float32) * (RAMP_PER_SECOND / rate)


def _combined_table(results_dir, rows):
    results_dir.mkdir(exist_ok=True)
    (results_dir / "BirdNET_SelectionTable.txt").write_text(
        "\n".join(
            [
                RAVEN_HEADER,
                *(
                    _raven_row(i, start, end, "American Robin", "T. m.", "x", 0.9, path)
                    for i, (path, start, end) in enumerate(rows, start=1)
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("rate", "detection", "seg_length", "speed", "cut"),
    [
        (48000, (4, 7), 3.0, 1.0, (4, 7)),
        (44100, (4, 7), 3.0, 1.0, (4, 7)),
        (48000, (4, 7), 5.0, 1.0, (3, 8)),
        (48000, (0, 3), 5.0, 1.0, (0, 4)),
        (48000, (18, 20), 4.0, 1.0, (17, 20)),
        (48000, (4, 10), 3.0, 2.0, (4, 10)),
        (48000, (4, 8), 3.0, 2.0, (3, 9)),
    ],
    ids=[
        "as-is",
        "resampled",
        "padded",
        "at-the-start",
        "at-the-end",
        "sped-up",
        "sped-up-and-padded",
    ],
)
def test_a_clip_holds_the_part_of_the_recording_around_its_detection(
    tmp_path, rate, detection, seg_length, speed, cut
):
    audio_dir, results_dir, out = (
        tmp_path / name for name in ("audio", "results", "segments")
    )
    audio_dir.mkdir()
    audio = audio_dir / "rec.flac"
    sf.write(audio, _ramp(20, rate), rate)
    _combined_table(results_dir, [(audio, *detection)])

    segments(
        str(audio_dir),
        str(out),
        str(results_dir),
        seg_length=seg_length,
        audio_speed=speed,
    )

    (clip,) = out.rglob("*.wav")
    sig, clip_rate = sf.read(clip)
    expected = RAMP_PER_SECOND * (cut[0] + np.arange(len(sig)) * speed / clip_rate)
    assert len(sig) / clip_rate == pytest.approx((cut[1] - cut[0]) / speed, abs=1e-3)
    # 0.001 of the ramp is 40 ms; the ends carry the resampler's ringing.
    assert np.abs(sig - expected)[100:-100].max() < 1e-3


def test_a_recording_is_not_decoded_as_a_whole_for_its_clips(tmp_path, monkeypatch):
    audio_dir, results_dir, out = (
        tmp_path / name for name in ("audio", "results", "segments")
    )
    audio_dir.mkdir()
    audio = audio_dir / "rec.wav"
    sf.write(audio, _ramp(20), 48000)
    _combined_table(results_dir, [(audio, 4, 7)])

    def decode_whole(*args, **kwargs):
        raise AssertionError("the whole recording was decoded")

    monkeypatch.setattr(segments_utils.audio, "open_audio_file", decode_whole)

    assert segments(str(audio_dir), str(out), str(results_dir)) == [(str(audio), True)]


def test_a_recording_soundfile_cannot_read_is_decoded_another_way(
    tmp_path, monkeypatch
):
    audio_dir, results_dir, out = (
        tmp_path / name for name in ("audio", "results", "segments")
    )
    audio_dir.mkdir()
    audio = audio_dir / "rec.m4a"
    audio.write_bytes(b"not a format of libsndfile")
    _combined_table(results_dir, [(audio, 4, 7)])
    monkeypatch.setattr(
        segments_utils.audio,
        "open_audio_file",
        lambda path, sample_rate, speed: (_ramp(20), sample_rate),
    )

    segments(str(audio_dir), str(out), str(results_dir))

    (clip,) = out.rglob("*.wav")
    sig, _ = sf.read(clip)
    ends = [4 * RAMP_PER_SECOND, 7 * RAMP_PER_SECOND]
    assert sig[[0, -1]] == pytest.approx(ends, abs=1e-3)


def test_recordings_moved_since_the_analysis_are_cut_from_the_audio_folder(tmp_path):
    audio_dir, results_dir, out = (
        tmp_path / name for name in ("audio", "results", "segments")
    )
    audio = audio_dir / "site_a" / "rec.wav"
    audio.parent.mkdir(parents=True)
    sf.write(audio, _ramp(10), 48000)
    analyzed = tmp_path / "drive_of_the_analysis" / "site_a" / "rec.wav"
    _combined_table(results_dir, [(analyzed, 4, 7)])

    done = segments(str(audio_dir), str(out), str(results_dir))

    assert done == [(str(audio), True)]
    assert len(list(out.rglob("*.wav"))) == 1


def _files(folder, *names):
    for name in names:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).touch()

    return [str(folder / name) for name in names]


def test_a_moved_recording_is_told_from_its_namesakes_by_its_folders(tmp_path):
    audio_dir = tmp_path / "audio"
    _, site_b = _files(audio_dir, "site_a/rec.wav", "site_b/rec.wav")
    analyzed = str(tmp_path / "gone" / "site_b" / "rec.wav")

    assert relocate_audio([(analyzed, [])], str(audio_dir)) == [(site_b, [])]


@pytest.mark.parametrize(
    ("in_audio_dir", "analyzed"),
    [
        (["site_a/rec.wav", "site_b/rec.wav"], ["gone/rec.wav"]),
        (["rec.wav"], ["gone/site_a/rec.wav", "gone/site_b/rec.wav"]),
        (["other.wav"], ["gone/rec.wav"]),
    ],
    ids=["two-candidates", "two-claims", "no-candidate"],
)
def test_a_moved_recording_is_not_guessed(tmp_path, in_audio_dir, analyzed):
    audio_dir = tmp_path / "audio"
    _files(audio_dir, *in_audio_dir)
    file_list = [(str(tmp_path / path), []) for path in analyzed]

    assert relocate_audio(file_list, str(audio_dir)) == file_list


def test_a_recording_in_its_place_is_not_taken_for_a_moved_one(tmp_path):
    audio_dir = tmp_path / "audio"
    (in_place,) = _files(audio_dir, "rec.wav")
    file_list = [(in_place, []), (str(tmp_path / "gone" / "rec.wav"), [])]

    assert relocate_audio(file_list, str(audio_dir)) == file_list
