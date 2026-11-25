"""What a user reads on the command line while a command runs and when it ends."""

import io
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from birdnet_analyzer import logs
from birdnet_analyzer.analyze import cli as analyze_cli
from birdnet_analyzer.config import OUTPUT_RAVEN_FILENAME
from birdnet_analyzer.embeddings import cli as embeddings_cli
from birdnet_analyzer.progress import ConsoleProgress
from birdnet_analyzer.segments import cli as segments_cli

SOUNDSCAPE = "birdnet_analyzer/example/soundscape.wav"


@pytest.fixture
def run_cli(monkeypatch, capsys):
    def run(main, *argv):
        monkeypatch.setattr(sys, "argv", ["birdnet", *map(str, argv)])
        code = 0

        try:
            main()
        except SystemExit as e:
            code = e.code
        finally:
            logs._remove_installed_handlers()

        return (code, *capsys.readouterr())

    return run


@pytest.fixture
def recordings(tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    shutil.copy(SOUNDSCAPE, folder / "a.wav")
    shutil.copy(SOUNDSCAPE, folder / "b.wav")

    return folder


@pytest.mark.parametrize("quiet", [False, True])
def test_an_analysis_shows_its_progress_and_where_its_detections_are(
    recordings, tmp_path, run_cli, quiet
):
    output = tmp_path / "out"

    code, out, _ = run_cli(
        analyze_cli.main, recordings, "-o", output, *(["--quiet"] if quiet else [])
    )

    assert code == 0
    progress = [line for line in out.splitlines() if line.startswith("Analyzing:")]
    rows = (output / OUTPUT_RAVEN_FILENAME).read_text().splitlines()[1:]

    if quiet:
        assert progress == []
        assert "detections" not in out
    else:
        assert progress[-1] == "Analyzing: 100% done"
        assert len(progress) <= 11
        assert "\r" not in out
        assert out.splitlines()[-1] == (
            f"{len(rows)} detections in 2 audio files, saved in {output}"
        )


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def _update(pct, left, finished=False):
    return SimpleNamespace(
        progress_pct=pct, est_remaining_time_s=left, finished=finished
    )


@pytest.fixture
def console():
    logs.setup_logging()
    yield
    logs._remove_installed_handlers()


def test_on_a_terminal_the_progress_rewrites_one_line_in_plain_ascii(console):
    terminal = _Terminal()
    progress = ConsoleProgress("Analyzing", terminal)

    for update in (_update(9.5, 3725.2), _update(60, 2), _update(100, 0, True)):
        progress(update)

    progress.close()
    shown = terminal.getvalue()

    assert shown.encode("ascii").decode() == shown
    assert shown.split("\r")[1:] == [
        "Analyzing:   9% done, about 1:02:06 left",
        "Analyzing:  60% done, about 0:00:02 left",
        "Analyzing: 100% done" + " " * 20 + "\n",
    ]


def test_a_console_that_cannot_be_written_does_not_stop_the_analysis(console):
    class Gone(_Terminal):
        def write(self, text):
            raise BrokenPipeError

    progress = ConsoleProgress("Analyzing", Gone())

    progress(_update(10, 5))
    progress(_update(20, 4))
    progress.close()


@pytest.mark.parametrize(
    ("folder", "closing"),
    [
        (
            True,
            [
                "Stopped.",
                "Run the same command again to continue the analysis: the files "
                "that are done are kept and skipped.",
            ],
        ),
        (False, ["", "Stopped."]),
    ],
)
def test_a_stopped_analysis_ends_with_what_to_do_next(
    recordings, run_cli, folder, closing
):
    target = recordings if folder else recordings / "a.wav"

    with patch.object(analyze_cli, "analyze", side_effect=KeyboardInterrupt):
        code, _, err = run_cli(analyze_cli.main, target)

    assert code == 130
    assert err.splitlines()[-2:] == closing


def test_a_missing_species_list_is_named_in_plain_words(recordings, tmp_path, run_cli):
    missing = tmp_path / "my species.txt"

    code, _, err = run_cli(analyze_cli.main, recordings, "--slist", missing)

    assert code == 1
    assert err.splitlines()[0] == (
        f"Error: {missing} could not be found. It may have been moved, renamed or "
        "deleted, or the drive it is on is not connected."
    )


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs a folder its user cannot write to",
)
def test_an_output_folder_that_cannot_be_written_is_named_as_typed(
    recordings, tmp_path, run_cli
):
    output = tmp_path / "read-only"
    output.mkdir()
    output.chmod(0o555)

    try:
        code, _, err = run_cli(analyze_cli.main, recordings, "-o", output)
    finally:
        output.chmod(0o755)

    assert code == 1
    assert err.splitlines()[0].startswith(
        f"Error: {output} could not be opened or written."
    )


def test_a_command_of_an_extra_that_is_not_installed_names_the_install_command(
    tmp_path, run_cli, monkeypatch
):
    for name in [m for m in sys.modules if m.startswith("perch_hoplite")]:
        monkeypatch.setitem(sys.modules, name, None)

    monkeypatch.setitem(sys.modules, "perch_hoplite", None)
    monkeypatch.setitem(sys.modules, "perch_hoplite.db", None)

    code, _, err = run_cli(
        embeddings_cli.main, "-i", SOUNDSCAPE, "-db", tmp_path / "db"
    )

    assert code == 1
    assert err.splitlines()[0] == (
        "Error: This needs the optional embeddings features of BirdNET-Analyzer, "
        'which are not installed. Install them with: pip install "birdnet_analyzer'
        '[embeddings]"'
    )


def test_segments_of_a_folder_that_is_not_there_fail_in_plain_words(tmp_path, run_cli):
    missing = tmp_path / "recordings"

    code, out, err = run_cli(segments_cli.main, missing)

    assert code == 1
    assert "Found 0" not in out
    assert err.splitlines()[0].startswith(f"Error: {missing} could not be found.")


def test_segments_without_a_result_file_they_can_read_say_which_ones_they_read(
    recordings, run_cli
):
    (recordings / "BirdNET_CombinedTable.parquet").write_bytes(b"PAR1")

    code, _, err = run_cli(segments_cli.main, recordings)

    assert code == 1
    assert re.match(
        r"Error: No segments were extracted: .* Raven selection tables, csv and "
        r"Kaleidoscope results .*; parquet results cannot be read\.$",
        err.splitlines()[0],
    )


def test_segments_of_a_combined_table_count_the_recordings_it_names(
    recordings, tmp_path, run_cli, capsys
):
    from birdnet_analyzer import analyze

    analyze(str(recordings), str(tmp_path / "results"), min_conf=0.5)
    capsys.readouterr()

    code, out, _ = run_cli(
        segments_cli.main,
        recordings,
        "-r",
        tmp_path / "results",
        "-o",
        tmp_path / "segments",
        "--threads",
        1,
    )

    assert code == 0
    assert out.splitlines()[0] == (
        "Reading the combined result file "
        f"{Path(tmp_path / 'results' / OUTPUT_RAVEN_FILENAME)}."
    )
    assert re.fullmatch(r"Found \d+ segments in 2 audio files\.", out.splitlines()[1])
    assert out.splitlines()[-1] == f"The segments are saved in {tmp_path / 'segments'}"


def test_a_file_in_a_format_that_cannot_be_read_lists_the_formats_that_can(
    tmp_path, run_cli
):
    notes = tmp_path / "notes.txt"
    notes.write_text("not audio")

    code, _, err = run_cli(analyze_cli.main, notes)

    assert code == 1
    assert re.fullmatch(
        rf"Error: {re.escape(str(notes))} is not an audio file that BirdNET-Analyzer "
        r"can read\. These formats can be read: [a-z0-9]+(, [a-z0-9]+)+\.",
        err.splitlines()[0],
    )
    assert "wav" in err.splitlines()[0].split(", ")
