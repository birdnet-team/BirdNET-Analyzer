from unittest.mock import MagicMock, patch

import pytest

from birdnet_analyzer.cli import search_parser
from birdnet_analyzer.search.core import search


@patch("birdnet_analyzer.audio.save_signal")
@patch("birdnet_analyzer.audio.open_audio_file")
@patch("birdnet_analyzer.search.utils.get_search_results")
@patch("birdnet_analyzer.search.core.get_database")
def test_search_cli_accepts_full_parser_surface(
    mock_get_database,
    mock_get_search_results,
    mock_open_audio_file,
    mock_save_signal,
    tmp_path,
):
    (tmp_path / "output").mkdir()
    args = search_parser().parse_args(
        [
            "-q",
            str(tmp_path / "query.wav"),
            "-o",
            str(tmp_path / "output"),
            "--n_results",
            "5",
            "--score_function",
            "dot",
            "--crop_mode",
            "segments",
            "--audio_root",
            str(tmp_path / "input"),
            "-db",
            str(tmp_path / "database.sqlite"),
            "--overlap",
            "0.5",
        ]
    )

    mock_db = MagicMock()
    mock_db.get_metadata.return_value = {
        "BANDPASS_FMIN": 100,
        "BANDPASS_FMAX": 10000,
        "AUDIO_SPEED": 1.1,
        "SIG_LENGTH": 3.0,
    }
    mock_db.get_window.return_value = MagicMock(recording_id=7, offsets=[1.0, 2.0])
    mock_db.get_recording.return_value = MagicMock(filename="query_source.wav")
    mock_db.db = MagicMock()
    mock_get_database.return_value = mock_db

    mock_get_search_results.return_value = [MagicMock(window_id=11, sort_score=0.12345)]
    mock_open_audio_file.return_value = (b"signal", 48000)

    search(**vars(args))

    mock_get_search_results.assert_called_once()
    search_args = mock_get_search_results.call_args.args
    assert search_args[0] == args.queryfile
    assert search_args[2] == 5
    assert search_args[6] == "dot"
    assert search_args[7] == "segments"
    assert search_args[8] == 0.5
    mock_open_audio_file.assert_called_once()
    mock_save_signal.assert_called_once()


def test_search_cli_requires_query_output_and_audio_root(capsys):
    with pytest.raises(SystemExit) as exc:
        search_parser().parse_args(["-db", "x"])

    assert exc.value.code == 2
    assert (
        "the following arguments are required: -q/--queryfile, -o/--output, "
        "--audio_root" in capsys.readouterr().err
    )
