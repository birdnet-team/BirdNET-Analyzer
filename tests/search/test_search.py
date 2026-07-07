import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from birdnet_analyzer.cli import search_parser
from birdnet_analyzer.search.core import search


def _make_test_environment():
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
    }


@patch("birdnet_analyzer.audio.save_signal")
@patch("birdnet_analyzer.audio.open_audio_file")
@patch("birdnet_analyzer.search.utils.get_search_results")
@patch("birdnet_analyzer.search.core.get_database")
def test_search_cli_accepts_full_parser_surface(
    mock_get_database,
    mock_get_search_results,
    mock_open_audio_file,
    mock_save_signal,
):
    env = _make_test_environment()
    try:
        parser = search_parser()
        args = parser.parse_args(
            [
                "-q",
                os.path.join(env["test_dir"], "query.wav"),
                "-o",
                env["output_dir"],
                "--n_results",
                "5",
                "--score_function",
                "dot",
                "--crop_mode",
                "segments",
                "--audio_root",
                env["input_dir"],
                "-db",
                os.path.join(env["test_dir"], "database.sqlite"),
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

        mock_get_search_results.return_value = [
            MagicMock(window_id=11, sort_score=0.12345)
        ]
        mock_open_audio_file.return_value = (b"signal", 48000)

        search(**vars(args))

        mock_get_search_results.assert_called_once()
        search_kwargs = mock_get_search_results.call_args.args
        assert search_kwargs[0] == args.queryfile
        assert search_kwargs[2] == 5
        assert search_kwargs[6] == "dot"
        assert search_kwargs[7] == "segments"
        assert search_kwargs[8] == 0.5
        mock_open_audio_file.assert_called_once()
        mock_save_signal.assert_called_once()
    finally:
        shutil.rmtree(env["test_dir"])
