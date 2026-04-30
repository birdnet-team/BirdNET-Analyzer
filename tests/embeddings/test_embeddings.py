import os
import pathlib
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from birdnet_analyzer.cli import embeddings_parser
from birdnet_analyzer.embeddings.core import embeddings


def _make_empty_encoding_result():
    """Create a mock AcousticFileEncodingResult with zero inputs."""
    mock_result = MagicMock()
    mock_result.segment_duration_s = 3.0
    mock_result.overlap_duration_s = 0.0
    mock_result.n_inputs = 0
    mock_result.max_n_segments = 0
    mock_result.embeddings = np.zeros((0, 0, 1024), dtype=np.float32)
    mock_result.embeddings_masked = np.zeros((0, 0, 1024), dtype=bool)
    mock_result.inputs = np.array([], dtype="<U1")
    mock_result.input_durations = np.array([])
    return mock_result


@pytest.fixture
def setup_test_environment():
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
    }

    shutil.rmtree(test_dir)


@patch("birdnet_analyzer.embeddings.core.create_csv_output")
@patch("birdnet_analyzer.embeddings.core._check_database_settings")
@patch("birdnet_analyzer.embeddings.core.get_or_create_database")
@patch("birdnet_analyzer.model_utils.get_embeddings")
def test_embeddings_cli(
    mock_get_embeddings: MagicMock,
    mock_get_db: MagicMock,
    mock_check_settings: MagicMock,
    mock_csv_output: MagicMock,
    setup_test_environment,
):
    env = setup_test_environment

    mock_get_embeddings.return_value = _make_empty_encoding_result()
    mock_db = MagicMock()
    mock_get_db.return_value = mock_db

    parser = embeddings_parser()
    args = parser.parse_args(["--input", env["input_dir"], "-db", env["output_dir"]])

    embeddings(**vars(args))

    mock_get_embeddings.assert_called_once()
    call_kwargs = mock_get_embeddings.call_args
    assert call_kwargs[0][0] == env["input_dir"]
    assert call_kwargs[1]["version"] == "2.4"


@patch("birdnet_analyzer.embeddings.core._ensure_recording")
@patch("birdnet_analyzer.embeddings.core._ensure_deployment")
@patch("birdnet_analyzer.embeddings.core._check_database_settings")
@patch("birdnet_analyzer.embeddings.core.get_or_create_database")
@patch("birdnet_analyzer.model_utils.get_embeddings")
def test_embeddings_inserts_windows_per_file_in_batch(
    mock_get_embeddings: MagicMock,
    mock_get_db: MagicMock,
    mock_check_settings: MagicMock,
    mock_ensure_deployment: MagicMock,
    mock_ensure_recording: MagicMock,
):
    # Use a real temp directory so paths are valid on all platforms
    tmpdir = tempfile.mkdtemp()
    try:
        audio_input = tmpdir
        example_wav = os.path.join(tmpdir, "example.wav")

        mock_result = MagicMock()
        mock_result.segment_duration_s = 3.0
        mock_result.overlap_duration_s = 0.0
        mock_result.n_inputs = 1
        mock_result.max_n_segments = 3
        mock_result.embeddings = np.array(
            [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]], dtype=np.float32
        )
        mock_result.embeddings_masked = np.zeros((1, 3, 1), dtype=bool)
        mock_result.inputs = np.array([example_wav])
        mock_result.input_durations = np.array([5.0])
        mock_get_embeddings.return_value = mock_result

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_ensure_deployment.return_value = 10
        mock_ensure_recording.return_value = 20

        embeddings(
            audio_input=audio_input, database=os.path.join(tmpdir, "test.sqlite")
        )

        # Compute the expected fpath the same way the production code does:
        # fpath = str(pathlib.Path(input_file).relative_to(pathlib.Path(audio_input).parent))  # noqa: E501
        audio_root = str(pathlib.Path(audio_input).parent)
        expected_fpath = str(pathlib.Path(example_wav).relative_to(audio_root))

        mock_ensure_recording.assert_called_once_with(mock_db, expected_fpath, 10)
        mock_db.insert_windows_batch.assert_called_once()

        call_kwargs = mock_db.insert_windows_batch.call_args.kwargs
        windows_batch = call_kwargs["windows_batch"]
        embeddings_batch = call_kwargs["embeddings_batch"]

        assert windows_batch == [
            {"recording_id": 20, "offsets": [0.0, 3.0]},
            {"recording_id": 20, "offsets": [3.0, 5.0]},
        ]
        assert embeddings_batch.shape == (2, 4)
        np.testing.assert_array_equal(
            embeddings_batch[0], np.array([1, 2, 3, 4], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            embeddings_batch[1], np.array([5, 6, 7, 8], dtype=np.float32)
        )
        assert call_kwargs["handle_duplicates"] == "skip"
    finally:
        shutil.rmtree(tmpdir)
