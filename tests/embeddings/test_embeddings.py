import multiprocessing
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.cli import embeddings_parser
from birdnet_analyzer.embeddings.core import embeddings, get_database


@pytest.fixture
def setup_test_environment():
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Store original config values
    original_config = {attr: getattr(cfg, attr) for attr in dir(cfg) if not attr.startswith("_") and not callable(getattr(cfg, attr))}

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
    }

    # Clean up
    shutil.rmtree(test_dir)

    # Restore original config
    for attr, value in original_config.items():
        setattr(cfg, attr, value)


@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.embeddings.utils.run")
def test_embeddings_cli(mock_run_embeddings: MagicMock, mock_ensure_model: MagicMock, setup_test_environment):
    env = setup_test_environment

    mock_ensure_model.return_value = True

    parser = embeddings_parser()
    args = parser.parse_args(["--input", env["input_dir"], "-db", env["output_dir"]])

    embeddings(**vars(args))

    mock_ensure_model.assert_called_once()
    threads = min(8, max(1, multiprocessing.cpu_count() // 2))
    mock_run_embeddings.assert_called_once_with(env["input_dir"], env["output_dir"], 0, 1.0, 0, 15000, threads, 1, None)



@pytest.mark.parametrize(
    ("audio_speed", "overlap"),
    [(10, 1), (5, 2), (5, 0), (0.1, 1), (0.2, 0)],
)
def test_extract_embeddings_with_speed_up_and_overlap(setup_test_environment, audio_speed, overlap):
    """Test embeddings with speed up."""
    env = setup_test_environment

    input_dir = "birdnet_analyzer/example/soundscape/"
    db_path = os.path.join(env["output_dir"], "embedding_db")
    file_output_dir = os.path.join(env["output_dir"], "file_output")

    assert os.path.exists(os.path.join(input_dir, "soundscape.wav")), "Soundscape file does not exist"
    file_length = 120
    step_size = round(3 * audio_speed - overlap * audio_speed, 1)
    expected_start_timestamps = [e / 10 for e in range(0, int(file_length * 10), int(step_size * 10))]
    expected_end_timestamps = [e / 10 for e in range(int(3 * audio_speed * 10), int(file_length) * 10 + 1, int(step_size * 10))]

    while len(expected_end_timestamps) < len(expected_start_timestamps):
        if file_length - expected_start_timestamps[-1] >= 1 * audio_speed:
            expected_end_timestamps.append(file_length)
        else:
            expected_start_timestamps.pop()

    # Call function under test
    embeddings(input_dir, database=db_path, audio_speed=audio_speed, overlap=overlap, file_output=file_output_dir)
    # TODO: Check if correct number of embeddings are in file output and in the database

    db = get_database(db_path)
    assert db is not None, "Database should be created successfully"
    assert db.count_embeddings() == len(expected_start_timestamps), "Number of embeddings should match expected count"
