import csv
import multiprocessing
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from numpy.testing import assert_array_equal

import birdnet_analyzer.config as cfg
from birdnet_analyzer.audio import open_audio_file
from birdnet_analyzer.cli import embeddings_parser
from birdnet_analyzer.embeddings.core import embeddings, get_or_create_database
from birdnet_analyzer.search.core import search


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
        "data_dir": os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")),
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
    mock_run_embeddings.assert_called_once_with(env["input_dir"], env["output_dir"], 0, 1.0, 0, 15000, threads, 8, None)


@pytest.mark.parametrize(
    ("audio_speed", "overlap", "threads"),
    [(10, 1, 1), (5, 2, 2), (5, 0, 1), (0.1, 1, 4), (0.2, 0, 4)],
)
def test_extract_embeddings_with_speed_up_and_overlap(setup_test_environment, audio_speed, overlap, threads):
    """Test embeddings with speed up."""
    env = setup_test_environment

    input_dir = "tests/data/soundscape/"
    db_path = os.path.join(env["output_dir"], "embedding_db")
    file_output = os.path.join(env["output_dir"], "file_output/embeddings.csv")

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
    embeddings(input_dir, database=db_path, audio_speed=audio_speed, overlap=overlap, file_output=file_output, threads=threads)

    db = get_or_create_database(db_path)
    assert db is not None, "Database should be created successfully"
    assert db.count_embeddings() == len(expected_start_timestamps), "Number of embeddings should match expected count"

    embedding_ids = db.get_embedding_ids()
    assert len(embedding_ids) == len(expected_start_timestamps), "Number of embeddings should match expected count"

    for embedding_id in embedding_ids:
        source = db.get_embedding_source(embedding_id)
        start, end = source.offsets
        start = round(float(start), 1)
        end = round(float(end), 1)
        assert start in expected_start_timestamps, f"Start time mismatch for start timestamp {start}"
        assert end == expected_end_timestamps[expected_start_timestamps.index(start)]

    with open(file_output, newline="") as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        assert len(rows) == len(expected_start_timestamps) + 1, "CSV should contain header and all embeddings"
        for row in rows[1:]:
            start = round(float(row[1]), 1)
            end = round(float(row[2]), 1)
            assert start in expected_start_timestamps, f"CSV start time mismatch for start timestamp {start}"
            assert end == expected_end_timestamps[expected_start_timestamps.index(start)]


def test_search(setup_test_environment):
    """Test embeddings with speed up."""
    env = setup_test_environment

    audio_speed = 1.0
    overlap = 0

    input_dir = "tests/data/soundscape/"
    db_path = os.path.join(env["output_dir"], "embedding_db")
    search_output_dir = os.path.join(env["output_dir"], "search_output")
    query_file = "tests/data/embeddings/search_sample.mp3"

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
    embeddings(input_dir, database=db_path, audio_speed=audio_speed, overlap=overlap)

    n_results = len(expected_start_timestamps) - 1  # Currently querying the full database is not possible due to hoplite

    search(output=search_output_dir, database=db_path, queryfile=query_file, n_results=n_results, score_function="cosine", crop_mode="crop")

    output_files = []
    for root, _, files in os.walk(search_output_dir):
        output_files.extend([os.path.join(root, file) for file in files])

    assert len(output_files) == n_results, "Number of output files should match expected count"


def test_with_dataset(setup_test_environment):
    """Test embeddings with speed up."""
    env = setup_test_environment

    audio_speed = 1.0
    overlap = 0

    input_dir = "tests/data/embeddings/search-dataset"
    db_path = os.path.join(env["output_dir"], "embedding_db")
    search_output_dir = os.path.join(env["output_dir"], "search_output")
    query_file = "tests/data/embeddings/search_sample.mp3"

    # Call function under test
    embeddings(input_dir, database=db_path, audio_speed=audio_speed, overlap=overlap)
    search(output=search_output_dir, database=db_path, queryfile=query_file, n_results=10, score_function="cosine", crop_mode="crop")

    output_files = []
    for root, _, files in os.walk(search_output_dir):
        output_files.extend([os.path.join(root, file) for file in files])

    assert len(output_files) == 10, "Number of output files should match expected count"
    for file in output_files:
        filename = os.path.split(file)[-1]
        score, name, start, end = filename.split(".wav")[0].split("_")

        output_audio = open_audio_file(file)[0]
        original_audio = open_audio_file(os.path.join(input_dir, name + ".flac"), offset=float(start), duration=float(end) - float(start))[0]

        assert_array_equal(output_audio, original_audio, "Output audio should match original audio segment")


@pytest.mark.parametrize(("threads"), [1, 3])
def test_complete_run_multiprocessing(setup_test_environment, threads):
    env = setup_test_environment

    embeddings(os.path.join(env["data_dir"], "embeddings", "embeddings-dataset"), env["output_dir"], threads=threads)

    assert os.path.exists(
        os.path.join(env["output_dir"], "hoplite.sqlite"),
    ), "Database has noot been created"
