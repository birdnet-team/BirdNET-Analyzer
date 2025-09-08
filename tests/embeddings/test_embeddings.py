import multiprocessing
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.cli import embeddings_parser
from birdnet_analyzer.embeddings.core import embeddings


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


@pytest.mark.parametrize(("threads"), [1, 3])
def test_complete_run_multiprocessing(setup_test_environment, threads):
    env = setup_test_environment

    embeddings(os.path.join(env["data_dir"], "embeddings", "embeddings-dataset"), env["output_dir"], threads=threads)

    assert os.path.exists(
        os.path.join(env["output_dir"], "hoplite.sqlite"),
    ), "Database has noot been created"
