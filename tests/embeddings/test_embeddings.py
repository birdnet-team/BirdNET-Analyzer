import multiprocessing
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer import model
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


def test_model_embeddings_function_returns_expected_shape():
    # Create a dummy sample (e.g., 1D numpy array of audio data)
    sample = np.zeros(144000).astype(np.float32)
    # Reshape the sample to (1, 144000) as expected by the model
    sample = sample.reshape(1, 144000)
    # Call the embeddings function
    result = model.embeddings(sample)

    # Check that result is a numpy array and has expected shape (depends on model, e.g., (1, embedding_dim))
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
