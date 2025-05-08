import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.train.core import analyze


@pytest.fixture
def setup_test_environment():
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

     # Create dummy audio files
    test_file1 = os.path.join(input_dir, "test1.wav")
    test_file2 = os.path.join(input_dir, "test2.wav")
    with open(test_file1, "wb") as f:
        f.write(b"dummy audio data")
    with open(test_file2, "wb") as f:
        f.write(b"more dummy audio data")

    # Store original config values
    original_config = {
        attr: getattr(cfg, attr) for attr in dir(cfg) if not attr.startswith("_") and not callable(getattr(cfg, attr))
    }

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "test_file1": test_file1,
        "test_file2": test_file2,
    }

    # Clean up
    shutil.rmtree(test_dir)

    # Restore original config
    for attr, value in original_config.items():
        setattr(cfg, attr, value)

@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.train.utils.train_model")
def test_training_basic(mock_ensure_model, mock_train_model, setup_test_environment):
    env = setup_test_environment