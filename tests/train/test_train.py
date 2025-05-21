import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.cli import train_parser
from birdnet_analyzer.train.core import train


@pytest.fixture
def setup_test_environment():
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    classifier_output = os.path.join(output_dir, "classifier_output")

    # Store original config values
    original_config = {
        attr: getattr(cfg, attr) for attr in dir(cfg) if not attr.startswith("_") and not callable(getattr(cfg, attr))
    }

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "classifier_output": classifier_output,
    }

    # Clean up
    shutil.rmtree(test_dir)

    # Restore original config
    for attr, value in original_config.items():
        setattr(cfg, attr, value)

@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.train.utils.train_model")
def test_train_cli(mock_train_model, mock_ensure_model, setup_test_environment):
    env = setup_test_environment

    mock_ensure_model.return_value = True

    parser = train_parser()
    args = parser.parse_args([env["input_dir"], "--output", env["classifier_output"]])

    train(**vars(args))

    mock_ensure_model.assert_called_once()
    mock_train_model.assert_called_once_with()
