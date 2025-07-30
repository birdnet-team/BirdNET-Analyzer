import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.analyze.core import analyze
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

    classifier_output = os.path.join(output_dir, "classifier_output", "custom_classifier.tflite")

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

@pytest.mark.timeout(600)  # Increase timeout for training
def test_training(setup_test_environment):
    """Test the training process and prediction with dummy data."""
    env = setup_test_environment
    training_data_input = "tests/data/training"

    # Read class names from subfolders in the input directory, filtering out background classes
    dummy_classes = [
        d for d in os.listdir(training_data_input)
        if os.path.isdir(os.path.join(training_data_input, d)) and d.lower() not in cfg.NON_EVENT_CLASSES
    ]

    train(training_data_input, env["classifier_output"])

    assert os.path.isfile(env["classifier_output"]), "Classifier output file was not created."
    assert os.path.exists(env["classifier_output"].replace(".tflite", "_Labels.txt")), "Labels file was not created."
    assert os.path.exists(env["classifier_output"].replace(".tflite", "_Params.csv")), "Params file was not created."
    assert os.path.exists(env["classifier_output"].replace(".tflite", ".tflite_sample_counts.csv")), "Params file was not created."

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"
    analyze(soundscape_path, env["output_dir"], top_n=1, classifier=env["classifier_output"])

    output_file = os.path.join(env["output_dir"], "soundscape.BirdNET.selection.table.txt")
    with open(output_file) as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split("\t")
            assert parts[7] in dummy_classes, f"Detected class {parts[7]} not in expected classes {dummy_classes}"
