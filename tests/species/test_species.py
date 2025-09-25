import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.cli import species_parser
from birdnet_analyzer.species.core import species


@pytest.fixture
def setup_test_environment():
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(output_dir, exist_ok=True)

    # Store original config values
    original_config = {
        attr: getattr(cfg, attr) for attr in dir(cfg) if not attr.startswith("_") and not callable(getattr(cfg, attr))
    }

    yield {
        "test_dir": test_dir,
        "output_dir": output_dir,
    }

    # Clean up
    shutil.rmtree(test_dir)

    # Restore original config
    for attr, value in original_config.items():
        setattr(cfg, attr, value)

@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.species.utils.run")
def test_species_cli(mock_run_species, mock_ensure_model, setup_test_environment):
    env = setup_test_environment

    mock_ensure_model.return_value = True

    parser = species_parser()
    args = parser.parse_args([env["output_dir"]])

    species(**vars(args))

    mock_ensure_model.assert_called_once()
    mock_run_species.assert_called_once_with(env["output_dir"], -1, -1, -1, 0.03, "freq")
