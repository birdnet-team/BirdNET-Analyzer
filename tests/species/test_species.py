import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from birdnet_analyzer.cli import species_parser
from birdnet_analyzer.species.core import species


@pytest.fixture
def setup_test_environment():
    test_dir = tempfile.mkdtemp()
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(output_dir, exist_ok=True)

    yield {
        "test_dir": test_dir,
        "output_dir": output_dir,
    }

    shutil.rmtree(test_dir)


@patch("birdnet_analyzer.species.utils.get_species_list")
def test_species_cli(mock_get_species_list, setup_test_environment):
    env = setup_test_environment

    mock_get_species_list.return_value = ["Species1", "Species2"]

    parser = species_parser()
    args = parser.parse_args([env["output_dir"]])
    kwargs = vars(args)
    kwargs.pop("sortby", None)

    species(**kwargs)

    mock_get_species_list.assert_called_once_with(
        lat=-1, lon=-1, week=None, threshold=0.03, lang="en_us"
    )


@patch("birdnet_analyzer.species.utils.get_species_list")
def test_species_cli_accepts_full_parser_surface(
    mock_get_species_list, setup_test_environment
):
    env = setup_test_environment

    mock_get_species_list.return_value = ["Species1", "Species2"]

    parser = species_parser()
    args = parser.parse_args(
        [
            "--lat",
            "42.5",
            "--lon",
            "-76.45",
            "--week",
            "20",
            "--sf_thresh",
            "0.12",
            "--locale",
            "de",
            env["output_dir"],
        ]
    )

    species(**vars(args))

    mock_get_species_list.assert_called_once_with(
        lat=42.5, lon=-76.45, week=20, threshold=0.12, lang="de"
    )
