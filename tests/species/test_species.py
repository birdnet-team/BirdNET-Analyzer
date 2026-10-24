import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from birdnet_analyzer.cli import species_parser
from birdnet_analyzer.config import DEFAULT_ACOUSTIC_MODEL_VERSION
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


@pytest.mark.parametrize(
    ("locale_args", "locale"), [([], "en_us"), (["--locale", "de"], "de")]
)
@patch("birdnet_analyzer.model_utils.run_geomodel")
@patch("birdnet_analyzer.model_utils.acoustic_species_list")
def test_species_cli(
    mock_acoustic_species_list,
    mock_run_geomodel,
    locale_args,
    locale,
    setup_test_environment,
):
    env = setup_test_environment

    mock_acoustic_species_list.return_value = ["Species1", "Species2"]

    parser = species_parser()
    args = parser.parse_args([*locale_args, env["output_dir"]])

    species(**vars(args))

    mock_run_geomodel.assert_not_called()
    mock_acoustic_species_list.assert_called_once_with(
        DEFAULT_ACOUSTIC_MODEL_VERSION, locale
    )

    with open(
        os.path.join(env["output_dir"], "species_list.txt"), encoding="utf-8"
    ) as f:
        assert f.read().splitlines() == ["Species1", "Species2"]


@pytest.mark.parametrize("coordinate", [{"lat": 50.83}, {"lon": 12.92}])
@patch("birdnet_analyzer.model_utils.run_geomodel")
@patch("birdnet_analyzer.model_utils.acoustic_species_list")
def test_species_rejects_a_single_coordinate(
    mock_acoustic_species_list, mock_run_geomodel, coordinate, setup_test_environment
):
    env = setup_test_environment

    with pytest.raises(ValueError, match="both lat and lon"):
        species(env["output_dir"], **coordinate)

    mock_run_geomodel.assert_not_called()
    mock_acoustic_species_list.assert_not_called()
    assert not os.listdir(env["output_dir"])


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
