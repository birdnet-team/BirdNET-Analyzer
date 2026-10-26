from unittest.mock import patch

import pytest

from birdnet_analyzer.cli import species_parser
from birdnet_analyzer.config import DEFAULT_ACOUSTIC_MODEL_VERSION
from birdnet_analyzer.species.core import species


@pytest.mark.parametrize(
    ("locale_args", "locale"), [([], "en_us"), (["--locale", "de"], "de")]
)
@patch("birdnet_analyzer.model_utils.run_geomodel")
@patch("birdnet_analyzer.model_utils.acoustic_species_list")
def test_species_cli(
    mock_acoustic_species_list, mock_run_geomodel, locale_args, locale, tmp_path
):
    mock_acoustic_species_list.return_value = ["Species1", "Species2"]
    args = species_parser().parse_args([*locale_args, str(tmp_path)])

    species(**vars(args))

    mock_run_geomodel.assert_not_called()
    mock_acoustic_species_list.assert_called_once_with(
        DEFAULT_ACOUSTIC_MODEL_VERSION, locale
    )
    assert (tmp_path / "species_list.txt").read_text(encoding="utf-8").splitlines() == [
        "Species1",
        "Species2",
    ]


@pytest.mark.parametrize("coordinate", [{"lat": 50.83}, {"lon": 12.92}])
@patch("birdnet_analyzer.model_utils.run_geomodel")
@patch("birdnet_analyzer.model_utils.acoustic_species_list")
def test_species_rejects_a_single_coordinate(
    mock_acoustic_species_list, mock_run_geomodel, coordinate, tmp_path
):
    with pytest.raises(ValueError, match="both lat and lon"):
        species(str(tmp_path), **coordinate)

    mock_run_geomodel.assert_not_called()
    mock_acoustic_species_list.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@patch("birdnet_analyzer.species.utils.get_species_list")
def test_species_cli_accepts_full_parser_surface(mock_get_species_list, tmp_path):
    mock_get_species_list.return_value = ["Species1", "Species2"]
    args = species_parser().parse_args(
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
            str(tmp_path),
        ]
    )

    species(**vars(args))

    mock_get_species_list.assert_called_once_with(
        lat=42.5, lon=-76.45, week=20, threshold=0.12, lang="de"
    )
