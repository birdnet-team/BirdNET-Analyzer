"""Module for predicting a species list.

Can be used to predict a species list using coordinates and weeks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from birdnet_analyzer import model_utils

if TYPE_CHECKING:
    from birdnet.globals import MODEL_LANGUAGES


def get_species_list(
    lat: float, lon: float, week: int | None, threshold: float, lang: MODEL_LANGUAGES
) -> list[str]:
    """
    Generates a species list for a given location and time, and saves it to the
    specified output path.
    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        week (int | None): Week of the year (1-52) for which the species list is
                           generated. If None, all weeks are considered.
        threshold (float): Threshold for location filtering.
        lang (MODEL_LANGUAGES): Language code for species names.
    Returns:
        list[str]: Species list as numpy strings.
    """
    result = model_utils.run_geomodel(
        lat, lon, week, threshold=threshold, language=lang
    )

    return [str(species) for species, prob in result.to_structured_array()]
