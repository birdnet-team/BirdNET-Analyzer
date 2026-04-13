from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from birdnet.globals import MODEL_LANGUAGES


def species(
    output: str,
    *,
    lat: float | None = None,
    lon: float | None = None,
    week: int | None = None,
    sf_thresh: float = 0.03,
    locale: MODEL_LANGUAGES = "en_us",
):
    """
    Retrieves and processes species data based on the provided parameters.
    Args:
        output (str): The output directory or file path where the results will be
                      stored.
        lat (float | None, optional): Latitude of the location for species filtering.
                                      Defaults to None (no filtering by location).
        lon (float | None, optional): Longitude of the location for species filtering.
                                      Defaults to None (no filtering by location).
        week (int | None, optional): Week of the year for species filtering.
                                     Defaults to None (no filtering by time).
        sf_thresh (float, optional): Species frequency threshold for filtering.
                                     Defaults to 0.03.
        locale (MODEL_LANGUAGES, optional): Locale for species names.
                                            Defaults to "en_us".
    Raises:
        FileNotFoundError: If the required model files are not found.
        ValueError: If invalid parameters are provided.
    Notes:
        This function ensures that the required model files exist before processing.
        It delegates the main processing to the `run` function
        from `birdnet_analyzer.species.utils`.
    """
    from birdnet_analyzer.species.utils import get_species_list

    species_list = get_species_list(
        lat=-1 if lat is None else lat,
        lon=-1 if lon is None else lon,
        week=week,
        threshold=sf_thresh,
        lang=locale,
    )

    if os.path.isdir(output):
        output = os.path.join(output, "species_list.txt")

    with open(output, "w", encoding="utf-8") as f:
        f.writelines(s + "\n" for s in species_list)
