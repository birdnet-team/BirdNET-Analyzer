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
    Writes a species list file, one label per line.
    Args:
        output (str): Path of the list file, or a folder to write "species_list.txt"
                      into.
        lat (float | None, optional): Latitude of the location. Together with
                                      ``lon``, the list holds the species the geo
                                      model predicts there; with neither, every
                                      species of the default acoustic model.
                                      Defaults to None.
        lon (float | None, optional): Longitude of the location. Defaults to None.
        week (int | None, optional): Week of the year (1-48) for the geo model,
                                     ignored without a location. Defaults to None,
                                     the year-round list.
        sf_thresh (float, optional): Minimum occurrence frequency of a species in the
                                     geo model's prediction, ignored without a
                                     location. Defaults to 0.03.
        locale (MODEL_LANGUAGES, optional): Language of the species names.
                                            Defaults to "en_us".
    Raises:
        ValueError: If only one of ``lat`` and ``lon`` is given, or ``lat``, ``lon``
            or ``week`` is out of range.
    Notes:
        The geo model, or without a location the acoustic model, is downloaded on
        first use.
    """
    if lat is None and lon is None:
        from birdnet_analyzer.config import DEFAULT_ACOUSTIC_MODEL_VERSION
        from birdnet_analyzer.model_utils import acoustic_species_list

        species_list = acoustic_species_list(DEFAULT_ACOUSTIC_MODEL_VERSION, locale)
    elif lat is None or lon is None:
        raise ValueError(
            "Set both lat and lon for the species list of a location, or neither for "
            "every species of the model."
        )
    else:
        from birdnet_analyzer.species.utils import get_species_list

        species_list = get_species_list(
            lat=lat, lon=lon, week=week, threshold=sf_thresh, lang=locale
        )

    if os.path.isdir(output):
        output = os.path.join(output, "species_list.txt")

    with open(output, "w", encoding="utf-8") as f:
        f.writelines(s + "\n" for s in species_list)
