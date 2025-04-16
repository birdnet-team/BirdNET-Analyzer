from typing import Literal


def species(
    output: str,
    *,
    lat: float = -1,
    lon: float = -1,
    week: int = -1,
    sf_thresh: float = 0.03,
    sortby: Literal["freq", "alpha"] = "freq",
):
    """
    Retrieves and processes species data based on the provided parameters.
    Args:
        output (str): The output directory or file path where the results will be stored.
        lat (float, optional): Latitude of the location for species filtering. Defaults to -1 (no filtering by location).
        lon (float, optional): Longitude of the location for species filtering. Defaults to -1 (no filtering by location).
        week (int, optional): Week of the year for species filtering. Defaults to -1 (no filtering by time).
        sf_thresh (float, optional): Species frequency threshold for filtering. Defaults to 0.03.
        sortby (Literal["freq", "alpha"], optional): Sorting method for the species list.
            "freq" sorts by frequency, and "alpha" sorts alphabetically. Defaults to "freq".
    Raises:
        FileNotFoundError: If the required model files are not found.
        ValueError: If invalid parameters are provided.
    Notes:
        This function ensures that the required model files exist before processing.
        It delegates the main processing to the `run` function from `birdnet_analyzer.species.utils`.
    """
    from birdnet_analyzer.species.utils import run
    from birdnet_analyzer.utils import ensure_model_exists

    ensure_model_exists()

    run(output, lat, lon, week, sf_thresh, sortby)
