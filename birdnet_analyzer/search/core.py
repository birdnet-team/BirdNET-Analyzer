from typing import Literal


def search(
    output: str,
    database: str,
    queryfile: str,
    *,
    n_results: int = 10,
    score_function: Literal["cosine", "euclidean", "dot"] = "cosine",
    crop_mode: Literal["center", "first", "segments"] = "center",
    overlap: float = 0.0,
):
    """
    Executes a search query on a given database and saves the results as audio files.
    Args:
        output (str): Path to the output directory where the results will be saved.
        database (str): Path to the database file to search in.
        queryfile (str): Path to the query file containing the search input.
        n_results (int, optional): Number of top results to return. Defaults to 10.
        score_function (Literal["cosine", "euclidean", "dot"], optional):
            Scoring function to use for similarity calculation. Defaults to "cosine".
        crop_mode (Literal["center", "first", "segments"], optional):
            Mode for cropping audio segments. Defaults to "center".
        overlap (float, optional): Overlap ratio for audio segments. Defaults to 0.0.
    Raises:
        ValueError: If the database does not contain the required settings metadata.
    Notes:
        - The function creates the output directory if it does not exist.
        - It retrieves metadata from the database to configure the search, including
          bandpass filter settings and audio speed.
        - The results are saved as audio files in the specified output directory, with
          filenames containing the score, source file name, and time offsets.
    Returns:
        None
    """
    import os

    import birdnet_analyzer.config as cfg
    from birdnet_analyzer import audio
    from birdnet_analyzer.search.utils import get_search_results

    # Create output folder
    if not os.path.exists(output):
        os.makedirs(output)

    # Load the database
    db = get_database(database)

    try:
        settings = db.get_metadata("birdnet_analyzer_settings")
    except KeyError as e:
        raise ValueError("No settings present in database.") from e

    fmin = settings["BANDPASS_FMIN"]
    fmax = settings["BANDPASS_FMAX"]
    audio_speed = settings["AUDIO_SPEED"]

    # Execute the search
    results = get_search_results(queryfile, db, n_results, audio_speed, fmin, fmax, score_function, crop_mode, overlap)

    # Save the results
    for r in results:
        embedding_source = db.get_embedding_source(r.embedding_id)
        file = embedding_source.source_id
        filebasename = os.path.basename(file)
        filebasename = os.path.splitext(filebasename)[0]
        offset = embedding_source.offsets[0]
        duration = cfg.SIG_LENGTH * audio_speed
        sig, rate = audio.open_audio_file(file, offset=offset, duration=duration, sample_rate=None)
        result_path = os.path.join(output, f"{r.sort_score:.5f}_{filebasename}_{offset}_{offset + duration}.wav")
        audio.save_signal(sig, result_path, rate)


def get_database(database_path):
    from perch_hoplite.db import sqlite_usearch_impl

    return sqlite_usearch_impl.SQLiteUsearchDB.create(database_path).thread_split()
