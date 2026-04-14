from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from birdnet_analyzer.config import CROP_MODES, SCORE_FUNCTIONS


def search(
    output: str,
    database: str,
    audio_root: str,
    queryfile: str,
    *,
    n_results: int = 10,
    score_function: SCORE_FUNCTIONS = "cosine",
    crop_mode: CROP_MODES = "center",
    overlap: float = 0.0,
):
    """
    Executes a search query on a given database and saves the results as audio files.
    Args:
        output (str): Path to the output directory where the results will be saved.
        database (str): Path to the database file to search in.
        queryfile (str): Path to the query file containing the search input.
        n_results (int, optional): Number of top results to return. Defaults to 10.
        score_function (SCORE_FUNCTIONS, optional):
            Scoring function to use for similarity calculation. Defaults to "cosine".
        crop_mode (CROP_MODES, optional):
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

    from birdnet_analyzer import audio
    from birdnet_analyzer.embeddings.core import SETTINGS_KEY
    from birdnet_analyzer.search.utils import get_search_results

    if not os.path.exists(output):
        os.makedirs(output)

    db = get_database(database)

    try:
        settings = db.get_metadata(SETTINGS_KEY)
    except KeyError as e:
        raise ValueError("No settings present in database.") from e

    fmin: int = settings["BANDPASS_FMIN"]
    fmax: int = settings["BANDPASS_FMAX"]
    audio_speed: float = settings["AUDIO_SPEED"]
    sig_length: float = settings.get("SIG_LENGTH", 3.0)
    duration = sig_length * audio_speed
    results = get_search_results(
        queryfile,
        db,
        n_results,
        audio_speed,
        fmin,
        fmax,
        score_function,
        crop_mode,
        overlap,
        sig_length,
    )

    for r in results:
        window = db.get_window(r.window_id)
        recording = db.get_recording(window.recording_id)
        file = os.path.join(audio_root, recording.filename)
        filebasename = os.path.basename(file)
        filebasename = os.path.splitext(filebasename)[0]
        offset = window.offsets[0]
        sig, rate = audio.open_audio_file(
            file, offset=offset, duration=duration, sample_rate=None
        )
        result_path = os.path.join(
            output,
            f"{r.sort_score:.5f}_{filebasename}_{offset}_{offset + duration}.wav",
        )
        audio.save_signal(sig, result_path, rate)

    db.db.close()


def get_database(database_path):
    from perch_hoplite.db import sqlite_usearch_impl

    return sqlite_usearch_impl.SQLiteUSearchDB.create(database_path)
