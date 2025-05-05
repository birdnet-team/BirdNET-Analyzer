def embeddings(
    audio_input: str,
    database: str,
    *,
    overlap: float = 0.0,
    audio_speed: float = 1.0,
    fmin: int = 0,
    fmax: int = 15000,
    threads: int = 8,
    batch_size: int = 1,
    file_output: str | None = None,
):
    """
    Generates embeddings for audio files using the BirdNET-Analyzer.
    This function processes audio files to extract embeddings, which are
    representations of audio features. The embeddings can be used for
    further analysis or comparison.
    Args:
        audio_input (str): Path to the input audio file or directory containing audio files.
        database (str): Path to the database where embeddings will be stored.
        overlap (float, optional): Overlap between consecutive audio segments in seconds. Defaults to 0.0.
        audio_speed (float, optional): Speed factor for audio processing. Defaults to 1.0.
        fmin (int, optional): Minimum frequency (in Hz) for audio analysis. Defaults to 0.
        fmax (int, optional): Maximum frequency (in Hz) for audio analysis. Defaults to 15000.
        threads (int, optional): Number of threads to use for processing. Defaults to 8.
        batch_size (int, optional): Number of audio segments to process in a single batch. Defaults to 1.
    Raises:
        FileNotFoundError: If the input path or database path does not exist.
        ValueError: If any of the parameters are invalid.
    Note:
        Ensure that the required model files are downloaded and available before
        calling this function. The `ensure_model_exists` function is used to
        verify this.
    Example:
        embeddings(
            "path/to/audio",
            "path/to/database",
            overlap=0.5,
            audio_speed=1.0,
            fmin=500,
            fmax=10000,
            threads=4,
            batch_size=2
        )
    """
    from birdnet_analyzer.embeddings.utils import run
    from birdnet_analyzer.utils import ensure_model_exists

    ensure_model_exists()
    run(audio_input, database, overlap, audio_speed, fmin, fmax, threads, batch_size, file_output)


def get_database(db_path: str):
    """Get the database object. Creates or opens the databse.
    Args:
        db: The path to the database.
    Returns:
        The database object.
    """
    import os

    from perch_hoplite.db import sqlite_usearch_impl

    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return sqlite_usearch_impl.SQLiteUsearchDB.create(
            db_path=db_path,
            usearch_cfg=sqlite_usearch_impl.get_default_usearch_config(embedding_dim=1024),  # TODO: dont hardcode this
        )
    return sqlite_usearch_impl.SQLiteUsearchDB.create(db_path=db_path)
