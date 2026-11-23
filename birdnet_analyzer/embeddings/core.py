from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
    from birdnet.globals import ACOUSTIC_MODEL_VERSIONS
    from perch_hoplite.db import sqlite_usearch_impl

DATASET_NAME: str = "birdnet_analyzer_dataset"
COMMIT_BS_SIZE = 512
SETTINGS_KEY = "birdnet_analyzer_settings"
MODEL_VERSION_KEY = "MODEL_VERSION"


def embeddings(
    audio_input: str,
    database: str,
    *,
    overlap: float = 0.0,
    audio_speed: float = 1.0,
    fmin: int = 0,
    fmax: int = 15000,
    batch_size: int = 1,
    file_output: str | None = None,
    n_workers: int | None = None,
    n_producers: int = 1,
    birdnet: ACOUSTIC_MODEL_VERSIONS | None = None,
    on_update: Callable[[AcousticProgressStats], None] | None = None,
):
    """
    Generates embeddings for audio files using the BirdNET-Analyzer.
    This function processes audio files to extract embeddings, which are
    representations of audio features. The embeddings can be used for
    further analysis or comparison.
    Args:
        audio_input (str): Path to the input audio file or directory containing audio
            files.
        database (str): Path to the database where embeddings will be stored. It is
            created if it does not exist.
        overlap (float, optional): Overlap between consecutive audio segments in
            seconds. Defaults to 0.0.
        audio_speed (float, optional): Speed factor for audio processing.
            Defaults to 1.0.
        fmin (int, optional): Minimum frequency (in Hz) for audio analysis.
            Defaults to 0.
        fmax (int, optional): Maximum frequency (in Hz) for audio analysis.
            Defaults to 15000.
        batch_size (int, optional): Number of audio segments to process in a single
            batch. Defaults to 1.
        file_output (str | None, optional): Path to save the output embeddings. If None,
            embeddings are not saved to a file. Defaults to None.
        n_workers (int | None, optional): Number of worker processes to use for
            processing. Defaults to None: as many as fit the available memory, at
            most one per CPU core.
        n_producers (int, optional): Number of producer processes to use for
            processing. Defaults to 1.
        birdnet (Literal["2.4", "3.0"] | None, optional): The BirdNET version to
            extract the embeddings with. Defaults to None: the version of the
            database if it exists, the newest version for a new one.
        on_update (Callable[[AcousticProgressStats], None] | None, optional): Callback
            function to report progress updates. Defaults to None.
    Raises:
        ValueError: If ``audio_input`` does not exist or holds no supported audio
            file, or if the database was created with a different ``fmin``, ``fmax``,
            ``audio_speed``, parent folder of ``audio_input`` or BirdNET version.
    Example:
        embeddings(
            "path/to/audio",
            "path/to/database",
            overlap=0.5,
            audio_speed=1.0,
            fmin=500,
            fmax=10000,
            batch_size=2,
        )
    """
    from birdnet_analyzer.model_utils import embeddings_dim, get_embeddings

    version = _model_version_for(database, birdnet)
    result = get_embeddings(
        audio_input,
        version=version,
        batch_size=batch_size,
        overlap_duration_s=overlap,
        bandpass_fmin=fmin,
        bandpass_fmax=fmax,
        speed=audio_speed,
        n_workers=n_workers,
        n_producers=n_producers,
        callback=on_update,
    )

    audio_root = str(pathlib.Path(audio_input).parent)

    batchsize = COMMIT_BS_SIZE
    pending_since_commit = 0
    db = get_or_create_database(database, embeddings_dim(version))

    try:
        _check_database_settings(
            db,
            fmin=fmin,
            fmax=fmax,
            audio_speed=audio_speed,
            audio_root=audio_root,
            model_version=version,
        )
        deployment_id = _ensure_deployment(db)

        seg_dur = result.segment_duration_s
        seg_overlap = result.overlap_duration_s
        step = seg_dur - seg_overlap

        n_inputs = result.n_inputs
        n_segments = result.max_n_segments
        emb_masked = result.embeddings_masked
        input_durations = result.input_durations

        for i in tqdm(
            range(n_inputs), desc="Saving embeddings to database", total=n_inputs
        ):
            fpath = str(pathlib.Path(result.inputs[i]).relative_to(audio_root))
            file_dur = float(input_durations[i])
            recording_id = _ensure_recording(db, fpath, deployment_id)
            windows_batch = []
            embeddings_batch = []

            for j in range(n_segments):
                # Skip masked (invalid/padded) segments
                if emb_masked[i, j, 0]:
                    continue

                s_start = j * step
                s_end = s_start + seg_dur

                if s_start >= file_dur:
                    continue

                s_end = min(s_end, file_dur)

                windows_batch.append(
                    {
                        "recording_id": recording_id,
                        "offsets": [float(s_start), float(s_end)],
                    }
                )
                embeddings_batch.append(result.embeddings[i, j, :])

            if windows_batch:
                db.insert_windows_batch(
                    windows_batch=windows_batch,
                    embeddings_batch=np.asarray(embeddings_batch),
                    handle_duplicates="skip",
                )
                pending_since_commit += len(windows_batch)

                if pending_since_commit >= batchsize:
                    db.commit()
                    pending_since_commit = 0

        db.commit()
    finally:
        db.db.close()

    if file_output:
        create_csv_output(file_output, database)


def create_csv_output(output_path: str, database: str):
    """Creates a CSV output for the database.

    Args:
        output_path: Path to the output file.
        database: Path to the database.

    Raises:
        FileNotFoundError: If `database` is not a folder holding a database.
    """
    from birdnet_analyzer.search.core import open_database

    with open_database(database) as db:
        parent_dir = os.path.dirname(output_path)

        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        window_ids = db.match_window_ids()

        with open(output_path, "w") as f:
            f.write("file_path,start,end,embedding\n")

            for window_id in window_ids:
                embedding = db.get_embedding(window_id)
                window = db.get_window(window_id)
                recording = db.get_recording(window.recording_id)
                start, end = window.offsets

                f.write(
                    f"{recording.filename},{start},{end},"
                    f'"{",".join(map(str, embedding.tolist()))}"\n'
                )


def _ensure_deployment(
    db: sqlite_usearch_impl.SQLiteUSearchDB, dataset_name: str = DATASET_NAME
) -> int:
    """Ensure the BirdNET deployment exists and return its id."""
    from ml_collections import config_dict

    deployments = db.get_all_deployments(
        config_dict.create(eq={"name": "birdnet_default", "project": dataset_name})
    )
    if deployments:
        return deployments[0].id
    return db.insert_deployment(name="birdnet_default", project=dataset_name)


def _ensure_recording(
    db: sqlite_usearch_impl.SQLiteUSearchDB, fpath: str, deployment_id: int
) -> int:
    """Ensure the recording exists and return its id."""
    from ml_collections import config_dict

    recordings = db.get_all_recordings(
        config_dict.create(eq={"filename": fpath, "deployment_id": deployment_id})
    )
    if recordings:
        return recordings[0].id
    return db.insert_recording(filename=fpath, deployment_id=deployment_id)


def get_or_create_database(
    db_path: str, embedding_dim: int
) -> sqlite_usearch_impl.SQLiteUSearchDB:
    """Get the database object. Creates or opens the databse.
    Args:
        db: The path to the database.
    Returns:
        The database object.
    """
    import os

    from perch_hoplite.db import sqlite_usearch_impl

    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        return sqlite_usearch_impl.SQLiteUSearchDB.create(
            db_path=db_path,
            usearch_cfg=sqlite_usearch_impl.get_default_usearch_config(
                embedding_dim=embedding_dim
            ),
        )

    from birdnet_analyzer.search.core import reject_database_of_v2

    reject_database_of_v2(db_path)

    try:
        return sqlite_usearch_impl.SQLiteUSearchDB.create(db_path=db_path)
    except ValueError:
        return sqlite_usearch_impl.SQLiteUSearchDB.create(
            db_path=db_path,
            usearch_cfg=sqlite_usearch_impl.get_default_usearch_config(
                embedding_dim=embedding_dim
            ),
        )


def database_model_version(settings) -> ACOUSTIC_MODEL_VERSIONS:
    """The BirdNET version the embeddings of a database were extracted with.

    Databases written before the version was stored hold BirdNET 2.4 embeddings.
    """
    return settings.get(MODEL_VERSION_KEY, "2.4")


def _model_version_for(
    database: str, requested: ACOUSTIC_MODEL_VERSIONS | None
) -> ACOUSTIC_MODEL_VERSIONS:
    from birdnet_analyzer.config import DEFAULT_ACOUSTIC_MODEL_VERSION
    from birdnet_analyzer.search.core import read_database_settings

    try:
        stored = database_model_version(read_database_settings(database))
    except (FileNotFoundError, KeyError):
        return requested or DEFAULT_ACOUSTIC_MODEL_VERSION  # ty:ignore[invalid-return-type]

    if requested and requested != stored:
        raise ValueError(
            f"The database holds BirdNET {stored} embeddings and cannot take "
            f"BirdNET {requested} ones. Leave the version out to add to it, or "
            "name a new database."
        )

    return stored


def _check_database_settings(
    db: sqlite_usearch_impl.SQLiteUSearchDB,
    fmin: int = 0,
    fmax: int = 15000,
    audio_speed: float = 1.0,
    audio_root: str | None = None,
    model_version: ACOUSTIC_MODEL_VERSIONS = "3.0",
):
    from ml_collections import ConfigDict

    from birdnet_analyzer.embeddings.core import SETTINGS_KEY

    try:
        settings = db.get_metadata(SETTINGS_KEY)

        if (
            settings["BANDPASS_FMIN"] != fmin
            or settings["BANDPASS_FMAX"] != fmax
            or settings["AUDIO_SPEED"] != audio_speed
            or settings.get("AUDIO_ROOT") != audio_root
        ):
            raise ValueError(
                "Database settings do not match current configuration. DB Settings are:"
                f" fmin: {settings['BANDPASS_FMIN']}, fmax: {settings['BANDPASS_FMAX']}"
                f", audio_speed: {settings['AUDIO_SPEED']}, "
                f"audio_root: {settings.get('AUDIO_ROOT')}"
            )
    except KeyError:
        settings = ConfigDict(
            {
                "BANDPASS_FMIN": fmin,
                "BANDPASS_FMAX": fmax,
                "AUDIO_SPEED": audio_speed,
                "AUDIO_ROOT": audio_root,
                MODEL_VERSION_KEY: model_version,
            }
        )

        db.insert_metadata(SETTINGS_KEY, settings)
        db.commit()
