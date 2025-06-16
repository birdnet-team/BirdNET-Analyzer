"""Module used to extract embeddings for samples."""

import datetime
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from ml_collections import ConfigDict
from perch_hoplite.db import interface as hoplite
from perch_hoplite.db import sqlite_usearch_impl
from tqdm import tqdm

import birdnet_analyzer.config as cfg
from birdnet_analyzer import utils
from birdnet_analyzer.analyze.utils import iterate_audio_chunks
from birdnet_analyzer.embeddings.core import get_database

DATASET_NAME: str = "birdnet_analyzer_dataset"


def analyze_file(item, db: sqlite_usearch_impl.SQLiteUsearchDB):
    """Extracts the embeddings for a file.

    Args:
        item: (filepath, config)
    """

    # Get file path and restore cfg
    fpath: str = item[0]
    cfg.set_config(item[1])

    # Start time
    start_time = datetime.datetime.now()

    # Status
    print(f"Analyzing {fpath}", flush=True)

    source_id = fpath

    # Process each chunk
    try:
        for s_start, s_end, embeddings in iterate_audio_chunks(fpath, embeddings=True):
            # Check if embedding already exists
            existing_embedding = db.get_embeddings_by_source(DATASET_NAME, source_id, np.array([s_start, s_end]))

            if existing_embedding.size == 0:
                # Store embeddings
                embeddings_source = hoplite.EmbeddingSource(DATASET_NAME, source_id, np.array([s_start, s_end]))

                # Insert into database
                db.insert_embedding(embeddings, embeddings_source)
                db.commit()

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}.", flush=True)
        utils.write_error_log(ex)

        return

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Finished {fpath} in {delta_time:.2f} seconds", flush=True)


def check_database_settings(db: sqlite_usearch_impl.SQLiteUsearchDB):
    try:
        settings = db.get_metadata("birdnet_analyzer_settings")
        if settings["BANDPASS_FMIN"] != cfg.BANDPASS_FMIN or settings["BANDPASS_FMAX"] != cfg.BANDPASS_FMAX or settings["AUDIO_SPEED"] != cfg.AUDIO_SPEED:
            raise ValueError(
                "Database settings do not match current configuration. DB Settings are: fmin:"
                + f"{settings['BANDPASS_FMIN']}, fmax: {settings['BANDPASS_FMAX']}, audio_speed: {settings['AUDIO_SPEED']}"
            )
    except KeyError:
        settings = ConfigDict({"BANDPASS_FMIN": cfg.BANDPASS_FMIN, "BANDPASS_FMAX": cfg.BANDPASS_FMAX, "AUDIO_SPEED": cfg.AUDIO_SPEED})
        db.insert_metadata("birdnet_analyzer_settings", settings)
        db.commit()


def create_file_output(output_path: str, db: sqlite_usearch_impl.SQLiteUsearchDB):
    """Creates a file output for the database.

    Args:
        output_path: Path to the output file.
        db: Database object.
    """
    # Check if output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Get all embeddings
    embedding_ids = db.get_embedding_ids()

    # Write embeddings to file
    for embedding_id in embedding_ids:
        embedding = db.get_embedding(embedding_id)
        source = db.get_embedding_source(embedding_id)

        # Get start and end time
        start, end = source.offsets

        source_id = source.source_id.rsplit(".", 1)[0]

        filename = f"{source_id}_{start}_{end}.birdnet.embeddings.txt"

        # Get the common prefix between the output path and the filename
        common_prefix = os.path.commonpath([output_path, os.path.dirname(filename)])
        relative_filename = os.path.relpath(filename, common_prefix)
        target_path = os.path.join(output_path, relative_filename)

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Write embedding values to a text file
        with open(target_path, "w") as f:
            f.write(",".join(map(str, embedding.tolist())))


def run(audio_input, database, overlap, audio_speed, fmin, fmax, threads, batchsize, file_output):
    ### Make sure to comment out appropriately if you are not using args. ###

    # Set input and output path
    cfg.INPUT_PATH = audio_input

    # Parse input files
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = utils.collect_audio_files(cfg.INPUT_PATH)
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    # Set overlap
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(overlap)))

    # Set audio speed
    cfg.AUDIO_SPEED = max(0.01, audio_speed)

    # Set bandpass frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))

    # Set number of threads
    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = max(1, int(threads))
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = max(1, int(threads))

    cfg.CPU_THREADS = 1  # TODO: with the current implementation, we can't use more than 1 thread

    # Set batch size
    cfg.BATCH_SIZE = max(1, int(batchsize))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(f, cfg.get_config()) for f in cfg.FILE_LIST]

    db = get_database(database)
    check_database_settings(db)

    # Analyze files
    if cfg.CPU_THREADS < 2:
        for entry in tqdm(flist):
            analyze_file(entry, db)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            tqdm(p.imap(partial(analyze_file, db=db), flist))

    if file_output:
        create_file_output(file_output, db)

    db.db.close()
