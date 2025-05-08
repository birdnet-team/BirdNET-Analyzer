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
from birdnet_analyzer import audio, model, utils
from birdnet_analyzer.analyze.utils import get_raw_audio_from_file
from birdnet_analyzer.embeddings.core import get_database

DATASET_NAME: str = "birdnet_analyzer_dataset"


def analyze_file(item, db: sqlite_usearch_impl.SQLiteUsearchDB):
    """Extracts the embeddings for a file.

    Args:
        item: (filepath, config)
    """

    import birdnet_analyzer.model as model
    import birdnet_analyzer.analyze.utils as analyze

    # Get file path and restore cfg
    fpath: str = item[0]
    cfg.set_config(item[1])

    offset = 0
    duration = cfg.FILE_SPLITTING_DURATION

    try:
        fileLengthSeconds = int(audio.get_audio_file_length(fpath))
    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}. File corrupt?\n", flush=True)
        utils.write_error_log(ex)

        return

    # Start time
    start_time = datetime.datetime.now()

    # Status
    print(f"Analyzing {fpath}", flush=True)

    source_id = fpath

    # Process each chunk
    try:
        while offset < fileLengthSeconds:
            chunks = get_raw_audio_from_file(fpath, offset, duration)
            start, end = offset, cfg.SIG_LENGTH + offset
            samples = []
            timestamps = []

            for c in range(len(chunks)):
                # Add to batch
                samples.append(chunks[c])
                timestamps.append([start, end])

                # Advance start and end
                start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
                end = start + cfg.SIG_LENGTH

                # Check if batch is full or last chunk
                if len(samples) < cfg.BATCH_SIZE and c < len(chunks) - 1:
                    continue

                # Prepare sample and pass through model
                data = np.array(samples, dtype="float32")
                e = model.embeddings(data)

                # Add to results
                for i in range(len(samples)):
                    # Get timestamp
                    s_start, s_end = timestamps[i]

                    # Check if embedding already exists
                    existing_embedding = db.get_embeddings_by_source(DATASET_NAME, source_id, np.array([s_start, s_end]))

                    if existing_embedding.size == 0:
                        # Get prediction
                        embeddings = e[i]

                        # Store embeddings
                        embeddings_source = hoplite.EmbeddingSource(DATASET_NAME, source_id, np.array([s_start, s_end]))

                        # Insert into database
                        db.insert_embedding(embeddings, embeddings_source)
                        db.commit()

                # Reset batch
                samples = []
                timestamps = []

            offset = offset + duration

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


def run(audio_input, database, overlap, audio_speed, fmin, fmax, threads, batchsize):
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

    db.db.close()
