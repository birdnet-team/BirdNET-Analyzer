"""Module used to extract embeddings for samples."""

import multiprocessing as mp
import os
import time

import numpy as np
from ml_collections import ConfigDict
from perch_hoplite.db import interface as hoplite
from perch_hoplite.db import sqlite_usearch_impl
from tqdm import tqdm

import birdnet_analyzer.config as cfg
from birdnet_analyzer import utils
from birdnet_analyzer.analyze.utils import iterate_audio_chunks
from birdnet_analyzer.embeddings.core import get_or_create_database

DATASET_NAME: str = "birdnet_analyzer_dataset"
COMMIT_BS_SIZE = 512


def analyze_file_core(fpath, config):
    results = []
    cfg.set_config(config)

    # Process each chunk
    try:
        for s_start, s_end, embeddings in iterate_audio_chunks(fpath, embeddings=True):
            results.append((fpath, s_start, s_end, embeddings))
    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}.", flush=True)
        utils.write_error_log(ex)

    return results


def analyze_file(items):
    """Extracts the embeddings for a file.

    Args:
        item: (filepath, config)
    """
    results = []

    for fpath, config in items:
        results.extend(analyze_file_core(fpath, config))

    return results


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


def create_csv_output(output_path: str, database: str):
    """Creates a CSV output for the database.

    Args:
        output_path: Path to the output file.
        db: Database object.
    """

    db = get_or_create_database(database)
    parent_dir = os.path.dirname(output_path)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    embedding_ids = db.get_embedding_ids()

    csv_content = "file_path,start,end,embedding\n"

    for embedding_id in embedding_ids:
        embedding = db.get_embedding(embedding_id)
        source = db.get_embedding_source(embedding_id)

        start, end = source.offsets

        csv_content += f'{source.source_id},{start},{end},"{",".join(map(str, embedding.tolist()))}"\n'

    with open(output_path, "w") as f:
        f.write(csv_content)


def create_file_output(output_path: str, database: str):
    """Creates a file output for the database.

    Args:
        output_path: Path to the output file.
        db: Database object.
    """

    db = get_or_create_database(database)

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


def consume_embedding(fpath, s_start, s_end, embeddings, db: sqlite_usearch_impl.SQLiteUsearchDB):
    # Check if embedding already exists
    existing_embedding = db.get_embeddings_by_source(DATASET_NAME, fpath, np.array([s_start, s_end]))

    if existing_embedding.size == 0:
        # Store embeddings
        embeddings_source = hoplite.EmbeddingSource(DATASET_NAME, fpath, np.array([s_start, s_end]))

        # Insert into database
        db.insert_embedding(embeddings, embeddings_source)

        return True

    return False


def consumer(q: mp.Queue, stop_at, database: str):
    batchsize = COMMIT_BS_SIZE
    batch = 0
    break_signal = True
    db = get_or_create_database(database)

    check_database_settings(db)

    while break_signal:
        if not q.empty():
            results = q.get()

            for fpath, s_start, s_end, embeddings in results:
                if fpath == stop_at:
                    break_signal = False
                    break

                if consume_embedding(fpath, s_start, s_end, embeddings, db):
                    batch += 1

                if batch >= batchsize:
                    db.commit()
                    batch = 0
        else:
            time.sleep(0.1)

    db.commit()
    db.db.close()


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

    # Set batch size
    cfg.BATCH_SIZE = max(1, int(batchsize))

    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = [(f, cfg.get_config()) for f in cfg.FILE_LIST]

    if cfg.CPU_THREADS < 2:
        # Force single core
        batchsize = COMMIT_BS_SIZE
        batch = 0
        db = get_or_create_database(database)
        check_database_settings(db)

        for fpath, config in tqdm(flist, desc="Files processed"):
            for _, s_start, s_end, embeddings in analyze_file_core(fpath, config):
                if consume_embedding(fpath, s_start, s_end, embeddings, db):
                    batch += 1

                if batch >= batchsize:
                    db.commit()
                    batch = 0

        db.commit()
        db.db.close()
    else:
        chunksize = 2
        queue = mp.Queue(maxsize=10_000)
        consumer_process = mp.Process(target=consumer, args=(queue, "STOP", database))
        consumer_process.start()

        # One less process for the pool, because we use one extra for the consumer
        with mp.Pool(processes=cfg.CPU_THREADS - 1) as pool:
            delta = chunksize
            processed_files = set()
            with tqdm(total=len(flist), desc="Files processed") as pbar:
                # Instead of chunk_size arg, manual splitting, because this reduces the overhead for the iterable.
                for res in pool.imap_unordered(analyze_file, [flist[i : i + delta] for i in range(0, len(flist), delta)], chunksize=1):
                    num_already_processed = len(processed_files)
                    processed_files.update([r[0] for r in res])
                    delta = len(processed_files) - num_already_processed
                    queue.put(res)
                    pbar.update(delta)

        queue.put([("STOP", 0, 0, None)])
        consumer_process.join()

    if file_output:
        create_csv_output(file_output, database)
