"""Module to analyze audio samples."""

import json
import os

import numpy as np

import birdnet_analyzer.config as cfg
from birdnet_analyzer import audio, model

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def load_codes() -> dict[str, str]:
    """Loads the eBird codes.

    Returns:
        A dictionary containing the eBird codes.
    """
    with open(os.path.join(SCRIPT_DIR, cfg.CODES_FILE), encoding="utf-8") as cfile:
        return json.load(cfile)


def merge_consecutive_detections(results: dict[str, list], max_consecutive: int | None = None):
    """Merges consecutive detections of the same species.
    Uses the mean of the top-3 highest scoring predictions as
    confidence score for the merged detection.

    Args:
        results: The dictionary with {segment: scores}.
        max_consecutive: The maximum number of consecutive detections to merge.
                          If None, merge all consecutive detections.

    Returns:
        The dictionary with merged detections.
    """

    # If max_consecutive is 0 or 1, return original results
    if max_consecutive is not None and max_consecutive <= 1:
        return results

    # For each species, make list of timestamps and scores
    species = {}
    for timestamp, scores in results.items():
        for label, score in scores:
            if label not in species:
                species[label] = []
            species[label].append((timestamp, score))

    # Sort timestamps by start time for each species
    for label, timestamps in species.items():
        species[label] = sorted(timestamps, key=lambda t: float(t[0].split("-", 1)[0]))

    # Merge consecutive detections
    merged_results = {}
    for label in species:
        timestamps = species[label]

        # Check if end time of current detection is within the start time of the next detection
        i = 0
        while i < len(timestamps) - 1:
            start, end = timestamps[i][0].split("-", 1)
            next_start, next_end = timestamps[i + 1][0].split("-", 1)

            if float(end) >= float(next_start):
                # Merge detections
                merged_scores = [timestamps[i][1], timestamps[i + 1][1]]
                timestamps.pop(i)

                while i < len(timestamps) - 1 and float(next_end) >= float(timestamps[i + 1][0].split("-", 1)[0]):
                    if max_consecutive and len(merged_scores) >= max_consecutive:
                        break
                    merged_scores.append(timestamps[i + 1][1])
                    next_end = timestamps[i + 1][0].split("-", 1)[1]
                    timestamps.pop(i + 1)

                # Calculate mean of top 3 scores
                top_3_scores = sorted(merged_scores, reverse=True)[:3]
                merged_score = sum(top_3_scores) / len(top_3_scores)

                timestamps[i] = (f"{start}-{next_end}", merged_score)

            i += 1

        merged_results[label] = timestamps

    # Restore original format
    results = {}
    for label, timestamps in merged_results.items():
        for timestamp, score in timestamps:
            if timestamp not in results:
                results[timestamp] = []
            results[timestamp].append((label, score))

    return results


def get_raw_audio_from_file(fpath: str, offset, duration):
    """Reads an audio file and splits the signal into chunks.

    Args:
        fpath: Path to the audio file.

    Returns:
        The signal split into a list of chunks.
    """
    # Open file
    sig, rate = audio.open_audio_file(
        fpath,
        cfg.SAMPLE_RATE,
        offset,
        duration,
        cfg.BANDPASS_FMIN,
        cfg.BANDPASS_FMAX,
        cfg.AUDIO_SPEED,
    )

    # Split into raw audio chunks
    return audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)


def iterate_audio_chunks(fpath: str, embeddings: bool = False):
    """Iterates over audio chunks from a file.

    Args:
        fpath: Path to the audio file.
        offset: Offset in seconds to start reading the file.

    Yields:
        Chunks of audio data.
    """
    fileLengthSeconds = audio.get_audio_file_length(fpath)
    start, end = 0, cfg.SIG_LENGTH * cfg.AUDIO_SPEED
    duration = int(cfg.FILE_SPLITTING_DURATION / cfg.AUDIO_SPEED)

    while start < fileLengthSeconds and not np.isclose(start, fileLengthSeconds):
        chunks = get_raw_audio_from_file(fpath, start, duration)
        samples = []
        timestamps = []

        if not chunks:
            break

        for chunk_index, chunk in enumerate(chunks):
            t_start = start + (chunk_index * (cfg.SIG_LENGTH - cfg.SIG_OVERLAP) * cfg.AUDIO_SPEED)
            end = min(t_start + cfg.SIG_LENGTH * cfg.AUDIO_SPEED, fileLengthSeconds)

            # Add to batch
            samples.append(chunk)
            timestamps.append([round(t_start, 2), round(end, 2)])

            # Check if batch is full or last chunk
            if len(samples) < cfg.BATCH_SIZE and chunk_index < len(chunks) - 1:
                continue

            # Predict
            p = model.embeddings(samples) if embeddings else predict(samples)

            # Add to results
            for i in range(len(samples)):
                # Get timestamp
                s_start, s_end = timestamps[i]

                yield s_start, s_end, p[i]

            # Clear batch
            samples = []
            timestamps = []

        start += len(chunks) * (cfg.SIG_LENGTH - cfg.SIG_OVERLAP) * cfg.AUDIO_SPEED


def predict(samples):
    """Predicts the classes for the given samples.

    Args:
        samples: Samples to be predicted.

    Returns:
        The prediction scores.
    """
    # Prepare sample and pass through model
    data = np.array(samples, dtype="float32")
    prediction = model.predict(data)

    # Logits or sigmoid activations?
    if cfg.APPLY_SIGMOID and not cfg.USE_PERCH:
        prediction = model.flat_sigmoid(np.array(prediction), sensitivity=-1, bias=cfg.SIGMOID_SENSITIVITY)

    return prediction
