"""Module containing common function."""

import itertools
import os
import traceback
from pathlib import Path

import birdnet_analyzer.config as cfg


def batched(iterable, n, *, strict=False):
    # TODO: Remove this function when Python 3.12 is the minimum version
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def spectrogram_from_file(path, fig_num=None, fig_size=None, offset=0, duration=None, fmin=None, fmax=None, speed=1.0):
    """
    Generate a spectrogram from an audio file.

    Parameters:
    path (str): The path to the audio file.

    Returns:
    matplotlib.figure.Figure: The generated spectrogram figure.
    """
    import birdnet_analyzer.audio as audio

    # s, sr = librosa.load(path, offset=offset, duration=duration)
    s, sr = audio.open_audio_file(path, offset=offset, duration=duration, fmin=fmin, fmax=fmax, speed=speed)

    return spectrogram_from_audio(s, sr, fig_num, fig_size)


def spectrogram_from_audio(s, sr, fig_num=None, fig_size=None):
    """
    Generate a spectrogram from an audio signal.

    Parameters:
    s: The signal
    sr: The sample rate

    Returns:
    matplotlib.figure.Figure: The generated spectrogram figure.
    """
    import librosa
    import librosa.display
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("agg")

    if isinstance(fig_size, tuple):
        f = plt.figure(fig_num, figsize=fig_size)
    elif fig_size == "auto":
        duration = librosa.get_duration(y=s, sr=sr)
        width = min(12, max(3, duration / 10))
        f = plt.figure(fig_num, figsize=(width, 3))
    else:
        f = plt.figure(fig_num)

    f.clf()

    ax = f.add_subplot(111)

    ax.set_axis_off()
    f.tight_layout(pad=0)

    D = librosa.stft(s, n_fft=1024, hop_length=512)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    return librosa.display.specshow(S_db, ax=ax, n_fft=1024, hop_length=512).figure


def collect_audio_files(path: str, max_files: int = None):
    """Collects all audio files in the given directory.

    Args:
        path: The directory to be searched.

    Returns:
        A sorted list of all audio files in the directory.
    """
    # Get all files in directory with os.walk
    files = []

    for root, _, flist in os.walk(path):
        for f in flist:
            if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES:
                files.append(os.path.join(root, f))

                if max_files and len(files) >= max_files:
                    return sorted(files)

    return sorted(files)


def collect_all_files(path: str, filetypes: list[str], pattern: str = ""):
    """Collects all files of the given filetypes in the given directory.

    Args:
        path: The directory to be searched.
        filetypes: A list of filetypes to be collected.

    Returns:
        A sorted list of all files in the directory.
    """

    files = []

    for root, _, flist in os.walk(path):
        for f in flist:
            if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in filetypes and (pattern in f or not pattern):
                files.append(os.path.join(root, f))

    return sorted(files)


def read_lines(path: str):
    """Reads the lines into a list.

    Opens the file and reads its contents into a list.
    It is expected to have one line for each species or label.

    Args:
        path: Absolute path to the species file.

    Returns:
        A list of all species inside the file.
    """
    return Path(path).read_text(encoding="utf-8").splitlines() if path else []


def list_subdirectories(path: str):
    """Lists all directories inside a path.

    Retrieves all the subdirectories in a given path without recursion.

    Args:
        path: Directory to be searched.

    Returns:
        A filter sequence containing the absolute paths to all directories.
    """
    return filter(lambda el: os.path.isdir(os.path.join(path, el)), os.listdir(path))


def save_to_cache(cache_file: str, x_train, y_train, labels: list[str]):
    """Saves the training data to a cache file.

    Args:
        cache_file: The path to the cache file.
        x_train: The training samples.
        y_train: The training labels.
        labels: The list of labels.
    """
    import numpy as np

    # Create cache directory
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Save to cache
    np.savez_compressed(
        cache_file,
        x_train=x_train,
        y_train=y_train,
        labels=labels,
        binary_classification=cfg.BINARY_CLASSIFICATION,
        multi_label=cfg.MULTI_LABEL,
    )


def load_from_cache(cache_file: str):
    """Loads the training data from a cache file.

    Args:
        cache_file: The path to the cache file.

    Returns:
        A tuple of (x_train, y_train, labels).

    """
    import numpy as np

    # Load from cache
    cache = np.load(cache_file, allow_pickle=True)

    # Get data
    x_train = cache["x_train"]
    y_train = cache["y_train"]
    labels = cache["labels"]
    binary_classification = bool(cache["binary_classification"]) if "binary_classification" in cache.keys() else False
    multi_label = bool(cache["multi_label"]) if "multi_label" in cache.keys() else False

    return x_train, y_train, labels, binary_classification, multi_label


def clear_error_log():
    """Clears the error log file.

    For debugging purposes.
    """
    if os.path.isfile(cfg.ERROR_LOG_FILE):
        os.remove(cfg.ERROR_LOG_FILE)


def write_error_log(ex: Exception):
    """Writes an exception to the error log.

    Formats the stacktrace and writes it in the error log file configured in the config.

    Args:
        ex: An exception that occurred.
    """
    import datetime

    with open(cfg.ERROR_LOG_FILE, "a") as elog:
        elog.write(
            datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            + "\n"
            + "".join(traceback.TracebackException.from_exception(ex).format())
            + "\n"
        )


def img2base64(path):
    import base64

    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def save_params(file_path, headers, values):
    """Saves the params used to train the custom classifier.

    The hyperparams will be saved to disk in a file named 'model_params.csv'.

    Args:
        file_path: The path to the file.
        headers: The headers of the csv file.
        values: The values of the csv file.
    """
    import csv

    with open(file_path, "w", newline="") as paramsfile:
        paramswriter = csv.writer(paramsfile)
        paramswriter.writerow(headers)
        paramswriter.writerow(values)


def save_result_file(result_path: str, out_string: str):
    """Saves the result to a file.

    Args:
        result_path: The path to the result file.
        out_string: The string to be written to the file.
    """

    # Make directory if it doesn't exist
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # Write the result to the file
    with open(result_path, "w", encoding="utf-8") as rfile:
        rfile.write(out_string)
