"""Module containing common function."""

import itertools
import os
import sys
import traceback
from pathlib import Path

from birdnet_analyzer.config import ALLOWED_FILETYPES, CODES_FILE, ERROR_LOG_FILE

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
FROZEN = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def runtime_error_handler(f):
    """Decorator to catch runtime errors and write them to the error log.

    Args:
        f: The function to be decorated.

    Returns:
        The decorated function.
    """

    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as ex:
            write_error_log(ex)
            raise

    return wrapper


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


def spectrogram_from_file(
    path,
    fig_num=None,
    fig_size=None,
    offset=0,
    duration=None,
    fmin=None,
    fmax=None,
    speed=1.0,
    sig_fmin=0,
    sig_fmax=15000,
):
    """
    Generate a spectrogram from an audio file.

    Parameters:
    path (str): The path to the audio file.

    Returns:
    matplotlib.figure.Figure: The generated spectrogram figure.
    """
    from birdnet_analyzer import audio

    s, sr = audio.open_audio_file(
        path,
        offset=offset,
        duration=duration,
        fmin=fmin,
        fmax=fmax,
        speed=speed,
        sig_fmin=sig_fmin,
        sig_fmax=sig_fmax,
    )

    return spectrogram_from_audio(s, sr, fig_num=fig_num, fig_size=fig_size)


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

    D = librosa.stft(s, n_fft=1024, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    return librosa.display.specshow(S_db, ax=ax, n_fft=1024, hop_length=512).figure


def collect_audio_files(path: str, max_files: int | None = None):
    """Collects all audio files in the given directory.

    Args:
        path: The directory to be searched.

    Returns:
        A sorted list of all audio files in the directory.
    """
    files = []

    for root, _, flist in os.walk(path):
        for f in flist:
            if (
                not f.startswith(".")
                and f.rsplit(".", 1)[-1].lower() in ALLOWED_FILETYPES
            ):
                files.append(os.path.join(root, f))

                if max_files and len(files) >= max_files:
                    return sorted(files)

    return sorted(files)


def read_lines(
    path: str | Path | None, trim: bool = False, fail_on_blank_lines: bool = False
) -> list[str]:
    """Reads the lines into a list.

    Opens the file and reads its contents into a list.
    It is expected to have one line for each species or label.

    Args:
        path: Absolute path to the species file.

    Returns:
        A list of all species inside the file.
    """

    if not path:
        return []

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    cleaned_lines = []

    for line in lines:
        if not line and fail_on_blank_lines:
            raise ValueError(
                f"Blank lines are not allowed in species list\nFile: {path}"
            )

        cleaned_lines.append(line.strip() if trim else line)

    return cleaned_lines


def list_subdirectories(path: str):
    """Lists all directories inside a path.

    Retrieves all the subdirectories in a given path without recursion.

    Args:
        path: Directory to be searched.

    Returns:
        A filter sequence containing the absolute paths to all directories.
    """
    return filter(lambda el: os.path.isdir(os.path.join(path, el)), os.listdir(path))


def clear_error_log():
    """Clears the error log file.

    For debugging purposes.
    """
    if os.path.isfile(ERROR_LOG_FILE):
        os.remove(ERROR_LOG_FILE)


def write_error_log(ex: Exception):
    """Writes an exception to the error log.

    Formats the stacktrace and writes it in the error log file configured in the config.

    Args:
        ex: An exception that occurred.
    """
    import datetime

    with open(ERROR_LOG_FILE, "a") as elog:
        elog.write(
            datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            + "\n"
            + "".join(traceback.TracebackException.from_exception(ex).format())
            + "\n"
        )


def load_codes() -> dict[str, str]:
    """Loads the eBird codes.

    Returns:
        A dictionary containing the eBird codes.
    """
    import json

    with open(os.path.join(SCRIPT_DIR, CODES_FILE), encoding="utf-8") as cfile:
        return json.load(cfile)


def img2base64(path):
    import base64

    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def save_params_to_file(file_path, headers, values):
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
