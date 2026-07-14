"""Module containing common function."""

import itertools
import os
from pathlib import Path

from birdnet_analyzer.config import ALLOWED_FILETYPES, CODES_FILE
from birdnet_analyzer.settings import write_error_log

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


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
    show_freq_axis=False,
):
    """
    Generate a spectrogram from an audio file.

    Parameters:
    path (str): The path to the audio file.
    show_freq_axis (bool): Whether to display the frequency scale (y-axis).

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

    return spectrogram_from_audio(
        s, sr, fig_num=fig_num, fig_size=fig_size, show_freq_axis=show_freq_axis
    )


def spectrogram_from_audio(s, sr, fig_num=None, fig_size=None, show_freq_axis=False):
    """
    Generate a spectrogram from an audio signal.

    Parameters:
    s: The signal
    sr: The sample rate
    show_freq_axis (bool): Whether to display the frequency scale (y-axis).

    Returns:
    matplotlib.figure.Figure: The generated spectrogram figure.
    """
    import librosa
    import librosa.display
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker
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

    D = librosa.stft(s, n_fft=1024, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    spec = librosa.display.specshow(
        S_db,
        ax=ax,
        sr=sr,
        n_fft=1024,
        hop_length=512,
        y_axis="linear" if show_freq_axis else None,
    )

    if show_freq_axis:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}")
        )
        ax.set_ylabel("Frequency (kHz)")
        f.tight_layout(pad=.5)
    else:
        ax.set_axis_off()
        f.tight_layout(pad=0)

    return spec.figure


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


def count_audio_files(path: str) -> int:
    """Counts all audio files in the given directory.

    Faster than ``collect_audio_files`` when only the number of files is needed,
    as it neither stores nor sorts the paths.

    Args:
        path: The directory to be searched.

    Returns:
        The number of audio files in the directory (recursively).
    """
    count = 0

    for _, _, flist in os.walk(path):
        for f in flist:
            if (
                not f.startswith(".")
                and f.rsplit(".", 1)[-1].lower() in ALLOWED_FILETYPES
            ):
                count += 1

    return count


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


def read_classifier_labels(classifier_file: str) -> list[str] | None:
    """Reads the labels belonging to a custom classifier.

    Looks for the label file next to the classifier, following the naming used when a
    custom classifier is trained.

    Args:
        classifier_file: Absolute path to the classifier file.

    Returns:
        The labels, or None if no label file was found.
    """
    base_name = os.path.splitext(classifier_file)[0]
    labels_file = base_name + "_Labels.txt"

    if not os.path.isfile(labels_file):
        labels_file = classifier_file.replace("Model_FP32.tflite", "Labels.txt")

    if not os.path.isfile(labels_file):
        return None

    return read_lines(labels_file, fail_on_blank_lines=True)


def list_subdirectories(path: str):
    """Lists all directories inside a path.

    Retrieves all the subdirectories in a given path without recursion.

    Args:
        path: Directory to be searched.

    Returns:
        A filter sequence containing the absolute paths to all directories.
    """
    return filter(lambda el: os.path.isdir(os.path.join(path, el)), os.listdir(path))


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
