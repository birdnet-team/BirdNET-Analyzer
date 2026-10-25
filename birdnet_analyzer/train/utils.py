"""Module for training a custom classifier.

Can be used to train a custom classifier with new training data.
"""

from __future__ import annotations

import csv
import errno
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, NamedTuple, get_args

import numpy as np
import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold

from birdnet_analyzer import audio, base_models, model, model_utils, utils
from birdnet_analyzer.config import (
    ALLOWED_FILETYPES,
    AUTOTUNE_METRICS,
    DEFAULT_ACOUSTIC_MODEL_VERSION,
    NON_EVENT_CLASSES,
    RANDOM_SEED,
    TRAIN_PARAMS_SUFFIX,
)
from birdnet_analyzer.model_utils import GLOBAL_PREFETCH_RATIO
from birdnet_analyzer.params import TRAIN_PARAMS, by_header

# Encode batch size per base model, measured on CPU inside a warm session: the 2.4
# tflite interpreter re-plans on every batch-shape change, so small batches win
# (16 ms/segment at 1-4, 34 ms at 32); the 3.0 ONNX model gains from large ones
# (210 ms/segment at 1, 111 ms at 32).
ENCODE_BATCH_SIZES = {"2.4": 4, "3.0": 32}
# Decoded segments buffered before a batched encode flush. Bounds peak buffered-audio
# memory (~0.5 MiB per 3 s float32 segment at 48 kHz) while amortising call overhead.
ENCODE_CHUNK_SIZE = 1024

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Generator

    from birdnet_analyzer.config import (
        SAMPLE_CROP_MODES,
        TRAINED_MODEL_OUTPUT_FORMATS,
        TRAINED_MODEL_SAVE_MODES,
        UPSAMPLING_MODES,
    )


def save_sample_counts(labels, y_train, output_dir: str):
    """
    Saves the count of samples per label combination to a CSV file.

    The function creates a dictionary where the keys are label combinations
    (joined by '+') and the values are the counts of samples for each combination.
    It then writes this information to a CSV file named "sample_counts.csv" with two
    columns: "Label" and "Count".

    Args:
        labels (list of str): List of label names corresponding to the columns in
                              y_train.
        y_train (numpy.ndarray): 2D array where each row is a binary vector indicating
                                 the presence (1) or absence (0) of each label.
    """
    samples_per_label = {}
    label_combinations = np.unique(y_train, axis=0)

    for label_combination in label_combinations:
        label = "+".join(
            [
                labels[i]
                for i in range(len(label_combination))
                if label_combination[i] == 1
            ]
        )
        samples_per_label[label] = np.sum(np.all(y_train == label_combination, axis=1))

    csv_file_path = output_dir + "_sample_counts.csv"

    with open(csv_file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Label", "Count"])

        for label, count in samples_per_label.items():
            writer.writerow([label, count])


def _read_and_crop_file(
    f,
    label_vector,
    sample_rate=48000,
    fmin=0,
    fmax=15000,
    audio_speed=1.0,
    crop_mode: SAMPLE_CROP_MODES = "center",
    sig_length=3.0,
    overlap=0.0,
    min_len=1.0,
):
    """Decode and crop one audio file into fixed-length signal segments.

    This performs only the CPU-bound audio work (decode, resample, bandpass, crop) and
    intentionally does *not* compute embeddings, so it can be run in parallel across
    many files. The resulting segments are fed to the encoding pipeline in a single
    batched call by the caller.

    Any future per-sample audio augmentation should be applied to ``sig_splits`` here,
    before the segments leave this function.

    Args:
        f: Path to the audio file.
        label_vector: The label vector for the file.
        sample_rate: Target sample rate for decoding.
        fmin: Minimum frequency for the bandpass filter.
        fmax: Maximum frequency for the bandpass filter.
        audio_speed: Speed factor applied while decoding.
        crop_mode: Mode for cropping audio samples.
        sig_length: Segment length in seconds.
        overlap: Overlap between segments in seconds.
        min_len: Minimum length of a segment in seconds.

    Returns:
        A tuple ``(sig_splits, labels)`` where ``sig_splits`` is a list of float32
        segment arrays and ``labels`` repeats the file's ``label_vector`` once per
        segment. Both lists are empty if the file could not be loaded or produced no
        usable segment (e.g. a signal shorter than ``min_len``).
    """
    try:
        sig, rate = audio.open_audio_file(
            f,
            sample_rate=sample_rate,
            duration=sig_length if crop_mode == "first" else None,
            fmin=fmin,
            fmax=fmax,
            speed=audio_speed,
        )
    except Exception as e:
        logger.error(f"\t Error when loading file {f}\n\t {e}", exc_info=e)
        return [], []

    if crop_mode == "center":
        sig_splits = [audio.crop_center(sig, rate, sig_length)]
    elif crop_mode == "first":
        # split_signal returns [] for signals shorter than min_len; slicing (not
        # indexing [0]) keeps such files fail-soft instead of raising IndexError.
        sig_splits = audio.split_signal(sig, rate, sig_length, overlap, min_len)[:1]
    elif crop_mode == "smart":
        sig_splits = audio.smart_crop_signal(sig, rate, sig_length, overlap, min_len)
    else:
        sig_splits = audio.split_signal(sig, rate, sig_length, overlap, min_len)

    sig_splits = [np.asarray(s, dtype="float32") for s in sig_splits]
    labels = [label_vector] * len(sig_splits)

    return sig_splits, labels


def _check_input_folders(audio_input: str, train_folders: list[str]):
    """Reject training folders without supported audio files before model setup."""
    empty_folders = []

    for folder in train_folders:
        folder_path = os.path.join(audio_input, folder)
        has_audio_file = any(
            entry.is_file()
            and not entry.name.startswith(".")
            and entry.name.rsplit(".", 1)[-1].lower() in ALLOWED_FILETYPES
            for entry in os.scandir(folder_path)
        )

        if not has_audio_file:
            empty_folders.append(folder)

    if empty_folders:
        folder_list = ", ".join(sorted(empty_folders))
        raise ValueError(
            "The following training data folders do not contain any supported audio "
            f"files: {folder_list}",
            "validation-no-audio-files-in-training-folders",
            folder_list,
        )


def _check_cache_destination(path: str):
    """Fail before the data is embedded if the cache file cannot be written."""
    if not path.endswith(".npz"):
        path += ".npz"  # as np.savez does

    if os.path.isdir(path):
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)

    directory = os.path.dirname(os.path.abspath(path))

    while not os.path.isdir(directory) and os.path.dirname(directory) != directory:
        if os.path.exists(directory):
            raise NotADirectoryError(
                errno.ENOTDIR, os.strerror(errno.ENOTDIR), directory
            )

        directory = os.path.dirname(directory)

    with tempfile.TemporaryFile(dir=directory):
        pass


def _load_training_data(
    audio_input: str,
    test_data: str | None = None,
    upsampling_ratio: float = 0.0,
    upsampling_mode: UPSAMPLING_MODES = "repeat",
    fmin=0,
    fmax=15000,
    audio_speed=1.0,
    crop_mode: SAMPLE_CROP_MODES = "center",
    overlap=0.0,
    min_len=1.0,
    threads=1,
    save_cache_to: str | None = None,
    progress_callback=None,
    model_version: str = DEFAULT_ACOUSTIC_MODEL_VERSION,
):
    """Loads the data for training.

    Reads all subdirectories of "audio_input" and uses their names as new labels.

    These directories should contain all the training data for each label.

    If a cache file is provided, the training data is loaded from there.

    Args:
        audio_input: Path to the training data directory or path to a cache file
                     ("train_cache.npz" for example).
        test_data: Path to the test data directory. Defaults to None.
        upsampling_ratio: Ratio for upsampling underrepresented classes.
                          Defaults to 0.0.
        progress_callback: A callback function for monitoring progress during data
                           loading.
        fmin: Minimum frequency for audio processing. Defaults to 0.0.
        fmax: Maximum frequency for audio processing. Defaults to 15000.0.
        audio_speed: Speed factor for audio playback. Defaults to 1.0.
        crop_mode: Mode for cropping audio samples. Defaults to "center".
        overlap: Overlap between audio chunks. Defaults to 0.0.
        min_len: Minimum length of audio chunks. Defaults to 1.0.
        model_version: The BirdNET version whose embeddings become the samples. A
                       cache file was embedded with a fixed version, which is
                       returned instead.

    Returns:
        A tuple of (x_train, y_train, x_test, y_test, labels, is_binary, is_multi_label,
        model_version)
    """

    if audio_input.endswith(".npz"):
        return _load_from_cache(audio_input)

    train_folders = list(utils.list_subdirectories(audio_input))
    labels: list[str] = []

    for folder in train_folders:
        for label in utils.parse_folder_labels(folder):
            if label not in labels:
                labels.append(label)

    labels = sorted(labels)
    valid_labels = [
        label
        for label in labels
        if label.lower() not in NON_EVENT_CLASSES and not label.startswith("-")
    ]
    is_binary = len(valid_labels) == 1
    is_multi_label = len(valid_labels) > 1 and any("," in f for f in train_folders)

    if is_binary:
        if any(f for f in train_folders if f.startswith("-")):
            raise ValueError(
                "Negative labels can't be used with binary classification",
                "validation-no-negative-samples-in-binary-classification",
            )
        if not any(f for f in train_folders if f.lower() in NON_EVENT_CLASSES):
            raise ValueError(
                "Non-event samples are required for binary classification",
                "validation-non-event-samples-required-in-binary-classification",
            )

    if is_multi_label and upsampling_ratio > 0 and upsampling_mode != "repeat":
        raise ValueError(
            "Only repeat-upsampling is available for multi-label",
            "validation-only-repeat-upsampling-for-multi-label",
        )

    _check_input_folders(audio_input, train_folders)

    if save_cache_to:
        _check_cache_destination(save_cache_to)

    x_train, y_train, x_test, y_test = [], [], [], []
    # Fail here rather than at export time, after the data has been embedded.
    model.base_model(model_version)
    acoustic_model = model_utils._load_acoustic_for_embeddings(model_version)
    model_sr = int(acoustic_model.get_sample_rate())

    # librosa/scipy release the GIL for the heavy numeric work, so threaded decode
    # workers parallelise well without multiprocessing's pickling/spawn overhead.
    n_decode_workers = threads if threads and threads > 0 else (os.cpu_count() or 1)

    # Bandpass and speed are already applied in open_audio_file (see
    # _read_and_crop_file), so the encoding session uses no-op defaults (fmin=0,
    # fmax=15000, speed=1.0) to avoid applying them twice.
    with acoustic_model.encode_session(
        batch_size=ENCODE_BATCH_SIZES[model_version],
        n_workers=None,
        n_producers=1,
        progress_callback=None,
        prefetch_ratio=GLOBAL_PREFETCH_RATIO,
    ) as session:

        def load_data(data_path, allowed_folders):
            # Accumulated embeddings/labels, plus a buffer of not-yet-encoded segments.
            x_chunks: list[np.ndarray] = []
            y_chunks: list[np.ndarray] = []
            seg_buffer: list[np.ndarray] = []
            lab_buffer: list[np.ndarray] = []

            def flush():
                """Encode buffered segments in one batched call, keeping valid rows."""
                if not seg_buffer:
                    return
                embeddings, valid = model_utils.encode_arrays_batched(
                    session, [(s, model_sr) for s in seg_buffer]
                )
                labels = np.asarray(lab_buffer, dtype="float32")
                x_chunks.append(embeddings[valid])
                y_chunks.append(labels[valid])
                seg_buffer.clear()
                lab_buffer.clear()

            folders = sorted(utils.list_subdirectories(data_path))

            for folder in folders:
                if folder not in allowed_folders:
                    continue

                label_vector = np.zeros((len(valid_labels),), dtype="float32")

                for label in utils.parse_folder_labels(folder):
                    if label.lower() not in NON_EVENT_CLASSES and not label.startswith(
                        "-"
                    ):
                        label_vector[valid_labels.index(label)] = 1
                    elif label.startswith("-") and label[1:] in valid_labels:
                        # Negative labels need to be contained in the valid labels
                        label_vector[valid_labels.index(label[1:])] = -1

                files = list(
                    filter(
                        os.path.isfile,
                        (
                            os.path.join(data_path, folder, f)
                            for f in sorted(os.listdir(os.path.join(data_path, folder)))
                            if not f.startswith(".")
                            and f.rsplit(".", 1)[-1].lower() in ALLOWED_FILETYPES
                        ),
                    )
                )

                num_files_processed = 0

                # Phase A: decode + crop all files in this folder in parallel.
                # Phase B: encode buffered segments in batched flushes (across folders).
                with (
                    tqdm.tqdm(
                        total=len(files), desc=f" - loading '{folder}'", unit="f"
                    ) as progress_bar,
                    ThreadPoolExecutor(max_workers=n_decode_workers) as executor,
                ):
                    futures = [
                        executor.submit(
                            _read_and_crop_file,
                            f,
                            label_vector,
                            sample_rate=model_sr,
                            fmin=fmin,
                            fmax=fmax,
                            audio_speed=audio_speed,
                            crop_mode=crop_mode,
                            overlap=overlap,
                            min_len=min_len,
                        )
                        for f in files
                    ]

                    # Consume in submission order (not as_completed) for a deterministic
                    # sample order: downstream stratified k-fold and shuffles use fixed
                    # seeds and rely on a stable input order.
                    for fut in futures:
                        sig_splits, labels = fut.result()
                        seg_buffer.extend(sig_splits)
                        lab_buffer.extend(labels)

                        num_files_processed += 1
                        progress_bar.update(1)

                        if progress_callback:
                            progress_callback(num_files_processed, len(files), folder)

                        if len(seg_buffer) >= ENCODE_CHUNK_SIZE:
                            flush()

            flush()

            if not x_chunks:
                return (
                    np.array([], dtype="float32"),
                    np.array([], dtype="float32"),
                )

            return (
                np.concatenate(x_chunks).astype("float32"),
                np.concatenate(y_chunks).astype("float32"),
            )

        x_train, y_train = load_data(audio_input, train_folders)

        if test_data and test_data != audio_input:
            test_folders = sorted(utils.list_subdirectories(test_data))
            allowed_test_folders = [
                folder
                for folder in test_folders
                if folder in train_folders and not folder.startswith("-")
            ]
            x_test, y_test = load_data(test_data, allowed_test_folders)
        else:
            x_test = np.array([])
            y_test = np.array([])

    if save_cache_to:
        try:
            _save_to_cache(
                save_cache_to,
                x_train,
                y_train,
                x_test,
                y_test,
                valid_labels,
                overlap=overlap,
                fmin=fmin,
                fmax=fmax,
                audio_speed=audio_speed,
                crop_mode=crop_mode,
                is_binary=is_binary,
                is_multi_label=is_multi_label,
                model_version=model_version,
            )
        except Exception as e:
            logger.error(f"\t...error saving cache: {e}")
            raise

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        valid_labels,
        is_binary,
        is_multi_label,
        model_version,
    )


class _FloatRange(NamedTuple):
    low: float
    high: float
    step: float | None = None
    log: bool = False


def _search_space(is_multi_label: bool) -> dict[str, list | _FloatRange]:
    return {
        "hidden_units": [0, 128, 256, 512, 1024, 2048],
        "dropout": [0.0, 0.25, 0.33, 0.5, 0.75, 0.9],
        "batch_size": [8, 16, 32, 64, 128],
        "learning_rate": _FloatRange(1e-4, 1e-1, log=True),
        "upsampling_ratio": _FloatRange(0.0, 1.0, step=0.25),
        "upsampling_mode": ["repeat"]
        if is_multi_label
        else ["repeat", "mean", "linear"],
        "weight_decay": _FloatRange(1e-6, 1e-2, log=True),
        "mixup": [True, False],
        "label_smoothing": [True, False],
        "focal_loss": [True, False],
        "focal_loss_gamma": _FloatRange(0.5, 4.0, step=0.5),
        "focal_loss_alpha": _FloatRange(0.1, 0.9, step=0.05),
    }


def _suggest(trial, space: dict[str, list | _FloatRange]) -> dict:
    return {
        name: trial.suggest_categorical(name, choices)
        if isinstance(choices, list)
        else trial.suggest_float(
            name, choices.low, choices.high, step=choices.step, log=choices.log
        )
        for name, choices in space.items()
    }


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _snap_to_search_space(defaults: dict, space: dict[str, list | _FloatRange]) -> dict:
    """Moves the defaults onto the search space, as optuna requires of a seed trial.

    Numbers go to the nearest value the search allows; any other value that is not a
    choice is left out, so the sampler picks it.
    """
    snapped = {}

    for name, value in defaults.items():
        choices = space[name]

        if isinstance(choices, _FloatRange):
            new = min(max(value, choices.low), choices.high)

            if choices.step:
                steps = round((new - choices.low) / choices.step)
                new = round(choices.low + steps * choices.step, 10)
        elif value in choices:
            new = value
        elif _is_number(value) and all(map(_is_number, choices)):
            new = min(choices, key=lambda choice: abs(choice - value))
        else:
            logger.info(
                "Autotune: %s=%r is not in the search space, the search picks it.",
                name,
                value,
            )
            continue

        if new != value:
            logger.info(
                "Autotune: starting with %s=%r, the nearest searched value to %r.",
                name,
                new,
                value,
            )

        snapped[name] = new

    return snapped


def _stratification_labels(
    y: np.ndarray, is_binary: bool, is_multi_label: bool
) -> np.ndarray:
    if is_binary:
        return (y[:, 0] > 0).astype(int)

    if is_multi_label:
        return np.sum(y > 0, axis=1)

    num_classes = y.shape[1]
    no_positive = np.where(np.any(y < 0, axis=1), num_classes + 1, num_classes)

    return np.where(np.any(y > 0, axis=1), np.argmax(y, axis=1), no_positive)


def _kfold_splits(
    y: np.ndarray, n_splits: int, n_repeats: int, is_binary: bool, is_multi_label: bool
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yields the training and validation row indices of each fold."""
    negative = np.any(y < 0, axis=1)
    candidates = np.flatnonzero(~negative)
    folds = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_SEED
    )
    strata = _stratification_labels(y[candidates], is_binary, is_multi_label)

    for train_idx, val_idx in folds.split(candidates, strata):
        yield (
            np.concatenate([candidates[train_idx], np.flatnonzero(negative)]),
            candidates[val_idx],
        )


def _without_negative_rows(x: np.ndarray, y: np.ndarray):
    if len(y) == 0:
        return x, y

    keep = ~np.any(y < 0, axis=1)

    return x[keep], y[keep]


def train_model(
    audio_input: str,
    output: str = "checkpoints/custom/Custom_Classifier",
    test_data: str | None = None,
    crop_mode: SAMPLE_CROP_MODES = "center",
    overlap: float = 0.0,
    epochs: int = 50,
    batch_size: int = 32,
    val_split: float = 0.2,
    learning_rate: float = 0.0001,
    weight_decay: float = 0.004,
    use_focal_loss: bool = False,
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: float = 0.25,
    hidden_units: int = 0,
    dropout: float = 0.0,
    label_smoothing: bool = False,
    mixup: bool = False,
    upsampling_ratio: float = 0.0,
    upsampling_mode: UPSAMPLING_MODES = "repeat",
    model_formats: list[TRAINED_MODEL_OUTPUT_FORMATS]
    | TRAINED_MODEL_OUTPUT_FORMATS = "tflite",
    model_save_mode: TRAINED_MODEL_SAVE_MODES = "replace",
    save_cache_to: str | None = None,
    threads: int = 1,
    fmin: float = 0.0,
    fmax: float = 15000.0,
    audio_speed: float = 1.0,
    autotune: bool = False,
    autotune_trials: int = 50,
    autotune_n_splits: int = 1,
    autotune_n_repeats: int = 1,
    autotune_metric: AUTOTUNE_METRICS = "val_AUPRC",
    birdnet: str = DEFAULT_ACOUSTIC_MODEL_VERSION,
    on_epoch_end=None,
    on_trial_result=None,
    on_data_load_end=None,
):
    """Trains a custom classifier.

    Args:
        audio_input: Training data folder, whose subfolder names are the labels, or a
            cache file.
        output: Path of the trained classifier. Defaults to
            "checkpoints/custom/Custom_Classifier".
        test_data: Test data folder, only used to evaluate the trained classifier.
            Defaults to None.
        crop_mode: "center", "first", "segments" or "smart". Defaults to "center".
        overlap: Overlap of the segments in seconds for the "segments" and "smart"
            crop modes. Defaults to 0.0.
        epochs: Maximum number of epochs. Defaults to 50.
        batch_size: Training batch size. Defaults to 32.
        val_split: Fraction of the training data used for validation, above 0.
            Defaults to 0.2.
        learning_rate: Learning rate. Defaults to 0.0001.
        weight_decay: AdamW weight decay. Defaults to 0.004.
        use_focal_loss: Whether to use focal loss. Defaults to False.
        focal_loss_gamma: Focal loss gamma. Defaults to 2.0.
        focal_loss_alpha: Focal loss alpha. Defaults to 0.25.
        hidden_units: Hidden units; above 0 the classifier has two layers.
            Defaults to 0.
        dropout: Dropout rate. Defaults to 0.0.
        label_smoothing: Whether to use label smoothing. Defaults to False.
        mixup: Whether to use mixup. Defaults to False.
        upsampling_ratio: Ratio for upsampling minority classes. Defaults to 0.0.
        upsampling_mode: "repeat", "linear", "mean" or "smote". Defaults to "repeat".
        model_formats: One or more of "tflite", "raven" and "detached" (the
            classifier alone, taking embeddings). Defaults to "tflite".
        model_save_mode: "replace" the base model's classes or "append" to them.
            Defaults to "replace".
        save_cache_to: Path of a cache file to save the training data to.
            Defaults to None.
        threads: Number of threads decoding the training audio. Defaults to 1.
        fmin: Lower bandpass frequency in Hz. Defaults to 0.0.
        fmax: Upper bandpass frequency in Hz. Defaults to 15000.0.
        audio_speed: Speed factor applied to the audio. Defaults to 1.0.
        autotune: Whether to tune the hyperparameters first. Defaults to False.
        autotune_trials: Number of tuning trials. Defaults to 50.
        autotune_n_splits: Cross-validation folds per trial; 1 uses the validation
            split. Defaults to 1.
        autotune_n_repeats: Repetitions of the folds per trial. Defaults to 1.
        autotune_metric: "val_AUPRC" or "val_AUROC", optimized by the tuning and the
            early stopping. Defaults to "val_AUPRC".
        birdnet: The BirdNET version to train on. Its embeddings are the classifier's
            input, so the classifier only works with that version. A cache file keeps
            the version it was embedded with. Defaults to the newest version.
        on_epoch_end: Called with ``(epoch, logs)`` after each epoch of the final
            training. Defaults to None.
        on_trial_result: Called with the number of finished tuning trials, 0 before
            the first. Defaults to None.
        on_data_load_end: Called with ``(files_done, files_total, folder)`` for each
            loaded file of a training or test data folder. Defaults to None.

    Returns:
        A tuple ``(history, metrics)``: the keras ``History`` of the final training,
        and a dict of metrics on ``test_data``, or None without test data.
    """
    if autotune_metric not in get_args(AUTOTUNE_METRICS):
        raise ValueError(
            f"Unsupported autotune metric {autotune_metric!r}, use one of "
            f"{', '.join(get_args(AUTOTUNE_METRICS))}."
        )

    if autotune and autotune_trials < 1:
        raise ValueError("Autotune needs at least one trial.")

    if val_split <= 0:
        raise ValueError(
            "The validation split must be above 0: early stopping validates on it, "
            "while test data only evaluates the trained classifier."
        )

    (
        x_train_full,
        y_train_full,
        x_test,
        y_test,
        labels,
        is_binary,
        is_multi_label,
        version,
    ) = _load_training_data(
        audio_input,
        test_data=test_data,
        upsampling_ratio=upsampling_ratio,
        upsampling_mode=upsampling_mode,
        fmin=fmin,
        fmax=fmax,
        audio_speed=audio_speed,
        crop_mode=crop_mode,
        overlap=overlap,
        min_len=1.0,
        threads=threads,
        save_cache_to=save_cache_to,
        progress_callback=on_data_load_end,
        model_version=birdnet,
    )

    if version != birdnet:
        logger.warning(
            "The training data cache holds BirdNET %s embeddings; training on that "
            "version instead of %s.",
            version,
            birdnet,
        )

    logger.info(
        f"...Done. Loaded {x_train_full.shape[0]} training samples "
        f"and {y_train_full.shape[1]} labels."
    )
    if len(x_test) > 0:
        logger.info(f"...Loaded {x_test.shape[0]} test samples.")

    if autotune:
        import gc

        import keras
        import optuna
        from optuna_integration.keras import KerasPruningCallback

        if on_trial_result:
            on_trial_result(0)

        study = optuna.create_study(
            direction="maximize",
            study_name="birdnet_analyzer",
            sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3, interval_steps=1),
        )
        space = _search_space(is_multi_label)
        folds = (
            list(
                _kfold_splits(
                    y_train_full,
                    autotune_n_splits,
                    autotune_n_repeats,
                    is_binary,
                    is_multi_label,
                )
            )
            if autotune_n_splits > 1
            else [(slice(None), None)]
        )

        def objective(trial: optuna.trial.Trial):
            params = _suggest(trial, space)
            scores: list[float] = []
            up_mode = (
                params["upsampling_mode"]
                if params["upsampling_ratio"] > 0
                else upsampling_mode
            )
            fg, fa = (
                (params["focal_loss_gamma"], params["focal_loss_alpha"])
                if params["focal_loss"]
                else (focal_loss_gamma, focal_loss_alpha)
            )

            for split, (train_idx, val_idx) in enumerate(folds):
                # The pruner's steps are epochs, which every split would report again.
                additional_callbacks = (
                    [KerasPruningCallback(trial, autotune_metric)] if split == 0 else []
                )
                history = None
                classifier = model.build_linear_classifier(
                    y_train_full.shape[1],
                    x_train_full.shape[1],
                    params["hidden_units"],
                    params["dropout"],
                )

                try:
                    classifier, history = model.train_linear_classifier(
                        classifier,
                        x_train_full[train_idx],
                        y_train_full[train_idx],
                        np.array([]) if val_idx is None else x_train_full[val_idx],
                        np.array([]) if val_idx is None else y_train_full[val_idx],
                        epochs=epochs,
                        batch_size=params["batch_size"],
                        learning_rate=params["learning_rate"],
                        val_split=val_split if val_idx is None else 0.0,
                        upsampling_ratio=params["upsampling_ratio"],
                        upsampling_mode=up_mode,
                        train_with_mixup=params["mixup"],
                        train_with_label_smoothing=params["label_smoothing"],
                        train_with_focal_loss=params["focal_loss"],
                        focal_loss_gamma=fg,
                        focal_loss_alpha=fa,
                        is_binary_classification=is_binary,
                        is_multi_label=is_multi_label,
                        additional_callbacks=additional_callbacks,
                        weight_decay=params["weight_decay"],
                        monitor=autotune_metric,
                        monitor_mode="max",
                    )
                    scores.append(
                        history.history[autotune_metric][history.restored_epoch]
                    )
                finally:
                    del classifier, history
                    keras.backend.clear_session()
                    gc.collect()

            if callable(on_trial_result):
                on_trial_result(trial.number + 1)

            return float(np.mean(scores))

        defaults = {
            "hidden_units": hidden_units,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "upsampling_ratio": upsampling_ratio,
            "upsampling_mode": upsampling_mode,
            "weight_decay": weight_decay,
            "mixup": mixup,
            "label_smoothing": label_smoothing,
            "focal_loss": use_focal_loss,
            "focal_loss_gamma": focal_loss_gamma,
            "focal_loss_alpha": focal_loss_alpha,
        }

        study.enqueue_trial(_snap_to_search_space(defaults, space))

        try:
            study.optimize(objective, n_trials=autotune_trials)
        except model.get_empty_class_exception() as e:
            e.message = (
                f"Class with label {labels[e.index]} is empty. "
                "Please remove it from the training data."
            )
            e.args = (e.message,)
            raise e

        best_params = study.best_params
        hidden_units = best_params["hidden_units"]
        dropout = best_params["dropout"]
        batch_size = best_params["batch_size"]
        learning_rate = best_params.get("learning_rate", learning_rate)
        upsampling_ratio = best_params["upsampling_ratio"]

        if upsampling_ratio > 0:
            upsampling_mode = best_params["upsampling_mode"]

        mixup = best_params["mixup"]
        label_smoothing = best_params["label_smoothing"]
        use_focal_loss = best_params["focal_loss"]
        weight_decay = best_params["weight_decay"]

        if use_focal_loss:
            focal_loss_alpha = best_params["focal_loss_alpha"]
            focal_loss_gamma = best_params["focal_loss_gamma"]

    classifier = model.build_linear_classifier(
        y_train_full.shape[1], x_train_full.shape[1], hidden_units, dropout
    )

    try:
        classifier, history = model.train_linear_classifier(
            classifier,
            x_train_full,
            y_train_full,
            np.array([]),
            np.array([]),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_split=val_split,
            upsampling_ratio=upsampling_ratio,
            upsampling_mode=upsampling_mode,
            train_with_mixup=mixup,
            train_with_label_smoothing=label_smoothing,
            train_with_focal_loss=use_focal_loss,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_alpha=focal_loss_alpha,
            on_epoch_end=on_epoch_end,
            is_binary_classification=is_binary,
            is_multi_label=is_multi_label,
            weight_decay=weight_decay,
            monitor=autotune_metric,
            monitor_mode="max",
        )
    except model.get_empty_class_exception() as e:
        e.message = (
            f"Class with label {labels[e.index]} is empty. "
            "Please remove it from the training data."
        )
        e.args = (e.message,)
        raise e

    # Remove activation from last layer before saving
    classifier.pop()
    formats = [model_formats] if isinstance(model_formats, str) else model_formats
    classifier_path = output.removesuffix(".tflite")

    # The settings the classifier was trained with, saved next to it both as a
    # record and so the GUI can load them again. When autotune ran, the values
    # are the tuned ones.
    utils.save_params_file(
        classifier_path + TRAIN_PARAMS_SUFFIX,
        by_header(
            TRAIN_PARAMS,
            {
                "classifier_name": os.path.basename(classifier_path),
                "model_formats": ", ".join(formats),
                "model_save_mode": model_save_mode,
                "fmin": fmin,
                "fmax": fmax,
                "audio_speed": audio_speed,
                "crop_mode": crop_mode,
                "overlap": overlap,
                "autotune": autotune,
                "autotune_trials": autotune_trials,
                "autotune_n_splits": autotune_n_splits,
                "autotune_n_repeats": autotune_n_repeats,
                "autotune_metric": autotune_metric,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "hidden_units": hidden_units,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "mixup": mixup,
                "use_focal_loss": use_focal_loss,
                "focal_loss_gamma": focal_loss_gamma,
                "focal_loss_alpha": focal_loss_alpha,
                "upsampling_mode": upsampling_mode,
                "upsampling_ratio": upsampling_ratio,
                "birdnet": version,
            },
        ),
    )

    if "tflite" in formats:
        model.save_linear_classifier(
            classifier, output, labels, mode=model_save_mode, version=version
        )
    if "raven" in formats:
        model.save_raven_model(
            classifier, output, labels, mode=model_save_mode, version=version
        )
    if "detached" in formats:
        model.save_detached_classifier(
            classifier, output, labels=labels, version=version
        )

    save_sample_counts(labels, y_train_full, output)

    metrics = None

    if len(x_test) > 0:
        metrics = evaluate_model(classifier, x_test, y_test, labels)

        if metrics:
            import csv

            eval_file_path = output + "_evaluation.csv"

            with open(eval_file_path, "w", newline="") as f:
                writer = csv.writer(f)

                header = [
                    "Class",
                    "Precision (0.5)",
                    "Recall (0.5)",
                    "F1 Score (0.5)",
                    "Precision (opt)",
                    "Recall (opt)",
                    "F1 Score (opt)",
                    "AUPRC",
                    "AUROC",
                    "Optimal Threshold",
                    "True Positives",
                    "False Positives",
                    "True Negatives",
                    "False Negatives",
                    "Samples",
                    "Percentage (%)",
                ]
                writer.writerow(header)

                writer.writerow(
                    [
                        "OVERALL (Macro-avg)",
                        f"{metrics['macro_precision_default']:.4f}",
                        f"{metrics['macro_recall_default']:.4f}",
                        f"{metrics['macro_f1_default']:.4f}",
                        f"{metrics['macro_precision_opt']:.4f}",
                        f"{metrics['macro_recall_opt']:.4f}",
                        f"{metrics['macro_f1_opt']:.4f}",
                        f"{metrics['macro_auprc']:.4f}",
                        f"{metrics['macro_auroc']:.4f}",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        # Empty cells for Threshold, TP, FP, TN, FN, Samples, Percentage
                    ]
                )

                for class_name, class_metrics in metrics["class_metrics"].items():
                    distribution = metrics["class_distribution"].get(
                        class_name, {"count": 0, "percentage": 0.0}
                    )
                    writer.writerow(
                        [
                            class_name,
                            f"{class_metrics['precision_default']:.4f}",
                            f"{class_metrics['recall_default']:.4f}",
                            f"{class_metrics['f1_default']:.4f}",
                            f"{class_metrics['precision_opt']:.4f}",
                            f"{class_metrics['recall_opt']:.4f}",
                            f"{class_metrics['f1_opt']:.4f}",
                            f"{class_metrics['auprc']:.4f}",
                            f"{class_metrics['auroc']:.4f}",
                            f"{class_metrics['threshold']:.2f}",
                            class_metrics["tp"],
                            class_metrics["fp"],
                            class_metrics["tn"],
                            class_metrics["fn"],
                            distribution["count"],
                            f"{distribution['percentage']:.2f}",
                        ]
                    )

    return history, metrics


def find_optimal_threshold(y_true, y_pred_prob):
    """
    Find the optimal classification threshold using the F1 score.

    For imbalanced datasets, the default threshold of 0.5 may not be optimal.
    This function finds the threshold that maximizes the F1 score for each class.

    Args:
        y_true: Ground truth labels
        y_pred_prob: Predicted probabilities

    Returns:
        The optimal threshold value
    """
    from sklearn.metrics import f1_score

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def evaluate_model(classifier, x_test, y_test, labels, threshold=None):
    """
    Evaluates the trained model on test data and prints detailed metrics.

    Args:
        classifier: The trained model
        x_test: Test features (embeddings)
        y_test: Test labels
        labels: List of label names
        threshold: Classification threshold (if None, will find optimal threshold for
                   each class)

    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    x_test, y_test = _without_negative_rows(x_test, y_test)

    if len(x_test) == 0:
        return {}

    y_pred_prob = classifier.predict(x_test)
    y_pred_prob = model.flat_sigmoid(y_pred_prob, sensitivity=-1, bias=1.0)

    metrics = {}

    logger.info("\nModel Evaluation:")
    logger.info("=================")

    precisions_default = []
    recalls_default = []
    f1s_default = []
    precisions_opt = []
    recalls_opt = []
    f1s_opt = []
    auprcs = []
    aurocs = []
    class_metrics = {}
    optimal_thresholds = {}

    logger.info(
        "\nNote: The AUPRC and AUROC metrics calculated during post-training evaluation"
        " may differ"
    )
    logger.info("from training history values due to different calculation methods:")
    logger.info("  - Training history uses Keras metrics calculated over batches")
    logger.info(
        "  - Evaluation uses scikit-learn metrics calculated over the entire dataset"
    )

    for i in range(y_test.shape[1]):
        try:
            y_pred_default = (y_pred_prob[:, i] >= 0.5).astype(int)

            class_precision_default = precision_score(y_test[:, i], y_pred_default)
            class_recall_default = recall_score(y_test[:, i], y_pred_default)
            class_f1_default = f1_score(y_test[:, i], y_pred_default)

            precisions_default.append(class_precision_default)
            recalls_default.append(class_recall_default)
            f1s_default.append(class_f1_default)

            if threshold is None:
                class_threshold = find_optimal_threshold(
                    y_test[:, i], y_pred_prob[:, i]
                )
                optimal_thresholds[labels[i]] = class_threshold
            else:
                class_threshold = threshold

            y_pred_opt = (y_pred_prob[:, i] >= class_threshold).astype(int)
            class_precision_opt = precision_score(y_test[:, i], y_pred_opt)
            class_recall_opt = recall_score(y_test[:, i], y_pred_opt)
            class_f1_opt = f1_score(y_test[:, i], y_pred_opt)
            class_auprc = average_precision_score(y_test[:, i], y_pred_prob[:, i])
            class_auroc = roc_auc_score(y_test[:, i], y_pred_prob[:, i])

            precisions_opt.append(class_precision_opt)
            recalls_opt.append(class_recall_opt)
            f1s_opt.append(class_f1_opt)
            auprcs.append(class_auprc)
            aurocs.append(class_auroc)

            tn, fp, fn, tp = confusion_matrix(y_test[:, i], y_pred_opt).ravel()
            class_metrics[labels[i]] = {
                "precision_default": class_precision_default,
                "recall_default": class_recall_default,
                "f1_default": class_f1_default,
                "precision_opt": class_precision_opt,
                "recall_opt": class_recall_opt,
                "f1_opt": class_f1_opt,
                "auprc": class_auprc,
                "auroc": class_auroc,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "threshold": class_threshold,
            }

            logger.info(f"\nClass: {labels[i]}")
            logger.info("  Default threshold (0.5):")
            logger.info(f"    Precision: {class_precision_default:.4f}")
            logger.info(f"    Recall:    {class_recall_default:.4f}")
            logger.info(f"    F1 Score:  {class_f1_default:.4f}")
            logger.info(f"  Optimized threshold ({class_threshold:.2f}):")
            logger.info(f"    Precision: {class_precision_opt:.4f}")
            logger.info(f"    Recall:    {class_recall_opt:.4f}")
            logger.info(f"    F1 Score:  {class_f1_opt:.4f}")
            logger.info(f"  AUPRC:     {class_auprc:.4f}")
            logger.info(f"  AUROC:     {class_auroc:.4f}")
            logger.info("  Confusion matrix (optimized threshold):")
            logger.info(f"    True Positives:  {tp}")
            logger.info(f"    False Positives: {fp}")
            logger.info(f"    True Negatives:  {tn}")
            logger.info(f"    False Negatives: {fn}")

        except Exception as e:
            logger.error(
                f"Error calculating metrics for class {labels[i]}: {e}", exc_info=e
            )

    metrics["macro_precision_default"] = np.mean(precisions_default)
    metrics["macro_recall_default"] = np.mean(recalls_default)
    metrics["macro_f1_default"] = np.mean(f1s_default)
    metrics["macro_precision_opt"] = np.mean(precisions_opt)
    metrics["macro_recall_opt"] = np.mean(recalls_opt)
    metrics["macro_f1_opt"] = np.mean(f1s_opt)
    metrics["macro_auprc"] = np.mean(auprcs)
    metrics["macro_auroc"] = np.mean(aurocs)
    metrics["class_metrics"] = class_metrics
    metrics["optimal_thresholds"] = optimal_thresholds

    logger.info("\nMacro-averaged metrics:")
    logger.info("  Default threshold (0.5):")
    logger.info(f"    Precision: {metrics['macro_precision_default']:.4f}")
    logger.info(f"    Recall:    {metrics['macro_recall_default']:.4f}")
    logger.info(f"    F1 Score:  {metrics['macro_f1_default']:.4f}")
    logger.info("  Optimized thresholds:")
    logger.info(f"    Precision: {metrics['macro_precision_opt']:.4f}")
    logger.info(f"    Recall:    {metrics['macro_recall_opt']:.4f}")
    logger.info(f"    F1 Score:  {metrics['macro_f1_opt']:.4f}")
    logger.info(f"  AUPRC:     {metrics['macro_auprc']:.4f}")
    logger.info(f"  AUROC:     {metrics['macro_auroc']:.4f}")

    class_counts = y_test.sum(axis=0)
    total_samples = len(y_test)
    class_distribution = {}

    logger.info("\nClass distribution in test set:")
    for i, count in enumerate(class_counts):
        percentage = count / total_samples * 100
        class_distribution[labels[i]] = {"count": int(count), "percentage": percentage}
        logger.info(f"  {labels[i]}: {int(count)} samples ({percentage:.2f}%)")

    metrics["class_distribution"] = class_distribution

    return metrics


def _save_to_cache(
    path,
    x_train,
    y_train,
    x_test,
    y_test,
    labels,
    overlap=0.0,
    fmin=0.0,
    fmax=15000.0,
    audio_speed=1.0,
    crop_mode="center",
    is_binary=False,
    is_multi_label=False,
    model_version: str = DEFAULT_ACOUSTIC_MODEL_VERSION,
):
    """Saves training data to cache.

    Args:
        path: Path to the cache file.
        x_train: Training samples.
        y_train: Training labels.
        x_test: Test samples.
        y_test: Test labels.
        labels: Labels.
        overlap: Overlap between samples.
        fmin: Minimum frequency for bandpass filter.
        fmax: Maximum frequency for bandpass filter.
        audio_speed: Speed of audio playback.
        crop_mode: Mode for cropping samples.
        is_binary: Whether it's a binary classification task.
        is_multi_label: Whether it's a multi-label classification task.
        model_version: The BirdNET version the samples were embedded with.
    """
    import numpy as np

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    np.savez(
        path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        labels=np.array(labels, dtype=object),
        binary_classification=is_binary,
        multi_label=is_multi_label,
        fmin=fmin,
        fmax=fmax,
        audio_speed=audio_speed,
        crop_mode=crop_mode,
        overlap=overlap,
        model_version=model_version,
    )


def _load_from_cache(path):
    """Loads training data from cache.

    Args:
        path: Path to the cache file.

    Returns:
        A tuple of (x_train, y_train, x_test, y_test, labels, binary_classification,
        multi_label, model_version).
    """
    import numpy as np

    data: dict = np.load(path, allow_pickle=True)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data.get("x_test", np.array([]))
    y_test = data.get("y_test", np.array([]))
    labels = data["labels"]
    binary_classification = bool(data.get("binary_classification", False))
    multi_label = bool(data.get("multi_label", False))
    model_version = (
        str(data["model_version"])
        if "model_version" in data
        else base_models.version_for_embeddings_dim(x_train.shape[1])
    )

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        labels,
        binary_classification,
        multi_label,
        model_version,
    )
