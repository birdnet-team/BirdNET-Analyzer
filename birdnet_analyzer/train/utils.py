"""Module for training a custom classifier.

Can be used to train a custom classifier with new training data.
"""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

import numpy as np
import tqdm
from birdnet import load
from sklearn.model_selection import RepeatedStratifiedKFold

from birdnet_analyzer import audio, model, model_utils, utils
from birdnet_analyzer.config import (
    ALLOWED_FILETYPES,
    AUTOTUNE_METRICS,
    NON_EVENT_CLASSES,
)
from birdnet_analyzer.model_utils import GLOBAL_PREFETCH_RATIO

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


def _load_audio_file(
    f,
    session,
    label_vector,
    sample_rate=48000,
    fmin=0,
    fmax=15000,
    audio_speed=1.0,
    crop_mode: SAMPLE_CROP_MODES = "center",
    sig_length=3.0,
    overlap=0.0,
    min_len=1.0,
    model_sample_rate=48000,
):
    """Load an audio file and extract features.
    Args:
        f: Path to the audio file.
        label_vector: The label vector for the file.
    Returns:
        A tuple of (x_train, y_train).
    """

    x_train = []
    y_train = []

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
        print(f"\t Error when loading file {f}", flush=True)
        print(f"\t {e}", flush=True)
        return np.array([]), np.array([])

    if crop_mode == "center":
        sig_splits = [audio.crop_center(sig, rate, sig_length)]
    elif crop_mode == "first":
        sig_splits = [audio.split_signal(sig, rate, sig_length, overlap, min_len)[0]]
    elif crop_mode == "smart":
        sig_splits = audio.smart_crop_signal(sig, rate, sig_length, overlap, min_len)
    else:
        sig_splits = audio.split_signal(sig, rate, sig_length, overlap, min_len)

    # turns out that batch size 1 is the fastest, probably because of having to resize
    # the model input when the number of samples in a batch changes
    # TODO: Still faster?
    batch_size = 1

    for i in range(0, len(sig_splits), batch_size):
        batch_sig = sig_splits[i : i + batch_size]
        batch_label = [label_vector] * len(batch_sig)
        embeddings = model_utils.get_embeddings_array_with_session(
            session, [(sig, model_sample_rate) for sig in batch_sig]
        )

        x_train.extend(embeddings)
        y_train.extend(batch_label)

    return x_train, y_train


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
):
    """Loads the data for training.

    Reads all subdirectories of "audio_input" and uses their names as new labels.

    These directories should contain all the training data for each label.

    If a cache file is provided, the training data is loaded from there.

    Args:
        audio_input: Path to the training data directory or path to a cache file
                     ("train_cache.npz" for example).
        test_data: Path to the test data directory. Defaults to None. If not specified,
                   a validation split will be used.
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

    Returns:
        A tuple of (x_train, y_train, x_test, y_test, labels, is_binary, is_multi_label)
    """

    if audio_input.endswith(".npz"):
        return _load_from_cache(audio_input)

    train_folders = list(utils.list_subdirectories(audio_input))
    labels: list[str] = []

    for folder in train_folders:
        labels_in_folder = folder.split(",")

        for label in labels_in_folder:
            cln_label = label.strip()

            if cln_label not in labels:
                labels.append(cln_label)

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
            raise Exception(
                "Negative labels can't be used with binary classification",
                "validation-no-negative-samples-in-binary-classification",
            )
        if not any(f for f in train_folders if f.lower() in NON_EVENT_CLASSES):
            raise Exception(
                "Non-event samples are required for binary classification",
                "validation-non-event-samples-required-in-binary-classification",
            )

    if is_binary and is_multi_label:
        raise Exception(
            "Error: Binary classification and multi-label not possible at the same time"
        )

    if is_multi_label and upsampling_ratio > 0 and upsampling_mode != "repeat":
        raise Exception(
            "Only repeat-upsampling ist available for multi-label",
            "validation-only-repeat-upsampling-for-multi-label",
        )

    x_train, y_train, x_test, y_test = [], [], [], []
    model = load("acoustic", "2.4", "tf")

    with model.encode_session(
        bandpass_fmin=fmin,
        bandpass_fmax=fmax,
        speed=audio_speed,
        batch_size=1,
        n_workers=None,
        progress_callback=None,
        prefetch_ratio=GLOBAL_PREFETCH_RATIO,
        n_feeders=1,
    ) as session:

        def load_data(data_path, allowed_folders):
            x = []
            y = []
            folders = sorted(utils.list_subdirectories(data_path))

            for folder in folders:
                if folder not in allowed_folders:
                    continue

                label_vector = np.zeros((len(valid_labels),), dtype="float32")
                folder_labels = folder.split(",")

                for label in folder_labels:
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

                with tqdm.tqdm(
                    total=len(files), desc=f" - loading '{folder}'", unit="f"
                ) as progress_bar:
                    for f in files:
                        result = _load_audio_file(
                            f,
                            session,
                            label_vector=label_vector,
                            fmin=fmin,
                            fmax=fmax,
                            audio_speed=audio_speed,
                            crop_mode=crop_mode,
                            overlap=overlap,
                            min_len=min_len,
                            model_sample_rate=model.get_sample_rate(),
                        )
                        if len(result[0]) > 0:
                            x += result[0]
                            y += result[1]

                        num_files_processed += 1
                        progress_bar.update(1)

                        if progress_callback:
                            progress_callback(num_files_processed, len(files), folder)

            return np.array(x, dtype="float32"), np.array(y, dtype="float32")

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
            )
        except Exception as e:
            print(f"\t...error saving cache: {e}", flush=True)

    return x_train, y_train, x_test, y_test, valid_labels, is_binary, is_multi_label


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
    use_focal_loss: bool = False,
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: float = 0.25,
    hidden_units: int = 0,
    dropout: float = 0.0,
    label_smoothing: bool = False,
    mixup: bool = False,
    upsampling_ratio: float = 0.0,
    upsampling_mode: UPSAMPLING_MODES = "repeat",
    model_format: TRAINED_MODEL_OUTPUT_FORMATS = "tflite",
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
    on_epoch_end=None,
    on_trial_result=None,
    on_data_load_end=None,
    save_detached_classifier: bool = False,
):
    """Trains a custom classifier.

    Args:
        on_epoch_end: A callback function that takes two arguments `epoch`, `logs`.
        on_trial_result: A callback function for hyperparameter tuning.
        on_data_load_end: A callback function for data loading progress.
        autotune_directory: Directory for autotune results.
        save_detached_classifier: Whether to save the detached classifier.

    Returns:
        A keras `History` object, whose `history` property contains all the metrics.
    """

    (
        x_train_full,
        y_train_full,
        x_val_full,
        y_val_full,
        labels,
        is_binary,
        is_multi_label,
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
    )
    print(
        f"...Done. Loaded {x_train_full.shape[0]} training samples "
        + f"and {y_train_full.shape[1]} labels.",
        flush=True,
    )
    if len(x_val_full) > 0:
        print(f"...Loaded {x_val_full.shape[0]} test samples.", flush=True)

    if autotune:
        import gc

        import keras
        import optuna

        if on_trial_result:
            on_trial_result(0)

        study = optuna.create_study(
            direction=optuna.study.StudyDirection.MAXIMIZE,
            study_name="birdnet_analyzer",
        )

        def objective(trial: optuna.trial.Trial):
            histories: list[float] = []
            h_units = trial.suggest_categorical(
                "hidden_units", [0, 128, 256, 512, 1024, 2048]
            )
            dr = trial.suggest_categorical("dropout", [0.0, 0.25, 0.33, 0.5, 0.75, 0.9])
            upsampling_choices = ["repeat", "mean", "linear"]

            if is_multi_label:
                upsampling_choices = ["repeat"]

            # Create stratified k-fold splits for cross-validation
            # For multi-label, create a pseudo-label based on number of active labels
            # TODO: Is this the best way to do stratification for multi-label data?
            if is_multi_label:
                stratify_labels = np.sum(y_train_full > 0, axis=1)
            else:
                stratify_labels = np.argmax(y_train_full, axis=1)

            def generate_splits(
                x_train: np.ndarray,
                y_train: np.ndarray,
                x_test: np.ndarray,
                y_test: np.ndarray,
                val_split: float,
                autotune_n_splits: int,
                autotune_n_repeats: int,
            ) -> Generator[
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], None, None
            ]:
                if x_test:
                    # If test data is available, use a single split with the test data
                    splits = [
                        (
                            np.arange(len(x_train)),
                            np.arange(len(x_test)),
                        )
                    ]
                    for train_idx, val_idx in splits:
                        yield (
                            x_train[train_idx],
                            y_train[train_idx],
                            x_test[val_idx],
                            y_test[val_idx],
                            0.0,
                        )
                elif autotune_n_splits == 1:
                    splits = [(np.arange(len(x_train)),)]
                    for train_idx in splits:
                        yield (
                            x_train[train_idx],
                            y_train[train_idx],
                            [],
                            [],
                            val_split,
                        )
                else:
                    # Repeated stratified k-fold cross-validation
                    skf = RepeatedStratifiedKFold(
                        n_splits=autotune_n_splits,
                        n_repeats=autotune_n_repeats,
                        random_state=42,  # TODO: use same state
                    )

                    for train_idx, val_idx in skf.split(x_train, stratify_labels):
                        yield (
                            x_train[train_idx],
                            y_train[train_idx],
                            x_train[val_idx],
                            y_train[val_idx],
                            0.0,
                        )

            for x_train, y_train, x_val, y_val, val_percentage in generate_splits(
                x_train_full,
                y_train_full,
                x_val_full,
                y_val_full,
                val_split,
                autotune_n_splits,
                autotune_n_repeats,
            ):
                bs = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
                # trial.suggest_loguniform

                if bs == 8:
                    lr = trial.suggest_categorical(
                        "learning_rate", [0.0005, 0.0002, 0.0001]
                    )
                elif bs == 16:
                    lr = trial.suggest_categorical(
                        "learning_rate", [0.005, 0.002, 0.001, 0.0005, 0.0002]
                    )
                elif bs == 32:
                    lr = trial.suggest_categorical(
                        "learning_rate", [0.01, 0.005, 0.001, 0.0005, 0.0001]
                    )
                elif bs == 64:
                    lr = trial.suggest_categorical(
                        "learning_rate", [0.01, 0.005, 0.002, 0.001]
                    )
                else:
                    lr = trial.suggest_categorical("learning_rate", [0.1, 0.01, 0.005])

                up_ratio = trial.suggest_categorical(
                    "upsampling_ratio", [0.0, 0.25, 0.33, 0.5, 0.75, 1.0]
                )

                if up_ratio > 0:
                    up_mode = trial.suggest_categorical(
                        "upsampling_mode", upsampling_choices
                    )
                else:
                    up_mode = upsampling_mode

                mix = trial.suggest_categorical("mixup", [True, False])
                ls = trial.suggest_categorical("label_smoothing", [True, False])
                focal = trial.suggest_categorical("focal_loss", [True, False])
                fg = focal_loss_gamma
                fa = focal_loss_alpha

                if focal:
                    fg = trial.suggest_categorical(
                        "focal_loss_gamma", [0.5, 1.0, 2.0, 3.0, 4.0]
                    )
                    fa = trial.suggest_categorical(
                        "focal_loss_alpha", [0.1, 0.25, 0.5, 0.75, 0.9]
                    )

                classifier = model.build_linear_classifier(
                    y_train.shape[1],
                    x_train.shape[1],
                    h_units,
                    dr,
                )
                classifier, history = model.train_linear_classifier(
                    classifier,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    epochs=epochs,
                    batch_size=bs,
                    learning_rate=lr,
                    val_split=val_percentage,
                    upsampling_ratio=up_ratio,
                    upsampling_mode=up_mode,
                    train_with_mixup=mix,
                    train_with_label_smoothing=ls,
                    train_with_focal_loss=focal,
                    focal_loss_gamma=fg,
                    focal_loss_alpha=fa,
                    is_binary_classification=is_binary,
                    is_multi_label=is_multi_label,
                )
                best_score = history.history[autotune_metric][
                    np.argmax(history.history[autotune_metric])
                ]

                histories.append(best_score)
                keras.backend.clear_session()
                del classifier
                del history
                gc.collect()

            if callable(on_trial_result):
                on_trial_result(trial.number + 1)

            return float(np.mean(histories))

        # enqueue the defaults as first trial so that the passed default
        # hyperparameters are evaluated even if tuning is skipped
        study.enqueue_trial(
            {
                "hidden_units": hidden_units,
                "dropout": dropout,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "upsampling_ratio": upsampling_ratio,
                "upsampling_mode": upsampling_mode,
                "mixup": mixup,
                "label_smoothing": label_smoothing,
                "focal_loss": use_focal_loss,
                "focal_loss_gamma": focal_loss_gamma,
                "focal_loss_alpha": focal_loss_alpha,
            }
        )

        try:
            study.optimize(objective, n_trials=autotune_trials)
        except model.get_empty_class_exception() as e:
            e.message = (
                f"Class with label {labels[e.index]} is empty. "  # type: ignore
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
            x_val_full,
            y_val_full,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_split=val_split if len(x_val_full) == 0 else 0.0,
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
        )
    except model.get_empty_class_exception() as e:
        e.message = (
            f"Class with label {labels[e.index]} is empty. "  # type: ignore
            "Please remove it from the training data."
        )
        e.args = (e.message,)
        raise e
    except Exception as e:
        raise Exception("Error training model") from e

    try:
        # Remove activation from last layer before saving
        classifier.pop()
        params = (
            [
                "Hidden units",
                "Dropout",
                "Batchsize",
                "Learning rate",
                "Crop mode",
                "Crop overlap",
                "Audio speed",
                "Upsampling mode",
                "Upsampling ratio",
                "use mixup",
                "use label smoothing",
                "use focal loss",
                "focal loss alpha",
                "focal loss gamma",
                "BirdNET Model version",
            ],
            [
                hidden_units,
                dropout,
                batch_size,
                learning_rate,
                crop_mode,
                overlap,
                audio_speed,
                upsampling_mode,
                upsampling_ratio,
                mixup,
                label_smoothing,
                use_focal_loss,
                focal_loss_alpha,
                focal_loss_gamma,
                "2.4",
            ],
        )

        if model_format == "both":
            model.save_raven_model(
                classifier, output, labels, mode=model_save_mode, params=params
            )
            model.save_linear_classifier(
                classifier, output, labels, mode=model_save_mode, params=params
            )
        elif model_format == "tflite":
            model.save_linear_classifier(
                classifier, output, labels, mode=model_save_mode, params=params
            )
        elif model_format == "raven":
            model.save_raven_model(
                classifier, output, labels, mode=model_save_mode, params=params
            )
        else:
            raise ValueError(f"Unknown model output format: {model_format}")

        if save_detached_classifier:
            model.save_detached_classifier(
                classifier, output, mode=model_save_mode
            )

    except Exception as e:
        raise Exception("Error saving model") from e

    save_sample_counts(labels, y_train_full, output)

    # Evaluate model on test data if available
    metrics = None

    if len(x_val_full) > 0:
        metrics = evaluate_model(classifier, x_val_full, y_val_full, labels)

        if metrics:
            import csv

            eval_file_path = os.path.join(output, "evaluation.csv")

            with open(eval_file_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Define all the metrics as columns, including both default and
                # optimized threshold metrics
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

                # Write macro-averaged metrics (overall scores) first
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

                # Write per-class metrics (one row per species)
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

    # Try different thresholds and find the one that gives the best F1 score
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

    if len(x_test) == 0:
        return {}

    y_pred_prob = classifier.predict(x_test)
    y_pred_prob = model.flat_sigmoid(y_pred_prob, sensitivity=-1, bias=1.0)

    metrics = {}

    print("\nModel Evaluation:")
    print("=================")

    # Calculate metrics for each class
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

    # Print the metric calculation method that's being used
    print(
        "\nNote: The AUPRC and AUROC metrics calculated during post-training evaluation"
        " may differ"
    )
    print("from training history values due to different calculation methods:")
    print("  - Training history uses Keras metrics calculated over batches")
    print("  - Evaluation uses scikit-learn metrics calculated over the entire dataset")

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

            print(f"\nClass: {labels[i]}")
            print("  Default threshold (0.5):")
            print(f"    Precision: {class_precision_default:.4f}")
            print(f"    Recall:    {class_recall_default:.4f}")
            print(f"    F1 Score:  {class_f1_default:.4f}")
            print(f"  Optimized threshold ({class_threshold:.2f}):")
            print(f"    Precision: {class_precision_opt:.4f}")
            print(f"    Recall:    {class_recall_opt:.4f}")
            print(f"    F1 Score:  {class_f1_opt:.4f}")
            print(f"  AUPRC:     {class_auprc:.4f}")
            print(f"  AUROC:     {class_auroc:.4f}")
            print("  Confusion matrix (optimized threshold):")
            print(f"    True Positives:  {tp}")
            print(f"    False Positives: {fp}")
            print(f"    True Negatives:  {tn}")
            print(f"    False Negatives: {fn}")

        except Exception as e:
            print(f"Error calculating metrics for class {labels[i]}: {e}")

    # Calculate macro-averaged metrics for both default and optimized thresholds
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

    print("\nMacro-averaged metrics:")
    print("  Default threshold (0.5):")
    print(f"    Precision: {metrics['macro_precision_default']:.4f}")
    print(f"    Recall:    {metrics['macro_recall_default']:.4f}")
    print(f"    F1 Score:  {metrics['macro_f1_default']:.4f}")
    print("  Optimized thresholds:")
    print(f"    Precision: {metrics['macro_precision_opt']:.4f}")
    print(f"    Recall:    {metrics['macro_recall_opt']:.4f}")
    print(f"    F1 Score:  {metrics['macro_f1_opt']:.4f}")
    print(f"  AUPRC:     {metrics['macro_auprc']:.4f}")
    print(f"  AUROC:     {metrics['macro_auroc']:.4f}")

    # Calculate class distribution in test set
    class_counts = y_test.sum(axis=0)
    total_samples = len(y_test)
    class_distribution = {}

    print("\nClass distribution in test set:")
    for i, count in enumerate(class_counts):
        percentage = count / total_samples * 100
        class_distribution[labels[i]] = {"count": int(count), "percentage": percentage}
        print(f"  {labels[i]}: {int(count)} samples ({percentage:.2f}%)")

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
    """
    import numpy as np

    # Make directory if needed
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save cache file with training data, test data, labels and configuration
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
    )


def _load_from_cache(path):
    """Loads training data from cache.

    Args:
        path: Path to the cache file.

    Returns:
        A tuple of (x_train, y_train, labels, binary_classification, multi_label).
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

    return x_train, y_train, x_test, y_test, labels, binary_classification, multi_label
