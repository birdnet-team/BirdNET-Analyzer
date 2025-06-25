"""Module for training a custom classifier.

Can be used to train a custom classifier with new training data.
"""

import csv
import os
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import tqdm

import birdnet_analyzer.config as cfg
from birdnet_analyzer import audio, model, utils


def save_sample_counts(labels, y_train):
    """
    Saves the count of samples per label combination to a CSV file.

    The function creates a dictionary where the keys are label combinations (joined by '+') and the values are the counts of samples for each combination.
    It then writes this information to a CSV file named "<cfg.CUSTOM_CLASSIFIER>_sample_counts.csv" with two columns: "Label" and "Count".

    Args:
        labels (list of str): List of label names corresponding to the columns in y_train.
        y_train (numpy.ndarray): 2D array where each row is a binary vector indicating the presence (1) or absence (0) of each label.
    """
    samples_per_label = {}
    label_combinations = np.unique(y_train, axis=0)

    for label_combination in label_combinations:
        label = "+".join([labels[i] for i in range(len(label_combination)) if label_combination[i] == 1])
        samples_per_label[label] = np.sum(np.all(y_train == label_combination, axis=1))

    csv_file_path = cfg.CUSTOM_CLASSIFIER + "_sample_counts.csv"

    with open(csv_file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Label", "Count"])

        for label, count in samples_per_label.items():
            writer.writerow([label, count])


def _load_audio_file(f, label_vector, config):
    """Load an audio file and extract features.
    Args:
        f: Path to the audio file.
        label_vector: The label vector for the file.
    Returns:
        A tuple of (x_train, y_train).
    """

    x_train = []
    y_train = []

    # restore config in case we're on Windows to be thread save
    cfg.set_config(config)

    # Try to load the audio file
    try:
        # Load audio
        sig, rate = audio.open_audio_file(
            f,
            sample_rate=cfg.SAMPLE_RATE,
            duration=cfg.SIG_LENGTH if cfg.SAMPLE_CROP_MODE == "first" else None,
            fmin=cfg.BANDPASS_FMIN,
            fmax=cfg.BANDPASS_FMAX,
            speed=cfg.AUDIO_SPEED,
        )

    # if anything happens print the error and ignore the file
    except Exception as e:
        # Print Error
        print(f"\t Error when loading file {f}", flush=True)
        print(f"\t {e}", flush=True)
        return np.array([]), np.array([])

    # Crop training samples
    if cfg.SAMPLE_CROP_MODE == "center":
        sig_splits = [audio.crop_center(sig, rate, cfg.SIG_LENGTH)]
    elif cfg.SAMPLE_CROP_MODE == "first":
        sig_splits = [audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)[0]]
    elif cfg.SAMPLE_CROP_MODE == "smart":
        # Smart cropping - detect peaks in audio energy to identify potential signals
        sig_splits = audio.smart_crop_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)
    else:
        sig_splits = audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

    # Get feature embeddings
    batch_size = 1  # turns out that batch size 1 is the fastest, probably because of having to resize the model input when the number of samples in a batch changes
    for i in range(0, len(sig_splits), batch_size):
        batch_sig = sig_splits[i : i + batch_size]
        batch_label = [label_vector] * len(batch_sig)
        embeddings = model.embeddings(batch_sig)

        # Add to training data
        x_train.extend(embeddings)
        y_train.extend(batch_label)

    return x_train, y_train


def _load_training_data(cache_mode=None, cache_file="", progress_callback=None):
    """Loads the data for training.

    Reads all subdirectories of "config.TRAIN_DATA_PATH" and uses their names as new labels.

    These directories should contain all the training data for each label.

    If a cache file is provided, the training data is loaded from there.

    Args:
        cache_mode: Cache mode. Can be 'load' or 'save'. Defaults to None.
        cache_file: Path to cache file.

    Returns:
        A tuple of (x_train, y_train, x_test, y_test, labels).
    """
    # Load from cache
    if cache_mode == "load":
        if os.path.isfile(cache_file):
            print(f"\t...loading from cache: {cache_file}", flush=True)
            x_train, y_train, x_test, y_test, labels, cfg.BINARY_CLASSIFICATION, cfg.MULTI_LABEL = (
                utils.load_from_cache(cache_file)
            )
            return x_train, y_train, x_test, y_test, labels

        print(f"\t...cache file not found: {cache_file}", flush=True)

    # Print train and test data path as confirmation
    print(f"\t...train data path: {cfg.TRAIN_DATA_PATH}", flush=True)
    print(f"\t...test data path: {cfg.TEST_DATA_PATH}", flush=True)

    # Get list of subfolders as labels
    train_folders = sorted(utils.list_subdirectories(cfg.TRAIN_DATA_PATH))

    # Read all individual labels from the folder names
    labels: list[str] = []

    for folder in train_folders:
        labels_in_folder = folder.split(",")
        for label in labels_in_folder:
            if label not in labels:
                labels.append(label)

    # Sort labels
    labels = sorted(labels)

    # Get valid labels
    valid_labels = [
        label for label in labels if label.lower() not in cfg.NON_EVENT_CLASSES and not label.startswith("-")
    ]

    # Check if binary classification
    cfg.BINARY_CLASSIFICATION = len(valid_labels) == 1

    # Validate the classes for binary classification
    if cfg.BINARY_CLASSIFICATION:
        if len([f for f in train_folders if f.startswith("-")]) > 0:
            raise Exception(
                "Negative labels can't be used with binary classification",
                "validation-no-negative-samples-in-binary-classification",
            )
        if len([f for f in train_folders if f.lower() in cfg.NON_EVENT_CLASSES]) == 0:
            raise Exception(
                "Non-event samples are required for binary classification",
                "validation-non-event-samples-required-in-binary-classification",
            )

    # Check if multi label
    cfg.MULTI_LABEL = len(valid_labels) > 1 and any("," in f for f in train_folders)

    # Check if multi-label and binary classficication
    if cfg.BINARY_CLASSIFICATION and cfg.MULTI_LABEL:
        raise Exception("Error: Binary classfication and multi-label not possible at the same time")

    # Only allow repeat upsampling for multi-label setting
    if cfg.MULTI_LABEL and cfg.UPSAMPLING_RATIO > 0 and cfg.UPSAMPLING_MODE != "repeat":
        raise Exception(
            "Only repeat-upsampling ist available for multi-label", "validation-only-repeat-upsampling-for-multi-label"
        )

    # Load training data
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    def load_data(data_path, allowed_folders):
        x = []
        y = []
        folders = sorted(utils.list_subdirectories(data_path))

        for folder in folders:
            if folder not in allowed_folders:
                print(f"Skipping folder {folder} because it is not in the training data.", flush=True)
                continue

            # Get label vector
            label_vector = np.zeros((len(valid_labels),), dtype="float32")
            folder_labels = folder.split(",")

            for label in folder_labels:
                if label.lower() not in cfg.NON_EVENT_CLASSES and not label.startswith("-"):
                    label_vector[valid_labels.index(label)] = 1
                elif (
                    label.startswith("-") and label[1:] in valid_labels
                ):  # Negative labels need to be contained in the valid labels
                    label_vector[valid_labels.index(label[1:])] = -1

            # Get list of files
            # Filter files that start with '.' because macOS seems to them for temp files.
            files = filter(
                os.path.isfile,
                (
                    os.path.join(data_path, folder, f)
                    for f in sorted(os.listdir(os.path.join(data_path, folder)))
                    if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES
                ),
            )

            # Load files using thread pool
            with Pool(cfg.CPU_THREADS) as p:
                tasks = []

                for f in files:
                    task = p.apply_async(
                        partial(_load_audio_file, f=f, label_vector=label_vector, config=cfg.get_config())
                    )
                    tasks.append(task)

                # Wait for tasks to complete and monitor progress with tqdm
                num_files_processed = 0

                with tqdm.tqdm(total=len(tasks), desc=f" - loading '{folder}'", unit="f") as progress_bar:
                    for task in tasks:
                        result = task.get()
                        # Make sure result is not empty
                        # Empty results might be caused by errors when loading the audio file
                        # TODO: We should check for embeddings size in result, otherwise we can't add them to the training data
                        if len(result[0]) > 0:
                            x += result[0]
                            y += result[1]

                        num_files_processed += 1
                        progress_bar.update(1)

                        if progress_callback:
                            progress_callback(num_files_processed, len(tasks), folder)
        return np.array(x, dtype="float32"), np.array(y, dtype="float32")

    x_train, y_train = load_data(cfg.TRAIN_DATA_PATH, train_folders)

    if cfg.TEST_DATA_PATH and cfg.TEST_DATA_PATH != cfg.TRAIN_DATA_PATH:
        test_folders = sorted(utils.list_subdirectories(cfg.TEST_DATA_PATH))
        allowed_test_folders = [
            folder for folder in test_folders if folder in train_folders and not folder.startswith("-")
        ]
        x_test, y_test = load_data(cfg.TEST_DATA_PATH, allowed_test_folders)
    else:
        x_test = np.array([])
        y_test = np.array([])

    # Save to cache?
    if cache_mode == "save":
        print(f"\t...saving training data to cache: {cache_file}", flush=True)
        try:
            # Only save the valid labels
            utils.save_to_cache(cache_file, x_train, y_train, x_test, y_test, valid_labels)
        except Exception as e:
            print(f"\t...error saving cache: {e}", flush=True)

    # Return only the valid labels for further use
    return x_train, y_train, x_test, y_test, valid_labels


def normalize_embeddings(embeddings):
    """
    Normalize embeddings to improve training stability and performance.

    This applies L2 normalization to each embedding vector, which can help
    with convergence and model performance, especially when training on
    embeddings from different sources or domains.

    Args:
        embeddings: numpy array of embedding vectors

    Returns:
        Normalized embeddings array
    """
    # Calculate L2 norm of each embedding vector
    norms = np.sqrt(np.sum(embeddings**2, axis=1, keepdims=True))
    # Avoid division by zero
    norms[norms == 0] = 1.0
    # Normalize each embedding vector
    return embeddings / norms


def train_model(on_epoch_end=None, on_trial_result=None, on_data_load_end=None, autotune_directory="autotune"):
    """Trains a custom classifier.

    Args:
        on_epoch_end: A callback function that takes two arguments `epoch`, `logs`.
        on_trial_result: A callback function for hyperparameter tuning.
        on_data_load_end: A callback function for data loading progress.
        autotune_directory: Directory for autotune results.

    Returns:
        A keras `History` object, whose `history` property contains all the metrics.
    """

    # Load training data
    print("Loading training data...", flush=True)
    x_train, y_train, x_test, y_test, labels = _load_training_data(
        cfg.TRAIN_CACHE_MODE, cfg.TRAIN_CACHE_FILE, on_data_load_end
    )
    print(f"...Done. Loaded {x_train.shape[0]} training samples and {y_train.shape[1]} labels.", flush=True)
    if len(x_test) > 0:
        print(f"...Loaded {x_test.shape[0]} test samples.", flush=True)

    # Normalize embeddings
    print("Normalizing embeddings...", flush=True)
    x_train = normalize_embeddings(x_train)
    if len(x_test) > 0:
        x_test = normalize_embeddings(x_test)

    if cfg.AUTOTUNE:
        import gc

        import keras
        import keras_tuner

        # Call callback to initialize progress bar
        if on_trial_result:
            on_trial_result(0)

        class BirdNetTuner(keras_tuner.BayesianOptimization):
            def __init__(self, x_train, y_train, x_test, y_test, max_trials, executions_per_trial, on_trial_result):
                super().__init__(
                    max_trials=max_trials,
                    executions_per_trial=executions_per_trial,
                    overwrite=True,
                    directory=autotune_directory,
                    project_name="birdnet_analyzer",
                )
                self.x_train = x_train
                self.y_train = y_train
                self.x_test = x_test
                self.y_test = y_test
                self.on_trial_result = on_trial_result

            def run_trial(self, trial, *args, **kwargs):
                histories = []
                hp: keras_tuner.HyperParameters = trial.hyperparameters
                trial_number = len(self.oracle.trials)

                for execution in range(int(self.executions_per_trial)):
                    print(f"Running Trial #{trial_number} execution #{execution + 1}", flush=True)

                    # Build model
                    print("Building model...", flush=True)
                    classifier = model.build_linear_classifier(
                        self.y_train.shape[1],
                        self.x_train.shape[1],
                        hidden_units=hp.Choice(
                            "hidden_units", [0, 128, 256, 512, 1024, 2048], default=cfg.TRAIN_HIDDEN_UNITS
                        ),
                        dropout=hp.Choice("dropout", [0.0, 0.25, 0.33, 0.5, 0.75, 0.9], default=cfg.TRAIN_DROPOUT),
                    )
                    print("...Done.", flush=True)

                    # Only allow repeat upsampling in multi-label setting
                    upsampling_choices = ["repeat", "mean", "linear"]  # SMOTE is too slow

                    if cfg.MULTI_LABEL:
                        upsampling_choices = ["repeat"]

                    batch_size = hp.Choice("batch_size", [8, 16, 32, 64, 128], default=cfg.TRAIN_BATCH_SIZE)

                    if batch_size == 8:
                        learning_rate = hp.Choice(
                            "learning_rate_8",
                            [0.0005, 0.0002, 0.0001],
                            default=0.0001,
                            parent_name="batch_size",
                            parent_values=[8],
                        )
                    elif batch_size == 16:
                        learning_rate = hp.Choice(
                            "learning_rate_16",
                            [0.005, 0.002, 0.001, 0.0005, 0.0002],
                            default=0.0005,
                            parent_name="batch_size",
                            parent_values=[16],
                        )
                    elif batch_size == 32:
                        learning_rate = hp.Choice(
                            "learning_rate_32",
                            [0.01, 0.005, 0.001, 0.0005, 0.0001],
                            default=0.0001,
                            parent_name="batch_size",
                            parent_values=[32],
                        )
                    elif batch_size == 64:
                        learning_rate = hp.Choice(
                            "learning_rate_64",
                            [0.01, 0.005, 0.002, 0.001],
                            default=0.001,
                            parent_name="batch_size",
                            parent_values=[64],
                        )
                    elif batch_size == 128:
                        learning_rate = hp.Choice(
                            "learning_rate_128",
                            [0.1, 0.01, 0.005],
                            default=0.005,
                            parent_name="batch_size",
                            parent_values=[128],
                        )

                    # Train model
                    print("Training model...", flush=True)
                    classifier, history = model.train_linear_classifier(
                        classifier,
                        self.x_train,
                        self.y_train,
                        self.x_test,
                        self.y_test,
                        epochs=cfg.TRAIN_EPOCHS,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        val_split=0.0 if len(self.x_test) > 0 else cfg.TRAIN_VAL_SPLIT,
                        upsampling_ratio=hp.Choice(
                            "upsampling_ratio", [0.0, 0.25, 0.33, 0.5, 0.75, 1.0], default=cfg.UPSAMPLING_RATIO
                        ),
                        upsampling_mode=hp.Choice(
                            "upsampling_mode",
                            upsampling_choices,
                            default=cfg.UPSAMPLING_MODE,
                            parent_name="upsampling_ratio",
                            parent_values=[0.25, 0.33, 0.5, 0.75, 1.0],
                        ),
                        train_with_mixup=hp.Boolean("mixup", default=cfg.TRAIN_WITH_MIXUP),
                        train_with_label_smoothing=hp.Boolean(
                            "label_smoothing", default=cfg.TRAIN_WITH_LABEL_SMOOTHING
                        ),
                        train_with_focal_loss=hp.Boolean("focal_loss", default=cfg.TRAIN_WITH_FOCAL_LOSS),
                        focal_loss_gamma=hp.Choice(
                            "focal_loss_gamma",
                            [0.5, 1.0, 2.0, 3.0, 4.0],
                            default=cfg.FOCAL_LOSS_GAMMA,
                            parent_name="focal_loss",
                            parent_values=[True],
                        ),
                        focal_loss_alpha=hp.Choice(
                            "focal_loss_alpha",
                            [0.1, 0.25, 0.5, 0.75, 0.9],
                            default=cfg.FOCAL_LOSS_ALPHA,
                            parent_name="focal_loss",
                            parent_values=[True],
                        ),
                    )

                    # Get the best validation AUPRC instead of loss
                    best_val_auprc = history.history["val_AUPRC"][np.argmax(history.history["val_AUPRC"])]
                    histories.append(best_val_auprc)

                    print(
                        f"Finished Trial #{trial_number} execution #{execution + 1}. Best validation AUPRC: {best_val_auprc}",
                        flush=True,
                    )

                keras.backend.clear_session()
                del classifier
                del history
                gc.collect()

                # Call the on_trial_result callback
                if self.on_trial_result:
                    self.on_trial_result(trial_number)

                # Return the negative AUPRC for minimization (keras-tuner minimizes by default)
                return [-h for h in histories]

        # Create the tuner instance
        tuner = BirdNetTuner(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            max_trials=cfg.AUTOTUNE_TRIALS,
            executions_per_trial=cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL,
            on_trial_result=on_trial_result,
        )
        try:
            tuner.search()
        except model.get_empty_class_exception() as e:
            e.message = f"Class with label {labels[e.index]} is empty. Please remove it from the training data."
            e.args = (e.message,)
            raise e

        best_params = tuner.get_best_hyperparameters()[0]

        cfg.TRAIN_HIDDEN_UNITS = best_params["hidden_units"]
        cfg.TRAIN_DROPOUT = best_params["dropout"]
        cfg.TRAIN_BATCH_SIZE = best_params["batch_size"]
        cfg.TRAIN_LEARNING_RATE = best_params[f"learning_rate_{cfg.TRAIN_BATCH_SIZE}"]
        if cfg.UPSAMPLING_RATIO > 0:
            cfg.UPSAMPLING_MODE = best_params["upsampling_mode"]
        cfg.UPSAMPLING_RATIO = best_params["upsampling_ratio"]
        cfg.TRAIN_WITH_MIXUP = best_params["mixup"]
        cfg.TRAIN_WITH_LABEL_SMOOTHING = best_params["label_smoothing"]

        print("Best params: ")
        print("hidden_units: ", cfg.TRAIN_HIDDEN_UNITS)
        print("dropout: ", cfg.TRAIN_DROPOUT)
        print("batch_size: ", cfg.TRAIN_BATCH_SIZE)
        print("learning_rate: ", cfg.TRAIN_LEARNING_RATE)
        print("upsampling_ratio: ", cfg.UPSAMPLING_RATIO)
        if cfg.UPSAMPLING_RATIO > 0:
            print("upsampling_mode: ", cfg.UPSAMPLING_MODE)
        print("mixup: ", cfg.TRAIN_WITH_MIXUP)
        print("label_smoothing: ", cfg.TRAIN_WITH_LABEL_SMOOTHING)

    # Build model
    print("Building model...", flush=True)
    classifier = model.build_linear_classifier(
        y_train.shape[1], x_train.shape[1], cfg.TRAIN_HIDDEN_UNITS, cfg.TRAIN_DROPOUT
    )
    print("...Done.", flush=True)

    # Train model
    print("Training model...", flush=True)
    try:
        classifier, history = model.train_linear_classifier(
            classifier,
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=cfg.TRAIN_EPOCHS,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            learning_rate=cfg.TRAIN_LEARNING_RATE,
            val_split=cfg.TRAIN_VAL_SPLIT if len(x_test) == 0 else 0.0,
            upsampling_ratio=cfg.UPSAMPLING_RATIO,
            upsampling_mode=cfg.UPSAMPLING_MODE,
            train_with_mixup=cfg.TRAIN_WITH_MIXUP,
            train_with_label_smoothing=cfg.TRAIN_WITH_LABEL_SMOOTHING,
            train_with_focal_loss=cfg.TRAIN_WITH_FOCAL_LOSS,
            focal_loss_gamma=cfg.FOCAL_LOSS_GAMMA,
            focal_loss_alpha=cfg.FOCAL_LOSS_ALPHA,
            on_epoch_end=on_epoch_end,
        )
    except model.get_empty_class_exception() as e:
        e.message = f"Class with label {labels[e.index]} is empty. Please remove it from the training data."
        e.args = (e.message,)
        raise e
    except Exception as e:
        raise Exception("Error training model") from e

    print("...Done.", flush=True)

    # Get best validation metrics based on AUPRC instead of loss for more reliable results with imbalanced data
    best_epoch = np.argmax(history.history["val_AUPRC"])
    best_val_auprc = history.history["val_AUPRC"][best_epoch]
    best_val_auroc = history.history["val_AUROC"][best_epoch]
    best_val_loss = history.history["val_loss"][best_epoch]

    print("Saving model...", flush=True)

    try:
        if cfg.TRAINED_MODEL_OUTPUT_FORMAT == "both":
            model.save_raven_model(classifier, cfg.CUSTOM_CLASSIFIER, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
            model.save_linear_classifier(classifier, cfg.CUSTOM_CLASSIFIER, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
        elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "tflite":
            model.save_linear_classifier(classifier, cfg.CUSTOM_CLASSIFIER, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
        elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "raven":
            model.save_raven_model(classifier, cfg.CUSTOM_CLASSIFIER, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
        else:
            raise ValueError(f"Unknown model output format: {cfg.TRAINED_MODEL_OUTPUT_FORMAT}")
    except Exception as e:
        raise Exception("Error saving model") from e

    save_sample_counts(labels, y_train)

    # Evaluate model on test data if available
    metrics = None
    if len(x_test) > 0:
        print("\nEvaluating model on test data...", flush=True)
        metrics = evaluate_model(classifier, x_test, y_test, labels)

        # Save evaluation results to file
        if metrics:
            import csv

            eval_file_path = cfg.CUSTOM_CLASSIFIER + "_evaluation.csv"
            with open(eval_file_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Define all the metrics as columns, including both default and optimized threshold metrics
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
                        "",  # Empty cells for Threshold, TP, FP, TN, FN, Samples, Percentage
                    ]
                )

                # Write per-class metrics (one row per species)
                for class_name, class_metrics in metrics["class_metrics"].items():
                    distribution = metrics["class_distribution"].get(class_name, {"count": 0, "percentage": 0.0})
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

            print(f"Evaluation results saved to {eval_file_path}", flush=True)
    else:
        print("\nNo separate test data provided for evaluation. Using validation metrics.", flush=True)

    print(
        f"...Done. Best AUPRC: {best_val_auprc}, Best AUROC: {best_val_auroc}, Best Loss: {best_val_loss} (epoch {best_epoch + 1}/{len(history.epoch)})",
        flush=True,
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
        threshold: Classification threshold (if None, will find optimal threshold for each class)

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

    # Skip evaluation if test set is empty
    if len(x_test) == 0:
        print("No test data available for evaluation.")
        return {}

    # Make predictions
    y_pred_prob = classifier.predict(x_test)

    # Calculate metrics for each class
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
    print("\nNote: The AUPRC and AUROC metrics calculated during post-training evaluation may differ")
    print("from training history values due to different calculation methods:")
    print("  - Training history uses Keras metrics calculated over batches")
    print("  - Evaluation uses scikit-learn metrics calculated over the entire dataset")

    for i in range(y_test.shape[1]):
        try:
            # Calculate metrics with default threshold (0.5)
            y_pred_default = (y_pred_prob[:, i] >= 0.5).astype(int)

            class_precision_default = precision_score(y_test[:, i], y_pred_default)
            class_recall_default = recall_score(y_test[:, i], y_pred_default)
            class_f1_default = f1_score(y_test[:, i], y_pred_default)

            precisions_default.append(class_precision_default)
            recalls_default.append(class_recall_default)
            f1s_default.append(class_f1_default)

            # Find optimal threshold for this class if needed
            if threshold is None:
                class_threshold = find_optimal_threshold(y_test[:, i], y_pred_prob[:, i])
                optimal_thresholds[labels[i]] = class_threshold
            else:
                class_threshold = threshold

            # Calculate metrics with optimized threshold
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

            # Confusion matrix with optimized threshold
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
