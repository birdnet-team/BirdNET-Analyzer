"""Module containing common function."""

import itertools
import os
import traceback
from pathlib import Path

import numpy as np

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


def spectrogram_from_file(path, fig_num=None, fig_size=None):
    """
    Generate a spectrogram from an audio file.

    Parameters:
    path (str): The path to the audio file.

    Returns:
    matplotlib.figure.Figure: The generated spectrogram figure.
    """
    import librosa
    import librosa.display
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('agg')


    s, sr = librosa.load(path)

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
    f.tight_layout()
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


def random_multilabel_split(x, y, val_ratio=0.2):
    """Splits the data into training and validation data.

    Makes sure that each combination of classes is represented in both sets.

    Args:
        x: Samples.
        y: One-hot labels.
        val_ratio: The ratio of validation data.

    Returns:
        A tuple of (x_train, y_train, x_val, y_val).

    """

    # Set numpy random seed
    np.random.seed(cfg.RANDOM_SEED)

    # Find all combinations of labels
    class_combinations = np.unique(y, axis=0)

    # Initialize training and validation data
    x_train, y_train, x_val, y_val = [], [], [], []

    # Split the data for each combination of labels
    for class_combination in class_combinations:
        # find all indices
        indices = np.where((y == class_combination).all(axis=1))[0]

        # When negative sample use only for training
        if -1 in class_combination:
            x_train.append(x[indices])
            y_train.append(y[indices])
        # Otherwise split according to the validation split
        else:
            # Get number of samples for each set
            num_samples = len(indices)
            num_samples_train = max(1, int(num_samples * (1 - val_ratio)))
            num_samples_val = max(0, num_samples - num_samples_train)
            # Randomly choose samples for training and validation
            np.random.shuffle(indices)
            train_indices = indices[:num_samples_train]
            val_indices = indices[num_samples_train : num_samples_train + num_samples_val]
            # Append samples to training and validation data
            x_train.append(x[train_indices])
            y_train.append(y[train_indices])
            x_val.append(x[val_indices])
            y_val.append(y[val_indices])

    # Concatenate data
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)

    # Shuffle data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(len(x_val))
    np.random.shuffle(indices)
    x_val = x_val[indices]
    y_val = y_val[indices]

    return x_train, y_train, x_val, y_val


def random_split(x, y, val_ratio=0.2):
    """Splits the data into training and validation data.

    Makes sure that each class is represented in both sets.

    Args:
        x: Samples.
        y: One-hot labels.
        val_ratio: The ratio of validation data.

    Returns:
        A tuple of (x_train, y_train, x_val, y_val).
    """

    # Set numpy random seed
    np.random.seed(cfg.RANDOM_SEED)

    # Get number of classes
    num_classes = y.shape[1]

    # Initialize training and validation data
    x_train, y_train, x_val, y_val = [], [], [], []

    # Split data
    for i in range(num_classes):
        # Get indices of positive samples of current class
        positive_indices = np.where(y[:, i] == 1)[0]

        # Get indices of negative samples of current class
        negative_indices = np.where(y[:, i] == -1)[0]

        # Get number of samples for each set
        num_samples = len(positive_indices)
        num_samples_train = max(1, int(num_samples * (1 - val_ratio)))
        num_samples_val = max(0, num_samples - num_samples_train)

        # Randomly choose samples for training and validation
        np.random.shuffle(positive_indices)
        train_indices = positive_indices[:num_samples_train]
        val_indices = positive_indices[num_samples_train : num_samples_train + num_samples_val]

        # Append samples to training and validation data
        x_train.append(x[train_indices])
        y_train.append(y[train_indices])
        x_val.append(x[val_indices])
        y_val.append(y[val_indices])

        # Append negative samples to training data
        x_train.append(x[negative_indices])
        y_train.append(y[negative_indices])

    # Add samples for non-event classes to training and validation data
    non_event_indices = np.where(np.sum(y[:, :], axis=1) == 0)[0]
    num_samples = len(non_event_indices)
    num_samples_train = max(1, int(num_samples * (1 - val_ratio)))
    num_samples_val = max(0, num_samples - num_samples_train)
    np.random.shuffle(non_event_indices)
    train_indices = non_event_indices[:num_samples_train]
    val_indices = non_event_indices[num_samples_train : num_samples_train + num_samples_val]
    x_train.append(x[train_indices])
    y_train.append(y[train_indices])
    x_val.append(x[val_indices])
    y_val.append(y[val_indices])

    # Concatenate data
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)

    # Shuffle data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(len(x_val))
    np.random.shuffle(indices)
    x_val = x_val[indices]
    y_val = y_val[indices]

    return x_train, y_train, x_val, y_val


def mixup(x, y, augmentation_ratio=0.25, alpha=0.2):
    """Apply mixup to the given data.

    Mixup is a data augmentation technique that generates new samples by
    mixing two samples and their labels.

    Args:
        x: Samples.
        y: One-hot labels.
        augmentation_ratio: The ratio of augmented samples.
        alpha: The beta distribution parameter.

    Returns:
        Augmented data.
    """

    # Set numpy random seed
    np.random.seed(cfg.RANDOM_SEED)

    # Get indices of all positive samples
    positive_indices = np.unique(np.where(y[:, :] == 1)[0])

    # Calculate the number of samples to augment based on the ratio
    num_samples_to_augment = int(len(positive_indices) * augmentation_ratio)

    # Indices of samples, that are already mixed up
    mixed_up_indices = []

    for _ in range(num_samples_to_augment):
        # Randomly choose one instance from the positive samples
        index = np.random.choice(positive_indices)

        # Choose another one, when the chosen one was already mixed up
        while index in mixed_up_indices:
            index = np.random.choice(positive_indices)

        x1, y1 = x[index], y[index]

        # Randomly choose a different instance from the dataset
        second_index = np.random.choice(positive_indices)

        # Choose again, when the same or an already mixed up sample was selected
        while second_index == index or second_index in mixed_up_indices:
            second_index = np.random.choice(positive_indices)
        x2, y2 = x[second_index], y[second_index]

        # Generate a random mixing coefficient (lambda)
        lambda_ = np.random.beta(alpha, alpha)

        # Mix the embeddings and labels
        mixed_x = lambda_ * x1 + (1 - lambda_) * x2
        mixed_y = lambda_ * y1 + (1 - lambda_) * y2

        # Replace one of the original samples and labels with the augmented sample and labels
        x[index] = mixed_x
        y[index] = mixed_y

        # Mark the sample as already mixed up
        mixed_up_indices.append(index)

    del mixed_x
    del mixed_y

    return x, y


def label_smoothing(y: np.ndarray, alpha=0.1):
    """
    Applies label smoothing to the given labels.
    Label smoothing is a technique used to prevent the model from becoming overconfident by adjusting the target labels.
    It subtracts a small value (alpha) from the correct label and distributes it among the other labels.
    Args:
        y (numpy.ndarray): Array of labels to be smoothed. The array should be of shape (num_labels,).
        alpha (float, optional): Smoothing parameter. Default is 0.1.
    Returns:
        numpy.ndarray: The smoothed labels.
    """
    # Subtract alpha from correct label when it is >0
    y[y > 0] -= alpha

    # Assigned alpha to all other labels
    y[y == 0] = alpha / y.shape[0]

    return y


def save_to_cache(cache_file: str, x_train: np.ndarray, y_train: np.ndarray, labels: list[str]):
    """Saves the training data to a cache file.

    Args:
        cache_file: The path to the cache file.
        x_train: The training samples.
        y_train: The training labels.
        labels: The list of labels.
    """
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
