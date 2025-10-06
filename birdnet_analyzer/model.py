# ruff: noqa: PLW0603
# pyright: reportOptionalMemberAccess=false
"""Contains functions to use the BirdNET models."""

import csv
import json
import logging
import os
from typing import Literal

import keras
import numpy as np
import tensorflow as tf

import birdnet_analyzer.config as cfg
from birdnet_analyzer import utils
from birdnet_analyzer.train import custom_models

try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ModuleNotFoundError:
    from tensorflow import lite as tflite

tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
INTERPRETER = None
C_INTERPRETER = None
M_INTERPRETER = None
OUTPUT_DETAILS = None
PBMODEL = None
PERCH_MODEL = None
C_PBMODEL = None
EMPTY_CLASS_EXCEPTION_REF = None


class WrappedSavedModel(keras.layers.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, inputs):
        outputs = self.fn(inputs)
        return list(outputs.values())[0]


def _load_interpreter(mpath, threads):
    return tflite.Interpreter(
        model_path=mpath,
        num_threads=threads,
        # XNNPACK disabled, because it does not support variable inputsize anyway (ie batchsize)
        experimental_op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )


def get_empty_class_exception():
    import keras_tuner.errors

    global EMPTY_CLASS_EXCEPTION_REF

    if EMPTY_CLASS_EXCEPTION_REF:
        return EMPTY_CLASS_EXCEPTION_REF

    class EmptyClassException(keras_tuner.errors.FatalError):
        """
        Exception raised when a class is found to be empty.

        Attributes:
            index (int): The index of the empty class.
            message (str): The error message indicating which class is empty.
        """

        def __init__(self, *args, index=None):
            super().__init__(*args)
            self.index = index
            self.message = f"Class {index} is empty."

    EMPTY_CLASS_EXCEPTION_REF = EmptyClassException
    return EMPTY_CLASS_EXCEPTION_REF


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
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    # Get indices of all positive samples
    positive_indices = np.unique(np.where(y[:, :] == 1)[0])

    # Calculate the number of samples to augment based on the ratio
    num_samples_to_augment = int(len(positive_indices) * augmentation_ratio)

    # Indices of samples, that are already mixed up
    mixed_up_indices = []

    for _ in range(num_samples_to_augment):
        # Randomly choose one instance from the positive samples
        index = rng.choice(positive_indices)

        # Choose another one, when the chosen one was already mixed up
        while index in mixed_up_indices:
            index = rng.choice(positive_indices)

        x1, y1 = x[index], y[index]

        # Randomly choose a different instance from the dataset
        second_index = rng.choice(positive_indices)

        # Choose again, when the same or an already mixed up sample was selected
        while second_index == index or second_index in mixed_up_indices:
            second_index = rng.choice(positive_indices)
        x2, y2 = x[second_index], y[second_index]

        # Generate a random mixing coefficient (lambda)
        lambda_ = rng.beta(alpha, alpha)

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
    rng = np.random.default_rng(cfg.RANDOM_SEED)

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
        rng.shuffle(positive_indices)
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
    rng.shuffle(non_event_indices)
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
    rng.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(len(x_val))
    rng.shuffle(indices)
    x_val = x_val[indices]
    y_val = y_val[indices]

    return x_train, y_train, x_val, y_val


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
    rng = np.random.default_rng(cfg.RANDOM_SEED)

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
            rng.shuffle(indices)
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
    rng.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(len(x_val))
    rng.shuffle(indices)
    x_val = x_val[indices]
    y_val = y_val[indices]

    return x_train, y_train, x_val, y_val


def upsample_core(x: np.ndarray, y: np.ndarray, min_samples: int, apply, size=2):
    """
    Upsamples the minority class in the dataset using the specified apply function.
    Parameters:
        x (np.ndarray): The feature matrix.
        y (np.ndarray): The target labels.
        min_samples (int): The minimum number of samples required for the minority class.
        apply (callable): A function that applies the SMOTE or any other algorithm to the data.
        size (int, optional): The number of samples to generate in each iteration. Default is 2.
    Returns:
        tuple: A tuple containing the upsampled feature matrix and target labels.
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    y_temp = []
    x_temp = []

    if cfg.BINARY_CLASSIFICATION:
        # Determine if 1 or 0 is the minority class
        minority_label = 1 if y.sum(axis=0) < len(y) - y.sum(axis=0) else 0

        while np.where(y == minority_label)[0].shape[0] + len(y_temp) < min_samples:
            # Randomly choose a sample from the minority class
            random_index = rng.choice(np.where(y == minority_label)[0], size=size)

            # Apply SMOTE
            x_app, y_app = apply(x, y, random_index)
            y_temp.append(y_app)
            x_temp.append(x_app)
    else:
        for i in range(y.shape[1]):
            while y[:, i].sum() + len(y_temp) < min_samples:
                try:
                    # Randomly choose a sample from the minority class
                    random_index = rng.choice(np.where(y[:, i] == 1)[0], size=size)
                except ValueError as e:
                    raise get_empty_class_exception()(index=i) from e

                # Apply SMOTE
                x_app, y_app = apply(x, y, random_index)
                y_temp.append(y_app)
                x_temp.append(x_app)

    return x_temp, y_temp


def upsampling(x: np.ndarray, y: np.ndarray, ratio=0.5, mode="repeat"):
    """Balance data through upsampling.

    We upsample minority classes to have at least 10% (ratio=0.1) of the samples of the majority class.

    Args:
        x: Samples.
        y: One-hot labels.
        ratio: The minimum ratio of minority to majority samples.
        mode: The upsampling mode. Either 'repeat', 'mean', 'linear' or 'smote'.

    Returns:
        Upsampled data.
    """

    # Set numpy random seed
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    # Determine min number of samples
    min_samples = int(max(y.sum(axis=0), len(y) - y.sum(axis=0)) * ratio) if cfg.BINARY_CLASSIFICATION else int(np.max(y.sum(axis=0)) * ratio)

    x_temp = []
    y_temp = []

    if mode == "repeat":

        def applyRepeat(x, y, random_index):
            return x[random_index[0]], y[random_index[0]]

        x_temp, y_temp = upsample_core(x, y, min_samples, applyRepeat, size=1)

    elif mode == "mean":
        # For each class with less than min_samples
        # select two random samples and calculate the mean
        def applyMean(x, y, random_indices):
            # Calculate the mean of the two samples
            mean = np.mean(x[random_indices], axis=0)

            # Append the mean and label to a temp list
            return mean, y[random_indices[0]]

        x_temp, y_temp = upsample_core(x, y, min_samples, applyMean)

    elif mode == "linear":
        # For each class with less than min_samples
        # select two random samples and calculate the linear combination
        def applyLinearCombination(x, y, random_indices):
            # Calculate the linear combination of the two samples
            alpha = rng.uniform(0, 1)
            new_sample = alpha * x[random_indices[0]] + (1 - alpha) * x[random_indices[1]]

            # Append the new sample and label to a temp list
            return new_sample, y[random_indices[0]]

        x_temp, y_temp = upsample_core(x, y, min_samples, applyLinearCombination)

    elif mode == "smote":
        # For each class with less than min_samples apply SMOTE
        def applySmote(x, y, random_index, k=5):
            # Get the k nearest neighbors
            distances = np.sqrt(np.sum((x - x[random_index[0]]) ** 2, axis=1))
            indices = np.argsort(distances)[1 : k + 1]

            # Randomly choose one of the neighbors
            random_neighbor = rng.choice(indices)

            # Calculate the difference vector
            diff = x[random_neighbor] - x[random_index[0]]

            # Randomly choose a weight between 0 and 1
            weight = rng.uniform(0, 1)

            # Calculate the new sample
            new_sample = x[random_index[0]] + weight * diff

            # Append the new sample and label to a temp list
            return new_sample, y[random_index[0]]

        x_temp, y_temp = upsample_core(x, y, min_samples, applySmote, size=1)

    # Append the temp list to the original data
    if len(x_temp) > 0:
        x = np.vstack((x, np.array(x_temp)))
        y = np.vstack((y, np.array(y_temp)))

    # Shuffle data
    indices = np.arange(len(x))
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]

    del x_temp
    del y_temp

    return x, y


def save_model_params(path):
    """Saves the model parameters to a file.

    Args:
        path: Path to the file.
    """
    utils.save_params(
        path,
        (
            "Hidden units",
            "Dropout",
            "Batchsize",
            "Learning rate",
            "Crop mode",
            "Crop overlap",
            "Audio speed",
            "Upsamling mode",
            "Upsamling ratio",
            "use mixup",
            "use label smoothing",
            "use focal loss",
            "focal loss alpha",
            "focal loss gamma",
            "BirdNET Model version",
        ),
        (
            cfg.TRAIN_HIDDEN_UNITS,
            cfg.TRAIN_DROPOUT,
            cfg.TRAIN_BATCH_SIZE,
            cfg.TRAIN_LEARNING_RATE,
            cfg.SAMPLE_CROP_MODE,
            cfg.SIG_OVERLAP,
            cfg.AUDIO_SPEED,
            cfg.UPSAMPLING_MODE,
            cfg.UPSAMPLING_RATIO,
            cfg.TRAIN_WITH_MIXUP,
            cfg.TRAIN_WITH_LABEL_SMOOTHING,
            cfg.TRAIN_WITH_FOCAL_LOSS,
            cfg.FOCAL_LOSS_ALPHA,
            cfg.FOCAL_LOSS_GAMMA,
            cfg.MODEL_VERSION,
        ),
    )


def reset_custom_classifier():
    """
    Resets the custom classifier by setting the global variables C_INTERPRETER and C_PBMODEL to None.
    This function is used to clear any existing custom classifier models and interpreters, effectively
    resetting the state of the custom classifier.
    """
    global C_INTERPRETER
    global C_PBMODEL

    C_INTERPRETER = None
    C_PBMODEL = None


def load_model(class_output=True):
    """
    Loads the machine learning model based on the configuration provided.
    This function loads either a TensorFlow Lite (TFLite) model or a protobuf model
    depending on the file extension of the model path specified in the configuration.
    It sets up the global variables for the model interpreter and input/output layer indices.

    Args:
        class_output (bool): If True, sets the output layer index to the classification output.
                             If False, sets the output layer index to the feature embeddings.
    """
    global PBMODEL
    global INTERPRETER
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global OUTPUT_DETAILS

    if cfg.MODEL_PATH is None:
        raise ValueError("MODEL_PATH is not set.")

    # Do we have to load the tflite or protobuf model?
    if cfg.MODEL_PATH.endswith(".tflite"):
        if not INTERPRETER:
            # Load TFLite model and allocate tensors.
            INTERPRETER = _load_interpreter(os.path.join(SCRIPT_DIR, cfg.MODEL_PATH), cfg.TFLITE_THREADS)
            INTERPRETER.allocate_tensors()

            # Get input and output tensors.
            input_details = INTERPRETER.get_input_details()
            OUTPUT_DETAILS = INTERPRETER.get_output_details()

            # Get input tensor index
            INPUT_LAYER_INDEX = input_details[0]["index"]

        # Get classification output or feature embeddings
        OUTPUT_LAYER_INDEX = OUTPUT_DETAILS[0]["index"] if class_output else OUTPUT_DETAILS[0]["index"] - 1

    elif not PBMODEL:
        # Load protobuf model
        # Note: This will throw a bunch of warnings about custom gradients
        # which we will ignore until TF lets us block them
        # TODO: pretty sure this doesn't work anymore, but I don't think anyone uses it
        PBMODEL = keras.models.load_model(os.path.join(SCRIPT_DIR, cfg.MODEL_PATH), compile=False)


def load_custom_classifier():
    """
    Loads a custom classifier model based on the file extension of the provided model path.
    If the model file ends with ".tflite", it loads a TensorFlow Lite model and sets up the interpreter,
    input layer index, output layer index, and input size.
    If the model file does not end with ".tflite", it loads a TensorFlow SavedModel.
    """
    global C_INTERPRETER
    global C_INPUT_LAYER_INDEX
    global C_OUTPUT_LAYER_INDEX
    global C_INPUT_SIZE
    global C_PBMODEL

    if cfg.CUSTOM_CLASSIFIER is None:
        raise ValueError("CUSTOM_CLASSIFIER is not set.")

    if cfg.CUSTOM_CLASSIFIER.endswith(".tflite"):
        # Load TFLite model and allocate tensors.
        C_INTERPRETER = _load_interpreter(cfg.CUSTOM_CLASSIFIER, cfg.TFLITE_THREADS)
        C_INTERPRETER.allocate_tensors()

        # Get input and output tensors.
        input_details = C_INTERPRETER.get_input_details()
        output_details = C_INTERPRETER.get_output_details()

        # Get input tensor index
        C_INPUT_LAYER_INDEX = input_details[0]["index"]

        C_INPUT_SIZE = input_details[0]["shape"][-1]

        # Get classification output
        C_OUTPUT_LAYER_INDEX = output_details[0]["index"]
    else:
        C_PBMODEL = tf.saved_model.load(cfg.CUSTOM_CLASSIFIER)


def load_meta_model():
    """Loads the model for species prediction.

    Initializes the model used to predict species list, based on coordinates and week of year.
    """
    global M_INTERPRETER
    global M_INPUT_LAYER_INDEX
    global M_OUTPUT_LAYER_INDEX

    # Load TFLite model and allocate tensors.
    M_INTERPRETER = _load_interpreter(os.path.join(SCRIPT_DIR, cfg.MDATA_MODEL_PATH), cfg.TFLITE_THREADS)
    M_INTERPRETER.allocate_tensors()

    # Get input and output tensors.
    input_details = M_INTERPRETER.get_input_details()
    output_details = M_INTERPRETER.get_output_details()

    # Get input tensor index
    M_INPUT_LAYER_INDEX = input_details[0]["index"]
    M_OUTPUT_LAYER_INDEX = output_details[0]["index"]


def build_linear_classifier(num_labels, input_size, hidden_units=0, dropout=0.0):
    """Builds a classifier.

    Args:
        num_labels: Output size.
        input_size: Size of the input.
        hidden_units: If > 0, creates another hidden layer with the given number of units.
        dropout: Dropout rate.

    Returns:
        A new classifier.
    """
    # Build a simple one- or two-layer linear classifier
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.InputLayer(input_shape=(input_size,)))

    # Batch normalization on input to standardize embeddings
    model.add(keras.layers.BatchNormalization())

    # Optional L2 regularization for all dense layers
    regularizer = keras.regularizers.l2(1e-5)

    # Hidden layer with improved architecture
    if hidden_units > 0:
        # Dropout layer before hidden layer
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))

        # Add a hidden layer with L2 regularization
        model.add(keras.layers.Dense(hidden_units, activation="relu", kernel_regularizer=regularizer, kernel_initializer="he_normal"))

        # Add another batch normalization after the hidden layer
        model.add(keras.layers.BatchNormalization())

    # Dropout layer before output
    if dropout > 0:
        model.add(keras.layers.Dropout(dropout))

    # Classification layer with L2 regularization
    model.add(keras.layers.Dense(num_labels, kernel_regularizer=regularizer, kernel_initializer="glorot_uniform"))

    # Activation layer
    model.add(keras.layers.Activation("sigmoid"))

    return model


def train_linear_classifier(
    classifier,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs,
    batch_size,
    learning_rate,
    val_split,
    upsampling_ratio,
    upsampling_mode,
    train_with_mixup,
    train_with_label_smoothing,
    train_with_focal_loss=False,
    focal_loss_gamma=2.0,
    focal_loss_alpha=0.25,
    on_epoch_end=None,
):
    """Trains a custom classifier.

    Trains a new classifier for BirdNET based on the given data.

    Args:
        classifier: The classifier to be trained.
        x_train: Samples.
        y_train: Labels.
        x_test: Validation samples.
        y_test: Validation labels.
        epochs: Number of epochs to train.
        batch_size: Batch size.
        learning_rate: The learning rate during training.
        val_split: Validation split ratio (is 0 when using test data).
        upsampling_ratio: Upsampling ratio.
        upsampling_mode: Upsampling mode.
        train_with_mixup: If True, applies mixup to the training data.
        train_with_label_smoothing: If True, applies label smoothing to the training data.
        train_with_focal_loss: If True, uses focal loss instead of binary cross-entropy loss.
        focal_loss_gamma: Focal loss gamma parameter.
        focal_loss_alpha: Focal loss alpha parameter.
        on_epoch_end: Optional callback `function(epoch, logs)`.

    Returns:
        (classifier, history)
    """

    class FunctionCallback(keras.callbacks.Callback):
        def __init__(self, on_epoch_end=None) -> None:
            super().__init__()
            self.on_epoch_end_fn = on_epoch_end

        def on_epoch_end(self, epoch, logs=None):
            if self.on_epoch_end_fn:
                self.on_epoch_end_fn(epoch, logs)

    # Set random seed
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    # Shuffle data
    idx = np.arange(x_train.shape[0])
    rng.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]

    # Random val split
    if val_split > 0:
        if not cfg.MULTI_LABEL:
            x_train, y_train, x_val, y_val = random_split(x_train, y_train, val_split)
        else:
            x_train, y_train, x_val, y_val = random_multilabel_split(x_train, y_train, val_split)
    else:
        x_val = x_test
        y_val = y_test

    print(
        f"Training on {x_train.shape[0]} samples, validating on {x_val.shape[0]} samples.",
        flush=True,
    )

    # Upsample training data
    if upsampling_ratio > 0:
        x_train, y_train = upsampling(x_train, y_train, upsampling_ratio, upsampling_mode)
        print(f"Upsampled training data to {x_train.shape[0]} samples.", flush=True)

    # Apply mixup to training data
    if train_with_mixup and not cfg.BINARY_CLASSIFICATION:
        x_train, y_train = mixup(x_train, y_train)

    # Apply label smoothing
    if train_with_label_smoothing and not cfg.BINARY_CLASSIFICATION:
        y_train = label_smoothing(y_train)

    # Early stopping with patience depending on dataset size
    patience = min(10, max(5, int(epochs / 10)))
    min_delta = 0.001

    callbacks = [
        # EarlyStopping with restore_best_weights
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            verbose=1,
            min_delta=min_delta,
            restore_best_weights=True,
        ),
        # Function callback for progress tracking
        FunctionCallback(on_epoch_end=on_epoch_end),
    ]

    # Learning rate schedule - use cosine decay with warmup
    warmup_epochs = min(5, int(epochs * 0.1))

    def lr_schedule(epoch, lr):
        if epoch < warmup_epochs:
            # Linear warmup
            return learning_rate * (epoch + 1) / warmup_epochs

        # Cosine decay
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return learning_rate * (0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2)

    # Add LR scheduler callback
    callbacks.append(keras.callbacks.LearningRateScheduler(lr_schedule))

    def _focal_loss(y_true, y_pred):
        return focal_loss(y_true, y_pred, gamma=focal_loss_gamma, alpha=focal_loss_alpha)

    # Choose the loss function based on config
    loss_function = _focal_loss if train_with_focal_loss else custom_loss

    # Compile model with appropriate metrics for classification task
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=[
            keras.metrics.AUC(
                curve="PR",
                multi_label=cfg.MULTI_LABEL,
                name="AUPRC",
                num_labels=y_train.shape[1] if cfg.MULTI_LABEL else None,
                from_logits=True,
            ),
            keras.metrics.AUC(
                curve="ROC",
                multi_label=cfg.MULTI_LABEL,
                name="AUROC",
                num_labels=y_train.shape[1] if cfg.MULTI_LABEL else None,
                from_logits=True,
            ),
        ],
    )

    # Train model
    history = classifier.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)

    return classifier, history


def save_linear_classifier(classifier, model_path: str, labels: list[str], mode: Literal["replace", "append"] = "replace"):
    """Saves the classifier as a tflite model, as well as the used labels in a .txt.

    Args:
        classifier: The custom classifier.
        model_path: Path the model will be saved at.
        labels: List of labels used for the classifier.
    """
    global PBMODEL

    if mode not in ("replace", "append"):
        raise ValueError("Model save mode must be either 'replace' or 'append'")

    if PBMODEL is None:
        PBMODEL = tf.saved_model.load(os.path.join(SCRIPT_DIR, cfg.PB_MODEL))

    saved_model = PBMODEL
    inputs = keras.Input(shape=(144000,), dtype=tf.float32, name="input_audio")
    wrapper = WrappedSavedModel(saved_model.signatures["embeddings"])(inputs)

    if mode == "replace":
        output = classifier(wrapper)
    elif mode == "append":
        basic = WrappedSavedModel(saved_model.signatures["basic"])(inputs)
        output = keras.layers.concatenate([basic, classifier(wrapper)], name="combined_output")

    combined_model = keras.Model(inputs=inputs, outputs=output, name="basic")

    # Append .tflite if necessary
    if not model_path.endswith(".tflite"):
        model_path += ".tflite"

    # Make folders
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model as tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
    tflite_model: bytes = converter.convert()

    with open(model_path, "wb") as f:
        f.write(tflite_model)

    if mode == "append":
        labels = [*utils.read_lines(os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)), *labels]

    # Save labels
    with open(model_path.replace(".tflite", "_Labels.txt"), "w", encoding="utf-8") as f:
        f.writelines(label + "\n" for label in labels)

    save_model_params(model_path.replace(".tflite", "_Params.csv"))


def save_raven_model(classifier, model_path: str, labels: list[str], mode: Literal["replace", "append"] = "replace"):
    """
    Save a TensorFlow model with a custom classifier and associated metadata for use with BirdNET.

    Args:
        classifier (tf.keras.Model): The custom classifier model to be saved.
        model_path (str): The path where the model will be saved.
        labels (list[str]): A list of labels associated with the classifier.
        mode (str, optional): The mode for saving the model. Can be either "replace" or "append".
                              Defaults to "replace".

    Raises:
        ValueError: If the mode is not "replace" or "append".

    Returns:
        None
    """
    global PBMODEL

    if PBMODEL is None:
        PBMODEL = tf.saved_model.load(os.path.join(SCRIPT_DIR, cfg.PB_MODEL))

    model_cls = custom_models.CombinedModelAppendWithSigmoid if mode == "append" else custom_models.CombinedModelReplaceWithSigmoid
    combined_model = model_cls(PBMODEL, classifier)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 144000], dtype=tf.float32)])  # pyright: ignore[reportCallIssue]
    def basic(inputs):
        return {"scores": combined_model(inputs)}

    signatures = {
        "basic": basic,
    }

    # Save signature model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_path = model_path.removesuffix(".tflite")
    tf.saved_model.save(combined_model, model_path, signatures=signatures)

    if mode == "append":
        labels = [*utils.read_lines(os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)), *labels]

    # Save label file
    labelIds = [label[:4].replace(" ", "") + str(i) for i, label in enumerate(labels, 1)]
    labels_dir = os.path.join(model_path, "labels")

    os.makedirs(labels_dir, exist_ok=True)

    with open(os.path.join(labels_dir, "label_names.csv"), "w", newline="") as labelsfile:
        labelwriter = csv.writer(labelsfile)
        labelwriter.writerows(zip(labelIds, labels, strict=True))

    # Save class names file
    classes_dir = os.path.join(model_path, "classes")

    os.makedirs(classes_dir, exist_ok=True)

    with open(os.path.join(classes_dir, "classes.csv"), "w", newline="") as classesfile:
        classeswriter = csv.writer(classesfile)
        for labelId in labelIds:
            classeswriter.writerow((labelId, 0.25, cfg.SIG_FMIN, cfg.SIG_FMAX, False))

    # Save model config
    model_config = os.path.join(model_path, "model_config.json")

    with open(model_config, "w") as modelconfigfile:
        modelconfig = {
            "specVersion": 1,
            "modelDescription": "Custom classifier trained with BirdNET "
            + cfg.MODEL_VERSION
            + " embeddings.\n"
            + "BirdNET was developed by the K. Lisa Yang Center for Conservation Bioacoustics"
            + "at the Cornell Lab of Ornithology in collaboration with Chemnitz University of Technology.\n\n"
            + "https://birdnet.cornell.edu",
            "modelTypeConfig": {"modelType": "RECOGNITION"},
            "signatures": [
                {
                    "signatureName": "basic",
                    "modelInputs": [
                        {
                            "inputName": "inputs",
                            "sampleRate": 48000.0,
                            "inputConfig": ["batch", "samples"],
                        }
                    ],
                    "modelOutputs": [{"outputName": "scores", "outputType": "SCORES"}],
                }
            ],
            "globalSemanticKeys": labelIds,
        }
        json.dump(modelconfig, modelconfigfile, indent=2)

        model_params = os.path.join(model_path, "model_params.csv")

        save_model_params(model_params)


def predict_filter(lat, lon, week):
    """Predicts the probability for each species.

    Args:
        lat: The latitude.
        lon: The longitude.
        week: The week of the year [1-48]. Use -1 for yearlong.

    Returns:
        A list of probabilities for all species.
    """
    # Does interpreter exist?
    if M_INTERPRETER is None:
        load_meta_model()

    # Prepare mdata as sample
    sample = np.expand_dims(np.array([lat, lon, week], dtype="float32"), 0)

    # Run inference
    M_INTERPRETER.set_tensor(M_INPUT_LAYER_INDEX, sample)
    M_INTERPRETER.invoke()

    return M_INTERPRETER.get_tensor(M_OUTPUT_LAYER_INDEX)[0]


def explore(lat: float, lon: float, week: int):
    """Predicts the species list.

    Predicts the species list based on the coordinates and week of year.

    Args:
        lat: The latitude.
        lon: The longitude.
        week: The week of the year [1-48]. Use -1 for yearlong.

    Returns:
        A sorted list of tuples with the score and the species.
    """
    # Make filter prediction
    l_filter = predict_filter(lat, lon, week)

    # Apply threshold
    l_filter = np.where(l_filter >= cfg.LOCATION_FILTER_THRESHOLD, l_filter, 0)

    # Zip with labels
    l_filter = list(zip(l_filter, cfg.LABELS, strict=True))

    # Sort by filter value
    return sorted(l_filter, key=lambda x: x[0], reverse=True)


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, epsilon=1e-7):
    """
    Focal loss for better handling of class imbalance.

    This loss function gives more weight to hard examples and down-weights easy examples.
    Particularly helpful for imbalanced datasets where some classes have few samples.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted probabilities.
        gamma: Focusing parameter. Higher values mean more focus on hard examples.
        alpha: Balance parameter. Controls weight of positive vs negative examples.
        epsilon: Small constant to prevent log(0).

    Returns:
        Focal loss value.
    """
    # Apply sigmoid if not already applied
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Calculate cross entropy
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

    # Calculate focal weight
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_weight = tf.pow(1 - p_t, gamma)

    # Apply alpha balancing
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    # Calculate focal loss
    focal_loss = alpha_factor * focal_weight * cross_entropy

    # Sum over all classes
    return tf.reduce_sum(focal_loss, axis=-1)


def custom_loss(y_true, y_pred, epsilon=1e-7):
    # Calculate loss for positive labels with epsilon
    positive_loss = -tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)), axis=-1)

    # Calculate loss for negative labels with epsilon
    negative_loss = -tf.reduce_sum((1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, epsilon, 1.0 - epsilon)), axis=-1)

    # Combine both loss terms
    return positive_loss + negative_loss


def flat_sigmoid(x, sensitivity=-1, bias=1.0):
    """
    Applies a flat sigmoid function to the input array with a bias shift.

    The flat sigmoid function is defined as:
        f(x) = 1 / (1 + exp(sensitivity * clip(x + bias, -20, 20)))

    We transform the bias parameter to a range of [-100, 100] with the formula:
        transformed_bias = (bias - 1.0) * 10.0

    Thus, higher bias values will shift the sigmoid function to the right on the x-axis, making it more "sensitive".

    Note: Not sure why we are clipping, must be for numerical stability somewhere else in the code.

    Args:
        x (array-like): Input data.
        sensitivity (float, optional): Sensitivity parameter for the sigmoid function. Default is -1.
        bias (float, optional): Bias parameter to shift the sigmoid function on the x-axis. Must be in the range [0.01, 1.99]. Default is 1.0.

    Returns:
        numpy.ndarray: Transformed data after applying the flat sigmoid function.
    """

    transformed_bias = (bias - 1.0) * 10.0

    return 1 / (1.0 + np.exp(sensitivity * np.clip(x + transformed_bias, -20, 20)))


def predict_with_perch(data: np.ndarray):
    global PERCH_MODEL

    if PERCH_MODEL is None:
        PERCH_MODEL = tf.saved_model.load(
            cfg.MODEL_PATH,
        )

    result = PERCH_MODEL.signatures["serving_default"](inputs=data)

    return tf.nn.softmax(result["label"], axis=-1).numpy()


def predict(sample):
    """Uses the main net to predict a sample.

    Args:
        sample: Audio sample.

    Returns:
        The prediction scores for the sample.
    """
    # Has custom classifier?
    if cfg.CUSTOM_CLASSIFIER is not None:
        return predict_with_custom_classifier(sample)

    if cfg.USE_PERCH:
        return predict_with_perch(sample)

    load_model()

    if PBMODEL is None:
        # Reshape input tensor
        INTERPRETER.resize_tensor_input(INPUT_LAYER_INDEX, [len(sample), *sample[0].shape])
        INTERPRETER.allocate_tensors()

        # Make a prediction (Audio only for now)
        INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample, dtype="float32"))
        INTERPRETER.invoke()
        return INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)

    # Make a prediction (Audio only for now)
    return PBMODEL.basic(sample)["scores"]


def predict_with_custom_classifier(sample):
    """Uses the custom classifier to make a prediction.

    Args:
        sample: Audio sample.

    Returns:
        The prediction scores for the sample.
    """
    # Does interpreter exist?
    if C_INTERPRETER is None and C_PBMODEL is None:
        load_custom_classifier()

    if C_PBMODEL is None:
        vector = embeddings(sample) if C_INPUT_SIZE != 144000 else sample

        # Reshape input tensor
        C_INTERPRETER.resize_tensor_input(C_INPUT_LAYER_INDEX, [len(vector), *vector[0].shape])
        C_INTERPRETER.allocate_tensors()

        # Make a prediction
        C_INTERPRETER.set_tensor(C_INPUT_LAYER_INDEX, np.array(vector, dtype="float32"))
        C_INTERPRETER.invoke()

        return C_INTERPRETER.get_tensor(C_OUTPUT_LAYER_INDEX)

    return C_PBMODEL.basic(sample)["scores"]


def embeddings(sample):
    """Extracts the embeddings for a sample.

    Args:
        sample: Audio samples.

    Returns:
        The embeddings.
    """
    load_model(False)

    sample = np.array(sample, dtype="float32")

    # Reshape input tensor
    INTERPRETER.resize_tensor_input(INPUT_LAYER_INDEX, [len(sample), *sample[0].shape])
    INTERPRETER.allocate_tensors()

    # Extract feature embeddings
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, sample)
    INTERPRETER.invoke()

    return INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)
