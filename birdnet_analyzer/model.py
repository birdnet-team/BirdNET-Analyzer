# pyright: reportOptionalMemberAccess=false
"""Contains functions to use the BirdNET models."""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import TYPE_CHECKING, Literal

import keras
import numpy as np
import tensorflow as tf

# from birdnet.acoustic.models.v2_4.pb import AcousticPBDownloaderV2_4 # 0.2.13
from birdnet.acoustic_models.v2_4.pb import AcousticPBDownloaderV2_4

from birdnet_analyzer import utils
from birdnet_analyzer.config import RANDOM_SEED
from birdnet_analyzer.train import custom_models

if TYPE_CHECKING:
    from numpy.random import Generator

tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
EMPTY_CLASS_EXCEPTION_REF = None


class WrappedSavedModel(keras.layers.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, inputs):
        outputs = self.fn(inputs)
        return list(outputs.values())[0]


def get_empty_class_exception():
    """Return a reusable exception class for signaling empty classes.

    The previous implementation subclassed
    :class:`keras_tuner.errors.FatalError` simply because the tuner was
    responsible for raising the error. After switching to ``optuna`` we no
    longer have a dependency on ``keras_tuner``; a plain :class:`Exception` is
    sufficient.
    """
    global EMPTY_CLASS_EXCEPTION_REF  # noqa: PLW0603

    if EMPTY_CLASS_EXCEPTION_REF:
        return EMPTY_CLASS_EXCEPTION_REF

    class EmptyClassException(Exception):
        """Error raised when a label channel contains no samples.

        Attributes:
            index (int): The index of the empty class.
            message (str): Human readable error message.
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
    Label smoothing is a technique used to prevent the model from becoming overconfident
    by adjusting the target labels. It subtracts a small value (alpha) from the correct
    label and distributes it among the other labels.
    Args:
        y (numpy.ndarray): Array of labels to be smoothed. The array should be of shape
            (num_labels,).
        alpha (float, optional): Smoothing parameter. Default is 0.1.
    Returns:
        numpy.ndarray: The smoothed labels.
    """
    y[y > 0] -= alpha
    y[y == 0] = alpha / y.shape[0]

    return y


def mixup(x, y, rng: Generator, augmentation_ratio=0.25, alpha=0.2):
    """Apply mixup to the given data.

    Mixup is a data augmentation technique that generates new samples by
    mixing two samples and their labels.

    Args:
        x: Samples.
        y: One-hot labels.
        rng: Random number generator.
        augmentation_ratio: The ratio of augmented samples.
        alpha: The beta distribution parameter.

    Returns:
        Augmented data.
    """
    positive_indices = np.unique(np.where(y[:, :] == 1)[0])
    num_samples_to_augment = int(len(positive_indices) * augmentation_ratio)
    mixed_up_indices = []

    for _ in range(num_samples_to_augment):
        index = rng.choice(positive_indices)

        while index in mixed_up_indices:
            index = rng.choice(positive_indices)

        x1, y1 = x[index], y[index]

        second_index = rng.choice(positive_indices)

        while second_index == index or second_index in mixed_up_indices:
            second_index = rng.choice(positive_indices)

        x2, y2 = x[second_index], y[second_index]
        lambda_ = rng.beta(alpha, alpha)
        mixed_x = lambda_ * x1 + (1 - lambda_) * x2
        mixed_y = lambda_ * y1 + (1 - lambda_) * y2
        x[index] = mixed_x
        y[index] = mixed_y

        mixed_up_indices.append(index)

        del mixed_x
        del mixed_y

    return x, y


def random_split(x, y, rng: Generator, val_ratio=0.2):
    """Splits the data into training and validation data.

    Makes sure that each class is represented in both sets.

    Args:
        x: Samples.
        y: One-hot labels.
        rng: Random number generator.
        val_ratio: The ratio of validation data.

    Returns:
        A tuple of (x_train, y_train, x_val, y_val).
    """
    num_classes = y.shape[1]
    x_train, y_train, x_val, y_val = [], [], [], []

    for i in range(num_classes):
        positive_indices = np.where(y[:, i] == 1)[0]
        negative_indices = np.where(y[:, i] == -1)[0]

        num_samples = len(positive_indices)
        num_samples_train = max(1, int(num_samples * (1 - val_ratio)))
        num_samples_val = max(0, num_samples - num_samples_train)

        rng.shuffle(positive_indices)

        train_indices = positive_indices[:num_samples_train]
        val_indices = positive_indices[
            num_samples_train : num_samples_train + num_samples_val
        ]

        x_train.append(x[train_indices])
        y_train.append(y[train_indices])
        x_val.append(x[val_indices])
        y_val.append(y[val_indices])

        x_train.append(x[negative_indices])
        y_train.append(y[negative_indices])

    non_event_indices = np.where(np.sum(y[:, :], axis=1) == 0)[0]
    num_samples = len(non_event_indices)
    num_samples_train = max(1, int(num_samples * (1 - val_ratio)))
    num_samples_val = max(0, num_samples - num_samples_train)

    rng.shuffle(non_event_indices)

    train_indices = non_event_indices[:num_samples_train]
    val_indices = non_event_indices[
        num_samples_train : num_samples_train + num_samples_val
    ]

    x_train.append(x[train_indices])
    y_train.append(y[train_indices])
    x_val.append(x[val_indices])
    y_val.append(y[val_indices])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)

    indices = np.arange(len(x_train))

    rng.shuffle(indices)

    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(len(x_val))

    rng.shuffle(indices)

    x_val = x_val[indices]
    y_val = y_val[indices]

    return x_train, y_train, x_val, y_val


def random_multilabel_split(x, y, rng: Generator, val_ratio=0.2):
    """Splits the data into training and validation data.

    Makes sure that each combination of classes is represented in both sets.

    Args:
        x: Samples.
        y: One-hot labels.
        rng: Random number generator.
        val_ratio: The ratio of validation data.

    Returns:
        A tuple of (x_train, y_train, x_val, y_val).

    """
    class_combinations = np.unique(y, axis=0)
    x_train, y_train, x_val, y_val = [], [], [], []

    for class_combination in class_combinations:
        indices = np.where((y == class_combination).all(axis=1))[0]

        if -1 in class_combination:
            x_train.append(x[indices])
            y_train.append(y[indices])
        else:
            num_samples = len(indices)
            num_samples_train = max(1, int(num_samples * (1 - val_ratio)))
            num_samples_val = max(0, num_samples - num_samples_train)

            rng.shuffle(indices)

            train_indices = indices[:num_samples_train]
            val_indices = indices[
                num_samples_train : num_samples_train + num_samples_val
            ]

            x_train.append(x[train_indices])
            y_train.append(y[train_indices])
            x_val.append(x[val_indices])
            y_val.append(y[val_indices])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)

    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(len(x_val))
    rng.shuffle(indices)
    x_val = x_val[indices]
    y_val = y_val[indices]

    return x_train, y_train, x_val, y_val


def upsample_core(
    x: np.ndarray,
    y: np.ndarray,
    min_samples: int,
    rng: Generator,
    apply,
    is_binary: bool,
    size=2,
):
    """
    Upsamples the minority class in the dataset using the specified apply function.
    Parameters:
        x (np.ndarray): The feature matrix.
        y (np.ndarray): The target labels.
        min_samples (int): The minimum number of samples required for the minority
            class.
        rng (Generator): A random number generator.
        apply (callable): A function that applies the SMOTE or any other algorithm to
            the data.
        is_binary (bool): Whether the classification is binary.
        size (int, optional): The number of samples to generate in each iteration.
            Default is 2.
    Returns:
        tuple: A tuple containing the upsampled feature matrix and target labels.
    """
    y_temp = []
    x_temp = []

    if is_binary:
        minority_label = 1 if y.sum(axis=0) < len(y) - y.sum(axis=0) else 0

        while np.where(y == minority_label)[0].shape[0] + len(y_temp) < min_samples:
            random_index = rng.choice(np.where(y == minority_label)[0], size=size)
            x_app, y_app = apply(x, y, random_index)

            y_temp.append(y_app)
            x_temp.append(x_app)
    else:
        for i in range(y.shape[1]):
            while y[:, i].sum() + len(y_temp) < min_samples:
                try:
                    random_index = rng.choice(np.where(y[:, i] == 1)[0], size=size)
                except ValueError as e:
                    raise get_empty_class_exception()(index=i) from e

                x_app, y_app = apply(x, y, random_index)

                y_temp.append(y_app)
                x_temp.append(x_app)

    return x_temp, y_temp


def upsampling(
    x: np.ndarray,
    y: np.ndarray,
    rng: Generator,
    is_binary: bool,
    ratio=0.5,
    mode="repeat",
):
    """Balance data through upsampling.

    We upsample minority classes to have at least 10% (ratio=0.1) of the samples of the
    majority class.

    Args:
        x: Samples.
        y: One-hot labels.
        rng: Random number generator.
        is_binary: Whether the classification is binary.
        ratio: The minimum ratio of minority to majority samples.
        mode: The upsampling mode. Either 'repeat', 'mean', 'linear' or 'smote'.

    Returns:
        Upsampled data.
    """
    min_samples = (
        int(max(y.sum(axis=0), len(y) - y.sum(axis=0)) * ratio)
        if is_binary
        else int(np.max(y.sum(axis=0)) * ratio)
    )
    x_temp = []
    y_temp = []

    if mode == "repeat":

        def applyRepeat(x, y, random_index):
            return x[random_index[0]], y[random_index[0]]

        x_temp, y_temp = upsample_core(
            x, y, min_samples, rng, applyRepeat, is_binary, size=1
        )

    elif mode == "mean":

        def applyMean(x, y, random_indices):
            mean = np.mean(x[random_indices], axis=0)

            return mean, y[random_indices[0]]

        x_temp, y_temp = upsample_core(x, y, min_samples, rng, applyMean, is_binary)
    elif mode == "linear":

        def applyLinearCombination(x, y, random_indices):
            alpha = rng.uniform(0, 1)
            new_sample = (
                alpha * x[random_indices[0]] + (1 - alpha) * x[random_indices[1]]
            )

            return new_sample, y[random_indices[0]]

        x_temp, y_temp = upsample_core(
            x, y, min_samples, rng, applyLinearCombination, is_binary
        )

    elif mode == "smote":

        def applySmote(x, y, random_index, k=5):
            distances = np.sqrt(np.sum((x - x[random_index[0]]) ** 2, axis=1))
            indices = np.argsort(distances)[1 : k + 1]
            random_neighbor = rng.choice(indices)
            diff = x[random_neighbor] - x[random_index[0]]
            weight = rng.uniform(0, 1)
            new_sample = x[random_index[0]] + weight * diff

            return new_sample, y[random_index[0]]

        x_temp, y_temp = upsample_core(
            x, y, min_samples, rng, applySmote, is_binary, size=1
        )

    if len(x_temp) > 0:
        x = np.vstack((x, np.array(x_temp)))
        y = np.vstack((y, np.array(y_temp)))

    indices = np.arange(len(x))
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]

    del x_temp
    del y_temp

    return x, y


def build_linear_classifier(num_labels, input_size, hidden_units=0, dropout=0.0):
    """Builds a classifier.

    Args:
        num_labels: Output size.
        input_size: Size of the input.
        hidden_units: If > 0, creates another hidden layer with the given number of
            units.
        dropout: Dropout rate.

    Returns:
        A new classifier.
    """
    regularizer = keras.regularizers.l2(1e-5)
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(input_size,)))

    if hidden_units > 0:
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))

        model.add(
            keras.layers.Dense(
                hidden_units,
                activation="relu",
                kernel_regularizer=regularizer,
                kernel_initializer="he_normal",
            )
        )

    if dropout > 0:
        model.add(keras.layers.Dropout(dropout))

    model.add(
        keras.layers.Dense(
            num_labels,
            kernel_regularizer=regularizer,
            kernel_initializer="glorot_uniform",
        )
    )
    model.add(keras.layers.Activation("sigmoid"))

    return model


def train_linear_classifier(
    classifier: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
    upsampling_ratio: float,
    upsampling_mode: str,
    train_with_mixup: bool,
    train_with_label_smoothing: bool,
    train_with_focal_loss=False,
    focal_loss_gamma=2.0,
    focal_loss_alpha=0.25,
    is_multi_label=False,
    is_binary_classification=False,
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
        train_with_label_smoothing: If True, applies label smoothing to the training
            data.
        train_with_focal_loss: If True, uses focal loss instead of binary cross-entropy
            loss.
        focal_loss_gamma: Focal loss gamma parameter.
        focal_loss_alpha: Focal loss alpha parameter.
        is_multi_label: If True, multi-label classification is used.
        is_binary_classification: If True, binary classification is used.
        on_epoch_end: Optional callback `function(epoch, logs)`.

    Returns:
        (classifier, history)
    """
    setting_cache = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    class FunctionCallback(keras.callbacks.Callback):
        def __init__(self, on_epoch_end=None) -> None:
            super().__init__()
            self.on_epoch_end_fn = on_epoch_end

        def on_epoch_end(self, epoch, logs=None):
            if self.on_epoch_end_fn:
                self.on_epoch_end_fn(epoch, logs)

    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(x_train.shape[0])
    rng.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]

    if val_split > 0:
        if not is_multi_label:
            x_train, y_train, x_val, y_val = random_split(
                x_train, y_train, rng, val_split
            )
        else:
            x_train, y_train, x_val, y_val = random_multilabel_split(
                x_train, y_train, rng, val_split
            )

    if upsampling_ratio > 0:
        x_train, y_train = upsampling(
            x_train,
            y_train,
            rng,
            is_binary_classification,
            upsampling_ratio,
            upsampling_mode,
        )

    if train_with_mixup and not is_binary_classification:
        x_train, y_train = mixup(x_train, y_train, rng)

    if train_with_label_smoothing and not is_binary_classification:
        y_train = label_smoothing(y_train)

    patience = min(10, max(5, int(epochs / 10)))
    min_delta = 0.001
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            verbose=1,
            min_delta=min_delta,  # type: ignore
            restore_best_weights=True,
        ),
        FunctionCallback(on_epoch_end=on_epoch_end),
    ]
    warmup_epochs = min(5, int(epochs * 0.1))

    def lr_schedule(epoch, lr):
        if epoch < warmup_epochs:
            return learning_rate * (epoch + 1) / warmup_epochs

        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)

        return learning_rate * (0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2)

    callbacks.append(keras.callbacks.LearningRateScheduler(lr_schedule))

    def _focal_loss(y_true, y_pred):
        return focal_loss(
            y_true, y_pred, gamma=focal_loss_gamma, alpha=focal_loss_alpha
        )

    loss_function = _focal_loss if train_with_focal_loss else custom_loss

    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  # type: ignore
        loss=loss_function,
        metrics=[
            keras.metrics.AUC(
                curve="PR",
                multi_label=is_multi_label,
                name="AUPRC",
                num_labels=y_train.shape[1] if is_multi_label else None,
                from_logits=True,
            ),
            keras.metrics.AUC(
                curve="ROC",
                multi_label=is_multi_label,
                name="AUROC",
                num_labels=y_train.shape[1] if is_multi_label else None,
                from_logits=True,
            ),
        ],
    )

    history = classifier.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = setting_cache

    return classifier, history

def save_detached_classifier(
    classifier,
    model_path: str,
):
    """Saves the detached classifier head as a pb model.

    Args:
        classifier: The custom classifier.
        model_path: Path the model will be saved at.
    """
    if model_path.endswith(".tflite"):
        model_path = model_path.removesuffix(".tflite")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    detached_classifier_path = model_path + "_detached"

    detached_model_inputs = keras.Input(shape=(1024,), dtype=tf.float32, name="detached_input")
    detached_model_outputs = classifier(detached_model_inputs)
    detached_model = keras.Model(inputs=detached_model_inputs, outputs=detached_model_outputs, name="detached_classifier")

    detached_model.export(detached_classifier_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(detached_model)
    tflite_model: bytes = converter.convert()

    with open(detached_classifier_path + ".tflite", "wb") as f:
        f.write(tflite_model)

def save_linear_classifier(
    classifier,
    model_path: str,
    labels: list[str],
    mode: Literal["replace", "append"] = "replace",
    params: tuple[list[str], list] | None = None,
):
    """Saves the classifier as a tflite model, as well as the used labels in a .txt.

    Args:
        classifier: The custom classifier.
        model_path: Path the model will be saved at.
        labels: List of labels used for the classifier.
    """
    if mode not in ("replace", "append"):
        raise ValueError("Model save mode must be either 'replace' or 'append'")

    saved_model_path, original_labels = (
        AcousticPBDownloaderV2_4.get_model_path_and_labels("en_us")
    )
    saved_model = tf.saved_model.load(saved_model_path)
    inputs = keras.Input(shape=(144000,), dtype=tf.float32, name="input_audio")
    wrapper = WrappedSavedModel(saved_model.signatures["embeddings"])(inputs)

    if mode == "replace":
        output = classifier(wrapper)
    elif mode == "append":
        basic = WrappedSavedModel(saved_model.signatures["basic"])(inputs)
        output = keras.layers.concatenate(
            [basic, classifier(wrapper)], name="combined_output"
        )

    combined_model = keras.Model(inputs=inputs, outputs=output, name="basic")

    if not model_path.endswith(".tflite"):
        model_path += ".tflite"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
    tflite_model: bytes = converter.convert()

    with open(model_path, "wb") as f:
        f.write(tflite_model)

    if mode == "append":
        labels = [*original_labels, *labels]

    with open(model_path.replace(".tflite", "_Labels.txt"), "w", encoding="utf-8") as f:
        f.writelines(label + "\n" for label in labels)

    if params:
        utils.save_params_to_file(model_path.replace(".tflite", "_Params.csv"), *params)

def save_raven_model(
    classifier,
    model_path: str,
    labels: list[str],
    mode: Literal["replace", "append"] = "replace",
    sig_fmin=0,
    sig_fmax=15000,
    model_version="2.4",
    params: tuple[list[str], list] | None = None,
):
    """
    Save a TensorFlow model with a custom classifier and associated metadata for use
    with BirdNET.

    Args:
        classifier (tf.keras.Model): The custom classifier model to be saved.
        model_path (str): The path where the model will be saved.
        labels (list[str]): A list of labels associated with the classifier.
        mode (str, optional): The mode for saving the model. Can be either "replace" or
            "append". Defaults to "replace".

    Raises:
        ValueError: If the mode is not "replace" or "append".

    Returns:
        None
    """

    saved_model_path, original_labels = (
        AcousticPBDownloaderV2_4.get_model_path_and_labels("en_us")
    )
    saved_model = tf.saved_model.load(saved_model_path)
    model_cls = (
        custom_models.CombinedModelAppendWithSigmoid
        if mode == "append"
        else custom_models.CombinedModelReplaceWithSigmoid
    )
    combined_model = model_cls(saved_model, classifier)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 144000], dtype=tf.float32)]
    )  # pyright: ignore[reportCallIssue]
    def basic(inputs):
        return {"scores": combined_model(inputs)}

    signatures = {
        "basic": basic,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model_path = model_path.removesuffix(".tflite")

    tf.saved_model.save(combined_model, model_path, signatures=signatures)

    if mode == "append":
        labels = [*original_labels, *labels]

    labelIds = [
        label[:4].replace(" ", "") + str(i) for i, label in enumerate(labels, 1)
    ]
    labels_dir = os.path.join(model_path, "labels")

    os.makedirs(labels_dir, exist_ok=True)

    with open(
        os.path.join(labels_dir, "label_names.csv"), "w", newline=""
    ) as labelsfile:
        labelwriter = csv.writer(labelsfile)
        labelwriter.writerows(zip(labelIds, labels, strict=True))

    classes_dir = os.path.join(model_path, "classes")

    os.makedirs(classes_dir, exist_ok=True)

    with open(os.path.join(classes_dir, "classes.csv"), "w", newline="") as classesfile:
        classeswriter = csv.writer(classesfile)

        for labelId in labelIds:
            classeswriter.writerow((labelId, 0.25, sig_fmin, sig_fmax, False))

    model_config = os.path.join(model_path, "model_config.json")

    with open(model_config, "w") as modelconfigfile:
        modelconfig = {
            "specVersion": 1,
            "modelDescription": "Custom classifier trained with BirdNET "
            + model_version
            + " embeddings.\n"
            + "BirdNET was developed by the K. Lisa Yang Center for Conservation "
            + "Bioacoustics at the Cornell Lab of Ornithology in collaboration with "
            + "Chemnitz University of Technology.\n\nhttps://birdnet.cornell.edu",
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

        if params:
            utils.save_params_to_file(model_params, *params)


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, epsilon=1e-7):
    """
    Focal loss for better handling of class imbalance.

    This loss function gives more weight to hard examples and down-weights easy
    examples. Particularly helpful for imbalanced datasets where some classes have few
    samples.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted probabilities.
        gamma: Focusing parameter. Higher values mean more focus on hard examples.
        alpha: Balance parameter. Controls weight of positive vs negative examples.
        epsilon: Small constant to prevent log(0).

    Returns:
        Focal loss value.
    """
    y_false = 1 - y_true
    y_false_pred = 1 - y_pred
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred) - y_false * tf.math.log(y_false_pred)
    p_t = y_true * y_pred + y_false * y_false_pred
    focal_weight = tf.pow(1 - p_t, gamma)
    alpha_factor = y_true * alpha + y_false * (1 - alpha)
    focal_loss = alpha_factor * focal_weight * cross_entropy

    return tf.reduce_sum(focal_loss, axis=-1)


def custom_loss(y_true, y_pred, epsilon=1e-7):
    positive_loss = -tf.reduce_sum(
        y_true * tf.math.log(tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)), axis=-1
    )
    negative_loss = -tf.reduce_sum(
        (1 - y_true)
        * tf.math.log(tf.clip_by_value(1 - y_pred, epsilon, 1.0 - epsilon)),
        axis=-1,
    )

    return positive_loss + negative_loss


def flat_sigmoid(x, sensitivity=-1, bias=1.0):
    """
    Applies a flat sigmoid function to the input array with a bias shift.

    The flat sigmoid function is defined as:
        f(x) = 1 / (1 + exp(sensitivity * clip(x + bias, -20, 20)))

    We transform the bias parameter to a range of [-100, 100] with the formula:
        transformed_bias = (bias - 1.0) * 10.0

    Thus, higher bias values will shift the sigmoid function to the right on the x-axis,
    making it more "sensitive".

    Note: Not sure why we are clipping, must be for numerical stability somewhere else
    in the code.

    Args:
        x (array-like): Input data.
        sensitivity (float, optional): Sensitivity parameter for the sigmoid function.
            Default is -1.
        bias (float, optional): Bias parameter to shift the sigmoid function on the
            x-axis. Must be in the range [0.01, 1.99]. Default is 1.0.

    Returns:
        numpy.ndarray: Transformed data after applying the flat sigmoid function.
    """

    transformed_bias = (bias - 1.0) * 10.0

    return 1 / (1.0 + np.exp(sensitivity * np.clip(x + transformed_bias, -20, 20)))
