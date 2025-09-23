"""
Module containing functions to calculate various performance metrics using scikit-learn.

This script includes implementations for calculating accuracy, precision, recall, F1 score,
average precision, and AUROC for binary and multilabel classification tasks. It supports
various averaging methods and thresholds for predictions.

Functions:
    - calculate_accuracy: Computes accuracy for binary or multilabel classification.
    - calculate_recall: Computes recall for binary or multilabel classification.
    - calculate_precision: Computes precision for binary or multilabel classification.
    - calculate_f1_score: Computes the F1 score for binary or multilabel classification.
    - calculate_average_precision: Computes the average precision score (AP).
    - calculate_auroc: Computes the Area Under the Receiver Operating Characteristic curve (AUROC).
"""

from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    num_classes: int,
    threshold: float,
    averaging_method: Literal["micro", "macro", "weighted", "none"] | None = "macro",
) -> np.ndarray:
    """
    Calculate accuracy for the given predictions and labels.

    Args:
        predictions (np.ndarray): Model predictions as probabilities.
        labels (np.ndarray): True labels.
        task (Literal["binary", "multilabel"]): Type of classification task.
        num_classes (int): Number of classes (only for multilabel tasks).
        threshold (float): Threshold to binarize probabilities.
        averaging_method (Optional[Literal["micro", "macro", "weighted", "none"]], optional):
            Averaging method to compute accuracy for multilabel tasks. Defaults to "macro".

    Returns:
        np.ndarray: Accuracy metric(s) based on the task and averaging method.

    Raises:
        ValueError: If inputs are invalid or unsupported task/averaging method is specified.
    """
    # Input validation for predictions, labels, and threshold
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    # Handle binary and multilabel tasks separately
    if task == "binary":
        # Binary classification: Binarize predictions and compute accuracy
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        acc = accuracy_score(y_true, y_pred)
        acc = np.array([acc])

    elif task == "multilabel":
        # Multilabel classification: Handle based on the specified averaging method
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)

        if averaging_method == "micro":
            # Micro-averaging: Overall accuracy across all labels
            correct = (y_pred == y_true).sum()
            total = y_true.size
            acc = correct / total if total > 0 else np.nan
            acc = np.array([acc])

        elif averaging_method == "macro":
            # Macro-averaging: Compute accuracy per class and take the mean
            accuracies = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(num_classes)]
            acc = np.mean(accuracies)
            acc = np.array([acc])

        elif averaging_method == "weighted":
            # Weighted averaging: Weight class accuracies by class prevalence
            accuracies, weights = [], []
            for i in range(num_classes):
                accuracies.append(accuracy_score(y_true[:, i], y_pred[:, i]))
                weights.append(np.sum(y_true[:, i]))
            acc = np.average(accuracies, weights=weights) if sum(weights) > 0 else np.array([0.0])
            acc = np.array([acc])

        elif averaging_method in [None, "none"]:
            # No averaging: Return accuracy per class
            acc = np.array([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(num_classes)])

        else:
            # Unsupported averaging method
            raise ValueError(f"Invalid averaging method: {averaging_method}")
    else:
        # Unsupported task type
        raise ValueError(f"Unsupported task type: {task}")

    return acc


def calculate_recall(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    threshold: float,
    averaging_method: Literal["binary", "micro", "macro", "weighted", "samples", "none"] | None = None,
) -> np.ndarray:
    """
    Calculate recall for the given predictions and labels.

    Args:
        predictions (np.ndarray): Model predictions as probabilities.
        labels (np.ndarray): True labels.
        task (Literal["binary", "multilabel"]): Type of classification task.
        threshold (float): Threshold to binarize probabilities.
        averaging_method (Optional[Literal["binary", "micro", "macro", "weighted", "samples", "none"]], optional):
            Averaging method for multilabel recall. Defaults to None.

    Returns:
        np.ndarray: Recall metric(s).

    Raises:
        ValueError: If inputs are invalid or unsupported task type is specified.
    """
    # Validate inputs for size, threshold, and shape
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    # Adjust averaging method for scikit-learn if none is specified
    averaging = None if averaging_method == "none" else averaging_method

    # Compute recall based on task type
    if task == "binary":
        averaging = averaging or "binary"
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        recall = recall_score(y_true, y_pred, average=averaging, zero_division=0)

    elif task == "multilabel":
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        recall = recall_score(y_true, y_pred, average=averaging, zero_division=0)

    else:
        # Unsupported task type
        raise ValueError(f"Unsupported task type: {task}")

    # Ensure return type is consistent
    if isinstance(recall, np.ndarray):
        return recall
    return np.array([recall])


def calculate_precision(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    threshold: float,
    averaging_method: Literal["binary", "micro", "macro", "weighted", "samples", "none"] | None = None,
) -> np.ndarray:
    """
    Calculate precision for the given predictions and labels.

    Args:
        predictions (np.ndarray): Model predictions as probabilities.
        labels (np.ndarray): True labels.
        task (Literal["binary", "multilabel"]): Type of classification task.
        threshold (float): Threshold to binarize probabilities.
        averaging_method (Optional[Literal["binary", "micro", "macro", "weighted", "samples", "none"]], optional):
            Averaging method for multilabel precision. Defaults to None.

    Returns:
        np.ndarray: Precision metric(s).

    Raises:
        ValueError: If inputs are invalid or unsupported task type is specified.
    """
    # Validate inputs for size, threshold, and shape
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    # Adjust averaging method for scikit-learn if none is specified
    averaging = None if averaging_method == "none" else averaging_method

    # Compute precision based on task type
    if task == "binary":
        averaging = averaging or "binary"
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        precision = precision_score(y_true, y_pred, average=averaging, zero_division=0)

    elif task == "multilabel":
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        precision = precision_score(y_true, y_pred, average=averaging, zero_division=0)

    else:
        # Unsupported task type
        raise ValueError(f"Unsupported task type: {task}")

    # Ensure return type is consistent
    if isinstance(precision, np.ndarray):
        return precision
    return np.array([precision])


def calculate_f1_score(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    threshold: float,
    averaging_method: Literal["binary", "micro", "macro", "weighted", "samples", "none"] | None = None,
) -> np.ndarray:
    """
    Calculate the F1 score for the given predictions and labels.

    Args:
        predictions (np.ndarray): Model predictions as probabilities.
        labels (np.ndarray): True labels.
        task (Literal["binary", "multilabel"]): Type of classification task.
        threshold (float): Threshold to binarize probabilities.
        averaging_method (Optional[Literal["binary", "micro", "macro", "weighted", "samples", "none"]], optional):
            Averaging method for multilabel F1 score. Defaults to None.

    Returns:
        np.ndarray: F1 score metric(s).

    Raises:
        ValueError: If inputs are invalid or unsupported task type is specified.
    """
    # Validate inputs for size, threshold, and shape
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    # Adjust averaging method for scikit-learn if none is specified
    averaging = None if averaging_method == "none" else averaging_method

    # Compute F1 score based on task type
    if task == "binary":
        averaging = averaging or "binary"
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        f1 = f1_score(y_true, y_pred, average=averaging, zero_division=0)

    elif task == "multilabel":
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        f1 = f1_score(y_true, y_pred, average=averaging, zero_division=0)

    else:
        # Unsupported task type
        raise ValueError(f"Unsupported task type: {task}")

    # Ensure return type is consistent
    if isinstance(f1, np.ndarray):
        return f1
    return np.array([f1])


def calculate_average_precision(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    averaging_method: Literal["micro", "macro", "weighted", "samples", "none"] | None = None,
) -> np.ndarray:
    """
    Calculate the average precision (AP) for the given predictions and labels.

    Args:
        predictions (np.ndarray): Model predictions as probabilities.
        labels (np.ndarray): True labels.
        task (Literal["binary", "multilabel"]): Type of classification task.
        averaging_method (Optional[Literal["micro", "macro", "weighted", "samples", "none"]], optional):
            Averaging method for AP. Defaults to None.

    Returns:
        np.ndarray: Average precision metric(s).

    Raises:
        ValueError: If inputs are invalid or unsupported task type is specified.
    """
    # Validate inputs for size and shape
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    # Adjust averaging method for scikit-learn if none is specified
    averaging = None if averaging_method == "none" else averaging_method

    # Compute average precision based on task type
    if task in ("binary", "multilabel"):
        y_true = labels.astype(int)
        y_scores = predictions
        ap = average_precision_score(y_true, y_scores, average=averaging)

    else:
        # Unsupported task type
        raise ValueError(f"Unsupported task type for average precision: {task}")

    # Ensure return type is consistent
    if isinstance(ap, np.ndarray):
        return ap
    return np.array([ap])


def calculate_auroc(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    averaging_method: Literal["macro", "weighted", "samples", "none"] | None = "macro",
) -> np.ndarray:
    """
    Calculate the Area Under the Receiver Operating Characteristic curve (AUROC).

    Args:
        predictions (np.ndarray): Model predictions as probabilities.
        labels (np.ndarray): True labels.
        task (Literal["binary", "multilabel"]): Type of classification task.
        averaging_method (Optional[Literal["macro", "weighted", "samples", "none"]], optional):
            Averaging method for multilabel AUROC. Defaults to "macro".

    Returns:
        np.ndarray: AUROC metric(s).

    Raises:
        ValueError: If inputs are invalid or unsupported task type is specified.
    """
    # Validate inputs for size and shape
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    # Adjust averaging method for scikit-learn if none is specified
    averaging = None if averaging_method == "none" else averaging_method

    try:
        # Compute AUROC based on task type
        if task == "binary":
            y_true = labels.astype(int)
            y_scores = predictions
            auroc = roc_auc_score(y_true, y_scores)

        elif task == "multilabel":
            y_true = labels.astype(int)
            y_scores = predictions
            auroc = roc_auc_score(y_true, y_scores, average=averaging)

        else:
            # Unsupported task type
            raise ValueError(f"Unsupported task type: {task}")

    except ValueError as e:
        # Handle edge cases where AUROC cannot be computed
        if "Only one class present in y_true" in str(e) or "Number of classes in y_true" in str(e):
            auroc = np.nan
        else:
            raise

    # Ensure return type is consistent
    if isinstance(auroc, np.ndarray):
        return auroc
    return np.array([auroc])
