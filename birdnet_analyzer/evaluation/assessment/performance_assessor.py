"""
PerformanceAssessor Module

This module defines the `PerformanceAssessor` class to evaluate classification model performance.
It includes methods to compute metrics like precision, recall, F1 score, AUROC, and accuracy,
as well as utilities for generating related plots.
"""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from birdnet_analyzer.evaluation.assessment import metrics, plotting


class PerformanceAssessor:
    """
    A class to assess the performance of classification models by computing metrics
    and generating visualizations for binary and multilabel classification tasks.
    """

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        classes: tuple[str, ...] | None = None,
        task: Literal["binary", "multilabel"] = "multilabel",
        metrics_list: tuple[str, ...] = (
            "recall",
            "precision",
            "f1",
            "ap",
            "auroc",
            "accuracy",
        ),
    ) -> None:
        """
        Initialize the PerformanceAssessor.

        Args:
            num_classes (int): The number of classes in the classification problem.
            threshold (float): The threshold for binarizing probabilities into class labels.
            classes (Optional[Tuple[str, ...]]): Optional tuple of class names.
            task (Literal["binary", "multilabel"]): The classification task type.
            metrics_list (Tuple[str, ...]): A tuple of metrics to compute.

        Raises:
            ValueError: If any of the inputs are invalid.
        """
        # Validate the number of classes
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        # Validate the threshold value
        if not isinstance(threshold, float) or not 0 < threshold < 1:
            raise ValueError("threshold must be a float between 0 and 1 (exclusive).")

        # Validate class names
        if classes is not None:
            if not isinstance(classes, tuple):
                raise ValueError("classes must be a tuple of strings.")
            if len(classes) != num_classes:
                raise ValueError(f"Length of classes ({len(classes)}) must match num_classes ({num_classes}).")
            if not all(isinstance(class_name, str) for class_name in classes):
                raise ValueError("All elements in classes must be strings.")

        # Validate the task type
        if task not in {"binary", "multilabel"}:
            raise ValueError("task must be 'binary' or 'multilabel'.")

        # Validate the metrics list
        valid_metrics = ["accuracy", "recall", "precision", "f1", "ap", "auroc"]
        if not metrics_list:
            raise ValueError("metrics_list cannot be empty.")
        if not all(metric in valid_metrics for metric in metrics_list):
            raise ValueError(f"Invalid metrics in {metrics_list}. Valid options are {valid_metrics}.")

        # Assign instance variables
        self.num_classes = num_classes
        self.threshold = threshold
        self.classes = classes
        self.task = task
        self.metrics_list = metrics_list

        # Set default colors for plotting
        self.colors = ["#3A50B1", "#61A83E", "#D74C4C", "#A13FA1", "#D9A544", "#F3A6E0"]

    def calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        per_class_metrics: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate multiple performance metrics for the given predictions and labels.

        Args:
            predictions (np.ndarray): Model predictions as a 2D NumPy array (probabilities or logits).
            labels (np.ndarray): Ground truth labels as a 2D NumPy array.
            per_class_metrics (bool): If True, compute metrics for each class individually.

        Returns:
            pd.DataFrame: A DataFrame containing the computed metrics.

        Raises:
            TypeError: If predictions or labels are not NumPy arrays.
            ValueError: If predictions and labels have mismatched dimensions or invalid shapes.
        """
        # Validate that predictions and labels are NumPy arrays
        if not isinstance(predictions, np.ndarray):
            raise TypeError("predictions must be a NumPy array.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be a NumPy array.")

        # Ensure predictions and labels have the same shape
        if predictions.shape != labels.shape:
            raise ValueError("predictions and labels must have the same shape.")
        if predictions.ndim != 2:
            raise ValueError("predictions and labels must be 2-dimensional arrays.")
        if predictions.shape[1] != self.num_classes:
            raise ValueError(f"The number of columns in predictions ({predictions.shape[1]}) " + f"must match num_classes ({self.num_classes}).")

        # Determine the averaging method for metrics
        if per_class_metrics and self.num_classes == 1:
            averaging_method = "macro"
        else:
            averaging_method = None if per_class_metrics else "macro"

        # Dictionary to store the results of each metric
        metrics_results = {}

        # Compute each metric in the metrics list
        for metric_name in self.metrics_list:
            if metric_name == "recall":
                result = metrics.calculate_recall(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
                metrics_results["Recall"] = np.atleast_1d(result)
            elif metric_name == "precision":
                result = metrics.calculate_precision(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
                metrics_results["Precision"] = np.atleast_1d(result)
            elif metric_name == "f1":
                result = metrics.calculate_f1_score(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
                metrics_results["F1"] = np.atleast_1d(result)
            elif metric_name == "ap":
                result = metrics.calculate_average_precision(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    averaging_method=averaging_method,
                )
                metrics_results["AP"] = np.atleast_1d(result)
            elif metric_name == "auroc":
                result = metrics.calculate_auroc(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    averaging_method=averaging_method,
                )
                metrics_results["AUROC"] = np.atleast_1d(result)
            elif metric_name == "accuracy":
                result = metrics.calculate_accuracy(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    num_classes=self.num_classes,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
                metrics_results["Accuracy"] = np.atleast_1d(result)

        # Define column names for the DataFrame
        columns = (self.classes if self.classes else [f"Class {i}" for i in range(self.num_classes)]) if per_class_metrics else ["Overall"]

        # Create a DataFrame to organize metric results
        metrics_data = {key: np.atleast_1d(value) for key, value in metrics_results.items()}
        return pd.DataFrame.from_dict(metrics_data, orient="index", columns=columns)

    def plot_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        per_class_metrics: bool = False,
    ):
        """
        Plot performance metrics for the given predictions and labels.

        Args:
            predictions (np.ndarray): Model output predictions as a 2D NumPy array (probabilities or logits).
            labels (np.ndarray): Ground truth labels as a 2D NumPy array.
            per_class_metrics (bool): If True, plots metrics for each class individually.

        Raises:
            ValueError: If the metrics cannot be calculated or plotting fails.

        Returns:
            None
        """
        # Calculate metrics using the provided predictions and labels
        metrics_df = self.calculate_metrics(predictions, labels, per_class_metrics)

        # Choose the plotting method based on whether per-class metrics are required
        return plotting.plot_metrics_per_class(metrics_df, self.colors) if per_class_metrics else plotting.plot_overall_metrics(metrics_df, self.colors)

    def plot_metrics_all_thresholds(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        per_class_metrics: bool = False,
    ):
        """
        Plot performance metrics across thresholds for the given predictions and labels.

        Args:
            predictions (np.ndarray): Model output predictions as a 2D NumPy array (probabilities or logits).
            labels (np.ndarray): Ground truth labels as a 2D NumPy array.
            per_class_metrics (bool): If True, plots metrics for each class individually.

        Raises:
            ValueError: If metrics calculation or plotting fails.

        Returns:
            None
        """
        # Save the original threshold value to restore it later
        original_threshold = self.threshold

        # Define a range of thresholds for analysis
        thresholds = np.arange(0.05, 1.0, 0.05)

        # Exclude metrics that are not threshold-dependent
        metrics_to_plot = [m for m in self.metrics_list if m not in ["auroc", "ap"]]

        if per_class_metrics:
            # Define class names for plotting
            class_names = list(self.classes) if self.classes else [f"Class {i}" for i in range(self.num_classes)]

            # Initialize a dictionary to store metric values per class
            metric_values_dict_per_class = {class_name: {metric: [] for metric in metrics_to_plot} for class_name in class_names}

            # Compute metrics for each threshold
            for thresh in thresholds:
                self.threshold = thresh
                metrics_df = self.calculate_metrics(predictions, labels, per_class_metrics=True)
                for metric_name in metrics_to_plot:
                    metric_label = metric_name.capitalize() if metric_name != "f1" else "F1"
                    for class_name in class_names:
                        value = metrics_df.loc[metric_label, class_name]
                        metric_values_dict_per_class[class_name][metric_name].append(value)

            # Restore the original threshold
            self.threshold = original_threshold

            # Plot metrics across thresholds per class
            fig = plotting.plot_metrics_across_thresholds_per_class(
                thresholds,
                metric_values_dict_per_class,
                metrics_to_plot,
                class_names,
                self.colors,
            )
        else:
            # Initialize a dictionary to store overall metric values
            metric_values_dict = {metric_name: [] for metric_name in metrics_to_plot}

            # Compute metrics for each threshold
            for thresh in thresholds:
                self.threshold = thresh
                metrics_df = self.calculate_metrics(predictions, labels, per_class_metrics=False)
                for metric_name in metrics_to_plot:
                    metric_label = metric_name.capitalize() if metric_name != "f1" else "F1"
                    value = metrics_df.loc[metric_label, "Overall"]
                    metric_values_dict[metric_name].append(value)

            # Restore the original threshold
            self.threshold = original_threshold

            # Plot metrics across thresholds
            fig = plotting.plot_metrics_across_thresholds(
                thresholds,
                metric_values_dict,
                metrics_to_plot,
                self.colors,
            )

        return fig

    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Plot confusion matrices for each class using scikit-learn's ConfusionMatrixDisplay.

        Args:
            predictions (np.ndarray): Model output predictions as a 2D NumPy array (probabilities or logits).
            labels (np.ndarray): Ground truth labels as a 2D NumPy array.

        Raises:
            TypeError: If predictions or labels are not NumPy arrays.
            ValueError: If predictions and labels have mismatched shapes or invalid dimensions.

        Returns:
            None
        """
        # Validate that predictions and labels are NumPy arrays and match in shape
        if not isinstance(predictions, np.ndarray):
            raise TypeError("predictions must be a NumPy array.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be a NumPy array.")
        if predictions.shape != labels.shape:
            raise ValueError("predictions and labels must have the same shape.")
        if predictions.ndim != 2:
            raise ValueError("predictions and labels must be 2-dimensional arrays.")
        if predictions.shape[1] != self.num_classes:
            raise ValueError(f"The number of columns in predictions ({predictions.shape[1]}) " + f"must match num_classes ({self.num_classes}).")

        if self.task == "binary":
            # Binarize predictions using the threshold
            y_pred = (predictions >= self.threshold).astype(int).flatten()
            y_true = labels.astype(int).flatten()

            # Compute and normalize the confusion matrix
            conf_mat = confusion_matrix(y_true, y_pred, normalize="true")
            conf_mat = np.round(conf_mat, 2)

            return plotting.plot_confusion_matrices(conf_mat, self.task, self.classes)

        if self.task == "multilabel":
            # Binarize predictions for multilabel classification
            y_pred = (predictions >= self.threshold).astype(int)
            y_true = labels.astype(int)

            # Compute confusion matrices for each class
            conf_mats = []
            class_names = self.classes if self.classes else [f"Class {i}" for i in range(self.num_classes)]

            for i in range(self.num_classes):
                conf_mat = confusion_matrix(y_true[:, i], y_pred[:, i], normalize="true")
                conf_mat = np.round(conf_mat, 2)
                conf_mats.append(conf_mat)

            return plotting.plot_confusion_matrices(np.array(conf_mats), self.task, class_names)

        raise ValueError(f"Unsupported task type: {self.task}")
