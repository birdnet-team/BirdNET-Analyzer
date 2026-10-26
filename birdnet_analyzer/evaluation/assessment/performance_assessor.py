"""Evaluate classification model performance and plot the results.

The ``PerformanceAssessor`` computes precision, recall, F1, AUROC and accuracy for
binary and multilabel tasks.
"""

from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from birdnet_analyzer.evaluation.assessment import metrics, plotting


class Metric(NamedTuple):
    """A metric the assessor computes, and the row it is reported in."""

    display_name: str
    function: Callable[..., np.ndarray]
    needs_threshold: bool


def _accuracy(predictions, labels, task, threshold, averaging_method):
    return metrics.calculate_accuracy(
        predictions=predictions,
        labels=labels,
        task=task,
        num_classes=predictions.shape[1],
        threshold=threshold,
        averaging_method=averaging_method,
    )


METRICS: dict[str, Metric] = {
    "accuracy": Metric("Accuracy", _accuracy, True),
    "recall": Metric("Recall", metrics.calculate_recall, True),
    "precision": Metric("Precision", metrics.calculate_precision, True),
    "f1": Metric("F1", metrics.calculate_f1_score, True),
    "ap": Metric("AP", metrics.calculate_average_precision, False),
    "auroc": Metric("AUROC", metrics.calculate_auroc, False),
}


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
            threshold (float): The threshold for binarizing probabilities into class
                labels.
            classes (Optional[Tuple[str, ...]]): Optional tuple of class names.
            task (Literal["binary", "multilabel"]): The classification task type.
            metrics_list (Tuple[str, ...]): A tuple of metrics to compute.

        Raises:
            ValueError: If any of the inputs are invalid.
        """
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1 (exclusive).")

        if classes is not None and len(classes) != num_classes:
            raise ValueError(
                f"Length of classes ({len(classes)}) must match "
                f"num_classes ({num_classes})."
            )

        if task not in {"binary", "multilabel"}:
            raise ValueError("task must be 'binary' or 'multilabel'.")

        if not metrics_list:
            raise ValueError("metrics_list cannot be empty.")
        if not all(metric in METRICS for metric in metrics_list):
            raise ValueError(
                f"Invalid metrics in {metrics_list}. Valid options are {list(METRICS)}."
            )

        self.num_classes = num_classes
        self.threshold = threshold
        self.classes = classes
        self.task: Literal["binary", "multilabel"] = task
        self.metrics_list = metrics_list

        self.colors = ["#3A50B1", "#61A83E", "#D74C4C", "#A13FA1", "#D9A544", "#F3A6E0"]

    def _class_names(self) -> list[str]:
        """The class labels to use as columns, falling back to ``Class i``."""
        return list(self.classes or [f"Class {i}" for i in range(self.num_classes)])

    def _check_tensors(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        if predictions.shape != labels.shape:
            raise ValueError("predictions and labels must have the same shape.")
        if predictions.ndim != 2:
            raise ValueError("predictions and labels must be 2-dimensional arrays.")
        if predictions.shape[1] != self.num_classes:
            raise ValueError(
                f"The number of columns in predictions ({predictions.shape[1]}) "
                + f"must match num_classes ({self.num_classes})."
            )

    def _compute_metrics_dict(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        averaging_method,
        metric_ids: list[str] | tuple[str, ...],
        threshold: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute the given metrics, keyed by their display names.

        Args:
            predictions (np.ndarray): Prediction probabilities.
            labels (np.ndarray): Ground truth labels, same shape as predictions.
            averaging_method: Averaging passed to the metric functions. ``None`` yields
                one value per class; ``"macro"``/``"micro"``/``"weighted"`` aggregate.
            metric_ids: The ids of the metrics to compute, in output order.
            threshold (Optional[float]): The threshold for binarizing probabilities,
                ``self.threshold`` if not given.

        Returns:
            An ordered mapping of display name to a 1D array of metric values.
        """
        threshold = self.threshold if threshold is None else threshold
        results: dict[str, np.ndarray] = {}

        for metric_id in metric_ids:
            metric = METRICS[metric_id]
            extra = {"threshold": threshold} if metric.needs_threshold else {}
            value = metric.function(
                predictions=predictions,
                labels=labels,
                task=self.task,
                averaging_method=averaging_method,
                **extra,
            )
            results[metric.display_name] = np.atleast_1d(value)

        return results

    def _metric_values(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metric_ids: list[str] | tuple[str, ...],
        per_class_metrics: bool,
        averaging: Literal["macro", "micro", "weighted"],
        threshold: float | None,
    ) -> dict[str, np.ndarray]:
        """The metrics by display name: one value per class, or the aggregate."""
        if per_class_metrics:
            # One value per class: no averaging across classes.
            return self._compute_metrics_dict(
                predictions, labels, None, metric_ids, threshold
            )

        # Aggregate into a single column. Undefined (empty) classes are dropped so they
        # cannot silently pull the average down.
        kept = np.flatnonzero(self.class_support(labels) > 0)

        if kept.size == 0:
            # Nothing scoreable -- report NaN rather than a misleading zero.
            return {METRICS[m].display_name: np.array([np.nan]) for m in metric_ids}

        # For a binary task the metric functions want no averaging (they use the
        # positive-class value). For multilabel we always aggregate across the kept
        # classes -- even a single kept column, where ``average=None`` would wrongly
        # return one value per binary outcome instead of a single score.
        averaging_method = None if self.task == "binary" else averaging

        return self._compute_metrics_dict(
            predictions[:, kept],
            labels[:, kept],
            averaging_method,
            metric_ids,
            threshold,
        )

    def class_support(self, labels: np.ndarray) -> np.ndarray:
        """Number of positive samples per class (the support of each class)."""
        return labels.astype(bool).sum(axis=0).astype(int)

    def empty_classes(self, labels: np.ndarray) -> tuple[str, ...]:
        """Names of the classes that have no positive annotation in ``labels``.

        These classes are excluded from the aggregate score, because precision,
        recall, F1 and AUROC are undefined without a single positive example.
        """
        support = self.class_support(labels)
        return tuple(
            name
            for name, count in zip(self._class_names(), support, strict=True)
            if count == 0
        )

    def calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        per_class_metrics: bool = False,
        averaging: Literal["macro", "micro", "weighted"] = "macro",
        include_support: bool = False,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Calculate multiple performance metrics for the given predictions and labels.

        Args:
            predictions (np.ndarray): Model predictions as a 2D NumPy array
                (probabilities or logits).
            labels (np.ndarray): Ground truth labels as a 2D NumPy array.
            per_class_metrics (bool): If True, compute metrics for each class
                individually. If False, return a single aggregate column.
            averaging (Literal["macro", "micro", "weighted"]): How to aggregate the
                per-class metrics into the single ``Overall`` column; classes without a
                positive label are left out of it. Ignored when ``per_class_metrics``
                is True.
            include_support (bool): When ``per_class_metrics`` is True, append a
                ``Support`` row with the number of positive samples per class.
            threshold (Optional[float]): The threshold for binarizing probabilities,
                instead of the assessor's own one.

        Returns:
            pd.DataFrame: A DataFrame containing the computed metrics.

        Raises:
            ValueError: If predictions and labels have mismatched dimensions or invalid
                shapes.
        """
        self._check_tensors(predictions, labels)

        results = self._metric_values(
            predictions,
            labels,
            self.metrics_list,
            per_class_metrics,
            averaging,
            threshold,
        )
        columns = self._class_names() if per_class_metrics else ["Overall"]
        metrics_df = pd.DataFrame.from_dict(
            results, orient="index", columns=pd.Index(columns)
        )
        if per_class_metrics and include_support:
            metrics_df.loc["Support"] = self.class_support(labels)

        return metrics_df

    def plot_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        per_class_metrics: bool = False,
    ):
        """
        Plot performance metrics for the given predictions and labels.

        Args:
            predictions (np.ndarray): Model output predictions as a 2D NumPy array
                (probabilities or logits).
            labels (np.ndarray): Ground truth labels as a 2D NumPy array.
            per_class_metrics (bool): If True, plots metrics for each class
                individually.

        Raises:
            ValueError: If the metrics cannot be calculated or plotting fails.

        Returns:
            None
        """
        metrics_df = self.calculate_metrics(predictions, labels, per_class_metrics)

        return (
            plotting.plot_metrics_per_class(metrics_df, self.colors)
            if per_class_metrics
            else plotting.plot_overall_metrics(metrics_df, self.colors)
        )

    def plot_metrics_all_thresholds(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        per_class_metrics: bool = False,
    ):
        """
        Plot performance metrics across thresholds for the given predictions and labels.

        Args:
            predictions (np.ndarray): Model output predictions as a 2D NumPy array
                (probabilities or logits).
            labels (np.ndarray): Ground truth labels as a 2D NumPy array.
            per_class_metrics (bool): If True, plots metrics for each class
                individually.

        Raises:
            ValueError: If metrics calculation or plotting fails.

        Returns:
            None
        """
        self._check_tensors(predictions, labels)

        thresholds = np.arange(0.05, 1.0, 0.05)
        metrics_to_plot = [m for m in self.metrics_list if METRICS[m].needs_threshold]
        columns = self.num_classes if per_class_metrics else 1

        values = np.array(
            [
                list(
                    self._metric_values(
                        predictions,
                        labels,
                        metrics_to_plot,
                        per_class_metrics,
                        "macro",
                        threshold,
                    ).values()
                )
                for threshold in thresholds
            ],
            dtype=float,
        ).reshape(len(thresholds), len(metrics_to_plot), columns)

        if per_class_metrics:
            class_names = self._class_names()

            return plotting.plot_metrics_across_thresholds_per_class(
                thresholds,
                {
                    class_name: {
                        metric: values[:, m, c]
                        for m, metric in enumerate(metrics_to_plot)
                    }
                    for c, class_name in enumerate(class_names)
                },
                metrics_to_plot,
                class_names,
                self.colors,
            )

        return plotting.plot_metrics_across_thresholds(
            thresholds,
            {metric: values[:, m, 0] for m, metric in enumerate(metrics_to_plot)},
            metrics_to_plot,
            self.colors,
        )

    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Plot confusion matrices for each class using scikit-learn's
            ConfusionMatrixDisplay.

        Args:
            predictions (np.ndarray): Model output predictions as a 2D NumPy array
                (probabilities or logits).
            labels (np.ndarray): Ground truth labels as a 2D NumPy array.

        Raises:
            ValueError: If predictions and labels have mismatched shapes or invalid
                dimensions.

        Returns:
            None
        """
        self._check_tensors(predictions, labels)

        if self.task == "binary":
            y_pred = (predictions >= self.threshold).astype(int).flatten()
            y_true = labels.astype(int).flatten()

            conf_mat = confusion_matrix(y_true, y_pred, normalize="true")
            conf_mat = np.round(conf_mat, 2)

            return plotting.plot_confusion_matrices(conf_mat, self.task, self.classes)  # ty:ignore[invalid-argument-type]

        if self.task == "multilabel":
            y_pred = (predictions >= self.threshold).astype(int)
            y_true = labels.astype(int)

            conf_mats = []
            class_names = self.classes or [
                f"Class {i}" for i in range(self.num_classes)
            ]

            for i in range(self.num_classes):
                conf_mat = confusion_matrix(
                    y_true[:, i], y_pred[:, i], normalize="true"
                )
                conf_mat = np.round(conf_mat, 2)
                conf_mats.append(conf_mat)

            return plotting.plot_confusion_matrices(
                np.array(conf_mats), self.task, class_names
            )

        raise ValueError(f"Unsupported task type: {self.task}")
