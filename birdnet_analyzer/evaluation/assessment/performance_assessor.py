"""Evaluate classification model performance and plot the results.

The ``PerformanceAssessor`` computes precision, recall, F1, AUROC and accuracy for
binary and multilabel tasks.
"""

from typing import ClassVar, Literal

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
            threshold (float): The threshold for binarizing probabilities into class
                labels.
            classes (Optional[Tuple[str, ...]]): Optional tuple of class names.
            task (Literal["binary", "multilabel"]): The classification task type.
            metrics_list (Tuple[str, ...]): A tuple of metrics to compute.

        Raises:
            ValueError: If any of the inputs are invalid.
        """
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        if not isinstance(threshold, float) or not 0 < threshold < 1:
            raise ValueError("threshold must be a float between 0 and 1 (exclusive).")

        if classes is not None:
            if not isinstance(classes, tuple):
                raise ValueError("classes must be a tuple of strings.")
            if len(classes) != num_classes:
                raise ValueError(
                    f"Length of classes ({len(classes)}) must match "
                    f"num_classes ({num_classes})."
                )
            if not all(isinstance(class_name, str) for class_name in classes):
                raise ValueError("All elements in classes must be strings.")

        if task not in {"binary", "multilabel"}:
            raise ValueError("task must be 'binary' or 'multilabel'.")

        valid_metrics = ["accuracy", "recall", "precision", "f1", "ap", "auroc"]
        if not metrics_list:
            raise ValueError("metrics_list cannot be empty.")
        if not all(metric in valid_metrics for metric in metrics_list):
            raise ValueError(
                f"Invalid metrics in {metrics_list}. Valid options are {valid_metrics}."
            )

        self.num_classes = num_classes
        self.threshold = threshold
        self.classes = classes
        self.task: Literal["binary", "multilabel"] = task
        self.metrics_list = metrics_list

        self.colors = ["#3A50B1", "#61A83E", "#D74C4C", "#A13FA1", "#D9A544", "#F3A6E0"]

    # Display labels for each supported metric id, in a stable order.
    _METRIC_DISPLAY_NAMES: ClassVar[dict[str, str]] = {
        "recall": "Recall",
        "precision": "Precision",
        "f1": "F1",
        "ap": "AP",
        "auroc": "AUROC",
        "accuracy": "Accuracy",
    }

    def _class_names(self) -> list[str]:
        """The class labels to use as columns, falling back to ``Class i``."""
        return list(self.classes or [f"Class {i}" for i in range(self.num_classes)])

    def _compute_metrics_dict(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        averaging_method,
        num_classes: int,
    ) -> dict[str, np.ndarray]:
        """Compute every requested metric, keyed by its display name.

        Args:
            predictions (np.ndarray): Prediction probabilities.
            labels (np.ndarray): Ground truth labels, same shape as predictions.
            averaging_method: Averaging passed to the metric functions. ``None`` yields
                one value per class; ``"macro"``/``"micro"``/``"weighted"`` aggregate.
            num_classes (int): Number of columns in the arrays (needed by accuracy).

        Returns:
            An ordered mapping of display name to a 1D array of metric values.
        """
        results: dict[str, np.ndarray] = {}

        for metric_name in self.metrics_list:
            if metric_name == "recall":
                value = metrics.calculate_recall(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
            elif metric_name == "precision":
                value = metrics.calculate_precision(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
            elif metric_name == "f1":
                value = metrics.calculate_f1_score(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
            elif metric_name == "ap":
                value = metrics.calculate_average_precision(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    averaging_method=averaging_method,
                )
            elif metric_name == "auroc":
                value = metrics.calculate_auroc(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    averaging_method=averaging_method,
                )
            elif metric_name == "accuracy":
                value = metrics.calculate_accuracy(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    num_classes=num_classes,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
            else:
                continue

            results[self._METRIC_DISPLAY_NAMES[metric_name]] = np.atleast_1d(value)

        return results

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
        drop_empty: bool = True,
        include_support: bool = False,
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
                per-class metrics into the single ``Overall`` column. Ignored when
                ``per_class_metrics`` is True.
            drop_empty (bool): When aggregating, exclude classes with no positive labels
                (their metrics are undefined) instead of counting them as zero. Applies
                only to the aggregate column.
            include_support (bool): When ``per_class_metrics`` is True, append a
                ``Support`` row with the number of positive samples per class.

        Returns:
            pd.DataFrame: A DataFrame containing the computed metrics.

        Raises:
            TypeError: If predictions or labels are not NumPy arrays.
            ValueError: If predictions and labels have mismatched dimensions or invalid
                shapes.
        """
        if not isinstance(predictions, np.ndarray):
            raise TypeError("predictions must be a NumPy array.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be a NumPy array.")

        if predictions.shape != labels.shape:
            raise ValueError("predictions and labels must have the same shape.")
        if predictions.ndim != 2:
            raise ValueError("predictions and labels must be 2-dimensional arrays.")
        if predictions.shape[1] != self.num_classes:
            raise ValueError(
                f"The number of columns in predictions ({predictions.shape[1]}) "
                + f"must match num_classes ({self.num_classes})."
            )

        if per_class_metrics:
            # One value per class: no averaging across classes.
            results = self._compute_metrics_dict(
                predictions, labels, averaging_method=None, num_classes=self.num_classes
            )
            metrics_df = pd.DataFrame.from_dict(
                results, orient="index", columns=pd.Index(self._class_names())
            )
            if include_support:
                metrics_df.loc["Support"] = self.class_support(labels)

            return metrics_df

        # Aggregate into a single column. Undefined (empty) classes are dropped so they
        # cannot silently pull the average down, unless the caller keeps them.
        keep = (
            self.class_support(labels) > 0
            if drop_empty
            else np.ones(self.num_classes, dtype=bool)
        )
        kept = np.flatnonzero(keep)

        if kept.size == 0:
            # Nothing scoreable -- report NaN rather than a misleading zero.
            results = {
                self._METRIC_DISPLAY_NAMES[m]: np.array([np.nan])
                for m in self.metrics_list
            }
            return pd.DataFrame.from_dict(
                results, orient="index", columns=pd.Index(["Overall"])
            )

        predictions_kept = predictions[:, kept]
        labels_kept = labels[:, kept]
        # For a binary task the metric functions want no averaging (they use the
        # positive-class value). For multilabel we always aggregate across the kept
        # classes -- even a single kept column, where ``average=None`` would wrongly
        # return one value per binary outcome instead of a single score.
        averaging_method = None if self.task == "binary" else averaging
        results = self._compute_metrics_dict(
            predictions_kept,
            labels_kept,
            averaging_method=averaging_method,
            num_classes=kept.size,
        )

        return pd.DataFrame.from_dict(
            results, orient="index", columns=pd.Index(["Overall"])
        )

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
        # Save the original threshold so the sweep never leaks out of this method,
        # even if a metric computation or the plotting call raises.
        original_threshold = self.threshold

        thresholds = np.arange(0.05, 1.0, 0.05)

        # Exclude metrics that are not threshold-dependent
        metrics_to_plot = [m for m in self.metrics_list if m not in ["auroc", "ap"]]

        try:
            if per_class_metrics:
                class_names = self._class_names()

                metric_values_dict_per_class = {
                    class_name: {metric: [] for metric in metrics_to_plot}
                    for class_name in class_names
                }

                for thresh in thresholds:
                    self.threshold = thresh
                    metrics_df = self.calculate_metrics(
                        predictions, labels, per_class_metrics=True
                    )
                    for metric_name in metrics_to_plot:
                        metric_label = self._METRIC_DISPLAY_NAMES[metric_name]
                        for class_name in class_names:
                            value = metrics_df.loc[metric_label, class_name]
                            metric_values_dict_per_class[class_name][
                                metric_name
                            ].append(value)

                metric_values_dict_per_class = {
                    class_name: {
                        metric: np.array(values)
                        for metric, values in metrics_dict.items()
                    }
                    for class_name, metrics_dict in (
                        metric_values_dict_per_class.items()
                    )
                }

                fig = plotting.plot_metrics_across_thresholds_per_class(
                    thresholds,
                    metric_values_dict_per_class,
                    metrics_to_plot,
                    class_names,
                    self.colors,
                )
            else:
                metric_values_dict = {
                    metric_name: [] for metric_name in metrics_to_plot
                }

                for thresh in thresholds:
                    self.threshold = thresh
                    metrics_df = self.calculate_metrics(
                        predictions, labels, per_class_metrics=False
                    )
                    for metric_name in metrics_to_plot:
                        metric_label = self._METRIC_DISPLAY_NAMES[metric_name]
                        value = metrics_df.loc[metric_label, "Overall"]
                        metric_values_dict[metric_name].append(value)

                metric_values_dict = {
                    metric_name: np.array(values)
                    for metric_name, values in metric_values_dict.items()
                }

                fig = plotting.plot_metrics_across_thresholds(
                    thresholds,
                    metric_values_dict,
                    metrics_to_plot,
                    self.colors,
                )
        finally:
            self.threshold = original_threshold

        return fig

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
            TypeError: If predictions or labels are not NumPy arrays.
            ValueError: If predictions and labels have mismatched shapes or invalid
                dimensions.

        Returns:
            None
        """
        if not isinstance(predictions, np.ndarray):
            raise TypeError("predictions must be a NumPy array.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be a NumPy array.")
        if predictions.shape != labels.shape:
            raise ValueError("predictions and labels must have the same shape.")
        if predictions.ndim != 2:
            raise ValueError("predictions and labels must be 2-dimensional arrays.")
        if predictions.shape[1] != self.num_classes:
            raise ValueError(
                f"The number of columns in predictions ({predictions.shape[1]}) "
                + f"must match num_classes ({self.num_classes})."
            )

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
