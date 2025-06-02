"""
Module containing functions to plot performance metrics.

This script provides a variety of functions to visualize performance metrics in different formats,
including bar charts, line plots, and heatmaps. These visualizations help analyze metrics such as
overall performance, per-class performance, and performance across thresholds.

Functions:
    - plot_overall_metrics: Plots a bar chart for overall performance metrics.
    - plot_metrics_per_class: Plots metric values per class with unique lines and colors.
    - plot_metrics_across_thresholds: Plots metrics across different thresholds.
    - plot_metrics_across_thresholds_per_class: Plots metrics across thresholds for each class.
    - plot_confusion_matrices: Visualizes confusion matrices for binary, multiclass, or multilabel tasks.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

MATPLOTLIB_BINARY_CONFUSION_MATRIX_FIGURE_NUM = "performance-tab-binary-confusion-matrix-plot"
MATPLOTLIB_MULTICLASS_CONFUSION_MATRIX_FIGURE_NUM = "performance-tab-multiclass-confusion-matrix-plot"
MATPLOTLIB_OVERALL_METRICS_FIGURE_NUM = "performance-tab-overall-metrics-plot"
MATPLOTLIB_PER_CLASS_METRICS_FIGURE_NUM = "performance-tab-per-class-metrics-plot"
MATPLOTLIB_ACROSS_METRICS_THRESHOLDS_FIGURE_NUM = "performance-tab-metrics-across-thresholds-plot"
MATPLOTLIB_ACROSS_METRICS_THRESHOLDS_PER_CLASS_FIGURE_NUM = "performance-tab-metrics-across-thresholds-per-class-plot"


def plot_overall_metrics(metrics_df: pd.DataFrame, colors: list[str]):
    """
    Plots a bar chart for overall performance metrics.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metric names as index and an 'Overall' column.
        colors (List[str]): List of colors for the bars.

    Raises:
        TypeError: If `metrics_df` is not a DataFrame or `colors` is not a list.
        KeyError: If 'Overall' column is missing in `metrics_df`.
        ValueError: If `metrics_df` is empty.

    Returns:
        plt.Figure
    """
    # Validate input types and content
    if not isinstance(metrics_df, pd.DataFrame):
        raise TypeError("metrics_df must be a pandas DataFrame.")
    if "Overall" not in metrics_df.columns:
        raise KeyError("metrics_df must contain an 'Overall' column.")
    if metrics_df.empty:
        raise ValueError("metrics_df is empty.")
    if not isinstance(colors, list):
        raise TypeError("colors must be a list.")
    if len(colors) == 0:
        # Default to matplotlib's color cycle if colors are not provided
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Extract metric names and values
    metrics = metrics_df.index  # Metric names
    values = metrics_df["Overall"].to_numpy()  # Metric values

    # Plot bar chart
    fig = plt.figure(MATPLOTLIB_OVERALL_METRICS_FIGURE_NUM, figsize=(10, 6))
    fig.clear()
    fig.tight_layout(pad=0)
    fig.set_dpi(300)

    plt.bar(metrics, values, color=colors[: len(metrics)])

    # Add titles, labels, and format
    plt.title("Overall Metric Scores", fontsize=16)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    return fig


def plot_metrics_per_class(metrics_df: pd.DataFrame, colors: list[str]):
    """
    Plots metric values per class, with each metric represented by a distinct color and line.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics as index and class names as columns.
        colors (List[str]): List of colors for the lines.

    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If `metrics_df` is empty.

    Returns:
        plt.Figure
    """
    # Validate inputs
    if not isinstance(metrics_df, pd.DataFrame):
        raise TypeError("metrics_df must be a pandas DataFrame.")
    if metrics_df.empty:
        raise ValueError("metrics_df is empty.")
    if not isinstance(colors, list):
        raise TypeError("colors must be a list.")
    if len(colors) == 0:
        # Default to matplotlib's color cycle if colors are not provided
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Line styles for distinction
    line_styles = ["-", "--", "-.", ":", (0, (5, 10)), (0, (5, 5)), (0, (3, 5, 1, 5))]
    fig = plt.figure(MATPLOTLIB_OVERALL_METRICS_FIGURE_NUM, figsize=(10, 6))
    fig.clear()
    fig.tight_layout(pad=0)
    fig.set_dpi(300)

    # Loop over each metric and plot it
    for i, metric_name in enumerate(metrics_df.index):
        values = metrics_df.loc[metric_name]  # Metric values for each class
        classes = metrics_df.columns  # Class labels
        plt.plot(
            classes,
            values,
            label=metric_name,
            marker="o",
            markersize=8,
            linewidth=2,
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i % len(colors)],
        )

    # Add titles, labels, legend, and format
    plt.title("Metric Scores per Class", fontsize=16)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True)

    return fig


def plot_metrics_across_thresholds(
    thresholds: np.ndarray,
    metric_values_dict: dict[str, np.ndarray],
    metrics_to_plot: list[str],
    colors: list[str],
):
    """
    Plots metrics across different thresholds.

    Args:
        thresholds (np.ndarray): Array of threshold values.
        metric_values_dict (Dict[str, np.ndarray]): Dictionary mapping metric names to their values.
        metrics_to_plot (List[str]): List of metric names to plot.
        colors (List[str]): List of colors for the lines.

    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If thresholds or metric values have mismatched lengths.

    Returns:
        plt.Figure
    """
    # Validate inputs
    if not isinstance(thresholds, np.ndarray):
        raise TypeError("thresholds must be a numpy ndarray.")
    if thresholds.size == 0:
        raise ValueError("thresholds array is empty.")
    if not isinstance(metric_values_dict, dict):
        raise TypeError("metric_values_dict must be a dictionary.")
    if not isinstance(metrics_to_plot, list):
        raise TypeError("metrics_to_plot must be a list.")
    if not isinstance(colors, list):
        raise TypeError("colors must be a list.")
    if len(colors) == 0:
        # Default to matplotlib's color cycle if colors are not provided
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Line styles for distinction
    line_styles = ["-", "--", "-.", ":", (0, (5, 10)), (0, (5, 5)), (0, (3, 5, 1, 5))]
    fig = plt.figure(MATPLOTLIB_ACROSS_METRICS_THRESHOLDS_FIGURE_NUM, figsize=(10, 6))
    fig.clear()
    fig.tight_layout(pad=0)
    fig.set_dpi(300)

    # Plot each metric against thresholds
    for i, metric_name in enumerate(metrics_to_plot):
        if metric_name not in metric_values_dict:
            raise KeyError(f"Metric '{metric_name}' not found in metric_values_dict.")
        metric_values = metric_values_dict[metric_name]
        if len(metric_values) != len(thresholds):
            raise ValueError(f"Length of metric '{metric_name}' values does not match length of thresholds.")
        plt.plot(
            thresholds,
            metric_values,
            label=metric_name.capitalize(),
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2,
            color=colors[i % len(colors)],
        )

    # Add titles, labels, legend, and format
    plt.title("Metrics across Different Thresholds", fontsize=16)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True)

    return fig


def plot_metrics_across_thresholds_per_class(
    thresholds: np.ndarray,
    metric_values_dict_per_class: dict[str, dict[str, np.ndarray]],
    metrics_to_plot: list[str],
    class_names: list[str],
    colors: list[str],
):
    """
    Plots metrics across different thresholds per class.

    Args:
        thresholds (np.ndarray): Array of threshold values.
        metric_values_dict_per_class (Dict[str, Dict[str, np.ndarray]]): Dictionary mapping class names
            to metric dictionaries, each containing metric names and their values across thresholds.
        metrics_to_plot (List[str]): List of metric names to plot.
        class_names (List[str]): List of class names.
        colors (List[str]): List of colors for the lines.

    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If inputs have mismatched lengths or are empty.

    Returns:
        plt.Figure
    """
    # Validate inputs
    if not isinstance(thresholds, np.ndarray):
        raise TypeError("thresholds must be a numpy ndarray.")
    if thresholds.size == 0:
        raise ValueError("thresholds array is empty.")
    if not isinstance(metric_values_dict_per_class, dict):
        raise TypeError("metric_values_dict_per_class must be a dictionary.")
    if not isinstance(metrics_to_plot, list):
        raise TypeError("metrics_to_plot must be a list.")
    if not isinstance(class_names, list):
        raise TypeError("class_names must be a list.")
    if not isinstance(colors, list):
        raise TypeError("colors must be a list.")
    if len(colors) == 0:
        # Default to matplotlib's color cycle if colors are not provided
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError("class_names list is empty.")

    # Determine grid size for subplots
    n_cols = int(np.ceil(np.sqrt(num_classes)))
    n_rows = int(np.ceil(num_classes / n_cols))

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), num=MATPLOTLIB_ACROSS_METRICS_THRESHOLDS_PER_CLASS_FIGURE_NUM)
    fig.clear()
    fig.tight_layout(pad=0)
    fig.set_dpi(300)

    # Flatten axes for easy indexing
    axes = [axes] if num_classes == 1 else axes.flatten()

    # Line styles for distinction
    line_styles = ["-", "--", "-.", ":", (0, (5, 10)), (0, (5, 5)), (0, (3, 5, 1, 5))]

    # Plot each class
    for class_idx, class_name in enumerate(class_names):
        if class_name not in metric_values_dict_per_class:
            raise KeyError(f"Class '{class_name}' not found in metric_values_dict_per_class.")
        ax = axes[class_idx]
        metric_values_dict = metric_values_dict_per_class[class_name]

        # Plot each metric for the current class
        for i, metric_name in enumerate(metrics_to_plot):
            if metric_name not in metric_values_dict:
                raise KeyError(f"Metric '{metric_name}' not found for class '{class_name}'.")
            metric_values = metric_values_dict[metric_name]
            if len(metric_values) != len(thresholds):
                raise ValueError(f"Length of metric '{metric_name}' values for class '{class_name}' " + "does not match length of thresholds.")
            ax.plot(
                thresholds,
                metric_values,
                label=metric_name.capitalize(),
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,
                color=colors[i % len(colors)],
            )

        # Add titles and labels for each subplot
        ax.set_title(f"{class_name}", fontsize=12)
        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel("Metric Score", fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True)

    return fig


def plot_confusion_matrices(
    conf_mat: np.ndarray,
    task: Literal["binary", "multiclass", "multilabel"],
    class_names: list[str],
):
    """
    Plots confusion matrices for each class in a single figure with multiple subplots.

    Args:
        conf_mat (np.ndarray): Confusion matrix or matrices. For binary classification, a single 2x2 matrix.
            For multilabel or multiclass, an array of shape (num_classes, 2, 2).
        task (Literal["binary", "multiclass", "multilabel"]): Task type.
        class_names (List[str]): List of class names.

    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If confusion matrix dimensions or task specifications are invalid.

    Returns:
        plt.Figure
    """
    # Validate inputs
    if not isinstance(conf_mat, np.ndarray):
        raise TypeError("conf_mat must be a numpy ndarray.")
    if conf_mat.size == 0:
        raise ValueError("conf_mat is empty.")
    if not isinstance(task, str) or task not in ["binary", "multiclass", "multilabel"]:
        raise ValueError("Invalid task. Expected 'binary', 'multiclass', or 'multilabel'.")

    if task == "binary":
        # Binary classification expects a single 2x2 matrix
        if conf_mat.shape != (2, 2):
            raise ValueError("For binary task, conf_mat must be of shape (2, 2).")

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["Negative", "Positive"])
        fig, ax = plt.subplots(num=MATPLOTLIB_BINARY_CONFUSION_MATRIX_FIGURE_NUM, figsize=(6, 6))

        fig.tight_layout()
        fig.set_dpi(300)
        disp.plot(cmap="Reds", ax=ax, colorbar=False, values_format=".2f")
        ax.set_title("Confusion Matrix")
    else:
        # Multilabel or multiclass expects a set of 2x2 matrices
        num_matrices = conf_mat.shape[0]

        if conf_mat.shape[1:] != (2, 2):
            raise ValueError("For multilabel or multiclass task, conf_mat must have shape (num_labels, 2, 2).")
        if len(class_names) != num_matrices:
            raise ValueError("Length of class_names must match number of labels in conf_mat.")

        # Determine grid size for subplots
        n_cols = int(np.ceil(np.sqrt(num_matrices)))
        n_rows = int(np.ceil(num_matrices / n_cols))

        # Create subplots for each confusion matrix
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), num=MATPLOTLIB_MULTICLASS_CONFUSION_MATRIX_FIGURE_NUM)
        fig.set_dpi(300)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # Plot each confusion matrix
        for idx, (cf, class_name) in enumerate(zip(conf_mat, class_names, strict=True)):
            disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=["Negative", "Positive"])
            disp.plot(cmap="Reds", ax=axes[idx], colorbar=False, values_format=".2f")
            axes[idx].set_title(f"{class_name}")
            axes[idx].set_xlabel("Predicted class")
            axes[idx].set_ylabel("True class")

        # Remove unused subplot axes
        for ax in axes[num_matrices:]:
            fig.delaxes(ax)

        plt.tight_layout()

    return fig
