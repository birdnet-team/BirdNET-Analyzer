import pytest
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from birdnet_analyzer.evaluation.assessment.plotting import (
    plot_overall_metrics,
    plot_metrics_per_class,
    plot_metrics_across_thresholds,
    plot_metrics_across_thresholds_per_class,
    plot_confusion_matrices,
)

# Set the matplotlib backend to 'Agg' to prevent GUI issues during testing
matplotlib.use("Agg")

# Mock plt.show to prevent plots from displaying during tests
plt.show = lambda: None


class TestPlotOverallMetrics:
    """
    Test suite for the plot_overall_metrics function.
    """

    def test_valid_input(self):
        """
        Test with valid inputs to ensure the function runs without errors.
        """
        metrics_df = pd.DataFrame(
            {"Overall": [0.8, 0.75, 0.9]}, index=["Precision", "Recall", "F1"]
        )
        colors = ["blue", "green", "red"]
        plot_overall_metrics(metrics_df, colors)

    def test_empty_metrics_df(self):
        """
        Test with an empty DataFrame to check handling of empty data.
        """
        metrics_df = pd.DataFrame({"Overall": []})
        colors = []
        with pytest.raises(ValueError):
            plot_overall_metrics(metrics_df, colors)

    def test_missing_overall_column(self):
        """
        Test with metrics_df missing 'Overall' column to ensure it raises KeyError.
        """
        metrics_df = pd.DataFrame(
            {"Value": [0.8, 0.75, 0.9]}, index=["Precision", "Recall", "F1"]
        )
        colors = ["blue", "green", "red"]
        with pytest.raises(KeyError):
            plot_overall_metrics(metrics_df, colors)

    def test_colors_shorter_than_metrics(self):
        """
        Test with fewer colors than metrics to check color assignment.
        """
        metrics_df = pd.DataFrame(
            {"Overall": [0.8, 0.75, 0.9]}, index=["Precision", "Recall", "F1"]
        )
        colors = ["blue", "green"]  # Only two colors for three metrics
        plot_overall_metrics(metrics_df, colors)

    def test_colors_longer_than_metrics(self):
        """
        Test with more colors than metrics to ensure extra colors are ignored.
        """
        metrics_df = pd.DataFrame(
            {"Overall": [0.8, 0.75]}, index=["Precision", "Recall"]
        )
        colors = ["blue", "green", "red", "yellow"]
        plot_overall_metrics(metrics_df, colors)

    def test_invalid_metrics_df_type(self):
        """
        Test with invalid type for metrics_df to ensure it raises TypeError.
        """
        metrics_df = [["Precision", 0.8], ["Recall", 0.75]]
        colors = ["blue", "green"]
        with pytest.raises(TypeError):
            plot_overall_metrics(metrics_df, colors)

    def test_invalid_colors_type(self):
        """
        Test with invalid type for colors to ensure it raises TypeError.
        """
        metrics_df = pd.DataFrame(
            {"Overall": [0.8, 0.75]}, index=["Precision", "Recall"]
        )
        colors = "blue"  # Should be a list
        with pytest.raises(TypeError):
            plot_overall_metrics(metrics_df, colors)

    def test_nan_values(self):
        """
        Test with NaN values in metrics_df to ensure it handles missing data.
        """
        metrics_df = pd.DataFrame(
            {"Overall": [0.8, np.nan, 0.9]}, index=["Precision", "Recall", "F1"]
        )
        colors = ["blue", "green", "red"]
        plot_overall_metrics(metrics_df, colors)

    def test_non_unique_metric_names(self):
        """
        Test with non-unique metric names to check handling of index duplication.
        """
        metrics_df = pd.DataFrame(
            {"Overall": [0.8, 0.75, 0.9]}, index=["Precision", "Precision", "F1"]
        )
        colors = ["blue", "green", "red"]
        plot_overall_metrics(metrics_df, colors)

    def test_extremely_large_values(self):
        """
        Test with extremely large values to check plot scaling.
        """
        metrics_df = pd.DataFrame(
            {"Overall": [1e10, 5e10, 1e11]}, index=["Metric1", "Metric2", "Metric3"]
        )
        colors = ["blue", "green", "red"]
        plot_overall_metrics(metrics_df, colors)


class TestPlotMetricsPerClass:
    """
    Test suite for the plot_metrics_per_class function.
    """

    def test_valid_input(self):
        """
        Test with valid inputs to ensure the function runs without errors.
        """
        metrics_df = pd.DataFrame(
            {
                "Class1": [0.8, 0.7, 0.9],
                "Class2": [0.85, 0.75, 0.95],
                "Class3": [0.9, 0.8, 0.96],
            },
            index=["Precision", "Recall", "F1"],
        )
        colors = ["blue", "green", "red"]
        plot_metrics_per_class(metrics_df, colors)

    def test_empty_metrics_df(self):
        """
        Test with an empty DataFrame to check handling of empty data.
        """
        metrics_df = pd.DataFrame()
        colors = []
        with pytest.raises(ValueError):
            plot_metrics_per_class(metrics_df, colors)

    def test_mismatched_colors_length(self):
        """
        Test with mismatched colors length to check color assignment.
        """
        metrics_df = pd.DataFrame(
            {"Class1": [0.8, 0.7, 0.9], "Class2": [0.85, 0.75, 0.95]},
            index=["Precision", "Recall", "F1"],
        )
        colors = ["blue"]  # Only one color provided
        plot_metrics_per_class(metrics_df, colors)

    def test_invalid_metrics_df_type(self):
        """
        Test with invalid type for metrics_df to ensure it raises TypeError.
        """
        metrics_df = [["Class1", 0.8], ["Class2", 0.85]]
        colors = ["blue", "green"]
        with pytest.raises(TypeError):
            plot_metrics_per_class(metrics_df, colors)

    def test_invalid_colors_type(self):
        """
        Test with invalid type for colors to ensure it raises TypeError.
        """
        metrics_df = pd.DataFrame(
            {"Class1": [0.8, 0.7, 0.9], "Class2": [0.85, 0.75, 0.95]},
            index=["Precision", "Recall", "F1"],
        )
        colors = "blue"  # Should be a list
        with pytest.raises(TypeError):
            plot_metrics_per_class(metrics_df, colors)

    def test_nan_values(self):
        """
        Test with NaN values to ensure it handles missing data.
        """
        metrics_df = pd.DataFrame(
            {"Class1": [0.8, np.nan, 0.9], "Class2": [0.85, 0.75, np.nan]},
            index=["Precision", "Recall", "F1"],
        )
        colors = ["blue", "green"]
        plot_metrics_per_class(metrics_df, colors)

    def test_inconsistent_lengths(self):
        """
        Test with inconsistent lengths per metric to check for errors.
        """
        # Create data with consistent lengths but insert NaNs to simulate missing data
        metrics_df = pd.DataFrame(
            {"Class1": [0.8, 0.7, np.nan], "Class2": [0.85, 0.75, 0.95]},
            index=["Precision", "Recall", "F1"],
        )
        colors = ["blue", "green"]
        plot_metrics_per_class(metrics_df, colors)

    def test_empty_colors_list(self):
        """
        Test with an empty colors list to check default color handling.
        """
        metrics_df = pd.DataFrame(
            {"Class1": [0.8, 0.7, 0.9], "Class2": [0.85, 0.75, 0.95]},
            index=["Precision", "Recall", "F1"],
        )
        colors = []
        plot_metrics_per_class(metrics_df, colors)

    def test_non_string_metric_names(self):
        """
        Test with non-string metric names to ensure labels are handled correctly.
        """
        metrics_df = pd.DataFrame(
            {"Class1": [0.8, 0.7, 0.9], "Class2": [0.85, 0.75, 0.95]}, index=[1, 2, 3]
        )
        colors = ["blue", "green"]
        plot_metrics_per_class(metrics_df, colors)

    def test_many_classes(self):
        """
        Test with many classes to check plotting scales correctly.
        """
        classes = [f"Class{i}" for i in range(20)]
        data = np.random.rand(3, 20)
        metrics_df = pd.DataFrame(
            data, index=["Precision", "Recall", "F1"], columns=classes
        )
        colors = ["blue", "green", "red"]
        plot_metrics_per_class(metrics_df, colors)


class TestPlotMetricsAcrossThresholds:
    """
    Test suite for the plot_metrics_across_thresholds function.
    """

    def test_valid_input(self):
        """
        Test with valid inputs to ensure the function runs without errors.
        """
        thresholds = np.linspace(0, 1, 10)
        metric_values_dict = {
            "precision": np.random.rand(10),
            "recall": np.random.rand(10),
            "f1": np.random.rand(10),
        }
        metrics_to_plot = ["precision", "recall", "f1"]
        colors = ["blue", "green", "red"]
        plot_metrics_across_thresholds(
            thresholds, metric_values_dict, metrics_to_plot, colors
        )

    def test_empty_thresholds(self):
        """
        Test with empty thresholds array to check handling of empty data.
        """
        thresholds = np.array([])
        metric_values_dict = {}
        metrics_to_plot = []
        colors = []
        with pytest.raises(ValueError):
            plot_metrics_across_thresholds(
                thresholds, metric_values_dict, metrics_to_plot, colors
            )

    def test_mismatched_lengths(self):
        """
        Test with mismatched lengths in metric values.
        """
        thresholds = np.linspace(0, 1, 10)
        metric_values_dict = {
            "precision": np.random.rand(8),  # Should be length 10
            "recall": np.random.rand(10),
            "f1": np.random.rand(10),
        }
        metrics_to_plot = ["precision", "recall", "f1"]
        colors = ["blue", "green", "red"]
        with pytest.raises(ValueError):
            plot_metrics_across_thresholds(
                thresholds, metric_values_dict, metrics_to_plot, colors
            )

    def test_invalid_thresholds_type(self):
        """
        Test with invalid type for thresholds to ensure it raises TypeError.
        """
        thresholds = "invalid_thresholds"
        metric_values_dict = {}
        metrics_to_plot = []
        colors = []
        with pytest.raises(TypeError):
            plot_metrics_across_thresholds(
                thresholds, metric_values_dict, metrics_to_plot, colors
            )

    def test_invalid_metrics_dict_type(self):
        """
        Test with invalid type for metric_values_dict.
        """
        thresholds = np.linspace(0, 1, 10)
        metric_values_dict = [("precision", np.random.rand(10))]
        metrics_to_plot = ["precision"]
        colors = ["blue"]
        with pytest.raises(TypeError):
            plot_metrics_across_thresholds(
                thresholds, metric_values_dict, metrics_to_plot, colors
            )

    def test_invalid_metrics_to_plot_type(self):
        """
        Test with invalid type for metrics_to_plot.
        """
        thresholds = np.linspace(0, 1, 10)
        metric_values_dict = {"precision": np.random.rand(10)}
        metrics_to_plot = "precision"  # Should be a list
        colors = ["blue"]
        with pytest.raises(TypeError):
            plot_metrics_across_thresholds(
                thresholds, metric_values_dict, metrics_to_plot, colors
            )

    def test_nan_values(self):
        """
        Test with NaN values in metric values.
        """
        thresholds = np.linspace(0, 1, 10)
        metric_values_dict = {
            "precision": np.append(np.random.rand(9), np.nan),
            "recall": np.random.rand(10),
            "f1": np.random.rand(10),
        }
        metrics_to_plot = ["precision", "recall", "f1"]
        colors = ["blue", "green", "red"]
        plot_metrics_across_thresholds(
            thresholds, metric_values_dict, metrics_to_plot, colors
        )

    def test_empty_colors_list(self):
        """
        Test with empty colors list to check default color handling.
        """
        thresholds = np.linspace(0, 1, 10)
        metric_values_dict = {
            "precision": np.random.rand(10),
            "recall": np.random.rand(10),
            "f1": np.random.rand(10),
        }
        metrics_to_plot = ["precision", "recall", "f1"]
        colors = []
        plot_metrics_across_thresholds(
            thresholds, metric_values_dict, metrics_to_plot, colors
        )

    def test_many_metrics(self):
        """
        Test with many metrics to check plotting scales correctly.
        """
        thresholds = np.linspace(0, 1, 10)
        metrics_to_plot = [f"metric{i}" for i in range(20)]
        metric_values_dict = {metric: np.random.rand(10) for metric in metrics_to_plot}
        colors = ["blue", "green", "red"] * 7
        plot_metrics_across_thresholds(
            thresholds, metric_values_dict, metrics_to_plot, colors
        )

    def test_mismatched_colors_length(self):
        """
        Test with mismatched colors length to check color assignment.
        """
        thresholds = np.linspace(0, 1, 10)
        metric_values_dict = {
            "precision": np.random.rand(10),
            "recall": np.random.rand(10),
        }
        metrics_to_plot = ["precision", "recall"]
        colors = ["blue"]  # Only one color provided
        plot_metrics_across_thresholds(
            thresholds, metric_values_dict, metrics_to_plot, colors
        )

    def test_large_thresholds_array(self):
        """
        Test with a large thresholds array to check performance.
        """
        thresholds = np.linspace(0, 1, 1000)
        metric_values_dict = {
            "precision": np.random.rand(1000),
            "recall": np.random.rand(1000),
            "f1": np.random.rand(1000),
        }
        metrics_to_plot = ["precision", "recall", "f1"]
        colors = ["blue", "green", "red"]
        plot_metrics_across_thresholds(
            thresholds, metric_values_dict, metrics_to_plot, colors
        )


class TestPlotMetricsAcrossThresholdsPerClass:
    """
    Test suite for the plot_metrics_across_thresholds_per_class function.
    """

    def test_valid_input(self):
        """
        Test with valid inputs to ensure the function runs without errors.
        """
        thresholds = np.linspace(0, 1, 10)
        class_names = ["Class1", "Class2"]
        metrics_to_plot = ["precision", "recall"]
        metric_values_dict_per_class = {
            "Class1": {"precision": np.random.rand(10), "recall": np.random.rand(10)},
            "Class2": {"precision": np.random.rand(10), "recall": np.random.rand(10)},
        }
        colors = ["blue", "green"]
        plot_metrics_across_thresholds_per_class(
            thresholds,
            metric_values_dict_per_class,
            metrics_to_plot,
            class_names,
            colors,
        )

    def test_empty_thresholds(self):
        """
        Test with empty thresholds array to check handling of empty data.
        """
        thresholds = np.array([])
        class_names = []
        metrics_to_plot = []
        metric_values_dict_per_class = {}
        colors = []
        with pytest.raises(ValueError):
            plot_metrics_across_thresholds_per_class(
                thresholds,
                metric_values_dict_per_class,
                metrics_to_plot,
                class_names,
                colors,
            )

    def test_invalid_class_names(self):
        """
        Test with invalid type for class_names.
        """
        thresholds = np.linspace(0, 1, 10)
        class_names = "Class1"  # Should be a list
        metrics_to_plot = ["precision"]
        metric_values_dict_per_class = {}
        colors = ["blue"]
        with pytest.raises(TypeError):
            plot_metrics_across_thresholds_per_class(
                thresholds,
                metric_values_dict_per_class,
                metrics_to_plot,
                class_names,
                colors,
            )

    def test_nan_values(self):
        """
        Test with NaN values in metric values.
        """
        thresholds = np.linspace(0, 1, 10)
        class_names = ["Class1"]
        metrics_to_plot = ["precision"]
        metric_values_dict_per_class = {
            "Class1": {"precision": np.append(np.random.rand(9), np.nan)}
        }
        colors = ["blue"]
        plot_metrics_across_thresholds_per_class(
            thresholds,
            metric_values_dict_per_class,
            metrics_to_plot,
            class_names,
            colors,
        )

    def test_empty_colors_list(self):
        """
        Test with empty colors list to check default color handling.
        """
        thresholds = np.linspace(0, 1, 10)
        class_names = ["Class1"]
        metrics_to_plot = ["precision"]
        metric_values_dict_per_class = {"Class1": {"precision": np.random.rand(10)}}
        colors = []
        plot_metrics_across_thresholds_per_class(
            thresholds,
            metric_values_dict_per_class,
            metrics_to_plot,
            class_names,
            colors,
        )

    def test_many_classes(self):
        """
        Test with many classes to check plotting scales correctly.
        """
        thresholds = np.linspace(0, 1, 10)
        class_names = [f"Class{i}" for i in range(20)]
        metrics_to_plot = ["precision"]
        metric_values_dict_per_class = {
            class_name: {"precision": np.random.rand(10)} for class_name in class_names
        }
        colors = ["blue", "green", "red"] * 7
        plot_metrics_across_thresholds_per_class(
            thresholds,
            metric_values_dict_per_class,
            metrics_to_plot,
            class_names,
            colors,
        )

    def test_mismatched_colors_length(self):
        """
        Test with mismatched colors length.
        """
        thresholds = np.linspace(0, 1, 10)
        class_names = ["Class1", "Class2"]
        metrics_to_plot = ["precision"]
        metric_values_dict_per_class = {
            "Class1": {"precision": np.random.rand(10)},
            "Class2": {"precision": np.random.rand(10)},
        }
        colors = ["blue"]  # Only one color provided
        plot_metrics_across_thresholds_per_class(
            thresholds,
            metric_values_dict_per_class,
            metrics_to_plot,
            class_names,
            colors,
        )

    def test_invalid_metrics_to_plot(self):
        """
        Test with invalid metrics_to_plot type.
        """
        thresholds = np.linspace(0, 1, 10)
        class_names = ["Class1"]
        metrics_to_plot = "precision"  # Should be a list
        metric_values_dict_per_class = {"Class1": {"precision": np.random.rand(10)}}
        colors = ["blue"]
        with pytest.raises(TypeError):
            plot_metrics_across_thresholds_per_class(
                thresholds,
                metric_values_dict_per_class,
                metrics_to_plot,
                class_names,
                colors,
            )

    def test_missing_class_in_dict(self):
        """
        Test with missing class in metric_values_dict_per_class.
        """
        thresholds = np.linspace(0, 1, 10)
        class_names = ["Class1", "Class2"]
        metrics_to_plot = ["precision"]
        metric_values_dict_per_class = {
            "Class1": {"precision": np.random.rand(10)}
            # 'Class2' is missing
        }
        colors = ["blue", "green"]
        with pytest.raises(KeyError):
            plot_metrics_across_thresholds_per_class(
                thresholds,
                metric_values_dict_per_class,
                metrics_to_plot,
                class_names,
                colors,
            )

    def test_mismatched_lengths(self):
        """
        Test with mismatched metric values lengths.
        """
        thresholds = np.linspace(0, 1, 10)
        class_names = ["Class1"]
        metrics_to_plot = ["precision"]
        metric_values_dict_per_class = {
            "Class1": {"precision": np.random.rand(9)}  # Length should be 10
        }
        colors = ["blue"]
        with pytest.raises(ValueError):
            plot_metrics_across_thresholds_per_class(
                thresholds,
                metric_values_dict_per_class,
                metrics_to_plot,
                class_names,
                colors,
            )

    def test_large_thresholds_array(self):
        """
        Test with a large thresholds array.
        """
        thresholds = np.linspace(0, 1, 1000)
        class_names = ["Class1"]
        metrics_to_plot = ["precision"]
        metric_values_dict_per_class = {"Class1": {"precision": np.random.rand(1000)}}
        colors = ["blue"]
        plot_metrics_across_thresholds_per_class(
            thresholds,
            metric_values_dict_per_class,
            metrics_to_plot,
            class_names,
            colors,
        )


class TestPlotConfusionMatrices:
    """
    Test suite for the plot_confusion_matrices function.
    """

    def test_binary_task(self):
        """
        Test with binary task to ensure it runs without errors.
        """
        conf_mat = np.array([[50, 10], [5, 35]])
        task = "binary"
        class_names = ["Positive", "Negative"]
        plot_confusion_matrices(conf_mat, task, class_names)

    def test_multilabel_task(self):
        """
        Test with multilabel task to ensure it runs without errors.
        """
        conf_mat = np.array([[[50, 10], [5, 35]], [[45, 15], [10, 30]]])
        task = "multilabel"
        class_names = ["Class1", "Class2"]
        plot_confusion_matrices(conf_mat, task, class_names)

    def test_invalid_task(self):
        """
        Test with an invalid task to ensure it raises ValueError.
        """
        conf_mat = np.array([[50, 10], [5, 35]])
        task = "invalid_task"
        class_names = ["Positive", "Negative"]
        with pytest.raises(ValueError):
            plot_confusion_matrices(conf_mat, task, class_names)

    def test_empty_conf_mat(self):
        """
        Test with empty confusion matrix to check handling of empty data.
        """
        conf_mat = np.array([])
        task = "binary"
        class_names = ["Positive", "Negative"]
        with pytest.raises(ValueError):
            plot_confusion_matrices(conf_mat, task, class_names)

    def test_mismatched_class_names(self):
        """
        Test with mismatched class_names length.
        """
        conf_mat = np.array([[50, 10], [5, 35]])
        task = "binary"
        class_names = ["Positive"]  # Should be two class names
        with pytest.raises(ValueError):
            plot_confusion_matrices(conf_mat, task, class_names)

    def test_invalid_conf_mat_type(self):
        """
        Test with invalid type for conf_mat to ensure it raises TypeError.
        """
        conf_mat = "invalid_conf_mat"
        task = "binary"
        class_names = ["Positive", "Negative"]
        with pytest.raises(TypeError):
            plot_confusion_matrices(conf_mat, task, class_names)

    def test_nan_values(self):
        """
        Test with NaN values in conf_mat.
        """
        conf_mat = np.array([[np.nan, 10], [5, 35]])
        task = "binary"
        class_names = ["Positive", "Negative"]
        plot_confusion_matrices(conf_mat, task, class_names)

    def test_single_class(self):
        """
        Test with single class confusion matrix.
        """
        conf_mat = np.array([[[50, 10], [5, 35]]])
        task = "multilabel"
        class_names = ["Class1"]
        plot_confusion_matrices(conf_mat, task, class_names)

    def test_many_classes(self):
        """
        Test with many classes to check plotting scales correctly.
        """
        num_classes = 10
        conf_mat = np.random.randint(0, 100, size=(num_classes, 2, 2))
        task = "multilabel"
        class_names = [f"Class{i}" for i in range(num_classes)]
        plot_confusion_matrices(conf_mat, task, class_names)

    def test_invalid_conf_mat_shape(self):
        """
        Test with invalid shape for conf_mat.
        """
        conf_mat = np.array([50, 10, 5, 35])  # Should be 2x2 or Nx2x2
        task = "binary"
        class_names = ["Positive", "Negative"]
        with pytest.raises(ValueError):
            plot_confusion_matrices(conf_mat, task, class_names)
