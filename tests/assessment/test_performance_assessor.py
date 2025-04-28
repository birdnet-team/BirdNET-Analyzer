import pytest
import numpy as np
import pandas as pd
import matplotlib

from birdnet_analyzer.evaluation.assessment.performance_assessor import PerformanceAssessor


matplotlib.use("Agg")  # Use non-interactive backend for plotting


class TestPerformanceAssessorInit:
    """
    Test suite for the PerformanceAssessor __init__ method.
    """

    def test_init_with_valid_inputs(self):
        """
        Test initializing PerformanceAssessor with valid inputs.
        """
        num_classes = 3
        threshold = 0.5
        classes = ("Class1", "Class2", "Class3")
        task = "multilabel"
        metrics_list = ("recall", "precision", "f1")
        assessor = PerformanceAssessor(
            num_classes, threshold, classes, task, metrics_list
        )
        assert assessor.num_classes == num_classes
        assert assessor.threshold == threshold
        assert assessor.classes == classes
        assert assessor.task == task
        assert assessor.metrics_list == metrics_list

    def test_init_with_invalid_num_classes(self):
        """
        Test initializing PerformanceAssessor with invalid num_classes (non-positive integer).
        """
        with pytest.raises(ValueError):
            PerformanceAssessor(num_classes=0)

    def test_init_with_invalid_threshold(self):
        """
        Test initializing PerformanceAssessor with invalid threshold (not between 0 and 1).
        """
        with pytest.raises(ValueError):
            PerformanceAssessor(num_classes=3, threshold=1.5)

    def test_init_with_invalid_classes_length(self):
        """
        Test initializing PerformanceAssessor when length of classes does not match num_classes.
        """
        with pytest.raises(ValueError):
            PerformanceAssessor(num_classes=2, classes=("Class1", "Class2", "Class3"))

    def test_init_with_invalid_classes_type(self):
        """
        Test initializing PerformanceAssessor when classes is not a tuple of strings.
        """
        with pytest.raises(ValueError):
            PerformanceAssessor(
                num_classes=2, classes=["Class1", "Class2"]
            )  # Should be tuple

    def test_init_with_invalid_task(self):
        """
        Test initializing PerformanceAssessor with invalid task type.
        """
        with pytest.raises(ValueError):
            PerformanceAssessor(num_classes=2, task="invalid_task")

    def test_init_with_invalid_metrics_list(self):
        """
        Test initializing PerformanceAssessor with invalid metrics_list containing unsupported metric.
        """
        with pytest.raises(ValueError):
            PerformanceAssessor(
                num_classes=2, metrics_list=("recall", "unsupported_metric")
            )

    def test_init_with_empty_metrics_list(self):
        """
        Test initializing PerformanceAssessor with empty metrics_list.
        """
        with pytest.raises(ValueError):
            PerformanceAssessor(num_classes=2, metrics_list=())

    def test_init_with_large_num_classes(self):
        """
        Test initializing PerformanceAssessor with a large number of classes.
        """
        num_classes = 1000
        assessor = PerformanceAssessor(num_classes=num_classes)
        assert assessor.num_classes == num_classes

    def test_init_with_default_parameters(self):
        """
        Test initializing PerformanceAssessor with default parameters.
        """
        assessor = PerformanceAssessor(num_classes=2)
        assert assessor.num_classes == 2
        assert assessor.threshold == 0.5
        assert assessor.classes is None
        assert assessor.task == "multilabel"
        assert assessor.metrics_list == (
            "recall",
            "precision",
            "f1",
            "ap",
            "auroc",
            "accuracy",
        )


class TestPerformanceAssessorCalculateMetrics:
    """
    Test suite for the PerformanceAssessor calculate_metrics method.
    """

    def test_calculate_metrics_with_valid_inputs(self):
        """
        Test calculate_metrics with valid predictions and labels.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100, num_classes))
        metrics_df = assessor.calculate_metrics(predictions, labels)
        assert isinstance(metrics_df, pd.DataFrame)
        assert not metrics_df.empty

    def test_calculate_metrics_with_per_class_metrics(self):
        """
        Test calculate_metrics with per_class_metrics=True.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100, num_classes))
        metrics_df = assessor.calculate_metrics(
            predictions, labels, per_class_metrics=True
        )
        assert isinstance(metrics_df, pd.DataFrame)
        assert not metrics_df.empty
        assert metrics_df.shape[1] == num_classes  # Columns should be per class

    def test_calculate_metrics_with_invalid_predictions_shape(self):
        """
        Test calculate_metrics with invalid predictions shape.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100)  # Invalid shape
        labels = np.random.randint(0, 2, size=(100, num_classes))
        with pytest.raises(ValueError):
            assessor.calculate_metrics(predictions, labels)

    def test_calculate_metrics_with_invalid_labels_shape(self):
        """
        Test calculate_metrics with invalid labels shape.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100,))  # Invalid shape
        with pytest.raises(ValueError):
            assessor.calculate_metrics(predictions, labels)

    def test_calculate_metrics_with_mismatched_predictions_and_labels(self):
        """
        Test calculate_metrics when predictions and labels have mismatched shapes.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(
            0, 2, size=(90, num_classes)
        )  # Different number of samples
        with pytest.raises(ValueError):
            assessor.calculate_metrics(predictions, labels)

    def test_calculate_metrics_with_invalid_predictions_type(self):
        """
        Test calculate_metrics with predictions of invalid type (not numpy array).
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = [[0.1, 0.2, 0.3]] * 100  # List instead of numpy array
        labels = np.random.randint(0, 2, size=(100, num_classes))
        with pytest.raises(TypeError):
            assessor.calculate_metrics(predictions, labels)

    def test_calculate_metrics_with_invalid_labels_type(self):
        """
        Test calculate_metrics with labels of invalid type (not numpy array).
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = [[0, 1, 0]] * 100  # List instead of numpy array
        with pytest.raises(TypeError):
            assessor.calculate_metrics(predictions, labels)

    def test_calculate_metrics_with_invalid_metric_in_metrics_list(self):
        """
        Test calculate_metrics when metrics_list contains an invalid metric.
        """
        num_classes = 3
        with pytest.raises(ValueError):
            PerformanceAssessor(
                num_classes=num_classes, metrics_list=("invalid_metric",)
            )

    def test_calculate_metrics_with_binary_task(self):
        """
        Test calculate_metrics with task='binary'.
        """
        num_classes = 1
        assessor = PerformanceAssessor(num_classes=num_classes, task="binary")
        predictions = np.random.rand(100, 1)
        labels = np.random.randint(0, 2, size=(100, 1))
        metrics_df = assessor.calculate_metrics(predictions, labels)
        assert isinstance(metrics_df, pd.DataFrame)
        assert not metrics_df.empty

    def test_calculate_metrics_with_no_classes(self):
        """
        Test calculate_metrics when no classes are provided (classes=None).
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes, classes=None)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100, num_classes))
        metrics_df = assessor.calculate_metrics(
            predictions, labels, per_class_metrics=True
        )
        assert isinstance(metrics_df, pd.DataFrame)
        assert not metrics_df.empty
        expected_columns = [f"Class {i}" for i in range(num_classes)]
        assert list(metrics_df.columns) == expected_columns


class TestPerformanceAssessorPlotMetrics:
    """
    Test suite for the PerformanceAssessor plot_metrics method.
    """

    def test_plot_metrics_with_valid_inputs(self):
        """
        Test plot_metrics with valid predictions and labels.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(50, num_classes)
        labels = np.random.randint(0, 2, size=(50, num_classes))
        assessor.plot_metrics(predictions, labels)

    def test_plot_metrics_with_per_class_metrics(self):
        """
        Test plot_metrics with per_class_metrics=True.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(50, num_classes)
        labels = np.random.randint(0, 2, size=(50, num_classes))
        assessor.plot_metrics(predictions, labels, per_class_metrics=True)

    def test_plot_metrics_with_invalid_predictions_shape(self):
        """
        Test plot_metrics with invalid predictions shape.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(50)  # Invalid shape
        labels = np.random.randint(0, 2, size=(50, num_classes))
        with pytest.raises(ValueError):
            assessor.plot_metrics(predictions, labels)

    def test_plot_metrics_with_invalid_labels_shape(self):
        """
        Test plot_metrics with invalid labels shape.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(50, num_classes)
        labels = np.random.randint(0, 2, size=(50,))  # Invalid shape
        with pytest.raises(ValueError):
            assessor.plot_metrics(predictions, labels)

    def test_plot_metrics_with_mismatched_predictions_and_labels(self):
        """
        Test plot_metrics when predictions and labels have mismatched shapes.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(50, num_classes)
        labels = np.random.randint(
            0, 2, size=(40, num_classes)
        )  # Different number of samples
        with pytest.raises(ValueError):
            assessor.plot_metrics(predictions, labels)

    def test_plot_metrics_with_binary_task(self):
        """
        Test plot_metrics with task='binary'.
        """
        num_classes = 1
        assessor = PerformanceAssessor(num_classes=num_classes, task="binary")
        predictions = np.random.rand(50, 1)
        labels = np.random.randint(0, 2, size=(50, 1))
        assessor.plot_metrics(predictions, labels)

    def test_plot_metrics_with_no_classes(self):
        """
        Test plot_metrics when no classes are provided (classes=None).
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes, classes=None)
        predictions = np.random.rand(50, num_classes)
        labels = np.random.randint(0, 2, size=(50, num_classes))
        assessor.plot_metrics(predictions, labels, per_class_metrics=True)

    def test_plot_metrics_with_invalid_predictions_type(self):
        """
        Test plot_metrics with predictions of invalid type (not numpy array).
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = [[0.1, 0.2, 0.3]] * 50  # List instead of numpy array
        labels = np.random.randint(0, 2, size=(50, num_classes))
        with pytest.raises(TypeError):
            assessor.plot_metrics(predictions, labels)

    def test_plot_metrics_with_invalid_labels_type(self):
        """
        Test plot_metrics with labels of invalid type (not numpy array).
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(50, num_classes)
        labels = [[0, 1, 0]] * 50  # List instead of numpy array
        with pytest.raises(TypeError):
            assessor.plot_metrics(predictions, labels)

    def test_plot_metrics_with_large_number_of_classes(self):
        """
        Test plot_metrics with a large number of classes.
        """
        num_classes = 100
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(50, num_classes)
        labels = np.random.randint(0, 2, size=(50, num_classes))
        assessor.plot_metrics(predictions, labels, per_class_metrics=True)


class TestPerformanceAssessorPlotMetricsAllThresholds:
    """
    Test suite for the PerformanceAssessor plot_metrics_all_thresholds method.
    """

    def test_plot_metrics_all_thresholds_with_valid_inputs(self):
        """
        Test plot_metrics_all_thresholds with valid predictions and labels.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100, num_classes))
        assessor.plot_metrics_all_thresholds(predictions, labels)

    def test_plot_metrics_all_thresholds_with_per_class_metrics(self):
        """
        Test plot_metrics_all_thresholds with per_class_metrics=True.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100, num_classes))
        assessor.plot_metrics_all_thresholds(
            predictions, labels, per_class_metrics=True
        )

    def test_plot_metrics_all_thresholds_with_invalid_predictions_shape(self):
        """
        Test plot_metrics_all_thresholds with invalid predictions shape.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100)  # Invalid shape
        labels = np.random.randint(0, 2, size=(100, num_classes))
        with pytest.raises(ValueError):
            assessor.plot_metrics_all_thresholds(predictions, labels)

    def test_plot_metrics_all_thresholds_with_invalid_labels_shape(self):
        """
        Test plot_metrics_all_thresholds with invalid labels shape.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100,))  # Invalid shape
        with pytest.raises(ValueError):
            assessor.plot_metrics_all_thresholds(predictions, labels)

    def test_plot_metrics_all_thresholds_with_mismatched_predictions_and_labels(self):
        """
        Test plot_metrics_all_thresholds when predictions and labels have mismatched shapes.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(
            0, 2, size=(90, num_classes)
        )  # Different number of samples
        with pytest.raises(ValueError):
            assessor.plot_metrics_all_thresholds(predictions, labels)

    def test_plot_metrics_all_thresholds_with_binary_task(self):
        """
        Test plot_metrics_all_thresholds with task='binary'.
        """
        num_classes = 1
        assessor = PerformanceAssessor(num_classes=num_classes, task="binary")
        predictions = np.random.rand(100, 1)
        labels = np.random.randint(0, 2, size=(100, 1))
        assessor.plot_metrics_all_thresholds(predictions, labels)

    def test_plot_metrics_all_thresholds_with_no_classes(self):
        """
        Test plot_metrics_all_thresholds when no classes are provided (classes=None).
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes, classes=None)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100, num_classes))
        assessor.plot_metrics_all_thresholds(
            predictions, labels, per_class_metrics=True
        )

    def test_plot_metrics_all_thresholds_with_invalid_predictions_type(self):
        """
        Test plot_metrics_all_thresholds with predictions of invalid type (not numpy array).
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = [[0.1, 0.2, 0.3]] * 100  # List instead of numpy array
        labels = np.random.randint(0, 2, size=(100, num_classes))
        with pytest.raises(TypeError):
            assessor.plot_metrics_all_thresholds(predictions, labels)

    def test_plot_metrics_all_thresholds_with_invalid_labels_type(self):
        """
        Test plot_metrics_all_thresholds with labels of invalid type (not numpy array).
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = [[0, 1, 0]] * 100  # List instead of numpy array
        with pytest.raises(TypeError):
            assessor.plot_metrics_all_thresholds(predictions, labels)

    def test_plot_metrics_all_thresholds_with_large_number_of_classes(self):
        """
        Test plot_metrics_all_thresholds with a large number of classes.
        """
        num_classes = 50
        assessor = PerformanceAssessor(num_classes=num_classes)
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100, num_classes))
        assessor.plot_metrics_all_thresholds(
            predictions, labels, per_class_metrics=True
        )


class TestPerformanceAssessorPlotConfusionMatrix:
    """
    Test suite for the PerformanceAssessor plot_confusion_matrix method.
    """

    def test_plot_confusion_matrix_with_valid_inputs(self):
        """
        Test plot_confusion_matrix with valid predictions and labels.
        """
        num_classes = 1
        assessor = PerformanceAssessor(num_classes=num_classes, task="binary")
        predictions = np.random.rand(100, 1)
        labels = np.random.randint(0, 2, size=(100, 1))
        assessor.plot_confusion_matrix(predictions, labels)

    def test_plot_confusion_matrix_with_multilabel_task(self):
        """
        Test plot_confusion_matrix with multilabel task.
        """
        num_classes = 3
        assessor = PerformanceAssessor(num_classes=num_classes, task="multilabel")
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100, num_classes))
        assessor.plot_confusion_matrix(predictions, labels)

    def test_plot_confusion_matrix_with_invalid_predictions_shape(self):
        """
        Test plot_confusion_matrix with invalid predictions shape.
        """
        num_classes = 1
        assessor = PerformanceAssessor(num_classes=num_classes, task="binary")
        predictions = np.random.rand(100)  # Invalid shape
        labels = np.random.randint(0, 2, size=(100, 1))
        with pytest.raises(ValueError):
            assessor.plot_confusion_matrix(predictions, labels)

    def test_plot_confusion_matrix_with_invalid_labels_shape(self):
        """
        Test plot_confusion_matrix with invalid labels shape.
        """
        num_classes = 1
        assessor = PerformanceAssessor(num_classes=num_classes, task="binary")
        predictions = np.random.rand(100, 1)
        labels = np.random.randint(0, 2, size=(100,))  # Invalid shape
        with pytest.raises(ValueError):
            assessor.plot_confusion_matrix(predictions, labels)

    def test_plot_confusion_matrix_with_mismatched_predictions_and_labels(self):
        """
        Test plot_confusion_matrix when predictions and labels have mismatched shapes.
        """
        num_classes = 1
        assessor = PerformanceAssessor(num_classes=num_classes, task="binary")
        predictions = np.random.rand(100, 1)
        labels = np.random.randint(0, 2, size=(90, 1))  # Different number of samples
        with pytest.raises(ValueError):
            assessor.plot_confusion_matrix(predictions, labels)

    def test_plot_confusion_matrix_with_invalid_predictions_type(self):
        """
        Test plot_confusion_matrix with predictions of invalid type (not numpy array).
        """
        num_classes = 1
        assessor = PerformanceAssessor(num_classes=num_classes, task="binary")
        predictions = [0.1] * 100  # List instead of numpy array
        labels = np.random.randint(0, 2, size=(100, 1))
        with pytest.raises(TypeError):
            assessor.plot_confusion_matrix(predictions, labels)

    def test_plot_confusion_matrix_with_invalid_labels_type(self):
        """
        Test plot_confusion_matrix with labels of invalid type (not numpy array).
        """
        num_classes = 1
        assessor = PerformanceAssessor(num_classes=num_classes, task="binary")
        predictions = np.random.rand(100, 1)
        labels = [0] * 100  # List instead of numpy array
        with pytest.raises(TypeError):
            assessor.plot_confusion_matrix(predictions, labels)

    def test_plot_confusion_matrix_with_large_number_of_classes(self):
        """
        Test plot_confusion_matrix with a large number of classes.
        """
        num_classes = 20
        assessor = PerformanceAssessor(num_classes=num_classes, task="multilabel")
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100, num_classes))
        assessor.plot_confusion_matrix(predictions, labels)

    def test_plot_confusion_matrix_with_no_classes(self):
        """
        Test plot_confusion_matrix when no classes are provided (classes=None).
        """
        num_classes = 3
        assessor = PerformanceAssessor(
            num_classes=num_classes, classes=None, task="multilabel"
        )
        predictions = np.random.rand(100, num_classes)
        labels = np.random.randint(0, 2, size=(100, num_classes))
        assessor.plot_confusion_matrix(predictions, labels)
