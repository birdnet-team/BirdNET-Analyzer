import pytest
import numpy as np

from birdnet_analyzer.evaluation.assessment.metrics import (
    calculate_accuracy,
    calculate_recall,
    calculate_precision,
    calculate_f1_score,
    calculate_average_precision,
    calculate_auroc,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


class TestCalculateAccuracy:
    def test_binary_classification_perfect(self):
        predictions = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 1, 0])
        result = calculate_accuracy(
            predictions,
            labels,
            task="binary",
            num_classes=1,
            threshold=0.5,
            averaging_method="micro",
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = np.array([0.6, 0.4, 0.3, 0.7])
        labels = np.array([1, 0, 1, 0])
        result = calculate_accuracy(
            predictions,
            labels,
            task="binary",
            num_classes=1,
            threshold=0.5,
            averaging_method="micro",
        )
        assert np.isclose(result, 0.5)

    def test_binary_classification_all_zeros(self):
        predictions = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0])
        result = calculate_accuracy(
            predictions,
            labels,
            task="binary",
            num_classes=1,
            threshold=0.5,
            averaging_method="micro",
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_all_ones(self):
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 1])
        result = calculate_accuracy(
            predictions,
            labels,
            task="binary",
            num_classes=1,
            threshold=0.5,
            averaging_method="micro",
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_perfect(self):
        predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
        labels = np.array([[1, 0], [0, 1], [1, 0]])
        num_classes = labels.shape[1]
        result = calculate_accuracy(
            predictions,
            labels,
            task="multilabel",
            num_classes=num_classes,
            threshold=0.5,
            averaging_method="micro",
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        predictions = np.array([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]])
        labels = np.array([[1, 0], [0, 1], [1, 0]])
        num_classes = labels.shape[1]
        result = calculate_accuracy(
            predictions,
            labels,
            task="multilabel",
            num_classes=num_classes,
            threshold=0.5,
            averaging_method="micro",
        )
        expected_result = 5 / 6  # Total correct predictions over total elements
        assert np.isclose(result, expected_result)

    def test_incorrect_shapes(self):
        predictions = np.array([0.9, 0.2, 0.8])
        labels = np.array([1, 0])
        with pytest.raises(ValueError):
            calculate_accuracy(
                predictions,
                labels,
                task="binary",
                num_classes=1,
                threshold=0.5,
            )

    def test_invalid_threshold(self):
        predictions = np.array([0.9, 0.2, 0.8, 0.1])
        labels = np.array([1, 0, 1, 0])
        with pytest.raises(ValueError):
            calculate_accuracy(
                predictions,
                labels,
                task="binary",
                num_classes=1,
                threshold=1.5,
            )

    def test_non_array_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        result = calculate_accuracy(
            np.array(predictions),
            np.array(labels),
            task="binary",
            num_classes=1,
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_empty_arrays(self):
        predictions = np.array([])
        labels = np.array([])
        with pytest.raises(ValueError):
            calculate_accuracy(
                predictions,
                labels,
                task="binary",
                num_classes=1,
                threshold=0.5,
            )

    def test_binary_classification_varying_threshold(self):
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([1, 0, 1, 0])
        result = calculate_accuracy(
            predictions,
            labels,
            task="binary",
            num_classes=1,
            threshold=0.75,
        )
        y_pred = (predictions >= 0.75).astype(int)
        acc = accuracy_score(labels, y_pred)
        assert np.isclose(result, acc)

    def test_multilabel_classification_macro_average(self):
        predictions = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
        labels = np.array([[1, 0], [1, 0], [0, 1]])
        num_classes = labels.shape[1]
        result = calculate_accuracy(
            predictions,
            labels,
            task="multilabel",
            num_classes=num_classes,
            threshold=0.5,
            averaging_method="macro",
        )
        y_pred = (predictions >= 0.5).astype(int)
        acc_class0 = accuracy_score(labels[:, 0], y_pred[:, 0])
        acc_class1 = accuracy_score(labels[:, 1], y_pred[:, 1])
        expected_result = (acc_class0 + acc_class1) / 2
        assert np.isclose(result, expected_result)


class TestCalculateRecall:
    def test_binary_classification_perfect(self):
        predictions = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 1, 0])
        result = calculate_recall(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = np.array([0.6, 0.4, 0.3, 0.7])
        labels = np.array([1, 0, 1, 0])
        result = calculate_recall(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 0.5)

    def test_binary_classification_all_zeros(self):
        predictions = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0])
        result = calculate_recall(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        # Recall is ill-defined and set to 0.0 due to no true positives
        assert np.isclose(result, 0.0)

    def test_binary_classification_all_ones(self):
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 1])
        result = calculate_recall(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_perfect(self):
        predictions = np.array([[0.9, 0.8], [0.8, 0.9], [0.7, 0.6]])
        labels = np.array([[1, 1], [1, 1], [1, 1]])
        result = calculate_recall(
            predictions,
            labels,
            task="multilabel",
            threshold=0.5,
        )
        assert np.allclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        predictions = np.array([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]])
        labels = np.array([[1, 0], [0, 1], [1, 0]])
        result = calculate_recall(
            predictions,
            labels,
            task="multilabel",
            threshold=0.5,
            averaging_method="macro",
        )
        expected_recall = recall_score(
            labels, (predictions >= 0.5).astype(int), average="macro", zero_division=0
        )
        assert np.isclose(result, expected_recall)

    def test_incorrect_shapes(self):
        predictions = np.array([0.9, 0.2, 0.8])
        labels = np.array([1, 0])
        with pytest.raises(ValueError):
            calculate_recall(
                predictions,
                labels,
                task="binary",
                threshold=0.5,
            )

    def test_invalid_threshold(self):
        predictions = np.array([0.9, 0.2, 0.8, 0.1])
        labels = np.array([1, 0, 1, 0])
        with pytest.raises(ValueError):
            calculate_recall(
                predictions,
                labels,
                task="binary",
                threshold=-0.1,
            )

    def test_non_array_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        result = calculate_recall(
            np.array(predictions),
            np.array(labels),
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_empty_arrays(self):
        predictions = np.array([])
        labels = np.array([])
        with pytest.raises(ValueError):
            calculate_recall(
                predictions,
                labels,
                task="binary",
                threshold=0.5,
            )

    def test_multilabel_classification_macro_average(self):
        predictions = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
        labels = np.array([[1, 0], [0, 1], [1, 0]])
        result = calculate_recall(
            predictions,
            labels,
            task="multilabel",
            threshold=0.5,
            averaging_method="macro",
        )
        expected_recall = recall_score(
            labels, (predictions >= 0.5).astype(int), average="macro", zero_division=0
        )
        assert np.isclose(result, expected_recall)

    def test_binary_classification_no_positive_predictions(self):
        predictions = np.array([0.0, 0.0, 0.0, 0.0])
        labels = np.array([1, 0, 1, 0])
        result = calculate_recall(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        expected_recall = recall_score(
            labels, (predictions >= 0.5).astype(int), zero_division=0
        )
        assert np.isclose(result, expected_recall)

    def test_binary_classification_no_positive_labels(self):
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 0])
        result = calculate_recall(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        # Recall is ill-defined and set to 0.0 due to no true positives
        assert np.isclose(result, 0.0)


class TestCalculatePrecision:
    def test_binary_classification_perfect(self):
        predictions = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 1, 0])
        result = calculate_precision(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = np.array([0.6, 0.4, 0.3, 0.7])
        labels = np.array([1, 0, 1, 0])
        result = calculate_precision(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 0.5)

    def test_binary_classification_all_zeros(self):
        predictions = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0])
        result = calculate_precision(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        # Precision is ill-defined and set to 0.0 due to no predicted positives
        assert np.isclose(result, 0.0)

    def test_binary_classification_all_ones(self):
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 1])
        result = calculate_precision(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_perfect(self):
        predictions = np.array([[0.9, 0.8], [0.8, 0.9], [0.7, 0.6]])
        labels = np.array([[1, 1], [1, 1], [1, 1]])
        result = calculate_precision(
            predictions,
            labels,
            task="multilabel",
            threshold=0.5,
        )
        assert np.allclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        predictions = np.array(
            [
                [0.9, 0.2],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.5, 0.5],
            ]
        )
        labels = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [1, 0],
                [0, 1],
            ]
        )
        result = calculate_precision(
            predictions,
            labels,
            task="multilabel",
            threshold=0.5,
            averaging_method="macro",
        )
        expected_precision = precision_score(
            labels, (predictions >= 0.5).astype(int), average="macro", zero_division=0
        )
        assert np.isclose(result, expected_precision, atol=1e-4)

    def test_incorrect_shapes(self):
        predictions = np.array([0.9, 0.2, 0.8])
        labels = np.array([1, 0])
        with pytest.raises(ValueError):
            calculate_precision(
                predictions,
                labels,
                task="binary",
                threshold=0.5,
            )

    def test_invalid_threshold(self):
        predictions = np.array([0.9, 0.2, 0.8, 0.1])
        labels = np.array([1, 0, 1, 0])
        with pytest.raises(ValueError):
            calculate_precision(
                predictions,
                labels,
                task="binary",
                threshold=-0.1,
            )

    def test_non_array_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        result = calculate_precision(
            np.array(predictions),
            np.array(labels),
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_empty_arrays(self):
        predictions = np.array([])
        labels = np.array([])
        with pytest.raises(ValueError):
            calculate_precision(
                predictions,
                labels,
                task="binary",
                threshold=0.5,
            )

    def test_multilabel_classification_weighted_average(self):
        predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
        labels = np.array([[1, 0], [0, 1], [1, 0]])
        result = calculate_precision(
            predictions,
            labels,
            task="multilabel",
            threshold=0.5,
            averaging_method="weighted",
        )
        expected_precision = precision_score(
            labels,
            (predictions >= 0.5).astype(int),
            average="weighted",
            zero_division=0,
        )
        assert np.isclose(result, expected_precision)

    def test_binary_classification_no_positive_predictions(self):
        predictions = np.array([0.0, 0.0, 0.0, 0.0])
        labels = np.array([1, 0, 1, 0])
        result = calculate_precision(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        expected_precision = precision_score(
            labels, (predictions >= 0.5).astype(int), zero_division=0
        )
        assert np.isclose(result, expected_precision)

    def test_binary_classification_no_positive_labels(self):
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 0])
        result = calculate_precision(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        # Precision is ill-defined and set to 0.0 due to no positive labels
        assert np.isclose(result, 0.0)


class TestCalculateF1Score:
    def test_binary_classification_perfect(self):
        predictions = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 1, 0])
        result = calculate_f1_score(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = np.array([0.6, 0.4, 0.3, 0.7])
        labels = np.array([1, 0, 1, 0])
        result = calculate_f1_score(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 0.5)

    def test_binary_classification_all_zeros(self):
        predictions = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0])
        result = calculate_f1_score(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        # F1 score is ill-defined and set to 0.0 due to no positive labels
        assert np.isclose(result, 0.0)

    def test_binary_classification_all_ones(self):
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 1])
        result = calculate_f1_score(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_perfect(self):
        predictions = np.array([[0.9, 0.8], [0.8, 0.9], [0.7, 0.6]])
        labels = np.array([[1, 1], [1, 1], [1, 1]])
        result = calculate_f1_score(
            predictions,
            labels,
            task="multilabel",
            threshold=0.5,
        )
        assert np.allclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        predictions = np.array([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]])
        labels = np.array([[1, 0], [0, 1], [1, 0]])
        result = calculate_f1_score(
            predictions,
            labels,
            task="multilabel",
            threshold=0.5,
            averaging_method="macro",
        )
        expected_f1 = f1_score(
            labels, (predictions >= 0.5).astype(int), average="macro", zero_division=0
        )
        assert np.isclose(result, expected_f1, atol=1e-4)

    def test_incorrect_shapes(self):
        predictions = np.array([0.9, 0.2, 0.8])
        labels = np.array([1, 0])
        with pytest.raises(ValueError):
            calculate_f1_score(
                predictions,
                labels,
                task="binary",
                threshold=0.5,
            )

    def test_invalid_threshold(self):
        predictions = np.array([0.9, 0.2, 0.8, 0.1])
        labels = np.array([1, 0, 1, 0])
        with pytest.raises(ValueError):
            calculate_f1_score(
                predictions,
                labels,
                task="binary",
                threshold=1.5,
            )

    def test_non_array_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        result = calculate_f1_score(
            np.array(predictions),
            np.array(labels),
            task="binary",
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_empty_arrays(self):
        predictions = np.array([])
        labels = np.array([])
        with pytest.raises(ValueError):
            calculate_f1_score(
                predictions,
                labels,
                task="binary",
                threshold=0.5,
            )

    def test_multilabel_classification_weighted_average(self):
        predictions = np.array([[0.9, 0.1], [0.8, 0.2], [0.5, 0.5]])
        labels = np.array([[1, 0], [1, 0], [1, 1]])
        result = calculate_f1_score(
            predictions,
            labels,
            task="multilabel",
            threshold=0.5,
            averaging_method="weighted",
        )
        expected_f1 = f1_score(
            labels,
            (predictions >= 0.5).astype(int),
            average="weighted",
            zero_division=0,
        )
        assert np.isclose(result, expected_f1)

    def test_binary_classification_no_positive_labels(self):
        predictions = np.array([0.0, 0.0, 0.0, 0.0])
        labels = np.array([0, 0, 0, 0])
        result = calculate_f1_score(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        # F1 score is ill-defined and set to 0.0 due to no positive labels
        assert np.isclose(result, 0.0)

    def test_binary_classification_no_positive_predictions(self):
        predictions = np.array([0.0, 0.0, 0.0, 0.0])
        labels = np.array([1, 0, 1, 0])
        result = calculate_f1_score(
            predictions,
            labels,
            task="binary",
            threshold=0.5,
        )
        expected_f1 = f1_score(
            labels, (predictions >= 0.5).astype(int), zero_division=0
        )
        assert np.isclose(result, expected_f1)


class TestCalculateAveragePrecision:
    def test_binary_classification_perfect(self):
        predictions = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([1, 1, 1, 1])
        result = calculate_average_precision(
            predictions,
            labels,
            task="binary",
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = np.array([0.9, 0.6, 0.3, 0.1])
        labels = np.array([1, 0, 1, 0])
        result = calculate_average_precision(
            predictions,
            labels,
            task="binary",
        )
        expected_ap = average_precision_score(labels, predictions)
        assert np.isclose(result, expected_ap)

    def test_binary_classification_all_zeros(self):
        predictions = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0])
        result = calculate_average_precision(
            predictions,
            labels,
            task="binary",
        )
        # AP is undefined when there are no positive labels; scikit-learn returns 0.0
        assert np.isclose(result, 0.0)

    def test_binary_classification_all_ones(self):
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 1])
        result = calculate_average_precision(
            predictions,
            labels,
            task="binary",
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_perfect(self):
        predictions = np.array([[0.9, 0.9], [0.8, 0.8], [0.7, 0.7]])
        labels = np.array([[1, 1], [1, 1], [1, 1]])
        result = calculate_average_precision(
            predictions,
            labels,
            task="multilabel",
            averaging_method="macro",
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        predictions = np.array(
            [
                [0.8, 0.2],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        )
        labels = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 1],
                [1, 0],
            ]
        )
        result = calculate_average_precision(
            predictions,
            labels,
            task="multilabel",
            averaging_method="macro",
        )
        expected_ap = average_precision_score(labels, predictions, average="macro")
        assert np.isclose(result, expected_ap)

    def test_incorrect_shapes(self):
        predictions = np.array([0.9, 0.2, 0.8])
        labels = np.array([1, 0])
        with pytest.raises(ValueError):
            calculate_average_precision(
                predictions,
                labels,
                task="binary",
            )

    def test_non_array_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        result = calculate_average_precision(
            np.array(predictions),
            np.array(labels),
            task="binary",
        )
        expected_ap = average_precision_score(labels, predictions)
        assert np.isclose(result, expected_ap)

    def test_empty_arrays(self):
        predictions = np.array([])
        labels = np.array([])
        with pytest.raises(ValueError):
            calculate_average_precision(
                predictions,
                labels,
                task="binary",
            )

    def test_binary_classification_no_positive_labels(self):
        predictions = np.array([0.2, 0.3, 0.4, 0.1])
        labels = np.array([0, 0, 0, 0])
        result = calculate_average_precision(
            predictions,
            labels,
            task="binary",
        )
        # AP is undefined when there are no positive labels; scikit-learn returns 0.0
        assert np.isclose(result, 0.0)

    def test_multilabel_classification_no_positive_labels(self):
        predictions = np.array([[0.1, 0.2], [0.3, 0.4]])
        labels = np.array([[0, 0], [0, 0]])
        result = calculate_average_precision(
            predictions,
            labels,
            task="multilabel",
            averaging_method="macro",
        )
        # AP is undefined when there are no positive labels; scikit-learn returns 0.0 for each class
        expected_result = np.array([0.0, 0.0])
        assert np.allclose(result, expected_result)

    def test_binary_classification_perfect_inverse(self):
        predictions = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0])
        result = calculate_average_precision(
            1 - predictions,
            1 - labels,
            task="binary",
        )
        assert np.isclose(result, 1.0)


class TestCalculateAUROC:
    def test_binary_classification_perfect(self):
        predictions = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([1, 1, 0, 0])
        result = calculate_auroc(
            predictions,
            labels,
            task="binary",
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = np.array([0.9, 0.6, 0.3, 0.1])
        labels = np.array([1, 0, 1, 0])
        result = calculate_auroc(
            predictions,
            labels,
            task="binary",
        )
        expected_auc = roc_auc_score(labels, predictions)
        assert np.isclose(result, expected_auc)

    def test_binary_classification_all_zeros(self):
        predictions = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0])
        result = calculate_auroc(
            predictions,
            labels,
            task="binary",
        )
        assert np.isnan(result)

    def test_binary_classification_all_ones(self):
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 1])
        result = calculate_auroc(
            predictions,
            labels,
            task="binary",
        )
        assert np.isnan(result)

    def test_multilabel_classification_perfect(self):
        predictions = np.array(
            [
                [0.99, 0.99],
                [0.99, 0.01],
                [0.01, 0.99],
                [0.99, 0.99],
                [0.01, 0.01],
            ]
        )
        labels = np.array(
            [
                [1, 1],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0],
            ]
        )
        result = calculate_auroc(
            predictions,
            labels,
            task="multilabel",
        )
        expected_auc = roc_auc_score(labels, predictions, average="macro")
        assert np.isclose(result, expected_auc)

    def test_multilabel_classification_imperfect(self):
        predictions = np.array(
            [
                [0.8, 0.2],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        )
        labels = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 1],
                [1, 0],
            ]
        )
        result = calculate_auroc(
            predictions,
            labels,
            task="multilabel",
        )
        expected_auc = roc_auc_score(labels, predictions, average="macro")
        assert np.isclose(result, expected_auc, atol=1e-4)

    def test_incorrect_shapes(self):
        predictions = np.array([0.9, 0.2])
        labels = np.array([1])
        with pytest.raises(ValueError):
            calculate_auroc(
                predictions,
                labels,
                task="binary",
            )

    def test_non_array_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        result = calculate_auroc(
            np.array(predictions),
            np.array(labels),
            task="binary",
        )
        expected_auc = roc_auc_score(labels, predictions)
        assert np.isclose(result, expected_auc)

    def test_empty_arrays(self):
        predictions = np.array([])
        labels = np.array([])
        with pytest.raises(ValueError):
            calculate_auroc(
                predictions,
                labels,
                task="binary",
            )

    def test_multilabel_classification_no_positive_labels(self):
        predictions = np.array([[0.1, 0.2], [0.2, 0.1]])
        labels = np.array([[0, 0], [0, 0]])
        result = calculate_auroc(
            predictions,
            labels,
            task="multilabel",
        )
        assert np.all(np.isnan(result))

    def test_binary_classification_no_positive_labels(self):
        predictions = np.array([0.2, 0.3, 0.4, 0.1])
        labels = np.array([0, 0, 0, 0])
        result = calculate_auroc(
            predictions,
            labels,
            task="binary",
        )
        assert np.isnan(result)

    def test_binary_classification_no_negative_labels(self):
        predictions = np.array([0.2, 0.3, 0.4, 0.1])
        labels = np.array([1, 1, 1, 1])
        result = calculate_auroc(
            predictions,
            labels,
            task="binary",
        )
        assert np.isnan(result)

    def test_multilabel_classification_mixed_classes(self):
        predictions = np.array([[0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
        labels = np.array([[1, 0], [1, 0], [0, 1]])
        result = calculate_auroc(
            predictions,
            labels,
            task="multilabel",
            averaging_method="macro",
        )
        expected_result = roc_auc_score(labels, predictions, average="macro")
        assert np.isclose(result, expected_result)
