import matplotlib
import numpy as np
import pandas as pd
import pytest

from birdnet_analyzer import evaluation
from birdnet_analyzer.evaluation.assessment import performance_assessor, plotting
from birdnet_analyzer.evaluation.assessment.performance_assessor import (
    PerformanceAssessor,
)

matplotlib.use("Agg")


def _tensors(num_classes=3):
    rng = np.random.default_rng(3)

    return rng.random((200, num_classes)), rng.integers(0, 2, (200, num_classes))


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"num_classes": 0}, "positive integer"),
        ({"num_classes": 3, "threshold": 1.5}, "between 0 and 1"),
        ({"num_classes": 2, "classes": ("a", "b", "c")}, "must match num_classes"),
        ({"num_classes": 2, "task": "multiclass"}, "'binary' or 'multilabel'"),
        ({"num_classes": 2, "metrics_list": ()}, "cannot be empty"),
        ({"num_classes": 2, "metrics_list": ("recall", "map")}, "Invalid metrics"),
    ],
    ids=["classes", "threshold", "class-names", "task", "no-metric", "unknown-metric"],
)
def test_settings_the_assessor_cannot_score_with_are_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        PerformanceAssessor(**settings)


@pytest.mark.parametrize(
    ("predictions_shape", "labels_shape", "message"),
    [((90, 3), (100, 3), "same shape"), ((100, 2), (100, 2), "must match num_classes")],
)
def test_tensors_of_the_wrong_shape_are_rejected(
    predictions_shape, labels_shape, message
):
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match=message):
        PerformanceAssessor(num_classes=3).calculate_metrics(
            rng.random(predictions_shape), rng.integers(0, 2, labels_shape)
        )


def test_the_rows_are_the_metrics_and_the_columns_the_classes_or_overall():
    predictions, labels = _tensors(2)
    named = PerformanceAssessor(num_classes=2, classes=("a", "b"))
    unnamed = PerformanceAssessor(num_classes=2)

    per_class = named.calculate_metrics(predictions, labels, per_class_metrics=True)
    numbered = unnamed.calculate_metrics(predictions, labels, per_class_metrics=True)
    overall = named.calculate_metrics(predictions, labels)

    assert per_class.columns.tolist() == ["a", "b"]
    assert numbered.columns.tolist() == ["Class 0", "Class 1"]
    assert overall.columns.tolist() == ["Overall"]
    assert overall.index.tolist() == [
        "Recall",
        "Precision",
        "F1",
        "AP",
        "AUROC",
        "Accuracy",
    ]


def _with_an_empty_class():
    # A: two positives, both found. Empty: never annotated, predicted three times.
    # C: one positive, found.
    labels = np.zeros((10, 3), dtype=int)
    labels[:2, 0] = 1
    labels[0, 2] = 1
    predictions = np.zeros((10, 3))
    predictions[:2, 0] = 0.9
    predictions[:3, 1] = 0.9
    predictions[0, 2] = 0.9
    assessor = PerformanceAssessor(
        num_classes=3,
        threshold=0.1,
        classes=("A", "Empty", "C"),
        metrics_list=("precision", "recall"),
    )

    return assessor, predictions, labels


def test_the_classes_without_a_positive_label_and_the_support_are_reported():
    assessor, _, labels = _with_an_empty_class()

    assert assessor.empty_classes(labels) == ("Empty",)
    assert assessor.class_support(labels).tolist() == [2, 0, 1]


def test_nothing_annotated_scores_nan_rather_than_zero():
    assessor, predictions, _ = _with_an_empty_class()

    overall = assessor.calculate_metrics(predictions, np.zeros((10, 3), dtype=int))

    assert np.isnan(overall.loc["Precision", "Overall"])


@pytest.mark.parametrize(
    ("classes", "task"), [(("a",), "binary"), (("a", "b", "c"), "multilabel")]
)
def test_the_plots_hold_one_panel_per_class_and_one_line_per_threshold_metric(
    classes, task
):
    predictions, labels = _tensors(len(classes))
    assessor = PerformanceAssessor(
        num_classes=len(classes),
        classes=classes,
        task=task,
        metrics_list=("recall", "precision", "ap"),
    )

    overall = assessor.plot_metrics(predictions, labels)
    per_class = assessor.plot_metrics(predictions, labels, per_class_metrics=True)
    sweep = assessor.plot_metrics_all_thresholds(predictions, labels)
    sweep_per_class = assessor.plot_metrics_all_thresholds(
        predictions, labels, per_class_metrics=True
    )
    confusion = assessor.plot_confusion_matrix(predictions, labels)

    assert len(overall.axes[0].patches) == 3
    assert len(per_class.axes[0].lines) == 3
    assert [len(ax.lines) for ax in sweep.axes] == [2]
    assert [ax.get_title() for ax in sweep_per_class.axes] == list(classes)
    assert [len(ax.lines) for ax in sweep_per_class.axes] == [2] * len(classes)
    assert len(confusion.axes) == len(classes)


def _recorded_sweep(monkeypatch, assessor, predictions, labels, per_class):
    recorded = []
    for name in (
        "plot_metrics_across_thresholds",
        "plot_metrics_across_thresholds_per_class",
    ):
        monkeypatch.setattr(plotting, name, lambda *args: recorded.append(args))

    assessor.plot_metrics_all_thresholds(
        predictions, labels, per_class_metrics=per_class
    )
    (args,) = recorded

    return args


@pytest.mark.parametrize("per_class", [False, True])
def test_metrics_at_another_threshold_match_an_assessor_built_with_it(per_class):
    predictions, labels = _tensors()
    assessor = PerformanceAssessor(num_classes=3, threshold=0.5)

    at_other = assessor.calculate_metrics(
        predictions, labels, per_class_metrics=per_class, threshold=0.7
    )

    expected = PerformanceAssessor(num_classes=3, threshold=0.7).calculate_metrics(
        predictions, labels, per_class_metrics=per_class
    )
    pd.testing.assert_frame_equal(at_other, expected)
    assert not at_other.equals(
        assessor.calculate_metrics(predictions, labels, per_class_metrics=per_class)
    )
    assert assessor.threshold == 0.5


@pytest.mark.parametrize("empty_class", [False, True])
@pytest.mark.parametrize("per_class", [False, True])
def test_the_sweep_plots_each_threshold_dependent_metric_at_each_threshold(
    monkeypatch, per_class, empty_class
):
    predictions, labels = _tensors()
    if empty_class:
        labels[:, 1] = 0
    assessor = PerformanceAssessor(
        num_classes=3,
        classes=("a", "b", "c"),
        metrics_list=("auroc", "f1", "ap", "recall"),
    )

    thresholds, values, plotted, *_ = _recorded_sweep(
        monkeypatch, assessor, predictions, labels, per_class
    )

    assert plotted == ["f1", "recall"]
    assert assessor.threshold == 0.5
    expected = [
        assessor.calculate_metrics(
            predictions, labels, per_class_metrics=per_class, threshold=threshold
        )
        for threshold in thresholds
    ]
    for metric, row in (("f1", "F1"), ("recall", "Recall")):
        if per_class:
            for name in ("a", "b", "c"):
                np.testing.assert_array_equal(
                    values[name][metric], [df.loc[row, name] for df in expected]
                )
        else:
            np.testing.assert_array_equal(
                values[metric], [df.loc[row, "Overall"] for df in expected]
            )


def test_the_cli_offers_the_metrics_the_assessor_registers():
    assert tuple(performance_assessor.METRICS) == evaluation.VALID_METRICS
    assert evaluation.VALID_METRICS == (
        "accuracy",
        "recall",
        "precision",
        "f1",
        "ap",
        "auroc",
    )
