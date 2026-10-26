import warnings

import numpy as np
import pytest

from birdnet_analyzer.evaluation.assessment import metrics

PERFECT = (np.array([0.9, 0.1, 0.8, 0.2]), np.array([1, 0, 1, 0]))
IMPERFECT = (np.array([0.6, 0.4, 0.3, 0.7]), np.array([1, 0, 1, 0]))
AT_THRESHOLD = (np.array([0.9, 0.5, 0.3, 0.7]), np.array([1, 0, 1, 0]))
NO_POSITIVES = (np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 0, 0]))
# Class 0: three positives, two of them found, two false alarms. Class 1: perfect.
MULTILABEL = (
    np.array([[0.9, 0.2], [0.8, 0.9], [0.3, 0.6], [0.7, 0.4], [0.6, 0.1]]),
    np.array([[1, 0], [1, 1], [1, 1], [0, 0], [0, 0]]),
)
# Class 1 is positive in every window, so its ranking cannot be scored.
ALWAYS_PRESENT = (
    np.array([[0.9, 0.2], [0.8, 0.9], [0.3, 0.6]]),
    np.array([[1, 1], [0, 1], [1, 1]]),
)

BINARY = {"task": "binary", "threshold": 0.5}
MULTI = {"task": "multilabel", "threshold": 0.5}


@pytest.mark.parametrize(
    ("score", "data", "kwargs", "expected"),
    [
        pytest.param(metrics.calculate_recall, PERFECT, BINARY, 1.0, id="perfect"),
        pytest.param(
            metrics.calculate_precision, IMPERFECT, BINARY, 0.5, id="precision"
        ),
        pytest.param(metrics.calculate_f1_score, IMPERFECT, BINARY, 0.5, id="f1"),
        pytest.param(
            metrics.calculate_accuracy,
            IMPERFECT,
            {**BINARY, "num_classes": 1},
            0.5,
            id="accuracy",
        ),
        pytest.param(
            metrics.calculate_average_precision,
            IMPERFECT,
            {"task": "binary"},
            0.5,
            id="ap",
        ),
        pytest.param(
            metrics.calculate_auroc, IMPERFECT, {"task": "binary"}, 0.25, id="auroc"
        ),
        pytest.param(
            metrics.calculate_precision,
            AT_THRESHOLD,
            BINARY,
            1 / 3,
            id="a score at the threshold is a detection",
        ),
        pytest.param(
            metrics.calculate_recall,
            NO_POSITIVES,
            BINARY,
            0.0,
            id="recall without positives is 0",
        ),
        pytest.param(
            metrics.calculate_precision,
            NO_POSITIVES,
            BINARY,
            0.0,
            id="precision without detections is 0",
        ),
        pytest.param(
            metrics.calculate_auroc,
            NO_POSITIVES,
            {"task": "binary"},
            np.nan,
            id="auroc of a single class is nan",
        ),
        pytest.param(
            metrics.calculate_auroc,
            ALWAYS_PRESENT,
            {"task": "multilabel", "averaging_method": "none"},
            [0.5, np.nan],
            id="auroc of an always present class is nan",
        ),
        pytest.param(
            metrics.calculate_auroc,
            ALWAYS_PRESENT,
            {"task": "multilabel"},
            np.nan,
            id="macro auroc with an always present class is nan",
        ),
        pytest.param(
            metrics.calculate_precision,
            MULTILABEL,
            {**MULTI, "averaging_method": "none"},
            [0.5, 1.0],
            id="one precision per class",
        ),
        pytest.param(
            metrics.calculate_precision,
            MULTILABEL,
            {**MULTI, "averaging_method": "macro"},
            0.75,
            id="macro precision",
        ),
        pytest.param(
            metrics.calculate_precision,
            MULTILABEL,
            {**MULTI, "averaging_method": "micro"},
            2 / 3,
            id="micro precision",
        ),
        pytest.param(
            metrics.calculate_precision,
            MULTILABEL,
            {**MULTI, "averaging_method": "weighted"},
            0.7,
            id="weighted precision",
        ),
        pytest.param(
            metrics.calculate_recall,
            MULTILABEL,
            {**MULTI, "averaging_method": "macro"},
            5 / 6,
            id="macro recall",
        ),
        pytest.param(
            metrics.calculate_f1_score,
            MULTILABEL,
            {**MULTI, "averaging_method": "macro"},
            11 / 14,
            id="macro f1",
        ),
        pytest.param(
            metrics.calculate_average_precision,
            MULTILABEL,
            {"task": "multilabel", "averaging_method": "macro"},
            14 / 15,
            id="macro ap",
        ),
        pytest.param(
            metrics.calculate_auroc,
            MULTILABEL,
            {"task": "multilabel"},
            5 / 6,
            id="macro auroc",
        ),
        pytest.param(
            metrics.calculate_accuracy,
            MULTILABEL,
            {**MULTI, "num_classes": 2, "averaging_method": "none"},
            [0.4, 1.0],
            id="one accuracy per class",
        ),
        pytest.param(
            metrics.calculate_accuracy,
            MULTILABEL,
            {**MULTI, "num_classes": 2, "averaging_method": "weighted"},
            0.64,
            id="weighted accuracy",
        ),
    ],
)
def test_a_hand_counted_example_scores_as_counted_without_a_warning(
    score, data, kwargs, expected
):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = score(*data, **kwargs)

    np.testing.assert_allclose(result, np.atleast_1d(expected))


@pytest.mark.parametrize(
    ("score", "scores", "labels", "kwargs", "message"),
    [
        (metrics.calculate_recall, [], [], BINARY, "must not be empty"),
        (metrics.calculate_recall, [0.9, 0.2, 0.8], [1, 0], BINARY, "same shape"),
        (
            metrics.calculate_recall,
            [0.9, 0.2],
            [1, 0],
            {"task": "binary", "threshold": 1.5},
            "Invalid threshold",
        ),
        (metrics.calculate_auroc, [0.9, 0.2, 0.8], [1, 0], {"task": "binary"}, "shape"),
    ],
    ids=["empty", "shape", "threshold", "shape-without-threshold"],
)
def test_unscorable_inputs_are_rejected(score, scores, labels, kwargs, message):
    with pytest.raises(ValueError, match=message):
        score(np.array(scores), np.array(labels), **kwargs)
