import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

from birdnet_analyzer import base_models
from birdnet_analyzer.model import (
    build_linear_classifier,
    focal_loss,
    get_empty_class_exception,
    label_smoothing,
    random_multilabel_split,
    random_split,
    train_linear_classifier,
    upsampling,
)


def test_random_split_adds_negative_samples_only_once():
    x = np.arange(10).reshape(-1, 1)
    y = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
        ]
    )

    x_train, _, x_val, y_val = random_split(
        x, y, np.random.default_rng(42), val_ratio=0.5
    )

    assert len(x_train) + len(x_val) == len(x)
    assert len(np.unique(x_train)) == len(x_train)
    assert not np.intersect1d(x_train, x_val).size
    assert not np.any(y_val == -1)


def test_random_multilabel_split_keeps_negative_combinations_in_training():
    x = np.arange(10).reshape(-1, 1)
    y = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    x_train, _, x_val, y_val = random_multilabel_split(
        x, y, np.random.default_rng(42), val_ratio=0.5
    )

    assert len(x_train) + len(x_val) == len(x)
    assert not np.intersect1d(x_train, x_val).size
    assert {6, 7}.issubset(set(x_train[:, 0]))
    assert not np.any(y_val == -1)


@pytest.mark.parametrize("mode", ["repeat", "mean", "linear"])
def test_upsampling_balances_each_class_to_requested_ratio(mode):
    x = np.arange(12, dtype="float32").reshape(6, 2)
    y = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1]])

    x_upsampled, y_upsampled = upsampling(
        x, y, np.random.default_rng(42), is_binary=False, ratio=1.0, mode=mode
    )

    assert len(x_upsampled) == 10
    assert np.array_equal(y_upsampled.sum(axis=0), [5, 5])


@pytest.mark.parametrize("mode", ["repeat", "mean", "linear"])
def test_upsampling_rejects_empty_classes(mode):
    x = np.arange(8, dtype="float32").reshape(4, 2)
    y = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])

    with pytest.raises(get_empty_class_exception()) as error:
        upsampling(
            x, y, np.random.default_rng(42), is_binary=False, ratio=1.0, mode=mode
        )

    assert error.value.index == 1


def test_smote_only_interpolates_between_samples_of_the_same_class():
    x = np.array([[0.0], [1.0], [100.0], [101.0], [102.0]])
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])

    x_upsampled, y_upsampled = upsampling(
        x, y, np.random.default_rng(42), is_binary=False, ratio=1.0, mode="smote"
    )

    assert len(x_upsampled) == 6
    assert np.array_equal(y_upsampled.sum(axis=0), [3, 3])
    assert 0 <= x_upsampled[-1, 0] <= 1


def test_label_smoothing_spreads_alpha_over_the_classes():
    y = np.zeros((1000, 5), dtype="float32")
    y[np.arange(1000), np.arange(1000) % 5] = 1

    smoothed = label_smoothing(y.copy(), alpha=0.1)

    np.testing.assert_allclose(smoothed[y == 0], 0.1 / 5)
    np.testing.assert_allclose(smoothed[y == 1], 1 - 0.1 + 0.1 / 5)


def test_label_smoothing_keeps_the_order_of_mixed_labels():
    y = np.array([[0.0, 0.05, 0.1, 0.11, 0.15, 1.0]], dtype="float32")

    smoothed = label_smoothing(y, alpha=0.1)

    assert np.all(np.diff(smoothed[0]) > 0)


def test_label_smoothing_keeps_mixed_labels_non_negative_and_negatives_intact():
    y = np.array([[0.95, 0.05, 0.0, -1.0]], dtype="float32")

    smoothed = label_smoothing(y, alpha=0.1)

    assert smoothed[0, 1] >= 0
    assert smoothed[0, 3] == -1


def test_focal_loss_is_finite_for_a_saturated_wrong_prediction():
    loss = focal_loss(tf.constant([[0.0]]), tf.constant([[1.0]]))

    assert np.isfinite(loss.numpy()).all()


def test_the_auc_metrics_score_the_classifiers_probabilities():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)].astype("float32").reshape(-1, 1)
    p = np.r_[rng.uniform(0.001, 0.06, 200), rng.uniform(0.04, 0.1, 200)]
    p = p.astype("float32").reshape(-1, 1)
    logits = np.log(p / (1 - p))

    classifier, _ = train_linear_classifier(
        build_linear_classifier(1, 1),
        logits,
        y,
        np.array([]),
        np.array([]),
        epochs=1,
        batch_size=32,
        learning_rate=0.001,
        val_split=0.2,
        upsampling_ratio=0.0,
        upsampling_mode="repeat",
        train_with_mixup=False,
        train_with_label_smoothing=False,
    )
    classifier.layers[0].set_weights([np.ones((1, 1)), np.zeros(1)])
    scores = classifier.evaluate(logits, y, return_dict=True, verbose=0)

    assert scores["AUPRC"] == pytest.approx(average_precision_score(y, p), abs=0.005)
    assert scores["AUROC"] == pytest.approx(roc_auc_score(y, p), abs=0.005)


def test_early_stopping_keeps_the_best_epoch_of_the_monitored_metric(monkeypatch):
    import keras

    stoppers = []

    class RecordingEarlyStopping(keras.callbacks.EarlyStopping):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            stoppers.append(self)

    monkeypatch.setattr(keras.callbacks, "EarlyStopping", RecordingEarlyStopping)
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    y = np.repeat(np.eye(2, dtype="float32"), 100, axis=0)
    x = (rng.normal(size=(200, 32)) + 0.5 * y[:, :1]).astype("float32")
    train, val = np.r_[0:10, 100:110], np.r_[10:100, 110:200]

    classifier, history = train_linear_classifier(
        build_linear_classifier(2, 32, hidden_units=64),
        x[train],
        y[train],
        x[val],
        y[val],
        epochs=40,
        batch_size=4,
        learning_rate=0.05,
        val_split=0.0,
        upsampling_ratio=0.0,
        upsampling_mode="repeat",
        train_with_mixup=False,
        train_with_label_smoothing=False,
        monitor="val_AUPRC",
        monitor_mode="max",
    )
    scores = history.history["val_AUPRC"]
    kept = classifier.evaluate(x[val], y[val], return_dict=True, verbose=0)["AUPRC"]

    assert stoppers[-1].monitor == "val_AUPRC"
    assert history.restored_epoch == stoppers[-1].best_epoch < len(scores) - 1
    assert scores[history.restored_epoch] == max(scores)
    assert kept == pytest.approx(scores[history.restored_epoch], abs=1e-6)
    assert kept != pytest.approx(scores[-1], abs=1e-3)


def _two_class_data():
    rng = np.random.default_rng(0)
    y = np.repeat(np.eye(2, dtype="float32"), 60, axis=0)
    x = (rng.normal(size=(120, 8)) + 1.5 * y[:, :1]).astype("float32")

    return x, y


def _train(classifier, x, y, **kwargs):
    return train_linear_classifier(
        classifier,
        x,
        y,
        np.array([]),
        np.array([]),
        epochs=2,
        batch_size=16,
        learning_rate=0.01,
        val_split=0.2,
        upsampling_ratio=0.0,
        upsampling_mode="repeat",
        train_with_mixup=False,
        **kwargs,
    )


@pytest.mark.parametrize("before", ["0", None])
def test_a_failed_training_restores_the_visible_cuda_devices(monkeypatch, before):
    import os

    if before is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", before)

    classifier = build_linear_classifier(2, 8)
    during = []

    def fail(*args, **kwargs):
        during.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        raise RuntimeError("boom")

    monkeypatch.setattr(classifier, "fit", fail)

    with pytest.raises(RuntimeError, match="boom"):
        _train(classifier, *_two_class_data(), train_with_label_smoothing=False)

    assert during == [""]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before


def test_the_training_metrics_score_smoothed_labels_as_their_hard_labels():
    import keras

    keras.utils.set_random_seed(0)

    _, history = _train(
        build_linear_classifier(2, 8),
        *_two_class_data(),
        train_with_label_smoothing=True,
    )

    assert history.history["AUROC"][-1] > 0.5
    assert history.history["AUPRC"][-1] < 1.0


def test_the_auc_metrics_count_a_negative_label_as_a_negative():
    from birdnet_analyzer.model import HardLabelAUC

    y_pred = np.array([[0.8], [0.9], [0.1]], dtype="float32")
    with_negative = HardLabelAUC(curve="ROC")
    with_negative.update_state(np.array([[1], [-1], [0]], "float32"), y_pred)
    with_zero = HardLabelAUC(curve="ROC")
    with_zero.update_state(np.array([[1], [0], [0]], "float32"), y_pred)

    assert float(with_negative.result()) == pytest.approx(float(with_zero.result()))


def test_base_models_carry_the_library_constants():
    from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
    from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0

    two_four = base_models.base_model("2.4")
    three_zero = base_models.base_model("3.0")

    assert two_four.segment_samples == AcousticModelV2_4.get_segment_size_samples()
    assert two_four.sample_rate == AcousticModelV2_4.get_sample_rate()
    assert two_four.embeddings_dim == AcousticModelV2_4.get_embeddings_dim()
    assert not two_four.outputs_probabilities
    assert three_zero.segment_samples == AcousticModelV3_0.get_segment_size_samples()
    assert three_zero.sample_rate == AcousticModelV3_0.get_sample_rate()
    assert three_zero.embeddings_dim == AcousticModelV3_0.get_embeddings_dim()
    assert three_zero.outputs_probabilities


def test_an_unknown_base_version_is_rejected():
    with pytest.raises(ValueError, match=r"9\.9"):
        base_models.base_model("9.9")


def test_embeddings_width_identifies_the_base_model():
    assert base_models.version_for_embeddings_dim(1024) == "2.4"
    assert base_models.version_for_embeddings_dim(1280) == "3.0"

    with pytest.raises(ValueError, match="512"):
        base_models.version_for_embeddings_dim(512)
