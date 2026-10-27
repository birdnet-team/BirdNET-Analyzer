import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from birdnet_analyzer.cli import train_parser
from birdnet_analyzer.train.core import train
from tests.models import pb_model_cached


@patch("birdnet_analyzer.train.utils.train_model")
def test_train_cli_accepts_full_parser_surface(mock_train_model, tmp_path):
    input_dir, test_data, classifier_output = (
        str(tmp_path / name) for name in ("input", "test", "cc")
    )
    cache_path = str(tmp_path / "train_cache.npz")
    args = train_parser().parse_args(
        [
            input_dir,
            "--test_data",
            test_data,
            "--crop_mode",
            "smart",
            "--birdnet",
            "2.4",
            "-o",
            classifier_output,
            "--epochs",
            "3",
            "--val_split",
            "0.25",
            "--learning_rate",
            "0.001",
            "--focal-loss",
            "--focal-loss-gamma",
            "2.5",
            "--focal-loss-alpha",
            "0.5",
            "--hidden_units",
            "16",
            "--dropout",
            "0.3",
            "--label_smoothing",
            "--mixup",
            "--upsampling_ratio",
            "0.5",
            "--upsampling_mode",
            "mean",
            "--model_formats",
            "tflite",
            "raven",
            "detached",
            "--model_save_mode",
            "append",
            "--save_cache_to",
            cache_path,
            "--fmin",
            "100",
            "--fmax",
            "10000",
            "--audio_speed",
            "1.1",
            "--threads",
            "2",
            "--overlap",
            "1.5",
            "-b",
            "4",
            "--autotune",
            "--autotune_trials",
            "3",
            "--autotune_n_repeats",
            "2",
            "--autotune_n_splits",
            "2",
            "--autotune_metric",
            "val_AUROC",
        ]
    )
    kwargs = vars(args)
    kwargs.pop("load_params")

    train(**kwargs)

    mock_train_model.assert_called_once()
    call_args, call_kwargs = mock_train_model.call_args
    assert call_args[0] == input_dir
    assert call_kwargs["output"] == classifier_output
    assert call_kwargs["test_data"] == test_data
    assert call_kwargs["crop_mode"] == "smart"
    assert call_kwargs["model_formats"] == ["tflite", "raven", "detached"]
    assert call_kwargs["model_save_mode"] == "append"
    assert call_kwargs["save_cache_to"] == cache_path
    assert call_kwargs["autotune"] is True
    assert call_kwargs["autotune_metric"] == "val_AUROC"
    assert call_kwargs["birdnet"] == "2.4"


@pytest.mark.skipif(not pb_model_cached("3.0"), reason="3.0 pb model not cached")
def test_a_classifier_trained_on_the_example_folder_analyses_the_soundscape(tmp_path):
    from birdnet_analyzer import analyze, params
    from birdnet_analyzer.config import OUTPUT_CSV_FILENAME, TRAIN_PARAMS_SUFFIX

    classifier = str(tmp_path / "cc")

    train("tests/data/training", output=classifier, epochs=1, model_formats=["tflite"])
    analyze(
        "birdnet_analyzer/example/soundscape.wav",
        str(tmp_path / "out"),
        classifier=classifier + ".tflite",
        rtype="csv",
        min_conf=0,
    )

    detections = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_FILENAME)
    assert len(detections) > 0
    assert set(detections["Common name"]) <= {"a", "b", "c"}
    assert params.load_train_params(classifier + TRAIN_PARAMS_SUFFIX)["epochs"] == 1


@patch("birdnet_analyzer.train.utils.train_model")
def test_the_cli_writes_no_training_cache_unless_asked(mock_train_model, tmp_path):
    kwargs = vars(train_parser().parse_args([str(tmp_path)]))
    kwargs.pop("load_params")

    train(**kwargs)

    assert mock_train_model.call_args[1]["save_cache_to"] is None


def test_the_api_and_the_cli_default_to_the_same_number_of_autotune_folds():
    api_default = inspect.signature(train).parameters["autotune_n_splits"].default

    assert api_default == train_parser().get_default("autotune_n_splits")


def test_val_loss_is_not_an_autotune_metric_on_the_cli():
    with pytest.raises(SystemExit):
        train_parser().parse_args(["input", "--autotune_metric", "val_loss"])


def test_val_loss_is_rejected_before_the_data_is_loaded(monkeypatch, tmp_path):
    from birdnet_analyzer.train import utils as train_utils

    def load(*args, **kwargs):
        pytest.fail("the training data was loaded")

    monkeypatch.setattr(train_utils, "_load_training_data", load)

    with pytest.raises(ValueError, match="val_loss"):
        train_utils.train_model(
            "unused",
            output=str(tmp_path / "cc"),
            autotune=True,
            autotune_metric="val_loss",
        )


def test_binary_data_is_stratified_on_its_column():
    from birdnet_analyzer.train.utils import _stratification_labels

    y = np.array([[1], [0], [0], [1]], dtype="float32")

    assert _stratification_labels(y, True, False).tolist() == [1, 0, 0, 1]


def test_single_label_strata_keep_non_events_and_negatives_apart():
    from birdnet_analyzer.train.utils import _stratification_labels

    y = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0], [-1, 0, 0], [0, 1, 0]])

    strata = _stratification_labels(y, False, False)

    assert strata[[0, 4, 1]].tolist() == [0, 1, 2]
    assert len(set(strata.tolist())) == 5


def test_multi_label_strata_count_the_active_labels():
    from birdnet_analyzer.train.utils import _stratification_labels

    y = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0], [1, -1, 1]])

    assert _stratification_labels(y, False, True).tolist() == [2, 1, 0, 2]


@pytest.mark.parametrize("is_multi_label", [False, True])
def test_negative_label_rows_train_in_every_fold_and_never_validate(is_multi_label):
    from birdnet_analyzer.train.utils import _kfold_splits

    y = np.zeros((24, 3), dtype="float32")
    y[:6, 0] = y[6:12, 1] = y[12:18, 2] = 1
    y[18:21, 0] = -1
    if is_multi_label:
        y[18:21, 1] = 1
    negatives = set(range(18, 21))
    validated = []

    for train_idx, val_idx in _kfold_splits(y, 3, 1, False, is_multi_label):
        assert negatives <= set(train_idx)
        assert not negatives & set(val_idx)
        assert not set(train_idx) & set(val_idx)
        validated.extend(val_idx)

    assert sorted(validated) == sorted(set(range(24)) - negatives)


def test_binary_folds_are_stratified():
    from birdnet_analyzer.train.utils import _kfold_splits

    y = np.zeros((30, 1), dtype="float32")
    y[:10] = 1

    for _, val_idx in _kfold_splits(y, 5, 1, True, False):
        assert y[val_idx].sum() == 2


def test_non_event_rows_form_their_own_stratum():
    from birdnet_analyzer.train.utils import _kfold_splits

    y = np.zeros((25, 2), dtype="float32")
    y[:10, 0] = y[10:20, 1] = 1

    for _, val_idx in _kfold_splits(y, 5, 1, False, False):
        assert np.sum(y[val_idx].sum(axis=1) == 0) == 1


def test_off_grid_defaults_are_snapped_to_the_search_space(caplog):
    from birdnet_analyzer.train.utils import _search_space, _snap_to_search_space

    defaults = {
        "hidden_units": 300,
        "dropout": 0.1,
        "batch_size": 32,
        "learning_rate": 0.5,
        "upsampling_ratio": 0.3,
        "upsampling_mode": "smote",
        "weight_decay": 0.0,
        "mixup": False,
        "focal_loss_alpha": 0.25,
    }

    with caplog.at_level("INFO", logger="birdnet_analyzer.train.utils"):
        snapped = _snap_to_search_space(defaults, _search_space(False))

    assert snapped == {
        "hidden_units": 256,
        "dropout": 0.0,
        "batch_size": 32,
        "learning_rate": 0.1,
        "upsampling_ratio": 0.25,
        "weight_decay": 1e-6,
        "mixup": False,
        "focal_loss_alpha": 0.25,
    }
    assert len(caplog.records) == 6


def test_multi_label_defaults_lose_upsampling_modes_other_than_repeat():
    from birdnet_analyzer.train.utils import _search_space, _snap_to_search_space

    space = _search_space(True)

    assert _snap_to_search_space({"upsampling_mode": "mean"}, space) == {}
    assert _snap_to_search_space({"upsampling_mode": "repeat"}, space) == {
        "upsampling_mode": "repeat"
    }


def test_the_evaluation_leaves_out_rows_with_negative_labels():
    from birdnet_analyzer.train.utils import evaluate_model

    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [1, -1]], dtype="float32")
    logits = np.where(y > 0, 3.0, -3.0).astype("float32")
    classifier = MagicMock()
    classifier.predict.side_effect = lambda rows: logits[rows[:, 0]]

    metrics = evaluate_model(classifier, np.arange(6).reshape(-1, 1), y, ["a", "b"])

    assert set(metrics["class_metrics"]) == {"a", "b"}
    assert metrics["class_distribution"]["b"]["count"] == 3
    assert metrics["class_distribution"]["a"]["count"] == 2


def _synthetic_training_data():
    rng = np.random.default_rng(0)
    y = np.zeros((40, 2), dtype="float32")
    y[:20, 0] = 1
    y[20:, 1] = 1
    x = (rng.normal(size=(40, 8)) + 2 * y[:, :1]).astype("float32")

    return x, y, np.array([]), np.array([]), ["a", "b"], False, False, "3.0"


@pytest.fixture
def synthetic_training_data(monkeypatch):
    from birdnet_analyzer.train import utils as train_utils

    monkeypatch.setattr(
        train_utils, "_load_training_data", lambda *a, **k: _synthetic_training_data()
    )


def _autotune(monkeypatch, tmp_path, **kwargs):
    """Runs a real two-epoch autotune; returns the pruning monitors and the study."""
    import keras
    import optuna
    import optuna_integration.keras

    from birdnet_analyzer.train.utils import train_model

    monitors = []
    studies = []

    class RecordingPruningCallback(keras.callbacks.Callback):
        def __init__(self, trial, monitor):
            super().__init__()
            monitors.append(monitor)

    create_study = optuna.create_study

    def recording_create_study(**study_kwargs):
        studies.append(create_study(**study_kwargs))
        return studies[-1]

    monkeypatch.setattr(
        optuna_integration.keras, "KerasPruningCallback", RecordingPruningCallback
    )
    monkeypatch.setattr(optuna, "create_study", recording_create_study)

    train_model(
        "unused",
        output=str(tmp_path / "cc"),
        epochs=2,
        model_formats=[],
        autotune=True,
        **kwargs,
    )

    return monitors, studies[0]


@pytest.mark.usefixtures("synthetic_training_data")
@pytest.mark.parametrize("metric", ["val_AUPRC", "val_AUROC"])
def test_autotune_prunes_and_maximises_the_tuned_metric(monkeypatch, tmp_path, metric):
    import optuna

    monitors, study = _autotune(
        monkeypatch, tmp_path, autotune_trials=2, autotune_metric=metric
    )

    assert monitors
    assert set(monitors) == {metric}
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE


@pytest.mark.usefixtures("synthetic_training_data")
@pytest.mark.parametrize(
    ("metric", "other"), [("val_AUPRC", "val_AUROC"), ("val_AUROC", "val_AUPRC")]
)
def test_a_trial_scores_the_mean_restored_epoch_over_its_splits(
    monkeypatch, tmp_path, metric, other
):
    import optuna

    from birdnet_analyzer.train import utils as train_utils

    x = _synthetic_training_data()[0]
    fits = iter([([0.5, 0.8, 0.8009, 0.3], 2), ([0.6, 0.7, 0.2, 0.1], 0)])
    calls = []
    studies = []

    def fake_train(classifier, x_train, y_train, x_val, y_val, **kwargs):
        calls.append((x_train, x_val, kwargs))
        curve, kept = next(fits, ([0.5, 0.6], 1))
        history = MagicMock(epoch=list(range(len(curve))), restored_epoch=kept)
        history.history = {
            metric: curve,
            other: [0.9] + [0.1] * (len(curve) - 1),
            "val_loss": [0.1] + [0.9] * (len(curve) - 1),
        }
        return classifier, history

    create_study = optuna.create_study

    def recording_create_study(**study_kwargs):
        studies.append(create_study(**study_kwargs))
        return studies[-1]

    monkeypatch.setattr(train_utils.model, "train_linear_classifier", fake_train)
    monkeypatch.setattr(optuna, "create_study", recording_create_study)

    train_utils.train_model(
        "unused",
        output=str(tmp_path / "cc"),
        epochs=4,
        model_formats=[],
        autotune=True,
        autotune_trials=1,
        autotune_n_splits=2,
        autotune_metric=metric,
    )

    assert studies[0].direction == optuna.study.StudyDirection.MAXIMIZE
    assert studies[0].trials[0].value == pytest.approx((0.8009 + 0.6) / 2)
    assert len(calls) == 3
    assert all(k["monitor"] == metric and k["monitor_mode"] == "max" for *_, k in calls)

    for x_train, x_val, _ in calls[:2]:
        trained = {row.tobytes() for row in x_train}

        assert len(x_val)
        assert not any(row.tobytes() in trained for row in x_val)
        assert len(trained) + len(x_val) == len(x)


def test_tuning_and_early_stopping_never_see_the_test_data(monkeypatch, tmp_path):
    from birdnet_analyzer.train import utils as train_utils

    x, y, *_ = _synthetic_training_data()
    x_test = np.full((6, 8), 99.0, dtype="float32")
    y_test = np.repeat(np.eye(2, dtype="float32"), 3, axis=0)
    monkeypatch.setattr(
        train_utils,
        "_load_training_data",
        lambda *a, **k: (x, y, x_test, y_test, ["a", "b"], False, False, "3.0"),
    )
    fits = []
    evaluated = []
    train = train_utils.model.train_linear_classifier

    def recording_train(classifier, x_train, y_train, x_val, y_val, **kwargs):
        fits.append((x_train, x_val))
        return train(classifier, x_train, y_train, x_val, y_val, **kwargs)

    def recording_evaluate(classifier, x_eval, y_eval, labels):
        evaluated.append(x_eval)
        return {}

    monkeypatch.setattr(train_utils.model, "train_linear_classifier", recording_train)
    monkeypatch.setattr(train_utils, "evaluate_model", recording_evaluate)

    train_utils.train_model(
        "unused",
        output=str(tmp_path / "cc"),
        epochs=2,
        model_formats=[],
        autotune=True,
        autotune_trials=1,
        autotune_n_splits=2,
    )

    assert len(fits) == 3
    assert not any(np.any(part == 99) for fit in fits for part in fit)
    assert len(evaluated) == 1
    assert np.all(evaluated[0] == 99)


@pytest.mark.parametrize("autotune", [False, True])
@pytest.mark.parametrize("test_data", [None, "test"])
def test_a_validation_split_of_zero_is_rejected_before_loading(
    monkeypatch, tmp_path, autotune, test_data
):
    from birdnet_analyzer.train import utils as train_utils

    def load(*args, **kwargs):
        pytest.fail("the training data was loaded")

    monkeypatch.setattr(train_utils, "_load_training_data", load)

    with pytest.raises(ValueError, match="validation split"):
        train_utils.train_model(
            "unused",
            output=str(tmp_path / "cc"),
            test_data=test_data,
            autotune=autotune,
            autotune_n_splits=3,
            val_split=0.0,
        )


def test_autotune_needs_at_least_one_trial(monkeypatch, tmp_path):
    from birdnet_analyzer.train import utils as train_utils

    def load(*args, **kwargs):
        pytest.fail("the training data was loaded")

    monkeypatch.setattr(train_utils, "_load_training_data", load)

    with pytest.raises(ValueError, match="trial"):
        train_utils.train_model(
            "unused", output=str(tmp_path / "cc"), autotune=True, autotune_trials=0
        )


@pytest.mark.usefixtures("synthetic_training_data")
def test_the_params_file_records_the_autotune_metric(tmp_path):
    from birdnet_analyzer import params
    from birdnet_analyzer.config import TRAIN_PARAMS_SUFFIX
    from birdnet_analyzer.train.utils import train_model

    train_model(
        "unused",
        output=str(tmp_path / "cc"),
        epochs=1,
        model_formats=[],
        autotune_metric="val_AUROC",
    )

    recorded = params.load_train_params(str(tmp_path / "cc") + TRAIN_PARAMS_SUFFIX)

    assert recorded["autotune_metric"] == "val_AUROC"


@pytest.mark.usefixtures("synthetic_training_data")
def test_the_params_file_reads_back_as_the_arguments_of_the_training(
    monkeypatch, tmp_path
):
    from birdnet_analyzer import params
    from birdnet_analyzer.config import TRAIN_PARAMS_SUFFIX
    from birdnet_analyzer.train import utils as train_utils

    for save in ("save_linear_classifier", "save_raven_model"):
        monkeypatch.setattr(train_utils.model, save, lambda *args, **kwargs: None)

    arguments = {
        "model_formats": ["tflite", "raven"],
        "model_save_mode": "append",
        "fmin": 150,
        "fmax": 12000,
        "audio_speed": 2.0,
        "crop_mode": "segments",
        "overlap": 1.5,
        "autotune": False,
        "autotune_trials": 7,
        "autotune_n_splits": 3,
        "autotune_n_repeats": 2,
        "autotune_metric": "val_AUROC",
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.0005,
        "hidden_units": 16,
        "dropout": 0.25,
        "label_smoothing": True,
        "mixup": True,
        "use_focal_loss": True,
        "focal_loss_gamma": 3.0,
        "focal_loss_alpha": 0.5,
        "upsampling_mode": "mean",
        "upsampling_ratio": 0.5,
        "birdnet": "3.0",
    }

    train_utils.train_model("unused", output=str(tmp_path / "cc"), **arguments)

    assert params.load_train_params(str(tmp_path / "cc") + TRAIN_PARAMS_SUFFIX) == {
        "classifier_name": "cc",
        **arguments,
    }


@pytest.mark.usefixtures("synthetic_training_data")
def test_off_grid_defaults_run_as_the_first_trial(monkeypatch, tmp_path):
    _, study = _autotune(
        monkeypatch,
        tmp_path,
        autotune_trials=1,
        dropout=0.1,
        upsampling_ratio=0.3,
        upsampling_mode="smote",
    )

    assert study.trials[0].state.name == "COMPLETE"
    assert study.trials[0].params["dropout"] == 0.0
    assert study.trials[0].params["upsampling_ratio"] == 0.25


@pytest.mark.usefixtures("synthetic_training_data")
def test_a_pruned_trial_still_clears_the_keras_session(monkeypatch, tmp_path):
    import keras
    import optuna
    import optuna_integration.keras

    from birdnet_analyzer.train.utils import train_model

    class PruneTheFirstTrial(keras.callbacks.Callback):
        def __init__(self, trial, monitor):
            super().__init__()
            self.trial = trial

        def on_epoch_end(self, epoch, logs=None):
            if self.trial.number == 0:
                raise optuna.TrialPruned

    cleared = []
    clear_session = keras.backend.clear_session

    def recording_clear_session(*args, **kwargs):
        cleared.append(True)
        return clear_session(*args, **kwargs)

    monkeypatch.setattr(
        optuna_integration.keras, "KerasPruningCallback", PruneTheFirstTrial
    )
    monkeypatch.setattr(keras.backend, "clear_session", recording_clear_session)

    train_model(
        "unused",
        output=str(tmp_path / "cc"),
        epochs=2,
        model_formats=[],
        autotune=True,
        autotune_trials=2,
    )

    assert len(cleared) == 2


@pytest.mark.usefixtures("synthetic_training_data")
def test_the_first_autotune_trial_runs_the_given_weight_decay(monkeypatch, tmp_path):
    _, study = _autotune(monkeypatch, tmp_path, autotune_trials=1, weight_decay=0.0005)

    assert study.trials[0].params["weight_decay"] == 0.0005


@pytest.mark.usefixtures("synthetic_training_data")
def test_autotune_accepts_a_weight_decay_of_zero(monkeypatch, tmp_path):
    _, study = _autotune(monkeypatch, tmp_path, autotune_trials=1, weight_decay=0.0)

    assert study.trials[0].params["weight_decay"] > 0


@pytest.mark.usefixtures("synthetic_training_data")
def test_only_the_first_split_of_a_trial_reports_to_the_pruner(monkeypatch, tmp_path):
    monitors, _ = _autotune(
        monkeypatch, tmp_path, autotune_trials=1, autotune_n_splits=2
    )

    assert len(monitors) == 1


@pytest.mark.usefixtures("synthetic_training_data")
@pytest.mark.parametrize(
    "failing", ["train_linear_classifier", "save_linear_classifier"]
)
def test_a_training_failure_reaches_the_caller_unchanged(
    monkeypatch, tmp_path, failing
):
    from birdnet_analyzer.train import utils as train_utils

    boom = RuntimeError("boom")

    def fail(*args, **kwargs):
        raise boom

    monkeypatch.setattr(train_utils.model, failing, fail)

    with pytest.raises(RuntimeError) as error:
        train_utils.train_model("unused", output=str(tmp_path / "cc"), epochs=1)

    assert error.value is boom
