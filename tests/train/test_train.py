import inspect
import os
import shutil
import tempfile
from typing import get_args
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from birdnet_analyzer.cli import train_parser
from birdnet_analyzer.config import UPSAMPLING_MODES
from birdnet_analyzer.train.core import train


@pytest.fixture
def setup_test_environment():
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    classifier_output = os.path.join(output_dir, "classifier_output")

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "classifier_output": classifier_output,
    }

    shutil.rmtree(test_dir)


@patch("birdnet_analyzer.train.utils.train_model")
def test_train_cli(mock_train_model, setup_test_environment):
    env = setup_test_environment

    parser = train_parser()
    args = parser.parse_args([env["input_dir"], "--output", env["classifier_output"]])

    # Remove CLI-only args not accepted by train()
    kwargs = vars(args)
    kwargs.pop("load_params")

    train(**kwargs)

    mock_train_model.assert_called_once()
    call_kwargs = mock_train_model.call_args[1]
    assert call_kwargs["output"] == env["classifier_output"]
    assert mock_train_model.call_args[0][0] == env["input_dir"]


@patch("birdnet_analyzer.train.utils.train_model")
def test_train_cli_accepts_full_parser_surface(
    mock_train_model, setup_test_environment
):
    env = setup_test_environment

    parser = train_parser()
    cache_path = os.path.join(env["test_dir"], "train_cache.npz")
    args = parser.parse_args(
        [
            env["input_dir"],
            "--test_data",
            env["output_dir"],
            "--crop_mode",
            "smart",
            "--birdnet",
            "2.4",
            "-o",
            env["classifier_output"],
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
            "val_loss",
        ]
    )

    # Remove CLI-only args not accepted by train()
    kwargs = vars(args)
    kwargs.pop("load_params")

    train(**kwargs)

    mock_train_model.assert_called_once()
    call_kwargs = mock_train_model.call_args[1]
    assert call_kwargs["test_data"] == env["output_dir"]
    assert call_kwargs["crop_mode"] == "smart"
    assert call_kwargs["model_formats"] == ["tflite", "raven", "detached"]
    assert call_kwargs["model_save_mode"] == "append"
    assert call_kwargs["save_cache_to"] == cache_path
    assert call_kwargs["autotune"] is True
    assert call_kwargs["autotune_metric"] == "val_loss"
    assert call_kwargs["birdnet"] == "2.4"


def _make_dummy_history():
    class DummyHistory:
        history = {"val_AUPRC": [0.123]}  # noqa: RUF012

    return DummyHistory()


@patch("birdnet_analyzer.train.utils.model.save_raven_model")
@patch("birdnet_analyzer.train.utils.model.save_linear_classifier")
@patch("birdnet_analyzer.train.utils.model.build_linear_classifier")
@patch("birdnet_analyzer.train.utils.model.train_linear_classifier")
@patch("birdnet_analyzer.train.utils._load_training_data")
@patch("birdnet_analyzer.train.utils.optuna", create=True)
def test_autotune_uses_optuna(
    mock_optuna,
    mock_load,
    mock_train,
    mock_build,
    mock_save_linear,
    mock_save_raven,
    setup_test_environment,
    monkeypatch,
):
    mock_load.return_value = (
        np.zeros((5, 10), dtype="float32"),
        np.zeros((5, 3), dtype="float32"),
        np.array([], dtype="float32"),  # no test samples to avoid evaluation step
        np.array([], dtype="float32"),
        ["a", "b", "c"],
        False,
        False,
        "2.4",
    )
    # use a mutable sequence so classifier.pop() used during save does not blow up
    mock_build.return_value = []
    # training returns (classifier, history); classifier must support pop()
    # give a classifier with at least one element so pop() succeeds
    mock_train.return_value = ([1], _make_dummy_history())

    # create fake study object which records calls
    import sys

    monkeypatch.setitem(sys.modules, "optuna", mock_optuna)
    monkeypatch.setitem(sys.modules, "optuna_integration.keras", MagicMock())

    dummy_study = type("S", (), {"enqueue_trial": lambda self, params: None})()
    calls = {}

    def fake_optimize(obj, n_trials):
        calls["optimized"] = n_trials

        # simulate a single trial evaluation (trial.number=0)
        class DummyTrial:
            def __init__(self):
                self.number = 0

            def suggest_categorical(self, name, choices):
                return choices[0]

            def suggest_float(self, name, low, high, **kwargs):
                return low

        obj(DummyTrial())
        dummy_study.best_params = {
            "hidden_units": 0,
            "dropout": 0.0,
            "batch_size": 8,
            "learning_rate": 0.0001,
            "upsampling_ratio": 0.0,
            "mixup": False,
            "label_smoothing": False,
            "focal_loss": False,
            "weight_decay": 0.0,
        }

    dummy_study.optimize = fake_optimize
    mock_optuna.create_study.return_value = dummy_study

    from birdnet_analyzer.train.utils import train_model

    env = setup_test_environment
    # ensure output directory exists so sample_counts can be written
    os.makedirs(env["classifier_output"], exist_ok=True)

    # call with autotune enabled
    try:
        train_model(
            env["input_dir"],
            output=env["classifier_output"],
            autotune=True,
            autotune_trials=3,
        )
    except Exception as e:  # pragma: no cover - we want failure message
        pytest.fail(f"train_model raised during autotune: {e!r}")

    mock_optuna.create_study.assert_called_once()
    create_study_kwargs = mock_optuna.create_study.call_args[1]
    assert create_study_kwargs["study_name"] == "birdnet_analyzer"
    assert calls.get("optimized") == 3
    # build_linear_classifier should be called at least once (during tuning)
    assert mock_build.called
    assert mock_train.called


@patch("birdnet_analyzer.train.utils.train_model")
def test_the_cli_writes_no_training_cache_unless_asked(mock_train_model, tmp_path):
    kwargs = vars(train_parser().parse_args([str(tmp_path)]))
    kwargs.pop("load_params")

    train(**kwargs)

    assert mock_train_model.call_args[1]["save_cache_to"] is None


def test_the_upsampling_modes_type_lists_every_mode_the_cli_accepts():
    action = next(a for a in train_parser()._actions if a.dest == "upsampling_mode")

    assert set(get_args(UPSAMPLING_MODES)) == set(action.choices)


def test_the_api_and_the_cli_default_to_the_same_number_of_autotune_folds():
    api_default = inspect.signature(train).parameters["autotune_n_splits"].default

    assert api_default == train_parser().get_default("autotune_n_splits")


@pytest.mark.parametrize(
    ("metric", "direction"),
    [("val_loss", "minimize"), ("val_AUPRC", "maximize"), ("val_AUROC", "maximize")],
)
def test_the_autotune_direction_follows_the_metric(metric, direction):
    from birdnet_analyzer.train.utils import _autotune_direction

    assert _autotune_direction(metric) == direction


@pytest.mark.parametrize(
    ("metric", "best"), [("val_loss", 0.2), ("val_AUPRC", 0.9), ("val_AUROC", 0.9)]
)
def test_the_best_epoch_is_the_one_the_metric_favours(metric, best):
    from birdnet_analyzer.train.utils import _best_epoch_score

    assert _best_epoch_score([0.5, 0.2, 0.9, 0.4], metric) == best


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
@pytest.mark.parametrize(
    ("metric", "direction"), [("val_loss", "MINIMIZE"), ("val_AUPRC", "MAXIMIZE")]
)
def test_autotune_prunes_and_optimises_the_tuned_metric(
    monkeypatch, tmp_path, metric, direction
):
    import optuna

    monitors, study = _autotune(
        monkeypatch, tmp_path, autotune_trials=2, autotune_metric=metric
    )

    assert monitors
    assert set(monitors) == {metric}
    assert study.direction == optuna.study.StudyDirection[direction]


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
