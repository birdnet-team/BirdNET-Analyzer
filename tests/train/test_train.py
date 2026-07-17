import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from birdnet_analyzer.cli import train_parser
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
):
    # prepare stubbed data and model training
    import numpy as np

    mock_load.return_value = (
        np.zeros((5, 10), dtype="float32"),
        np.zeros((5, 3), dtype="float32"),
        np.array([], dtype="float32"),  # no test samples to avoid evaluation step
        np.array([], dtype="float32"),
        ["a", "b", "c"],
        False,
        False,
    )
    # use a mutable sequence so classifier.pop() used during save does not blow up
    mock_build.return_value = []
    # training returns (classifier, history); classifier must support pop()
    # give a classifier with at least one element so pop() succeeds
    mock_train.return_value = ([1], _make_dummy_history())

    # create fake study object which records calls
    import sys

    # make sure the `import optuna` in train_model doesn't raise ImportError
    sys.modules.setdefault("optuna", mock_optuna)

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
