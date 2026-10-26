import pytest

from birdnet_analyzer import cli, params
from tests.params_files import (
    TRAIN_FILE,
    analysis_values,
    tall_file,
    train_values,
    wide_file,
)

EXPECTED_ANALYSIS_KWARGS = {
    "model": "birdnet",
    "birdnet": "2.4",
    "overlap": 0.5,
    "fmin": 150,
    "fmax": 12000,
    "merge_consecutive": 3,
    "audio_speed": 1.0,
    "min_conf": 0.3,
    "sensitivity": 1.2,
    "batch_size": 8,
    "n_workers": 4,
    "n_producers": 2,
    "rtype": ["table", "csv"],
    "additional_columns": ["lat", "lon"],
    "sf_thresh": 0.05,
    "locale": "en_us",
    "split_tables": False,
}


@pytest.mark.parametrize("write_file", [wide_file, tall_file])
def test_analysis_params_are_read_back_as_analyze_arguments(tmp_path, write_file):
    kwargs = params.load_analysis_params(write_file(tmp_path, analysis_values()))

    assert kwargs == EXPECTED_ANALYSIS_KWARGS


def test_the_values_are_returned_exactly_as_the_analysis_ran_with_them(tmp_path):
    # No GUI transformations: the speed factor stays a factor, and the placeholder
    # confidence of a top-N analysis is kept, so the original call is reproduced.
    file = tall_file(
        tmp_path,
        analysis_values(
            **{"Audio speed": "0.25", "Top N": "5", "Minimum confidence": "0"}
        ),
    )

    kwargs = params.load_analysis_params(file)

    assert kwargs["audio_speed"] == 0.25
    assert kwargs["top_n"] == 5
    assert kwargs["min_conf"] == 0


def test_an_empty_selection_is_read_back_as_one(tmp_path):
    file = tall_file(
        tmp_path, analysis_values(**{"Result type(s)": "", "Additional columns": ""})
    )

    kwargs = params.load_analysis_params(file)

    assert (kwargs["rtype"], kwargs["additional_columns"]) == ([], [])


def test_files_of_an_analysis_are_read_back(tmp_path):
    file = tall_file(
        tmp_path,
        analysis_values(
            **{
                "Species list file": "/data/birds.txt",
                "Custom classifier path": "/models/my.tflite",
            }
        ),
    )

    kwargs = params.load_analysis_params(file)

    assert kwargs["slist"] == "/data/birds.txt"
    assert kwargs["classifier"] == "/models/my.tflite"


def test_train_params_are_read_back_as_train_arguments(tmp_path):
    file = tall_file(tmp_path, train_values(), name=TRAIN_FILE)

    assert params.load_train_params(file) == {
        "classifier_name": "MyClassifier",
        "model_formats": ["tflite", "raven"],
        "model_save_mode": "replace",
        "fmin": 150,
        "fmax": 12000,
        "audio_speed": 2.0,
        "crop_mode": "segments",
        "overlap": 1.5,
        "autotune": False,
        "autotune_trials": 50,
        "autotune_n_splits": 5,
        "autotune_n_repeats": 1,
        "autotune_metric": "val_AUROC",
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "hidden_units": 512,
        "dropout": 0.25,
        "label_smoothing": True,
        "mixup": True,
        "use_focal_loss": True,
        "focal_loss_gamma": 2.0,
        "focal_loss_alpha": 0.25,
        "upsampling_mode": "repeat",
        "upsampling_ratio": 0.5,
        "birdnet": "2.4",
    }


def test_an_unknown_model_version_is_left_out(tmp_path):
    file = tall_file(
        tmp_path, train_values(**{"BirdNET model version": "9.9"}), name=TRAIN_FILE
    )

    assert "birdnet" not in params.load_train_params(file)


def test_an_old_train_params_file_is_still_understood(tmp_path):
    # The layout and names the versions before 2.x saved next to the classifier.
    old = {
        "Hidden units": 512,
        "Dropout": 0.25,
        "Batchsize": 32,
        "Learning rate": 0.0001,
        "Weight decay": 0.004,
        "Crop mode": "center",
        "Crop overlap": 0.0,
        "Audio speed": 0.5,
        "Upsampling mode": "repeat",
        "Upsampling ratio": 0.75,
        "use mixup": True,
        "use label smoothing": False,
        "use focal loss": False,
        "focal loss alpha": 0.25,
        "focal loss gamma": 2.0,
        "BirdNET Model version": "2.4",
    }

    kwargs = params.load_train_params(wide_file(tmp_path, old))

    assert kwargs == {
        "hidden_units": 512,
        "dropout": 0.25,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "crop_mode": "center",
        "overlap": 0.0,
        "audio_speed": 0.5,
        "upsampling_mode": "repeat",
        "upsampling_ratio": 0.75,
        "mixup": True,
        "label_smoothing": False,
        "use_focal_loss": False,
        "focal_loss_alpha": 0.25,
        "focal_loss_gamma": 2.0,
        "birdnet": "2.4",
    }


def test_files_that_are_no_params_files_are_rejected(tmp_path):
    results = tmp_path / "results.csv"
    results.write_text("Start (s),End (s),Confidence\n0,3,0.8\n", encoding="utf-8")

    for loader in (params.load_analysis_params, params.load_train_params):
        with pytest.raises(ValueError, match="parameters file"):
            loader(str(results))


def test_the_params_file_of_the_other_command_is_rejected(tmp_path):
    # The commands share a handful of parameter names (audio speed, bandpass limits,
    # batch size), which must not be enough to pass as the right file.
    analysis_file = tall_file(tmp_path, analysis_values())
    train_file = tall_file(tmp_path, train_values(), name="train-params.csv")

    with pytest.raises(ValueError, match="training parameters"):
        params.load_train_params(analysis_file)

    with pytest.raises(ValueError, match="analysis parameters"):
        params.load_analysis_params(train_file)


def test_analysis_params_become_cli_defaults_that_explicit_arguments_override(
    tmp_path,
):
    file = tall_file(
        tmp_path,
        analysis_values(
            **{
                "Minimum confidence": "0.4",
                "Audio speed": "0.25",
                "Split tables": "True",
                "Species list file": "/data/birds.txt",
            }
        ),
    )
    parser = cli.analyzer_parser()

    cli.apply_params_file_defaults(
        parser, params.load_analysis_params, argv=["--load_params", file]
    )
    args = parser.parse_args(["recordings"])

    assert args.min_conf == 0.4
    assert args.audio_speed == 0.25
    assert args.split_tables is True
    assert args.slist == "/data/birds.txt"
    assert args.rtype == ["table", "csv"]

    overridden = parser.parse_args(
        ["recordings", "--min_conf", "0.9", "--no-split_tables"]
    )

    assert overridden.min_conf == 0.9
    assert overridden.split_tables is False
    # Values without an explicit argument keep the file's value.
    assert overridden.audio_speed == 0.25


def test_train_params_become_cli_defaults_that_explicit_arguments_override(tmp_path):
    file = tall_file(tmp_path, train_values(Autotune=True, Epochs=75), name=TRAIN_FILE)
    parser = cli.train_parser()

    cli.apply_params_file_defaults(
        parser, params.load_train_params, argv=["--load_params", file]
    )
    args = parser.parse_args(["train_data"])

    assert args.epochs == 75
    assert args.autotune is True
    assert args.mixup is True
    assert args.model_formats == ["tflite", "raven"]
    # The classifier name has no train() argument, the output path is that identity.
    assert "classifier_name" not in vars(args)

    overridden = parser.parse_args(["train_data", "--no-autotune", "--epochs", "10"])

    assert overridden.autotune is False
    assert overridden.epochs == 10


def test_a_val_loss_metric_from_a_params_file_is_rejected_by_training(
    tmp_path, monkeypatch
):
    from birdnet_analyzer.train import core
    from birdnet_analyzer.train import utils as train_utils

    file = tall_file(
        tmp_path, train_values(**{"Autotune metric": "val_loss"}), name=TRAIN_FILE
    )
    parser = cli.train_parser()
    cli.apply_params_file_defaults(
        parser, params.load_train_params, argv=["--load_params", file]
    )
    kwargs = vars(parser.parse_args(["train_data"]))
    kwargs.pop("load_params")

    def load(*args, **kwargs):
        pytest.fail("the training data was loaded")

    monkeypatch.setattr(train_utils, "_load_training_data", load)

    with pytest.raises(ValueError, match="val_loss"):
        core.train(**kwargs)


def test_an_unreadable_params_file_stops_the_cli(tmp_path, capsys):
    results = tmp_path / "results.csv"
    results.write_text("Start (s),End (s),Confidence\n0,3,0.8\n", encoding="utf-8")
    parser = cli.analyzer_parser()

    with pytest.raises(SystemExit):
        cli.apply_params_file_defaults(
            parser, params.load_analysis_params, argv=["--load_params", str(results)]
        )

    assert "parameters file" in capsys.readouterr().err


def test_without_the_argument_the_defaults_stay_untouched():
    parser = cli.analyzer_parser()

    cli.apply_params_file_defaults(parser, params.load_analysis_params, argv=[])

    assert parser.parse_args(["recordings"]).min_conf == 0.25


def test_the_cli_arguments_stay_in_sync_with_the_api():
    # The mains pass the parsed arguments straight into analyze()/train(), which is
    # also what makes the loaded parameters files line up with the parsers.
    import inspect

    from birdnet_analyzer import analyze, train

    analyze_args = vars(cli.analyzer_parser().parse_args(["recordings"]))
    analyze_args.pop("use_perch")
    analyze_args.pop("load_params")

    assert set(analyze_args) <= set(inspect.signature(analyze).parameters)

    train_args = vars(cli.train_parser().parse_args(["train_data"]))
    train_args.pop("load_params")

    assert set(train_args) <= set(inspect.signature(train).parameters)


def test_a_train_params_file_without_a_version_is_a_2_4_one(tmp_path):
    values = {k: v for k, v in train_values().items() if k != "BirdNET model version"}
    file = tall_file(tmp_path, values, name=TRAIN_FILE)

    assert params.load_train_params(file)["birdnet"] == "2.4"
