import errno
import importlib
import os
import tempfile
import threading

import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import model_utils  # noqa: E402
from birdnet_analyzer.gui import analysis as ga  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402

SETTINGS = {
    "input_path": "recording.wav",
    "output_path": None,
    "use_top_n": False,
    "top_n": 1,
    "confidence": 0.25,
    "sensitivity": 1.0,
    "overlap": 0.0,
    "merge_consecutive": 1,
    "audio_speed": 0,
    "fmin": 0,
    "fmax": 15000,
    "species_list_choice": gu.ALL_SPECIES,
    "species_list_file": None,
    "lat": -1,
    "lon": -1,
    "week": -1,
    "use_yearlong": True,
    "sf_thresh": 0.03,
    "selected_model": gu.USE_BIRDNET_3_0,
    "custom_classifier_file": None,
    "output_types": "table",
    "additional_columns": None,
    "locale": "en_us",
    "batch_size": 1,
    "input_dir": None,
    "save_params": False,
    "n_producers": 1,
    "n_workers": 1,
    "progress": None,
}


def analyze_with(monkeypatch, **settings):
    seen = {}
    monkeypatch.setattr(gu, "validate_audio_input", lambda path: None)
    monkeypatch.setattr(
        importlib.import_module("birdnet_analyzer.analyze"),
        "analyze",
        lambda **kwargs: seen.update(kwargs),
    )

    ga.run_analysis(**{**SETTINGS, **settings})

    return seen


@pytest.mark.parametrize(("slider", "speed"), [(-2, 0.5), (0, 1.0), (3, 3.0)])
def test_the_speed_slider_reaches_the_analysis_as_a_factor(monkeypatch, slider, speed):
    assert analyze_with(monkeypatch, audio_speed=slider)["audio_speed"] == speed


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (gu.ALL_SPECIES, {"slist": None, "lat": None, "lon": None}),
        (gu.CUSTOM_SPECIES, {"slist": "birds.txt", "lat": None, "lon": None}),
        (gu.PREDICT_SPECIES, {"slist": None, "lat": 50.8, "lon": 12.9}),
    ],
)
def test_the_species_list_choice_decides_which_list_the_analysis_uses(
    monkeypatch, choice, expected
):
    seen = analyze_with(
        monkeypatch,
        species_list_choice=choice,
        species_list_file="birds.txt",
        lat=50.8,
        lon=12.9,
    )

    assert {key: seen[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (
            gu.USE_BIRDNET_3_0,
            {"model": "birdnet", "birdnet": "3.0", "classifier": None},
        ),
        (
            gu.USE_BIRDNET_2_4,
            {"model": "birdnet", "birdnet": "2.4", "classifier": None},
        ),
        (gu.USE_PERCH, {"model": "perch", "classifier": None}),
        (gu.CUSTOM_CLASSIFIER, {"model": "birdnet", "classifier": "my.tflite"}),
    ],
)
def test_the_model_choice_decides_which_model_the_analysis_runs(
    monkeypatch, choice, expected
):
    seen = analyze_with(
        monkeypatch, selected_model=choice, custom_classifier_file="my.tflite"
    )

    assert {key: seen[key] for key in expected} == expected


def _location_without_species_of_the_model():
    model_utils._reconcile_species_list(
        {"Corvus corax_Common Raven"}, ["Poecile atricapillus_Chickadee"], strict=False
    )


def _in_another_analysis():
    other = threading.Thread(target=_location_without_species_of_the_model)
    other.start()
    other.join()


@pytest.mark.parametrize(
    ("logs_a_warning", "shown"),
    [
        (
            _location_without_species_of_the_model,
            ["species-list-location-unmatched-warning"],
        ),
        (lambda: model_utils.effective_sensitivity(0.5, "perch"), []),
        (_in_another_analysis, []),
    ],
    ids=["location", "sensitivity", "another analysis"],
)
@pytest.mark.usefixtures("made_up_inputs")
def test_only_a_location_without_species_of_the_model_is_shown(
    monkeypatch, logs_a_warning, shown
):
    warned = []
    monkeypatch.setattr(gr, "Warning", lambda message, **_: warned.append(message))
    monkeypatch.setattr(
        importlib.import_module("birdnet_analyzer.analyze"),
        "analyze",
        lambda **kwargs: logs_a_warning(),
    )

    ga.run_analysis(**SETTINGS)

    assert warned == [loc.localize(key) for key in shown]


@pytest.fixture
def no_analysis(monkeypatch):
    def fail(**kwargs):
        raise AssertionError("the analysis started")

    monkeypatch.setattr(
        importlib.import_module("birdnet_analyzer.analyze"), "analyze", fail
    )


def test_a_recording_that_is_gone_is_reported_instead_of_analyzed(
    no_analysis, tmp_path
):
    gone = str(tmp_path / "May & June" / "recording.wav")

    with pytest.raises(gr.Error) as raised:
        ga.run_analysis(**{**SETTINGS, "input_path": gone})

    assert raised.value.message == loc.LANGUAGE_LOOKUP["error-path-not-found"].format(
        path=gone.replace("&", "&amp;")
    )


def test_a_folder_without_audio_is_reported_instead_of_analyzed(no_analysis, tmp_path):
    folder = tmp_path / "May & June"
    folder.mkdir()
    (folder / "recording.m4a").touch()
    (folder / "notes.txt").touch()

    with pytest.raises(gr.Error) as raised:
        ga.run_analysis(**{**SETTINGS, "input_path": None, "input_dir": str(folder)})

    assert raised.value.message == loc.LANGUAGE_LOOKUP[
        "validation-no-audio-files-found"
    ].format(path=str(folder).replace("&", "&amp;"))


@pytest.mark.parametrize("selected", ["input_path", "input_dir"])
def test_a_recording_or_a_folder_with_recordings_is_analyzed(
    monkeypatch, tmp_path, selected
):
    (tmp_path / "day 1").mkdir()
    recording = tmp_path / "day 1" / "recording.flac"
    recording.touch()
    path = str(recording if selected == "input_path" else tmp_path)
    seen = {}
    monkeypatch.setattr(
        importlib.import_module("birdnet_analyzer.analyze"),
        "analyze",
        lambda **kwargs: seen.update(kwargs),
    )

    ga.run_analysis(
        **{**SETTINGS, "input_path": None, "input_dir": None, selected: path}
    )

    assert seen["audio_input"] == path


def _denied(path):
    def fail(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    return fail


def _batch(tmp_path, **settings):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "recording.wav").touch()

    return {**SETTINGS, "input_path": None, "input_dir": str(recordings), **settings}


def test_a_folder_that_may_not_be_opened_is_reported_as_such(
    no_analysis, monkeypatch, tmp_path
):
    settings = _batch(tmp_path)
    monkeypatch.setattr(os, "scandir", _denied(settings["input_dir"]))

    with pytest.raises(gr.Error) as raised:
        ga.run_analysis(**settings)

    assert raised.value.message == loc.LANGUAGE_LOOKUP[
        "error-path-not-readable"
    ].format(path=settings["input_dir"])


def test_an_output_folder_on_a_drive_that_is_gone_is_reported_as_not_found(
    no_analysis, monkeypatch, tmp_path
):
    output = tmp_path / "SD card & more" / "results"
    monkeypatch.setattr(os, "makedirs", _denied(output.parent))

    with pytest.raises(gr.Error) as raised:
        ga.run_analysis(**_batch(tmp_path, output_path=str(output)))

    assert raised.value.message == loc.LANGUAGE_LOOKUP["error-path-not-found"].format(
        path=str(output.parent).replace("&", "&amp;")
    )


@pytest.mark.parametrize("chosen", [True, False], ids=["chosen", "the input folder"])
def test_an_output_folder_that_cannot_be_written_is_named_itself(
    no_analysis, monkeypatch, tmp_path, chosen
):
    settings = _batch(tmp_path)
    output = tmp_path / "R&D results" if chosen else tmp_path / "recordings"
    settings["output_path"] = str(output) if chosen else None
    monkeypatch.setattr(tempfile, "TemporaryFile", _denied(output / "tmp123"))

    with pytest.raises(gr.Error) as raised:
        ga.run_analysis(**settings)

    assert raised.value.message == loc.LANGUAGE_LOOKUP["error-path-no-access"].format(
        path=str(output).replace("&", "&amp;")
    )


def test_a_full_disk_is_not_taken_for_a_folder_without_write_access(
    no_analysis, monkeypatch, tmp_path
):
    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(tempfile, "TemporaryFile", full)

    with pytest.raises(OSError, match="No space left on device"):
        ga.run_analysis(**_batch(tmp_path))


def test_an_output_folder_that_was_deleted_is_created_again(monkeypatch, tmp_path):
    output = tmp_path / "deleted" / "results"
    started = []
    monkeypatch.setattr(
        importlib.import_module("birdnet_analyzer.analyze"),
        "analyze",
        lambda **kwargs: started.append(os.path.isdir(kwargs["output"])),
    )

    ga.run_analysis(**_batch(tmp_path, output_path=str(output)))

    assert started == [True]
    assert os.listdir(output) == []
