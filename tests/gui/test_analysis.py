import importlib

import pytest

gr = pytest.importorskip("gradio")

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
