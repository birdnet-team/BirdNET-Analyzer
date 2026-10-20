import importlib

import pytest

gr = pytest.importorskip("gradio")

from birdnet_analyzer.gui import analysis as ga  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.mark.parametrize(("slider", "speed"), [(-2, 0.5), (0, 1.0), (3, 3.0)])
def test_the_speed_slider_reaches_the_analysis_as_a_factor(monkeypatch, slider, speed):
    seen = {}
    monkeypatch.setattr(
        importlib.import_module("birdnet_analyzer.analyze"),
        "analyze",
        lambda **kwargs: seen.update(kwargs),
    )

    ga.run_analysis(
        input_path="recording.wav",
        output_path=None,
        use_top_n=False,
        top_n=1,
        confidence=0.25,
        sensitivity=1.0,
        overlap=0.0,
        merge_consecutive=1,
        audio_speed=slider,
        fmin=0,
        fmax=15000,
        species_list_choice=gu._ALL_SPECIES,
        species_list_file=None,
        lat=-1,
        lon=-1,
        week=-1,
        use_yearlong=True,
        sf_thresh=0.03,
        selected_model=gu._USE_BIRDNET_3_0,
        custom_classifier_file=None,
        output_types="table",
        additional_columns=None,
        locale="en_us",
        batch_size=1,
        input_dir=None,
        save_params=False,
        n_producers=1,
        n_workers=1,
        progress=None,
    )

    assert seen["audio_speed"] == speed
