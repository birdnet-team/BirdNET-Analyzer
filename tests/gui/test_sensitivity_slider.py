"""The sensitivity slider follows the selected model.

Only BirdNET 2.4 and custom classifiers take a sensitivity. Selecting BirdNET 3.0 or
Perch disables the slider and shows 1.0 (what the analysis will use); switching back
restores the value the user last set.
"""

import sys
from unittest.mock import MagicMock

import pytest

gr = pytest.importorskip("gradio")

sys.modules.setdefault("webview", MagicMock(settings={}))

from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.fixture(autouse=True)
def no_map_figure(monkeypatch):
    monkeypatch.setattr(gu, "plot_map_scatter_mapbox", lambda *a, **k: None)


@pytest.fixture
def appdir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "APPDIR", tmp_path)
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr(gs, "_PERSISTED", [])
    return tmp_path


def build():
    with gr.Blocks() as demo:
        sample, _, model = gu.sample_species_model_settings(gs.TabState("multi"))
    radio = model["model_selection_radio"]
    slider = sample["sensitivity_slider"]
    handler = next(
        event.fn
        for event in demo.fns.values()
        if event.targets
        and event.targets[0] == (radio._id, "change")
        and slider in event.outputs
    )
    return slider, handler


def test_slider_disabled_for_3_0_and_restored_for_2_4(appdir):
    settings.set_tab_setting("multi", "sensitivity_slider", 1.25)
    settings.set_tab_setting("multi", "model_selection_radio", gu._USE_BIRDNET_2_4)

    slider, on_model_change = build()
    assert slider.interactive
    assert slider.value == 1.25

    to_3_0, *_ = on_model_change(gu._USE_BIRDNET_3_0, gu._ALL_SPECIES)
    assert to_3_0["interactive"] is False
    assert to_3_0["value"] == 1.0

    back_to_2_4, *_ = on_model_change(gu._USE_BIRDNET_2_4, gu._ALL_SPECIES)
    assert back_to_2_4["interactive"] is True
    assert back_to_2_4["value"] == 1.25, "the user's sensitivity comes back"

    to_perch, *_ = on_model_change(gu._USE_PERCH, gu._ALL_SPECIES)
    assert to_perch["interactive"] is False
    assert to_perch["value"] == 1.0


def test_slider_built_disabled_at_1_0_when_3_0_is_persisted(appdir):
    settings.set_tab_setting("multi", "sensitivity_slider", 1.25)
    settings.set_tab_setting("multi", "model_selection_radio", gu._USE_BIRDNET_3_0)

    slider, _ = build()
    assert not slider.interactive
    assert slider.value == 1.0


def test_programmatic_value_on_disabled_slider_snaps_back_to_1_0(appdir):
    settings.set_tab_setting("multi", "model_selection_radio", gu._USE_BIRDNET_3_0)

    with gr.Blocks() as demo:
        sample, _, _ = gu.sample_species_model_settings(gs.TabState("multi"))
    slider = sample["sensitivity_slider"]
    on_slider_change = next(
        event.fn
        for event in demo.fns.values()
        if event.targets
        and event.targets[0] == (slider._id, "change")
        and slider in event.outputs
    )

    assert on_slider_change(1.3, gu._USE_BIRDNET_3_0) == gr.update(value=1.0)
    assert on_slider_change(1.0, gu._USE_BIRDNET_3_0) == gr.update()
    assert on_slider_change(1.3, gu._USE_BIRDNET_2_4) == gr.update()
    assert on_slider_change(1.3, gu._USE_PERCH) == gr.update(value=1.0)
