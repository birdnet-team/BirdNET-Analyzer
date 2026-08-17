"""The sensitivity slider follows the selected model.

Only BirdNET 2.4 and custom classifiers take a sensitivity. Selecting BirdNET 3.0 or
Perch disables the slider and shows 1.0 (what the analysis will use); switching back
restores the value the user last set.
"""

import sys
from unittest.mock import MagicMock

import pytest

gr = pytest.importorskip("gradio")

# gui.utils imports pywebview at module level, which the gui-tests extra lacks.
sys.modules.setdefault("webview", MagicMock(settings={}))

from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


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
    # The radio has several change handlers; the one that drives the slider is the one
    # listing it among its outputs.
    handler = next(
        event.fn
        for event in demo.fns.values()
        if event.targets
        and event.targets[0] == (radio._id, "change")
        and slider in event.outputs
    )
    return slider, handler


def test_slider_disabled_for_3_0_and_restored_for_2_4(appdir):
    # The user set 1.25 while on 2.4 (persisted on slider release).
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
    # A 1.25 persisted from a 2.4 session must not sit visibly on the disabled slider.
    settings.set_tab_setting("multi", "sensitivity_slider", 1.25)
    settings.set_tab_setting("multi", "model_selection_radio", gu._USE_BIRDNET_3_0)

    slider, _ = build()
    assert not slider.interactive
    assert slider.value == 1.0
