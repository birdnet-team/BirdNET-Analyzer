"""The sensitivity slider follows the selected model.

Every BirdNET model and custom classifier takes a sensitivity. Selecting Perch
disables the slider and shows 1.0 (what the analysis will use); switching back
restores the value the user last set.
"""

import pytest

gr = pytest.importorskip("gradio")

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


@pytest.fixture
def perch_offered(monkeypatch):
    """macOS drops Perch from the choices, and a persisted choice that is not
    offered falls back to the default model."""
    choices = [*gu.model_choices(), gu._USE_PERCH]
    monkeypatch.setattr(gu, "model_choices", lambda: list(dict.fromkeys(choices)))


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


def test_slider_disabled_for_perch_and_restored_for_birdnet(appdir):
    settings.set_tab_setting("multi", "sensitivity_slider", 1.25)
    settings.set_tab_setting("multi", "model_selection_radio", gu._USE_BIRDNET_2_4)

    slider, on_model_change = build()
    assert slider.interactive
    assert slider.value == 1.25

    to_3_0, *_ = on_model_change(gu._USE_BIRDNET_3_0, gu._ALL_SPECIES)
    assert to_3_0["interactive"] is True
    assert to_3_0["value"] == 1.25

    to_perch, *_ = on_model_change(gu._USE_PERCH, gu._ALL_SPECIES)
    assert to_perch["interactive"] is False
    assert to_perch["value"] == 1.0

    to_custom, *_ = on_model_change(gu._CUSTOM_CLASSIFIER, gu._ALL_SPECIES)
    assert to_custom["interactive"] is True
    assert to_custom["value"] == 1.25, "the user's sensitivity comes back"


def test_slider_built_disabled_at_1_0_when_perch_is_persisted(appdir, perch_offered):
    settings.set_tab_setting("multi", "sensitivity_slider", 1.25)
    settings.set_tab_setting("multi", "model_selection_radio", gu._USE_PERCH)

    slider, _ = build()
    assert not slider.interactive
    assert slider.value == 1.0


def test_programmatic_value_on_disabled_slider_snaps_back_to_1_0(appdir, perch_offered):
    settings.set_tab_setting("multi", "model_selection_radio", gu._USE_PERCH)

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

    assert on_slider_change(1.3, gu._USE_PERCH) == gr.update(value=1.0)
    assert on_slider_change(1.0, gu._USE_PERCH) == gr.update()
    assert on_slider_change(1.3, gu._USE_BIRDNET_3_0) == gr.update()
    assert on_slider_change(1.3, gu._CUSTOM_CLASSIFIER) == gr.update()


def test_an_unusable_classifier_is_not_selected(appdir, monkeypatch):
    import birdnet_analyzer.gui.localization as loc
    from birdnet_analyzer import model_utils

    def reject(path):
        raise ValueError("looks like a detached classifier head")

    warnings = []
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: "head_detached.tflite")
    monkeypatch.setattr(model_utils, "classifier_version", reject)
    monkeypatch.setattr(
        gr, "Warning", lambda message, *a, **k: warnings.append(message)
    )

    with gr.Blocks() as demo:
        _, _, model = gu.sample_species_model_settings(gs.TabState("multi"))
    state = model["selected_classifier_state"]
    on_select = next(
        fn.fn
        for fn in demo.fns.values()
        if fn.targets and fn.targets[0][1] == "click" and state in fn.outputs
    )

    assert on_select() == (gr.skip(), gr.skip(), gr.skip())
    assert warnings == [
        loc.localize("species-list-custom-classifier-invalid-warning").format(
            "looks like a detached classifier head"
        )
    ]
