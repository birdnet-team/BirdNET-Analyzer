"""The sensitivity slider follows the selected model."""

import pytest

gr = pytest.importorskip("gradio")

from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


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
    settings.set_tab_setting("multi", "model_selection_radio", gu.USE_BIRDNET_2_4)

    slider, on_model_change = build()
    assert slider.interactive
    assert slider.value == 1.25

    to_3_0, *_ = on_model_change(gu.USE_BIRDNET_3_0, gu.ALL_SPECIES, 0.0, "en_us", None)
    assert to_3_0["interactive"] is True
    assert to_3_0["value"] == 1.25

    to_perch, *_ = on_model_change(gu.USE_PERCH, gu.ALL_SPECIES, 0.0, "en_us", None)
    assert to_perch["interactive"] is False
    assert to_perch["value"] == 1.0

    to_custom, *_ = on_model_change(
        gu.CUSTOM_CLASSIFIER, gu.ALL_SPECIES, 0.0, "en_us", None
    )
    assert to_custom["interactive"] is True
    assert to_custom["value"] == 1.25, "the user's sensitivity comes back"


def test_slider_built_disabled_at_1_0_when_perch_is_persisted(appdir, perch_offered):
    settings.set_tab_setting("multi", "sensitivity_slider", 1.25)
    settings.set_tab_setting("multi", "model_selection_radio", gu.USE_PERCH)

    slider, _ = build()
    assert not slider.interactive
    assert slider.value == 1.0


@pytest.mark.parametrize(
    ("model", "shown"),
    [
        (gu.USE_PERCH, gr.update(interactive=False, value=1.0)),
        (gu.USE_BIRDNET_3_0, gr.update(interactive=True, value=1.3)),
        (gu.CUSTOM_CLASSIFIER, gr.update(interactive=True, value=1.3)),
    ],
)
def test_a_preset_shows_its_sensitivity_only_where_the_model_takes_one(
    appdir, perch_offered, model, shown
):
    settings.set_tab_setting("multi", "model_selection_radio", gu.USE_PERCH)
    state = gs.TabState("multi")

    with gr.Blocks():
        sample, _, _ = gu.sample_species_model_settings(state)
    current = {component: component.value for component in state.fitter_inputs()}

    updates, _ = state.updates_for(
        {"model_selection_radio": model, "sensitivity_slider": 1.3}, current
    )

    shown_by_component = dict(zip(state.components(), updates, strict=True))
    assert shown_by_component[sample["sensitivity_slider"]] == shown


def test_an_unusable_classifier_is_not_selected(appdir, monkeypatch):
    import birdnet_analyzer.gui.localization as loc
    from birdnet_analyzer import model_utils

    def reject(path, **kwargs):
        raise ValueError("looks like a detached classifier head")

    warnings = []
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: "head_detached.tflite")
    monkeypatch.setattr(model_utils, "classifier_info", reject)
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

    assert on_select() == gr.skip()
    assert warnings == [
        loc.localize("species-list-custom-classifier-invalid-warning").format(
            "looks like a detached classifier head"
        )
    ]
