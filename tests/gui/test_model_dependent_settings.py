"""The overlap, species list, locale and model summary follow the selected model."""

import asyncio
from types import SimpleNamespace
from typing import NamedTuple

import pytest

gr = pytest.importorskip("gradio")

from gradio.state_holder import SessionState  # noqa: E402

import birdnet_analyzer.config as cfg  # noqa: E402
import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import model_utils, settings, utils  # noqa: E402
from birdnet_analyzer.gui import presets  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402

ONLY_3_0_LOCALE = "es_es"


pytestmark = pytest.mark.usefixtures("perch_offered")


class Tab(NamedTuple):
    demo: gr.Blocks
    state: gs.TabState
    controls: presets.PresetControls
    model: gr.Radio
    species: gr.Radio
    overlap: gr.Slider
    locale: gr.Dropdown
    species_file_input: gr.File
    species_file_state: gr.State
    sensitivity: gr.Slider
    classifier_state: gr.State
    labels: gr.List
    facts_row: gr.Row
    facts: list[gr.Textbox]
    facts_warning: gr.Markdown


def build(**persisted) -> Tab:
    settings.update_tab_settings("multi", persisted)
    state = gs.TabState("multi")

    with gr.Blocks() as demo:
        controls = presets.PresetControls(
            "multi", params_loader=presets.load_analysis_params
        )
        sample, species, model = gu.sample_species_model_settings(state)
        controls.wire(
            state,
            species_file_state=species["species_file_state"],
            species_file_input=species["species_file_input"],
            classifier_state=model["selected_classifier_state"],
            classifier_file_input=model["classifier_file_input"],
            classifier_labels_df=model["classifier_labels_df"],
        )

    radio = model["model_selection_radio"]
    overlap = sample["overlap_slider"]

    return Tab(
        demo,
        state,
        controls,
        radio,
        species["species_list_radio"],
        overlap,
        model["locale_dropdown"],
        species["species_file_input"],
        species["species_file_state"],
        sample["sensitivity_slider"],
        model["selected_classifier_state"],
        model["classifier_labels_df"],
        model["model_facts_row"],
        model["model_facts"],
        model["model_facts_warning"],
    )


def current(tab: Tab) -> dict:
    return {component: component.value for component in tab.state.fitter_inputs()}


def apply(tab: Tab, values: dict):
    updates, skipped = tab.state.updates_for(values, current(tab))
    return dict(zip(tab.state.components(), updates, strict=True)), skipped


def change_model(
    tab: Tab,
    model,
    species=gu.ALL_SPECIES,
    overlap=0.0,
    locale="en_us",
    classifier=None,
) -> dict:
    event = find_event(tab.demo, (tab.model._id, "change"), tab.overlap)
    updates = event.fn(model, species, overlap, locale, classifier)
    return dict(zip(event.outputs, updates, strict=True))


def values_of(choices) -> list:
    return [
        choice[1] if isinstance(choice, list | tuple) else choice for choice in choices
    ]


def perch_params_file(tmp_path, overlap: float) -> str:
    path = tmp_path / "birdnet.analyze-params.csv"
    utils.save_params_file(
        path,
        {
            "Model": "perch",
            "Segment overlap": overlap,
            "Minimum confidence": 0.3,
            "Sensitivity": 1.0,
            "Merge consecutive detections": 1,
            "Audio speed": 1.0,
            "Locale": "en_us",
        },
    )
    return str(path)


def test_a_birdnet_preset_is_applied_on_a_perch_built_tab(appdir):
    tab = build(model_selection_radio=gu.USE_PERCH)
    assert gu.PREDICT_SPECIES not in values_of(tab.species.choices)

    updates, skipped = apply(
        tab,
        {
            "model_selection_radio": gu.USE_BIRDNET_3_0,
            "species_list_radio": gu.PREDICT_SPECIES,
            "overlap_slider": 2.0,
        },
    )

    assert skipped == []
    assert updates[tab.species]["value"] == gu.PREDICT_SPECIES
    assert gu.PREDICT_SPECIES in values_of(updates[tab.species]["choices"])
    assert updates[tab.overlap]["value"] == 2.0
    assert updates[tab.overlap]["maximum"] == 2.9


def test_a_perch_params_file_is_applied_on_a_birdnet_built_tab(appdir, tmp_path):
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)
    assert tab.overlap.maximum == 2.9

    updates, skipped = apply(
        tab, presets.load_analysis_params(perch_params_file(tmp_path, 4.0))
    )

    assert skipped == []
    assert updates[tab.model]["value"] == gu.USE_PERCH
    assert updates[tab.overlap]["value"] == 4.0
    assert updates[tab.overlap]["maximum"] == 4.9


def test_switching_from_perch_to_birdnet_pulls_the_overlap_down(appdir):
    tab = build(model_selection_radio=gu.USE_PERCH, overlap_slider=4.0)
    assert tab.overlap.value == 4.0

    overlap = change_model(tab, gu.USE_BIRDNET_3_0, overlap=4.0)[tab.overlap]

    assert overlap["maximum"] == 2.9
    assert overlap["value"] == 2.9


def test_an_overlap_pulled_down_by_a_model_switch_survives_a_restart(appdir):
    tab = build(model_selection_radio=gu.USE_PERCH, overlap_slider=4.0)
    change_model(tab, gu.USE_BIRDNET_3_0, overlap=4.0)
    settings.set_tab_setting("multi", "model_selection_radio", gu.USE_BIRDNET_3_0)

    assert build().overlap.value == 2.9


def test_a_model_switch_after_a_reset_keeps_the_settings_empty(appdir):
    tab = build(model_selection_radio=gu.USE_PERCH)
    gs.reset_to_defaults()

    change_model(tab, gu.default_model())

    assert settings.get_tab_settings("multi") == {}


def test_a_birdnet_preset_pulls_the_overlap_of_a_perch_tab_down(appdir):
    tab = build(model_selection_radio=gu.USE_PERCH, overlap_slider=4.0)

    updates, skipped = apply(tab, {"model_selection_radio": gu.USE_BIRDNET_3_0})

    assert skipped == []
    assert updates[tab.overlap]["maximum"] == 2.9
    assert updates[tab.overlap]["value"] == 2.9
    assert settings.get_tab_settings("multi")["overlap_slider"] == 2.9


def test_the_sensitivity_of_a_birdnet_preset_survives_the_fitting(appdir):
    tab = build(model_selection_radio=gu.USE_PERCH)

    updates, _ = apply(
        tab,
        {"model_selection_radio": gu.USE_BIRDNET_3_0, "sensitivity_slider": 1.3},
    )

    assert updates[tab.sensitivity]["value"] == 1.3
    assert updates[tab.sensitivity]["interactive"] is True


def test_a_fitted_value_is_persisted_as_shown(appdir):
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)

    updates, _ = apply(tab, {"overlap_slider": 4.0})

    assert updates[tab.overlap]["value"] == 2.9
    assert settings.get_tab_settings("multi")["overlap_slider"] == 2.9


def test_a_locale_only_3_0_offers_is_applied_on_a_2_4_built_tab(appdir):
    tab = build(model_selection_radio=gu.USE_BIRDNET_2_4)
    assert ONLY_3_0_LOCALE not in values_of(tab.locale.choices)

    updates, skipped = apply(
        tab,
        {
            "model_selection_radio": gu.USE_BIRDNET_3_0,
            "locale_dropdown": ONLY_3_0_LOCALE,
        },
    )

    assert skipped == []
    assert updates[tab.locale]["value"] == ONLY_3_0_LOCALE
    assert updates[tab.locale]["choices"] == gu.model_languages(gu.USE_BIRDNET_3_0)
    assert updates[tab.locale]["visible"] is True


def test_a_locale_the_new_model_lacks_falls_back_to_english(appdir):
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)

    updates, skipped = apply(
        tab,
        {
            "model_selection_radio": gu.USE_BIRDNET_2_4,
            "locale_dropdown": ONLY_3_0_LOCALE,
        },
    )

    assert skipped == []
    assert updates[tab.locale]["value"] == "en_us"
    assert updates[tab.locale]["choices"] == gu.model_languages(gu.USE_BIRDNET_2_4)


def test_perch_hides_the_locale_and_keeps_it(appdir):
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)

    locale = change_model(tab, gu.USE_PERCH, locale=ONLY_3_0_LOCALE)[tab.locale]

    assert locale == gr.update(visible=False, choices=cfg.ALL_MODEL_LANGUAGES)


@pytest.mark.parametrize("model", [gu.USE_PERCH, gu.USE_BIRDNET_3_0])
def test_the_species_choices_keep_their_order_across_models(appdir, model):
    tab = build(model_selection_radio=model)

    species = change_model(tab, model)[tab.species]

    assert values_of(species["choices"]) == values_of(tab.species.choices)


@pytest.mark.parametrize(
    ("persisted", "component", "expected"),
    [
        (
            {"model_selection_radio": gu.USE_BIRDNET_3_0, "overlap_slider": 4.0},
            "overlap",
            0.0,
        ),
        (
            {
                "model_selection_radio": gu.USE_PERCH,
                "species_list_radio": gu.PREDICT_SPECIES,
            },
            "species",
            gu.ALL_SPECIES,
        ),
        (
            {
                "model_selection_radio": gu.USE_BIRDNET_2_4,
                "locale_dropdown": ONLY_3_0_LOCALE,
            },
            "locale",
            "en_us",
        ),
    ],
)
def test_a_persisted_value_the_built_model_does_not_take_is_dropped_at_build(
    appdir, persisted, component, expected
):
    tab = build(**persisted)

    assert getattr(tab, component).value == expected


def test_a_value_outside_what_any_model_takes_is_still_skipped(appdir):
    tab = build(model_selection_radio=gu.USE_PERCH)

    _, skipped = apply(
        tab,
        {
            "overlap_slider": 7.0,
            "locale_dropdown": "xx_yy",
            "species_list_radio": "Nonexistent list",
        },
    )

    assert sorted(skipped) == [
        "locale_dropdown",
        "overlap_slider",
        "species_list_radio",
    ]


def test_a_locale_outside_the_analysis_tabs_keeps_its_own_limits(appdir):
    state = gs.TabState("species")

    with gr.Blocks():
        gu.locale(state, cfg.GEO_MODEL_LANGUAGES)

    only_2_4 = next(
        language
        for language in cfg.ALL_MODEL_LANGUAGES
        if language not in cfg.GEO_MODEL_LANGUAGES
    )
    _, skipped = state.updates_for({"locale_dropdown": only_2_4})

    assert skipped == ["locale_dropdown"]


def find_event(demo, trigger, output=None, takes=None):
    return next(
        event
        for event in demo.fns.values()
        if event.targets
        and event.targets[0] == trigger
        and (output is None or output in event.outputs)
        and (takes is None or takes in event.inputs)
    )


class Session:
    """Runs handlers through gradio's own input and output processing."""

    def __init__(self, tab: Tab):
        self.tab = tab
        self.state = SessionState(tab.demo)

    def run(self, event, inputs: list) -> list:
        return asyncio.run(self.tab.demo.process_api(event, inputs, self.state))["data"]

    def load(self, button, *args) -> dict:
        """Loads through a preset button and returns the updates by component."""
        event = find_event(self.tab.demo, (button._id, "click"), self.tab.overlap)
        live = [component.value for component in self.tab.state.fitter_inputs()]
        data = self.run(event, [*args, *live])
        return dict(zip(event.outputs, data, strict=True))

    def model_change(self, updates: dict):
        """The model change the frontend fires with the values the load showed."""
        tab = self.tab
        shown = [
            updates[c].get("value", c.value)
            for c in (tab.model, tab.species, tab.overlap, tab.locale)
        ]
        return self.run(
            find_event(tab.demo, (tab.model._id, "change"), tab.overlap),
            [*shown, None],
        )


def test_a_preset_for_another_model_passes_gradios_input_checks(appdir):
    settings.save_preset(
        "multi",
        "BirdNET",
        {
            "model_selection_radio": gu.USE_BIRDNET_3_0,
            "species_list_radio": gu.PREDICT_SPECIES,
            "overlap_slider": 2.0,
            "locale_dropdown": ONLY_3_0_LOCALE,
        },
    )
    tab = build(model_selection_radio=gu.USE_PERCH)
    session = Session(tab)
    show_species_choice = find_event(
        tab.demo, (tab.species._id, "change"), takes=tab.species_file_state
    )

    updates = session.load(tab.controls.load_button, "BirdNET")
    session.run(show_species_choice, [gu.PREDICT_SPECIES, None])
    session.model_change(updates)

    assert updates[tab.species]["value"] == gu.PREDICT_SPECIES
    assert updates[tab.overlap]["value"] == 2.0
    assert updates[tab.overlap]["maximum"] == 2.9
    assert updates[tab.locale]["value"] == ONLY_3_0_LOCALE


def test_a_preset_the_model_change_narrows_leaves_nothing_stale(appdir):
    # The species radio fires its change with what the load showed, which gradio may
    # process after the model change has narrowed the choices.
    settings.save_preset(
        "multi",
        "Perch",
        {
            "model_selection_radio": gu.USE_PERCH,
            "species_list_radio": gu.PREDICT_SPECIES,
        },
    )
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)
    session = Session(tab)
    show_species_choice = find_event(
        tab.demo, (tab.species._id, "change"), takes=tab.species_file_state
    )

    updates = session.load(tab.controls.load_button, "Perch")
    session.model_change(updates)
    session.run(show_species_choice, [updates[tab.species]["value"], None])

    assert updates[tab.species]["value"] == gu.ALL_SPECIES


def test_a_locale_the_model_change_narrows_leaves_nothing_stale(appdir):
    settings.save_preset("multi", "2.4", {"model_selection_radio": gu.USE_BIRDNET_2_4})
    tab = build(
        model_selection_radio=gu.USE_BIRDNET_3_0, locale_dropdown=ONLY_3_0_LOCALE
    )
    session = Session(tab)
    warn_unmatched_species = find_event(
        tab.demo, (tab.species_file_input._id, "change"), takes=tab.locale
    )

    updates = session.load(tab.controls.load_button, "2.4")
    session.model_change(updates)
    session.run(
        warn_unmatched_species, [None, gu.USE_BIRDNET_2_4, updates[tab.locale]["value"]]
    )

    assert updates[tab.locale]["value"] == "en_us"


def test_a_hidden_locale_takes_any_models_language(appdir):
    settings.save_preset(
        "multi",
        "Perch",
        {"model_selection_radio": gu.USE_PERCH, "locale_dropdown": ONLY_3_0_LOCALE},
    )
    tab = build(model_selection_radio=gu.USE_BIRDNET_2_4)
    session = Session(tab)
    warn_unmatched_species = find_event(
        tab.demo, (tab.species_file_input._id, "change"), takes=tab.locale
    )

    updates = session.load(tab.controls.load_button, "Perch")
    session.model_change(updates)
    session.run(warn_unmatched_species, [None, gu.USE_PERCH, ONLY_3_0_LOCALE])

    assert updates[tab.locale]["value"] == ONLY_3_0_LOCALE
    assert updates[tab.locale]["visible"] is False


def test_a_params_file_arrives_fitted(appdir, tmp_path, monkeypatch):
    path = perch_params_file(tmp_path, 4.0)
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: path)
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)
    session = Session(tab)

    updates = session.load(tab.controls.load_params_file_button)
    session.model_change(updates)

    assert updates[tab.overlap]["maximum"] == 4.9
    assert updates[tab.overlap]["value"] == 4.0
    assert values_of(updates[tab.species]["choices"]) == [
        gu.ALL_SPECIES,
        gu.CUSTOM_SPECIES,
    ]


def classifier_with_labels(tmp_path, count: int) -> str:
    path = tmp_path / "my.tflite"
    path.write_bytes(b"")
    (tmp_path / "my_Labels.txt").write_text(
        "\n".join(f"Genus species{i}_Bird {i}" for i in range(count)), encoding="utf-8"
    )
    return str(path)


def fake_classifier(monkeypatch, version="3.0", mode="replace", n_species=12):
    monkeypatch.setattr(
        model_utils,
        "classifier_info",
        lambda path, **kwargs: SimpleNamespace(
            version=version, mode=mode, n_species=n_species
        ),
    )


def summary(tab: Tab, updates: dict) -> dict:
    return {
        "visible": updates[tab.facts_row]["visible"],
        "tiles": [
            (updates[tile].get("label"), updates[tile].get("value"))
            for tile in tab.facts
        ],
        "warning": updates[tab.facts_warning],
    }


def test_the_summary_describes_the_built_model(appdir):
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)

    assert tab.facts_row.visible
    assert [(box.label, box.value) for box in tab.facts] == [
        ("Segment length", "3 s"),
        ("Classes", "11\u202f560"),
        ("Label languages", str(len(gu.model_languages(gu.USE_BIRDNET_3_0)))),
        ("Species by location", "Available"),
    ]
    assert not tab.facts_warning.visible


def test_switching_to_perch_updates_the_summary(appdir):
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)

    shown = summary(tab, change_model(tab, gu.USE_PERCH))

    assert shown["visible"]
    assert shown["tiles"] == [
        ("Segment length", "5 s"),
        ("Classes", "14\u202f795"),
        ("Label languages", "Scientific names only"),
        ("Species by location", "Not available"),
    ]
    assert shown["warning"]["visible"] is False


def test_a_custom_classifier_shows_what_it_was_trained_on(
    appdir, tmp_path, monkeypatch
):
    fake_classifier(monkeypatch, version="2.4", mode="append", n_species=6522 + 4)
    classifier = classifier_with_labels(tmp_path, 6522 + 4)
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)

    shown = summary(tab, change_model(tab, gu.CUSTOM_CLASSIFIER, classifier=classifier))

    assert shown["tiles"] == [
        ("Based on", "BirdNET 2.4"),
        ("Classes", "4"),
        ("Output", "Official classes + custom"),
        ("Segment length", "3 s"),
    ]
    assert shown["warning"]["visible"] is False


def page_load(tab: Tab, model, classifier) -> dict:
    event = find_event(tab.demo, (tab.demo._id, "load"), tab.facts_row)
    return dict(zip(event.outputs, event.fn(model, classifier), strict=True))


def test_a_persisted_classifier_is_summarized_once_the_page_loads(
    appdir, tmp_path, monkeypatch
):
    fake_classifier(monkeypatch, n_species=12)
    classifier = classifier_with_labels(tmp_path, 12)

    tab = build(
        model_selection_radio=gu.CUSTOM_CLASSIFIER, custom_classifier_file=classifier
    )
    assert not tab.facts_row.visible

    shown = summary(tab, page_load(tab, gu.CUSTOM_CLASSIFIER, classifier))

    assert shown["visible"]
    assert shown["tiles"] == [
        ("Based on", "BirdNET 3.0"),
        ("Classes", "12"),
        ("Output", "Custom classes only"),
        ("Segment length", "3 s"),
    ]


def test_a_label_file_of_the_wrong_length_is_flagged(appdir, tmp_path, monkeypatch):
    fake_classifier(monkeypatch, n_species=12)
    classifier = classifier_with_labels(tmp_path, 11)
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)

    shown = summary(tab, change_model(tab, gu.CUSTOM_CLASSIFIER, classifier=classifier))

    assert shown["visible"]
    assert shown["warning"]["visible"] is True
    assert shown["warning"]["value"] == loc.localize(
        "model-selection-labels-mismatch-warning"
    ).format(labels=11, classes=12, file="my_Labels.txt")


def test_without_a_readable_classifier_there_is_no_summary(
    appdir, tmp_path, monkeypatch
):
    tab = build(model_selection_radio=gu.USE_BIRDNET_3_0)

    assert not summary(tab, change_model(tab, gu.CUSTOM_CLASSIFIER))["visible"]

    gone = str(tmp_path / "gone.tflite")
    shown = summary(tab, change_model(tab, gu.CUSTOM_CLASSIFIER, classifier=gone))

    assert not shown["visible"]
    assert shown["warning"]["visible"] is True
    assert "gone.tflite" in shown["warning"]["value"]
    assert tmp_path.name not in shown["warning"]["value"]

    def reject(path, **kwargs):
        raise ValueError(
            f"Failed to inspect custom TFLite model '{path}': not a classifier"
        ) from ValueError("not a classifier")

    monkeypatch.setattr(model_utils, "classifier_info", reject)
    shown = summary(
        tab, change_model(tab, gu.CUSTOM_CLASSIFIER, classifier="head.tflite")
    )

    assert shown["warning"]["value"] == loc.localize(
        "model-selection-classifier-unreadable-warning"
    ).format(file="head.tflite", error="not a classifier")


def test_an_unreadable_persisted_classifier_is_flagged_once_the_page_loads(
    appdir, tmp_path, monkeypatch
):
    def reject(path, **kwargs):
        raise ValueError("not a classifier")

    monkeypatch.setattr(model_utils, "classifier_info", reject)
    classifier = classifier_with_labels(tmp_path, 1)
    tab = build(
        model_selection_radio=gu.CUSTOM_CLASSIFIER, custom_classifier_file=classifier
    )

    shown = summary(tab, page_load(tab, gu.CUSTOM_CLASSIFIER, classifier))

    assert not shown["visible"]
    assert shown["warning"]["visible"] is True
    assert "my.tflite" in shown["warning"]["value"]


def test_a_persisted_mismatched_label_file_is_flagged_once_the_page_loads(
    appdir, tmp_path, monkeypatch
):
    fake_classifier(monkeypatch, n_species=12)
    classifier = classifier_with_labels(tmp_path, 11)
    tab = build(
        model_selection_radio=gu.CUSTOM_CLASSIFIER, custom_classifier_file=classifier
    )

    shown = summary(tab, page_load(tab, gu.CUSTOM_CLASSIFIER, classifier))

    assert shown["visible"]
    assert shown["warning"]["visible"] is True
    assert shown["warning"]["value"] == loc.localize(
        "model-selection-labels-mismatch-warning"
    ).format(labels=11, classes=12, file="my_Labels.txt")


def test_a_classifier_stored_with_a_birdnet_preset_keeps_the_models_view(
    appdir, tmp_path, monkeypatch
):
    fake_classifier(monkeypatch, n_species=12)
    classifier = classifier_with_labels(tmp_path, 12)
    tab = build()
    event = find_event(tab.demo, (tab.classifier_state._id, "change"), tab.facts_row)

    updates = dict(
        zip(event.outputs, event.fn(gu.USE_BIRDNET_3_0, classifier), strict=True)
    )

    shown = summary(tab, updates)
    assert shown["tiles"][0] == ("Segment length", "3 s")
    assert shown["tiles"][1][0] == "Classes"
    assert updates[tab.labels]["visible"] is False


def test_a_persisted_classifier_is_inspected_before_the_first_switch(
    appdir, tmp_path, monkeypatch
):
    calls = []

    def record(path, **kwargs):
        calls.append(path)
        return SimpleNamespace(version="3.0", mode="replace", n_species=12)

    monkeypatch.setattr(model_utils, "classifier_info", record)
    classifier = classifier_with_labels(tmp_path, 12)
    tab = build(
        model_selection_radio=gu.USE_BIRDNET_3_0, custom_classifier_file=classifier
    )
    assert calls == []

    load = find_event(tab.demo, (tab.demo._id, "load"), tab.facts_row)

    assert load.fn(gu.USE_BIRDNET_3_0, classifier) == gr.skip()
    assert calls == [classifier]


def test_repicking_the_same_classifier_refreshes_the_summary(
    appdir, tmp_path, monkeypatch
):
    fake_classifier(monkeypatch, n_species=12)
    classifier = classifier_with_labels(tmp_path, 11)
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: classifier)
    tab = build(model_selection_radio=gu.CUSTOM_CLASSIFIER)
    pick = pick_event(tab)

    first = dict(zip(pick.outputs, pick.fn(), strict=True))
    classifier_with_labels(tmp_path, 12)
    second = dict(zip(pick.outputs, pick.fn(), strict=True))

    assert summary(tab, first)["warning"]["visible"] is True
    assert summary(tab, second)["warning"]["visible"] is False


def test_picking_a_classifier_again_retries_a_failed_inspection(
    appdir, tmp_path, monkeypatch
):
    calls = []

    def inspect(path, *, library):
        calls.append(path)
        if len(calls) == 1:
            raise ValueError("Failed to inspect model due to timeout.")
        return SimpleNamespace(version="3.0", mode="replace", n_species=12)

    monkeypatch.setattr(model_utils.birdnet, "inspect_custom_tflite", inspect)
    model_utils._INSPECTIONS.clear()
    classifier = classifier_with_labels(tmp_path, 12)
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: classifier)
    tab = build(model_selection_radio=gu.CUSTOM_CLASSIFIER)
    pick = pick_event(tab)

    assert pick.fn() == gr.skip()
    second = dict(zip(pick.outputs, pick.fn(), strict=True))

    assert summary(tab, second)["tiles"][1] == ("Classes", "12")
    assert len(calls) == 2


def pick_event(tab: Tab):
    return next(
        event
        for event in tab.demo.fns.values()
        if event.targets[0][1] == "click" and tab.facts_row in event.outputs
    )


def test_cancelling_the_file_dialog_changes_nothing(appdir, monkeypatch):
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: None)
    tab = build(model_selection_radio=gu.CUSTOM_CLASSIFIER)

    assert pick_event(tab).fn() == gr.skip()


def test_a_model_switch_is_a_single_update(appdir):
    tab = build()

    listeners = [
        event
        for event in tab.demo.fns.values()
        if (tab.model._id, "change") in event.targets
    ]

    assert len(listeners) == 1
    assert all(
        component in listeners[0].outputs
        for component in (tab.sensitivity, tab.locale, tab.facts_row)
    )


def test_the_summary_follows_the_picked_classifier(appdir, tmp_path, monkeypatch):
    fake_classifier(monkeypatch, n_species=12)
    classifier = classifier_with_labels(tmp_path, 12)
    tab = build()
    event = find_event(tab.demo, (tab.classifier_state._id, "change"), tab.facts_row)

    updates = event.fn(gu.CUSTOM_CLASSIFIER, classifier)

    shown = summary(tab, dict(zip(event.outputs, updates, strict=True)))
    assert shown["tiles"][1] == ("Classes", "12")
