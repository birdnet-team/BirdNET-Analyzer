"""A restart keeps the custom classifier and species list the user worked with."""

import inspect
import os

import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import model_utils, settings  # noqa: E402
from birdnet_analyzer.gui import multi_file, presets, single_file  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402

BLACKBIRD = "Turdus merula_Common Blackbird"


class Tab:
    def __init__(self, **persisted):
        settings.update_tab_settings("multi", persisted)
        state = gs.TabState("multi")

        with gr.Blocks() as self.demo:
            self.controls = presets.PresetControls("multi")
            _, species, model = gu.sample_species_model_settings(state)
            self.controls.wire(
                state,
                species_file_state=species["species_file_state"],
                species_file_input=species["species_file_input"],
                classifier_state=model["selected_classifier_state"],
                classifier_file_input=model["classifier_file_input"],
                classifier_labels_df=model["classifier_labels_df"],
            )

        self.model = model["model_selection_radio"]
        self.classifier = model["selected_classifier_state"]
        self.classifier_file = model["classifier_file_input"]
        self.labels = model["classifier_labels_df"]
        self.species = species["species_list_radio"]
        self.species_file_state = species["species_file_state"]
        self.species_file = species["species_file_input"]
        self.fitter_inputs = state.fitter_inputs()

    def handler(self, trigger, output):
        return next(
            event.fn
            for event in self.demo.fns.values()
            if event.targets and event.targets[0] == trigger and output in event.outputs
        )

    def select(self, outputs, path, monkeypatch):
        monkeypatch.setattr(gu, "select_file", lambda *a, **k: path)
        on_select = next(
            event.fn
            for event in self.demo.fns.values()
            if event.outputs[: len(outputs)] == outputs
        )
        on_select()

    def pick_classifier(self, path, monkeypatch):
        self.select(
            [self.classifier, self.classifier_file, self.labels], path, monkeypatch
        )

    def pick_species_list(self, path, monkeypatch):
        self.select([self.species_file_state, self.species_file], path, monkeypatch)

    def load_preset(self, name):
        on_load = self.handler((self.controls.load_button._id, "click"), self.model)
        on_load(name, *(component.value for component in self.fitter_inputs))


@pytest.fixture
def classifier(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        model_utils,
        "classifier_info",
        lambda path, **kwargs: SimpleNamespace(
            version="2.4", mode="replace", n_species=1
        ),
    )
    path = tmp_path / "my.tflite"
    path.write_bytes(b"")
    (tmp_path / "my_Labels.txt").write_text(BLACKBIRD, encoding="utf-8")
    return str(path)


@pytest.fixture
def species_list(tmp_path):
    path = tmp_path / "birds.txt"
    path.write_text(BLACKBIRD, encoding="utf-8")
    return str(path)


@pytest.fixture
def language(monkeypatch):
    monkeypatch.setattr(loc, "TARGET_LANGUAGE", loc.TARGET_LANGUAGE)
    monkeypatch.setattr(loc, "LANGUAGE_LOOKUP", loc.LANGUAGE_LOOKUP)

    def switch(language):
        settings.set_setting("language-id", language)
        loc.load_local_state()

    return switch


def values_of(radio) -> list:
    return [value for _, value in radio.choices]


def test_a_picked_classifier_is_selected_after_a_restart(
    appdir, classifier, monkeypatch
):
    Tab().pick_classifier(classifier, monkeypatch)
    settings.set_tab_setting("multi", "model_selection_radio", gu.CUSTOM_CLASSIFIER)

    tab = Tab()

    assert tab.model.value == gu.CUSTOM_CLASSIFIER
    assert tab.classifier.value == classifier
    assert tab.labels.visible
    assert gu._STARTUP_WARNINGS == []


def test_files_applied_by_a_preset_are_selected_after_a_restart(
    appdir, classifier, species_list
):
    settings.save_preset(
        "multi",
        "Own model",
        {
            "model_selection_radio": gu.CUSTOM_CLASSIFIER,
            gu.CLASSIFIER_FILE_KEY: classifier,
            "species_list_radio": gu.CUSTOM_SPECIES,
            gu.SPECIES_FILE_KEY: species_list,
        },
    )
    Tab().load_preset("Own model")

    tab = Tab()

    assert tab.model.value == gu.CUSTOM_CLASSIFIER
    assert tab.classifier.value == classifier
    assert tab.species.value == gu.CUSTOM_SPECIES
    assert tab.species_file_state.value == species_list


@pytest.mark.parametrize("saved", [{}, {gu.CLASSIFIER_FILE_KEY: "gone.tflite"}])
def test_a_missing_classifier_falls_back_to_the_default_model(appdir, saved):
    tab = Tab(model_selection_radio=gu.CUSTOM_CLASSIFIER, **saved)

    shown = {value: label for label, value in tab.model.choices}[tab.model.value]

    assert tab.model.value == gu.default_model()
    assert settings.get_tab_settings("multi")["model_selection_radio"] == (
        gu.default_model()
    )
    assert [
        loc.localize("startup-classifier-missing-warning").format(model=shown)
    ] == gu._STARTUP_WARNINGS


def test_a_picked_species_list_is_selected_after_a_restart(
    appdir, species_list, monkeypatch
):
    Tab().pick_species_list(species_list, monkeypatch)
    settings.set_tab_setting("multi", "species_list_radio", gu.CUSTOM_SPECIES)

    tab = Tab()

    assert tab.species.value == gu.CUSTOM_SPECIES
    assert tab.species_file_state.value == species_list
    assert tab.species_file.value["orig_name"] == "birds.txt"
    assert gu._STARTUP_WARNINGS == []


def test_a_species_list_with_blank_lines_is_not_selected(appdir, tmp_path, monkeypatch):
    path = tmp_path / "gappy.txt"
    path.write_text(f"{BLACKBIRD}\n\n{BLACKBIRD}", encoding="utf-8")

    with pytest.raises(gr.Error, match="Blank lines"):
        Tab().pick_species_list(str(path), monkeypatch)

    assert Tab().species_file_state.value is None


def test_a_deleted_species_list_falls_back_to_all_species(
    appdir, species_list, monkeypatch
):
    Tab().pick_species_list(species_list, monkeypatch)
    settings.set_tab_setting("multi", "species_list_radio", gu.CUSTOM_SPECIES)
    os.remove(species_list)

    tab = Tab()

    assert tab.species.value == gu.ALL_SPECIES
    assert [
        loc.localize("startup-species-list-missing-warning")
    ] == gu._STARTUP_WARNINGS


def test_the_saved_choices_do_not_change_with_the_language(appdir, language):
    language("de")
    german = Tab()
    language("en")
    english = Tab()

    assert german.species.choices != english.species.choices
    assert values_of(german.species) == values_of(english.species)
    assert values_of(german.model) == values_of(english.model)


@pytest.mark.parametrize(
    ("build", "analysis"),
    [
        (multi_file.build_multi_analysis_tab, "run_batch_analysis"),
        (single_file.build_single_analysis_tab, "run_single_file_analysis"),
    ],
)
def test_the_analysis_reads_the_picked_species_list_itself(appdir, build, analysis):
    with gr.Blocks() as demo:
        build()

    events = {getattr(event.fn, "__name__", None): event for event in demo.fns.values()}
    run = events[analysis]
    parameters = list(inspect.signature(run.fn).parameters)

    assert (
        run.inputs[parameters.index("species_list_file")]
        is events["on_species_file_selection_click"].outputs[0]
    )
