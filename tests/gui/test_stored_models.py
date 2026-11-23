"""The settings tab lists the models, and a model that is not there says so."""

from types import SimpleNamespace

import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import model_store, settings  # noqa: E402
from birdnet_analyzer.gui import (  # noqa: E402
    embeddings,
    multi_file,
    single_file,
    train,
)
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402
from tests.models import put_acoustic_2_4  # noqa: E402

DOWNLOADED = "settings-tab-models-status-downloaded"
MISSING = "settings-tab-models-status-missing"


@pytest.fixture
def model_directory(appdir, monkeypatch, tmp_path):
    directory = tmp_path / "models"
    monkeypatch.setenv(settings.MODEL_DIR_ENV_VAR, str(directory))

    return directory


def note(*keys) -> str:
    size = gu._format_bytes(model_store.missing_download_size(*keys))

    return loc.localize("model-download-note").format(size=size)


def statuses(handler) -> tuple[dict, str]:
    on_tab_select = handler(gu.build_settings, "on_tab_select")
    _error_log, rows, info = on_tab_select(SimpleNamespace(selected=True))

    return {row[0]: row[2] for row in rows}, info


def test_the_settings_tab_lists_which_models_are_on_the_computer(
    model_directory, handler
):
    status, _info = statuses(handler)

    assert len(status) == len(model_store._PROBES)
    assert set(status.values()) == {loc.localize(MISSING)}

    put_acoustic_2_4(model_directory)
    (model_directory / "recordings").mkdir()
    (model_directory / "recordings" / "a.wav").write_bytes(bytes(5_000_000))
    status, info = statuses(handler)

    assert status.pop("BirdNET 2.4") == loc.localize(DOWNLOADED)
    assert set(status.values()) == {loc.localize(MISSING)}
    assert "52 MB" in info


def test_another_model_directory_is_listed_as_soon_as_it_is_picked(
    model_directory, handler, monkeypatch, tmp_path
):
    monkeypatch.delenv(settings.MODEL_DIR_ENV_VAR)
    monkeypatch.setattr(settings, "FROZEN", False)
    monkeypatch.setattr(gr, "Info", lambda *a, **k: None)
    other = tmp_path / "other"
    put_acoustic_2_4(other)
    monkeypatch.setattr(gu, "select_folder", lambda **_: str(other))

    _shown, rows, _info = handler(gu.build_settings, "on_model_dir_select")()

    assert [row[2] for row in rows if row[0] == "BirdNET 2.4"] == [
        loc.localize(DOWNLOADED)
    ]

    monkeypatch.setattr(
        "birdnet.utils.local_data.get_app_data_path", lambda: tmp_path / "default"
    )

    _shown, rows, _info = handler(gu.build_settings, "on_model_dir_reset")()

    assert [row[2] for row in rows if row[0] == "BirdNET 2.4"] == [
        loc.localize(MISSING)
    ]


def test_the_open_button_shows_the_directory_in_use(
    model_directory, handler, monkeypatch
):
    opened = []
    monkeypatch.setattr(gu, "_open_directory", opened.append)

    handler(gu.build_settings, "on_model_dir_open")()

    assert opened == [gu._effective_model_directory()]


def test_a_model_that_is_not_downloaded_says_so_where_it_is_chosen(model_directory):
    settings.update_tab_settings("multi", {"model_selection_radio": gu.USE_BIRDNET_2_4})

    with gr.Blocks():
        model = gu.model_selection(gs.TabState("multi"))

    (line,) = model["refresh_download_note"]["outputs"]

    assert line.value == note(model_store.ACOUSTIC_2_4)
    assert "77 MB" in line.value

    switched = model["updates_for_model"](gu.USE_PERCH, None)[-1]

    assert switched["value"] == note(model_store.PERCH)
    assert switched["visible"]

    classifier = model["updates_for_model"](gu.CUSTOM_CLASSIFIER, None)[-1]

    assert not classifier["visible"]


def test_the_note_goes_once_the_model_is_there(model_directory):
    settings.update_tab_settings("multi", {"model_selection_radio": gu.USE_BIRDNET_2_4})

    with gr.Blocks():
        model = gu.model_selection(gs.TabState("multi"))

    refresh = model["refresh_download_note"]["fn"]

    assert refresh(gu.USE_BIRDNET_2_4, None)[0]["visible"]

    put_acoustic_2_4(model_directory)

    assert not refresh(gu.USE_BIRDNET_2_4, None)[0]["visible"]
    assert not model["updates_for_model"](gu.USE_BIRDNET_2_4, None)[-1]["visible"]


def test_a_downloaded_model_is_chosen_without_a_note(model_directory):
    put_acoustic_2_4(model_directory)
    settings.update_tab_settings("multi", {"model_selection_radio": gu.USE_BIRDNET_2_4})

    with gr.Blocks():
        model = gu.model_selection(gs.TabState("multi"))

    (line,) = model["refresh_download_note"]["outputs"]

    assert not line.visible
    assert not line.value


def test_a_location_filter_says_when_its_model_is_not_downloaded(
    model_directory, monkeypatch
):
    settings.update_tab_settings("multi", {"model_selection_radio": gu.USE_BIRDNET_2_4})

    with gr.Blocks() as demo:
        _, species, model = gu.sample_species_model_settings(gs.TabState("multi"))

    line = species["location_download_note"]
    for_model = model["updates_for_model"]

    assert line.value == note(model_store.GEO_2_4)
    assert "46 MB" in line.value
    assert for_model(gu.USE_BIRDNET_3_0, None)[-1]["value"] == note(model_store.GEO_3_0)
    assert not for_model(gu.USE_PERCH, None)[-1]["visible"]
    assert len(for_model(gu.USE_PERCH, None)) == len(model["model_outputs"])
    assert [
        line in event.outputs
        for event in demo.fns.values()
        if _name(event) == "on_model_selection_change"
    ] == [True]

    monkeypatch.setitem(model_store._PROBES, model_store.GEO_2_4, lambda: (True, 1))

    assert not for_model(gu.USE_BIRDNET_2_4, None)[-1]["visible"]


def test_a_classifier_filters_by_location_with_the_geo_model_of_its_version(
    model_directory, monkeypatch
):
    from birdnet_analyzer import model_utils

    monkeypatch.setattr(
        model_utils,
        "classifier_info",
        lambda file, **_: SimpleNamespace(version=file.partition("-")[0]),
    )

    with gr.Blocks():
        _, _species, model = gu.sample_species_model_settings(gs.TabState("multi"))

    refresh = model["refresh_download_note"]["fn"]

    assert [
        refresh(gu.CUSTOM_CLASSIFIER, file)[-1]["value"]
        for file in ("2.4-based.tflite", "3.0-based.tflite")
    ] == [note(model_store.GEO_2_4), note(model_store.GEO_3_0)]
    assert not refresh(gu.CUSTOM_CLASSIFIER, None)[-1]["visible"]


def test_the_training_tab_counts_the_export_model_too(model_directory, handler):
    refresh = handler(train.build_train_tab, "refresh_download_note")

    assert refresh("2.4")["value"] == note(*model_store.training_keys("2.4"))
    assert "201 MB" in refresh("2.4")["value"]

    put_acoustic_2_4(model_directory)

    assert refresh("2.4")["value"] == note(model_store.TRAINING_2_4)
    assert "125 MB" in refresh("2.4")["value"]


def test_the_embeddings_tab_names_the_model_of_the_chosen_database(
    model_directory, handler, hoplite_database
):
    refresh = handler(embeddings.build_embeddings_tab, "refresh_download_note")

    assert refresh(None)["value"] == note(model_store.ACOUSTIC_3_0)
    assert refresh(hoplite_database({}))["value"] == note(model_store.ACOUSTIC_2_4)

    put_acoustic_2_4(model_directory)

    assert not refresh(hoplite_database({}))["visible"]


def _name(event) -> str:
    fn = getattr(event.fn, "func", event.fn)

    return getattr(fn, "__name__", "")


@pytest.mark.parametrize(
    ("build", "run"),
    [
        (single_file.build_single_analysis_tab, "run_single_file_analysis"),
        (multi_file.build_multi_analysis_tab, "run_batch_analysis"),
        (train.build_train_tab, "train_and_show_metrics"),
        (embeddings.build_embeddings_tab, "run_embeddings_with_tqdm_tracking"),
    ],
)
def test_the_note_is_brought_up_to_date_after_a_run(appdir, build, run):
    with gr.Blocks() as demo:
        build()

    ran_before = set()

    for refresh in demo.fns.values():
        if _name(refresh) != "refresh_download_note":
            continue

        earlier = refresh.trigger_after

        while earlier is not None:
            ran_before.add(_name(demo.fns[earlier]))
            earlier = demo.fns[earlier].trigger_after

    assert run in ran_before


@pytest.mark.parametrize(
    "build", [train.build_train_tab, embeddings.build_embeddings_tab]
)
def test_the_note_follows_the_choice_it_depends_on(appdir, build):
    with gr.Blocks() as demo:
        build()

    triggers = {
        trigger
        for event in demo.fns.values()
        if _name(event) == "refresh_download_note"
        for _target, trigger in event.targets
    }

    assert "change" in triggers


def test_the_note_is_brought_up_to_date_when_its_tab_is_selected(appdir):
    with gr.Blocks() as demo:
        single_file.build_single_analysis_tab()

    triggers = {
        trigger
        for event in demo.fns.values()
        if _name(event) == "refresh_download_note"
        for _target, trigger in event.targets
    }

    assert "select" in triggers
