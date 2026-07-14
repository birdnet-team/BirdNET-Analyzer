import csv

import pytest

# Building components needs gradio, which only comes with the gui and gui-tests extras.
gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import presets  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402

# The header row of a BirdNET_analysis_params.csv, as the analysis writes it.
PARAMS_HEADERS = [
    "Model",
    "BirdNET version",
    "Segment length",
    "Sample rate",
    "Segment overlap",
    "Bandpass filter minimum",
    "Bandpass filter maximum",
    "Merge consecutive detections",
    "Audio speed",
    "Minimum confidence",
    "Sensitivity",
    "Top N",
    "Batch size",
    "Number of workers",
    "Number of producers",
    "Result type(s)",
    "Additional columns",
    "Latitude",
    "Longitude",
    "Week",
    "Species filter threshold",
    "Species list file",
    "Locale",
    "Custom classifier path",
    "Custom classifier species list",
    "Split tables",
]


@pytest.fixture
def appdir(monkeypatch, tmp_path):
    """Points the presets and the GUI state at tmp_path."""
    monkeypatch.setattr(settings, "APPDIR", tmp_path)
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr(gs, "_PERSISTED", [])

    return tmp_path


def params_file(tmp_path, **overrides):
    """Writes an analysis parameters file like the one an analysis saves."""
    values = dict.fromkeys(PARAMS_HEADERS, "")
    values.update(
        {
            "Model": "birdnet",
            "BirdNET version": "2.4",
            "Segment length": "3.0",
            "Sample rate": "48000",
            "Segment overlap": "0.5",
            "Bandpass filter minimum": "150",
            "Bandpass filter maximum": "12000",
            "Merge consecutive detections": "3",
            "Audio speed": "1.0",
            "Minimum confidence": "0.3",
            "Sensitivity": "1.2",
            "Batch size": "8",
            "Number of workers": "4",
            "Number of producers": "2",
            "Result type(s)": "table, csv",
            "Additional columns": "lat, lon",
            "Species filter threshold": "0.05",
            "Locale": "en_us",
            "Split tables": "False",
        }
    )
    values.update(overrides)
    path = tmp_path / "BirdNET_analysis_params.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PARAMS_HEADERS)
        writer.writerow([values[header] for header in PARAMS_HEADERS])

    return str(path)


def test_a_saved_preset_can_be_loaded_and_deleted(appdir):
    settings.save_preset("multi", "Project A", {"confidence_slider": 0.5})

    assert settings.list_presets("multi") == ["Project A"]
    assert settings.load_preset("multi", "Project A") == {"confidence_slider": 0.5}

    settings.delete_preset("multi", "Project A")

    assert settings.list_presets("multi") == []
    assert settings.load_preset("multi", "Project A") is None


def test_presets_are_kept_per_tab(appdir):
    settings.save_preset("multi", "Wetlands", {"confidence_slider": 0.5})
    settings.save_preset("train", "Wetlands", {"epochs_number": 100})

    assert settings.load_preset("multi", "Wetlands") == {"confidence_slider": 0.5}
    assert settings.load_preset("train", "Wetlands") == {"epochs_number": 100}


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "..",
        "../escape",
        "a/b",
        "a\\b",
        "trailing.",
        "trailing ",
        ".hidden",
        "x" * (settings.PRESET_NAME_MAX_LENGTH + 1),
    ],
)
def test_a_name_that_does_not_fit_a_file_is_rejected(appdir, name):
    with pytest.raises(ValueError, match="Invalid preset name"):
        settings.save_preset("multi", name, {})

    assert settings.list_presets("multi") == []


@pytest.mark.parametrize("name", ["Project A", "wetlands_2026", "v1.2-final"])
def test_a_usable_name_is_accepted(appdir, name):
    settings.save_preset("multi", name, {})

    assert settings.list_presets("multi") == [name]


def build_tab():
    tab = gs.TabState("multi")

    with gr.Blocks() as demo:
        controls = presets.PresetControls("multi")
        tab.persist(
            "confidence_slider", gr.Slider, minimum=0.05, maximum=0.95, value=0.25
        )
        tab.persist(
            "output_type_checkboxgroup",
            gr.CheckboxGroup,
            choices=[("Raven", "table"), ("CSV", "csv")],
            value=["table"],
        )
        controls.wire(tab)

    return tab, controls, demo


def test_snapshot_pairs_the_values_with_their_keys(appdir):
    tab, _, _ = build_tab()

    assert tab.snapshot([0.5, ["csv"]]) == {
        "confidence_slider": 0.5,
        "output_type_checkboxgroup": ["csv"],
    }


def test_updates_apply_only_the_values_the_components_take(appdir):
    tab, _, _ = build_tab()

    updates, skipped = tab.updates_for(
        {
            "confidence_slider": 0.5,
            "output_type_checkboxgroup": ["csv", "parquet"],  # parquet is not offered
            "epochs_number": 100,  # not a setting of this tab
        }
    )

    assert updates == [gr.update(value=0.5), gr.update()]
    assert sorted(skipped) == ["epochs_number", "output_type_checkboxgroup"]
    # What was applied is persisted, like a user edit.
    assert settings.get_tab_settings("multi") == {"confidence_slider": 0.5}


def find_click_handler(demo, button):
    events = [
        event
        for event in demo.fns.values()
        if event.targets and event.targets[0] == (button._id, "click")
    ]

    assert len(events) == 1

    return events[0].fn


def test_saving_and_loading_a_preset_through_the_controls(appdir):
    _, controls, demo = build_tab()

    dropdown_update = find_click_handler(demo, controls.save_button)(
        " Project A ", 0.5, ["csv"]
    )

    assert dropdown_update["choices"] == ["Project A"]
    assert dropdown_update["value"] == "Project A"
    assert settings.load_preset("multi", "Project A") == {
        "confidence_slider": 0.5,
        "output_type_checkboxgroup": ["csv"],
    }

    updates = find_click_handler(demo, controls.load_button)("Project A")

    assert updates == [gr.update(value=0.5), gr.update(value=["csv"])]

    dropdown_update = find_click_handler(demo, controls.delete_button)("Project A")

    assert dropdown_update["choices"] == []
    assert settings.list_presets("multi") == []


def test_saving_under_an_invalid_name_shows_an_error(appdir):
    _, controls, demo = build_tab()

    with pytest.raises(gr.Error):
        find_click_handler(demo, controls.save_button)("../escape", 0.5, ["csv"])

    assert settings.list_presets("multi") == []


def test_loading_without_a_selection_changes_nothing(appdir):
    _, controls, demo = build_tab()

    updates = find_click_handler(demo, controls.load_button)(None)

    assert updates == [gr.skip(), gr.skip()]


def test_the_analysis_params_of_a_previous_run_are_read_back(appdir, tmp_path):
    values = presets.load_analysis_params(params_file(tmp_path))

    assert values == {
        "confidence_slider": 0.3,
        "sensitivity_slider": 1.2,
        "overlap_slider": 0.5,
        "merge_consecutive_slider": 3,
        "audio_speed_slider": 1,
        "fmin_number": 150,
        "fmax_number": 12000,
        "sf_thresh_number": 0.05,
        "batch_size_number": 8,
        "workers_number": 4,
        "producers_number": 2,
        "use_top_n_checkbox": False,
        "yearlong_checkbox": True,
        "locale_dropdown": "en_us",
        "output_type_checkboxgroup": ["table", "csv"],
        "additional_columns_checkboxgroup": ["lat", "lon"],
        "split_tables_checkbox": False,
        "species_list_radio": loc.localize("species-list-radio-option-all"),
        "model_selection_radio": "BirdNET 2.4",
    }


def test_a_top_n_analysis_does_not_restore_the_confidence_placeholder(appdir, tmp_path):
    # With top N in use the analysis runs without a confidence threshold and stores 0.
    values = presets.load_analysis_params(
        params_file(tmp_path, **{"Top N": "5", "Minimum confidence": "0"})
    )

    assert values["use_top_n_checkbox"] is True
    assert values["top_n_input"] == 5
    assert "confidence_slider" not in values


def test_a_location_based_analysis_restores_the_predicted_species_list(
    appdir, tmp_path
):
    values = presets.load_analysis_params(
        params_file(tmp_path, Latitude="42.5", Longitude="-76.4", Week="20")
    )

    assert values["species_list_radio"] == loc.localize(
        "species-list-radio-option-predict-list"
    )
    assert values["lat_number"] == 42.5
    assert values["lon_number"] == -76.4
    assert values["week_number"] == 20
    assert values["yearlong_checkbox"] is False


def test_a_species_list_analysis_restores_the_custom_list(appdir, tmp_path):
    values = presets.load_analysis_params(
        params_file(tmp_path, **{"Species list file": "/data/birds.txt"})
    )

    assert values["species_list_radio"] == loc.localize(
        "species-list-radio-option-custom-list"
    )
    assert values[presets.SPECIES_FILE_KEY] == "/data/birds.txt"


def test_a_custom_classifier_analysis_restores_the_classifier(appdir, tmp_path):
    values = presets.load_analysis_params(
        params_file(tmp_path, **{"Custom classifier path": "/models/my.tflite"})
    )

    assert values["model_selection_radio"] == loc.localize(
        "species-list-radio-option-custom-classifier"
    )
    assert values[presets.CLASSIFIER_FILE_KEY] == "/models/my.tflite"


@pytest.mark.parametrize(
    ("stored_speed", "slider"),
    [
        ("2.0", 2),  # faster than realtime stays on the positive side
        ("0.5", -2),  # slower than realtime is undone back to the negative side
        ("1.0", 1),
    ],
)
def test_the_audio_speed_factor_is_undone_to_the_slider_value(
    appdir, tmp_path, stored_speed, slider
):
    values = presets.load_analysis_params(
        params_file(tmp_path, **{"Audio speed": stored_speed})
    )

    assert values["audio_speed_slider"] == slider


def test_a_file_that_is_no_params_file_is_rejected(appdir, tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("Start (s),End (s),Confidence\n0,3,0.8\n", encoding="utf-8")

    with pytest.raises(gr.Error):
        presets.load_analysis_params(str(path))
