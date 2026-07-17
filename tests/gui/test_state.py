import json

import pytest

# Building components needs gradio, which only comes with the gui and gui-tests extras.
gr = pytest.importorskip("gradio")

from birdnet_analyzer import logs, settings  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402


@pytest.fixture
def state_file(monkeypatch, tmp_path):
    """Points the GUI state at a state.json inside tmp_path.

    The error log goes to tmp_path too: reading a broken state file logs the error,
    which only reaches a file when logging is configured the way the apps do it.
    """
    path = tmp_path / "state.json"
    monkeypatch.setattr(settings, "APPDIR", tmp_path)
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(path))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr(gs, "_PERSISTED", [])
    logs.setup_logging()

    yield path

    logs._remove_installed_handlers()


def test_settings_are_kept_per_tab(state_file):
    settings.set_tab_setting("single", "confidence_slider", 0.1)
    settings.set_tab_setting("multi", "confidence_slider", 0.9)

    assert settings.get_tab_settings("single") == {"confidence_slider": 0.1}
    assert settings.get_tab_settings("multi") == {"confidence_slider": 0.9}
    assert settings.get_tab_settings("train") == {}


def test_tab_settings_do_not_touch_the_dialog_directories(state_file):
    settings.set_state("train-data-dir", "/tmp/train")
    settings.set_tab_setting("train", "epochs_number", 120)

    settings.reset_tab_settings()

    assert settings.get_tab_settings("train") == {}
    assert settings.get_state("train-data-dir") == "/tmp/train"


def test_a_corrupted_state_file_falls_back_to_the_defaults(state_file, tmp_path):
    state_file.write_text("{not json", encoding="utf-8")

    assert settings.get_state_dict() == {}
    assert gs.TabState("multi").get("confidence_slider", 0.25) == 0.25
    assert (tmp_path / "error_log.txt").exists()


@pytest.mark.parametrize(
    ("persisted", "expected"),
    [
        (0.5, 0.5),  # in range
        (0.05, 0.05),  # on the lower bound
        (2.0, 0.25),  # above the maximum
        (-1.0, 0.25),  # below the minimum
        ("0.5", 0.25),  # not a number
        (True, 0.25),  # a bool is an int, but not a confidence
        (None, 0.25),
    ],
)
def test_a_number_is_only_restored_within_its_bounds(state_file, persisted, expected):
    settings.set_tab_setting("multi", "confidence_slider", persisted)

    value = gs.TabState("multi").get(
        "confidence_slider", 0.25, minimum=0.05, maximum=0.95
    )

    assert value == expected


@pytest.mark.parametrize(
    ("persisted", "expected"),
    [
        ("load", "load"),
        ("none", "none"),
        ("gone", "none"),  # not offered (anymore)
        (2, "none"),  # not even a choice
    ],
)
def test_a_choice_is_only_restored_if_it_is_offered(state_file, persisted, expected):
    settings.set_tab_setting("train", "cache_mode_radio", persisted)

    value = gs.TabState("train").get(
        "cache_mode_radio",
        "none",
        choices=[("None", "none"), ("Load", "load"), ("Save", "save")],
    )

    assert value == expected


@pytest.mark.parametrize(
    ("persisted", "expected"),
    [
        (["csv", "table"], ["csv", "table"]),
        ([], []),
        (["csv", "gone"], ["table"]),  # one unknown format invalidates the selection
        ("csv", ["table"]),  # a single value is not a selection
    ],
)
def test_a_selection_is_only_restored_if_every_choice_is_offered(
    state_file, persisted, expected
):
    settings.set_tab_setting("multi", "output_type_checkboxgroup", persisted)

    value = gs.TabState("multi").get(
        "output_type_checkboxgroup",
        ["table"],
        choices=[("Raven", "table"), ("CSV", "csv")],
    )

    assert value == expected


def test_a_component_is_built_with_the_value_of_the_last_session(state_file):
    settings.set_tab_setting("multi", "confidence_slider", 0.75)
    tab = gs.TabState("multi")

    with gr.Blocks():
        restored = tab.persist(
            "confidence_slider", gr.Slider, minimum=0.05, maximum=0.95, value=0.25
        )
        default = tab.persist("sensitivity_slider", gr.Slider, value=1.0)

    assert restored.value == 0.75
    assert default.value == 1.0


def test_editing_a_component_persists_its_value(state_file):
    tab = gs.TabState("multi")

    with gr.Blocks() as demo:
        slider = tab.persist(
            "confidence_slider", gr.Slider, minimum=0.05, maximum=0.95, value=0.25
        )

    # A slider is persisted when it is released, everything else when it is edited.
    events = [
        event
        for event in demo.fns.values()
        if slider in event.inputs and event.targets[0][1] == "release"
    ]

    assert len(events) == 1

    events[0].fn(0.6)

    assert json.loads(state_file.read_text(encoding="utf-8")) == {
        settings.TAB_SETTINGS_KEY: {"multi": {"confidence_slider": 0.6}}
    }


def test_resetting_discards_the_persisted_values(state_file):
    settings.set_tab_setting("multi", "confidence_slider", 0.75)
    tab = gs.TabState("multi")

    with gr.Blocks():
        tab.persist(
            "confidence_slider", gr.Slider, minimum=0.05, maximum=0.95, value=0.25
        )

    assert [update["value"] for update in gs.reset_to_defaults()] == [0.25]
    assert settings.get_tab_settings("multi") == {}
