import os
import re

import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.embeddings import core as embeddings_core  # noqa: E402
from birdnet_analyzer.gui import embeddings as gui_embeddings  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402

DB_SETTINGS = {
    "AUDIO_SPEED": 2.0,
    "BANDPASS_FMIN": 500,
    "BANDPASS_FMAX": 8000,
    "AUDIO_ROOT": "recordings",
}


@pytest.fixture
def appdir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "APPDIR", tmp_path)
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(settings, "GUI_SETTINGS_PATH", str(tmp_path / "gui.json"))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr(gs, "_PERSISTED", [])
    return tmp_path


@pytest.fixture
def embeddings_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(gr, "Info", lambda *a, **k: None)
    monkeypatch.setattr(
        embeddings_core, "embeddings", lambda *a, **k: calls.append((a, k))
    )
    return calls


def _run(db_directory, fmin=200, fmax=9000):
    return gui_embeddings.run_embeddings(
        input_path="recordings",
        db_directory=db_directory,
        overlap=0.5,
        producers_number=3,
        workers_number=5,
        batch_size=7,
        audio_speed=-2,
        fmin=fmin,
        fmax=fmax,
        file_output=None,
        progress=None,
    )


def test_a_new_database_is_built_with_the_tabs_settings(embeddings_calls, tmp_path):
    db_directory = tmp_path / "new_db"

    _run(str(db_directory))

    assert embeddings_calls == [
        (
            ("recordings", str(db_directory)),
            {
                "overlap": 0.5,
                "n_producers": 3,
                "n_workers": 5,
                "batch_size": 7,
                "audio_speed": 0.5,
                "fmin": 200,
                "fmax": 9000,
                "file_output": None,
            },
        )
    ]
    assert not db_directory.exists()


def _passed_bandpass_and_speed(embeddings_calls):
    [(_, kwargs)] = embeddings_calls
    return kwargs["audio_speed"], kwargs["fmin"], kwargs["fmax"]


def test_a_database_without_settings_gets_the_tabs_settings(
    embeddings_calls, hoplite_database
):
    _run(hoplite_database())

    assert _passed_bandpass_and_speed(embeddings_calls) == (0.5, 200, 9000)


def test_a_leftover_folder_gets_the_tabs_settings(embeddings_calls, hoplite_leftover):
    _run(hoplite_leftover)

    assert _passed_bandpass_and_speed(embeddings_calls) == (0.5, 200, 9000)


def test_an_invalid_bandpass_is_rejected_before_inference(embeddings_calls, tmp_path):
    message = loc.localize("validation-no-valid-frequency")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _run(str(tmp_path / "new_db"), fmin=9000, fmax=200)

    assert embeddings_calls == []


def test_an_existing_database_keeps_its_settings(embeddings_calls, hoplite_database):
    _run(hoplite_database(DB_SETTINGS))

    [(_, kwargs)] = embeddings_calls
    assert kwargs["audio_speed"] == DB_SETTINGS["AUDIO_SPEED"]
    assert kwargs["fmin"] == DB_SETTINGS["BANDPASS_FMIN"]
    assert kwargs["fmax"] == DB_SETTINGS["BANDPASS_FMAX"]
    assert (kwargs["n_producers"], kwargs["n_workers"]) == (3, 5)


def test_a_failing_run_is_neither_retried_nor_masked(monkeypatch, hoplite_database):
    raised = []

    def fail(*args, **kwargs):
        raised.append(RuntimeError("inference failed"))
        raise raised[-1]

    monkeypatch.setattr(embeddings_core, "embeddings", fail)

    with pytest.raises(gr.Error, match="inference failed") as excinfo:
        _run(hoplite_database(DB_SETTINGS))

    assert len(raised) == 1
    assert excinfo.value.__cause__ is raised[0]


def _select_db_directory_handler():
    with gr.Blocks() as demo:
        gui_embeddings.build_embeddings_tab()

    return next(
        event.fn
        for event in demo.fns.values()
        if getattr(event.fn, "__name__", "") == "select_directory_and_update_tb"
    )


def test_picking_a_folder_for_a_new_database_leaves_it_empty(
    appdir, monkeypatch, tmp_path
):
    folder = tmp_path / "new_db"
    folder.mkdir()
    monkeypatch.setattr(gu, "select_directory", lambda *a, **k: str(folder))

    _, _, speed, fmin, fmax, _ = _select_db_directory_handler()(None)

    assert list(folder.iterdir()) == []
    assert [u["interactive"] for u in (speed, fmin, fmax)] == [True, True, True]


def test_picking_a_leftover_folder_treats_it_as_new_and_leaves_it_as_is(
    appdir, monkeypatch, hoplite_leftover
):
    monkeypatch.setattr(gu, "select_directory", lambda *a, **k: hoplite_leftover)

    _, _, speed, fmin, fmax, _ = _select_db_directory_handler()(None)

    assert os.listdir(hoplite_leftover) == ["hoplite.sqlite"]
    assert [u["interactive"] for u in (speed, fmin, fmax)] == [True, True, True]


def test_picking_an_existing_database_shows_and_locks_its_settings(
    appdir, monkeypatch, hoplite_database, hoplite_connections
):
    path = hoplite_database(DB_SETTINGS)
    monkeypatch.setattr(gu, "select_directory", lambda *a, **k: path)
    monkeypatch.setattr(gr, "Info", lambda *a, **k: None)

    _, _, speed, fmin, fmax, _ = _select_db_directory_handler()(None)

    assert speed == gr.update(value=2.0, interactive=False)
    assert fmin == gr.update(value=500, interactive=False)
    assert fmax == gr.update(value=8000, interactive=False)
    assert hoplite_connections.still_open() == []
