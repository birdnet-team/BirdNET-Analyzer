import re

import numpy as np
import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import audio, settings, utils  # noqa: E402
from birdnet_analyzer.gui import search as gui_search  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402
from birdnet_analyzer.search import utils as search_utils  # noqa: E402

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


def _handler(name):
    with gr.Blocks() as demo:
        gui_search.build_search_tab()

    for event in demo.fns.values():
        # gr.render registers the bound apply() of a Renderable wrapping the function.
        fn = getattr(getattr(event.fn, "__self__", None), "fn", event.fn)
        if getattr(fn, "__name__", "") == name:
            return fn

    raise LookupError(name)


def _insert_window(path) -> int:
    from birdnet_analyzer.embeddings.core import get_or_create_database

    db = get_or_create_database(path)
    deployment_id = db.insert_deployment(name="deployment", project="project")
    recording_id = db.insert_recording(filename="rec.wav", deployment_id=deployment_id)
    window_id = db.insert_window(recording_id, [0.0, 3.0], np.zeros(8, np.float32))
    db.commit()
    db.db.close()
    return window_id


def test_a_failing_search_closes_the_database(
    hoplite_database, hoplite_connections, monkeypatch, tmp_path
):
    def fail(*args, **kwargs):
        raise RuntimeError("search failed")

    monkeypatch.setattr(search_utils, "get_search_results", fail)

    with pytest.raises(gr.Error, match="search failed"):
        gui_search.run_search(
            hoplite_database(DB_SETTINGS),
            str(tmp_path),
            "query.wav",
            10,
            "cosine",
            "center",
            0.0,
            progress=None,
        )

    assert hoplite_connections.still_open() == []


def test_picking_a_folder_without_a_database_is_rejected_and_left_as_is(
    appdir, monkeypatch, tmp_path
):
    folder = tmp_path / "recordings"
    folder.mkdir()
    monkeypatch.setattr(gu, "select_folder", lambda *a, **k: str(folder))
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)
    message = loc.localize("embeddings-search-db-selection-error")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _handler("on_db_selection_click")()

    assert list(folder.iterdir()) == []
    assert logged == []


def test_picking_a_database_without_settings_is_rejected_and_closed(
    appdir, monkeypatch, hoplite_database, hoplite_connections
):
    path = hoplite_database()
    monkeypatch.setattr(gu, "select_folder", lambda *a, **k: path)
    message = loc.localize("embeddings-search-db-selection-error")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _handler("on_db_selection_click")()

    assert hoplite_connections.still_open() == []


def test_picking_a_database_shows_its_settings_and_closes_it(
    appdir, monkeypatch, hoplite_database, hoplite_connections
):
    path = hoplite_database(DB_SETTINGS)
    monkeypatch.setattr(gu, "select_folder", lambda *a, **k: path)

    folder, audio_root, count, bandpass, speed, *_ = _handler("on_db_selection_click")()

    assert (folder, audio_root) == (path, "recordings")
    assert count["value"] == 0
    assert bandpass["value"] == "500 - 8000 Hz"
    assert speed["value"] == 2.0
    assert hoplite_connections.still_open() == []


def test_the_query_spectrogram_uses_the_database_settings(
    appdir, monkeypatch, hoplite_database, hoplite_connections
):
    opened = []

    def open_audio_file(path, **kwargs):
        opened.append(kwargs)
        return np.zeros(48000 * 6, np.float32), 48000

    monkeypatch.setattr(audio, "open_audio_file", open_audio_file)
    monkeypatch.setattr(utils, "spectrogram_from_audio", lambda *a, **k: "spec")

    spectrogram, *_ = _handler("update_query_spectrogram")(
        "query.wav", hoplite_database(DB_SETTINGS), "center", 0.0
    )

    assert spectrogram["value"] == "spec"
    [kwargs] = opened
    assert (kwargs["speed"], kwargs["fmin"], kwargs["fmax"]) == (2.0, 500, 8000)
    assert hoplite_connections.still_open() == []


def test_the_query_spectrogram_reports_a_missing_database_without_creating_it(
    appdir, tmp_path
):
    missing = tmp_path / "moved_db"
    message = loc.localize("embeddings-search-db-selection-error")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _handler("update_query_spectrogram")("query.wav", str(missing), "center", 0.0)

    assert not missing.exists()


def test_rendering_results_closes_the_database_when_drawing_fails(
    appdir, monkeypatch, hoplite_database, hoplite_connections, tmp_path
):
    from perch_hoplite.db.search_results import SearchResult

    path = hoplite_database(DB_SETTINGS)
    window_id = _insert_window(path)

    def fail(*args, **kwargs):
        raise RuntimeError("drawing failed")

    monkeypatch.setattr(utils, "spectrogram_from_file", fail)
    render_results = _handler("render_results")

    with gr.Blocks(), pytest.raises(gr.Error, match="drawing failed"):
        render_results([[SearchResult(window_id, 0.5)]], 0, path, {}, str(tmp_path))

    assert hoplite_connections.still_open() == []


def test_a_failing_export_is_logged_and_shown(appdir, monkeypatch, tmp_path):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)
    monkeypatch.setattr(gu, "select_folder", lambda *a, **k: str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(audio, "open_audio_file", fail)

    with pytest.raises(gr.Error, match="disk"):
        _handler("run_export")({0: ["clip.wav", 0.0, 3.0, 0, 0.5, 1.0, 0, 15000]})

    assert len(logged) == 1
