import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.multi_file as mfa  # noqa: E402
from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.fixture
def appdir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "APPDIR", tmp_path)
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr(gs, "_PERSISTED", [])
    monkeypatch.setattr(gu, "plot_map_scatter_mapbox", lambda *a, **k: None)
    return tmp_path


def _handler(name):
    with gr.Blocks() as demo:
        mfa.build_multi_analysis_tab()

    return next(
        event.fn
        for event in demo.fns.values()
        if getattr(event.fn, "__name__", "") == name
    )


def test_a_failing_folder_pick_is_logged_and_shown(appdir, monkeypatch, tmp_path):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)
    monkeypatch.setattr(gu, "select_folder", lambda *a, **k: str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(gu, "get_audio_files_and_durations", fail)

    with pytest.raises(gr.Error, match="disk"):
        _handler("select_directory_on_empty")()

    assert len(logged) == 1
