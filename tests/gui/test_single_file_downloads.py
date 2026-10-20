import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.config as cfg  # noqa: E402
from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import single_file  # noqa: E402
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


def test_the_kaleidoscope_dialog_offers_the_csv_it_saves(appdir, monkeypatch):
    with gr.Blocks() as demo:
        single_file.build_single_analysis_tab()
    download = next(
        event.fn
        for event in demo.fns.values()
        if getattr(event.fn, "__name__", "") == "download_kaleidoscope"
    )
    dialogs = []
    monkeypatch.setattr(
        gu, "save_file_dialog", lambda **kwargs: dialogs.append(kwargs) or None
    )

    download({"table": None})

    (dialog,) = dialogs
    assert dialog["default_filename"] == cfg.OUTPUT_KALEIDOSCOPE_FILENAME
    assert cfg.OUTPUT_KALEIDOSCOPE_FILENAME.endswith(".csv")
    assert dialog["filetypes"] == ("Kaleidoscope (*.csv)",)
