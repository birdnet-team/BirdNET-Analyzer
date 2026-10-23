import numpy as np
import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import audio, settings, utils  # noqa: E402
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


@pytest.fixture
def logged(monkeypatch):
    records = []
    monkeypatch.setattr(gu.utils, "write_error_log", records.append)
    return records


def _handler(name):
    with gr.Blocks() as demo:
        single_file.build_single_analysis_tab()

    return next(
        event.fn
        for event in demo.fns.values()
        if getattr(event.fn, "__name__", "") == name
    )


def _fail(error):
    def raiser(*args, **kwargs):
        raise error

    return raiser


def test_an_unreadable_file_is_reported_as_such(appdir, monkeypatch, logged):
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: "clip.wav")
    monkeypatch.setattr(audio, "open_audio_file", _fail(OSError("disk")))

    with pytest.raises(gr.Error) as raised:
        _handler("select_and_load_audio_file")(True)

    assert "disk" in raised.value.message
    assert loc.localize("single-tab-generate-spectrogram-error") not in (
        raised.value.message
    )
    assert len(logged) == 1


def test_a_failed_spectrogram_keeps_its_message(appdir, monkeypatch, logged):
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: "clip.wav")
    monkeypatch.setattr(audio, "open_audio_file", lambda *a, **k: (np.zeros(10), 48000))
    monkeypatch.setattr(utils, "spectrogram_from_file", _fail(RuntimeError("boom")))

    with pytest.raises(gr.Error) as raised:
        _handler("select_and_load_audio_file")(True)

    assert raised.value.message == loc.localize("single-tab-generate-spectrogram-error")
    assert len(logged) == 1


def test_a_failed_export_is_logged_and_shown(appdir, monkeypatch, logged, tmp_path):
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: str(tmp_path / "out.csv"))
    monkeypatch.setattr(single_file, "save_as_csv", _fail(OSError("disk")))

    with pytest.raises(gr.Error, match="disk"):
        _handler("download_csv")({"table": None})

    assert len(logged) == 1
