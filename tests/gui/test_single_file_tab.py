import numpy as np
import pytest
import soundfile as sf

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import utils  # noqa: E402
from birdnet_analyzer.gui import single_file  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.fixture
def logged(monkeypatch):
    records = []
    monkeypatch.setattr(gu.utils, "write_error_log", records.append)
    return records


def _fail(error):
    def raiser(*args, **kwargs):
        raise error

    return raiser


def test_an_unreadable_file_is_reported_as_such(appdir, handler, monkeypatch, logged):
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: "clip.wav")
    monkeypatch.setattr(sf, "info", _fail(OSError("disk")))

    with pytest.raises(gr.Error) as raised:
        handler(single_file.build_single_analysis_tab, "select_and_load_audio_file")(
            True
        )

    assert "disk" in raised.value.message
    assert loc.localize("single-tab-generate-spectrogram-error") not in (
        raised.value.message
    )
    assert len(logged) == 1


def test_a_failed_spectrogram_keeps_its_message(appdir, handler, monkeypatch, logged):
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: "clip.wav")
    monkeypatch.setattr(sf, "info", lambda *a, **k: None)
    monkeypatch.setattr(utils, "spectrogram_from_file", _fail(RuntimeError("boom")))

    with pytest.raises(gr.Error) as raised:
        handler(single_file.build_single_analysis_tab, "select_and_load_audio_file")(
            True
        )

    assert raised.value.message == loc.localize("single-tab-generate-spectrogram-error")
    assert len(logged) == 1


def test_the_player_gets_the_file_itself(appdir, handler, monkeypatch, tmp_path):
    clip = tmp_path / "clip.wav"
    sf.write(clip, np.zeros(4800, dtype="float32"), 48000)
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: str(clip))

    *_, player, _, _, analyze = handler(
        single_file.build_single_analysis_tab, "select_and_load_audio_file"
    )(False)

    assert player["value"] == str(clip)
    assert player["label"] == "clip.wav"
    assert analyze["interactive"] is True


def test_a_format_the_browser_cannot_play_is_decoded_for_it(
    appdir, handler, monkeypatch, tmp_path
):
    clip = tmp_path / "clip.aiff"
    sf.write(clip, np.full(4800, 0.5, dtype="float32"), 48000)
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: str(clip))

    player = handler(
        single_file.build_single_analysis_tab, "select_and_load_audio_file"
    )(False)[2]

    rate, data = player["value"]
    assert rate == 48000
    assert data.shape == (4800,)
    assert data.max() == pytest.approx(0.5)


def test_a_failed_export_is_logged_and_shown(
    appdir, handler, monkeypatch, logged, tmp_path
):
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: str(tmp_path / "out.csv"))
    monkeypatch.setattr(single_file, "save_as_csv", _fail(OSError("disk")))

    with pytest.raises(gr.Error, match="disk"):
        handler(single_file.build_single_analysis_tab, "download_csv")({"table": None})

    assert len(logged) == 1
