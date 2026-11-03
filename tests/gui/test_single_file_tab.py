import numpy as np
import pytest
import soundfile as sf

gr = pytest.importorskip("gradio")

from gradio import processing_utils  # noqa: E402

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
        handler(single_file.build_single_analysis_tab, "select_and_load_audio_file")()

    assert "disk" in raised.value.message
    assert loc.localize("single-tab-generate-spectrogram-error") not in (
        raised.value.message
    )
    assert len(logged) == 1


def test_a_failed_spectrogram_keeps_its_message(appdir, handler, monkeypatch, logged):
    monkeypatch.setattr(utils, "spectrogram_from_file", _fail(RuntimeError("boom")))

    with pytest.raises(gr.Error) as raised:
        handler(
            single_file.build_single_analysis_tab, "draw_spectrogram_for_selection"
        )("clip.wav", True)

    assert raised.value.message == loc.localize("single-tab-generate-spectrogram-error")
    assert len(logged) == 1


def test_the_file_loads_before_its_spectrogram_is_drawn(appdir):
    with gr.Blocks() as demo:
        single_file.build_single_analysis_tab()

    events = {getattr(e.fn, "__name__", None): e for e in demo.fns.values()}
    select = events["select_and_load_audio_file"]
    draw = events["draw_spectrogram_for_selection"]

    assert any(isinstance(c, gr.Audio) for c in select.outputs)
    assert not any(isinstance(c, gr.Plot) for c in select.outputs)
    assert any(isinstance(c, gr.Plot) for c in draw.outputs)
    assert draw.trigger_after == next(k for k, e in demo.fns.items() if e is select)


def test_cancelling_the_dialog_changes_nothing(appdir, handler, monkeypatch):
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: None)

    *kept, request = handler(
        single_file.build_single_analysis_tab, "select_and_load_audio_file"
    )()
    spectrogram = handler(
        single_file.build_single_analysis_tab, "draw_spectrogram_for_selection"
    )(request, True)

    assert request is None
    assert kept == [gr.skip()] * 4
    assert spectrogram == (gr.skip(), gr.skip())


def test_the_player_gets_the_file_itself(appdir, handler, monkeypatch, tmp_path):
    clip = tmp_path / "clip.wav"
    sf.write(clip, np.zeros(4800, dtype="float32"), 48000)
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: str(clip))

    path, label, player, analyze, request = handler(
        single_file.build_single_analysis_tab, "select_and_load_audio_file"
    )()

    assert path == label == request == str(clip)
    assert player["value"] == str(clip)
    assert player["label"] == "clip.wav"
    assert analyze["interactive"] is True
    served = processing_utils.move_files_to_cache(
        gr.Audio(type="filepath").postprocess(player["value"]).model_dump(), gr.Audio()
    )
    assert served["path"] == str(clip), "served in place, not copied into the cache"


@pytest.mark.parametrize(
    ("name", "subtype"),
    [("clip.aiff", None), ("clip.ogg", None), ("clip.wav", "FLOAT")],
)
def test_a_format_the_browser_cannot_play_is_decoded_for_it(
    appdir, handler, monkeypatch, tmp_path, name, subtype
):
    clip = tmp_path / name
    sf.write(clip, np.full(4800, 0.5, dtype="float32"), 48000, subtype=subtype)
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: str(clip))

    player = handler(
        single_file.build_single_analysis_tab, "select_and_load_audio_file"
    )()[2]

    rate, data = player["value"]
    assert rate == 48000
    assert data.shape == (4800,)
    assert data.max() == pytest.approx(0.5, abs=0.05)


def test_a_failed_export_is_logged_and_shown(
    appdir, handler, monkeypatch, logged, tmp_path
):
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: str(tmp_path / "out.csv"))
    monkeypatch.setattr(single_file, "save_as_csv", _fail(OSError("disk")))

    with pytest.raises(gr.Error, match="disk"):
        handler(single_file.build_single_analysis_tab, "download_csv")({"table": None})

    assert len(logged) == 1
