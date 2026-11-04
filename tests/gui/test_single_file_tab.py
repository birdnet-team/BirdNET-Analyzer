import numpy as np
import pandas as pd
import pytest
import soundfile as sf

gr = pytest.importorskip("gradio")
pytest.importorskip("signavis")

from signavis.gradio import SignaVisPlayer  # noqa: E402

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


def _detections():
    return pd.DataFrame(
        {
            "start_time": [3.0, 9.0],
            "end_time": [6.0, 12.0],
            "species_name": ["Turdus merula_Eurasian Blackbird", "Noise"],
            "confidence": np.array([0.91234, 0.5], dtype="float16"),
        }
    )


def test_an_unreadable_file_is_reported_as_such(appdir, handler, monkeypatch, logged):
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: "clip.wav")
    monkeypatch.setattr(sf, "info", _fail(OSError("disk")))

    with pytest.raises(gr.Error) as raised:
        handler(single_file.build_single_analysis_tab, "select_and_load_audio_file")()

    assert "disk" in raised.value.message
    assert len(logged) == 1


def test_the_file_is_shown_in_the_spectrogram_player(appdir):
    with gr.Blocks() as demo:
        single_file.build_single_analysis_tab()

    events = {getattr(e.fn, "__name__", None): e for e in demo.fns.values()}

    assert any(
        isinstance(c, SignaVisPlayer)
        for c in events["select_and_load_audio_file"].outputs
    )
    assert not any(isinstance(c, (gr.Audio, gr.Plot)) for c in demo.blocks.values())


def test_cancelling_the_dialog_changes_nothing(appdir, handler, monkeypatch):
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: None)

    result = handler(
        single_file.build_single_analysis_tab, "select_and_load_audio_file"
    )()

    assert list(result) == [gr.skip()] * 6


def test_the_player_gets_the_file_itself(appdir, handler, monkeypatch, tmp_path):
    clip = tmp_path / "clip.wav"
    sf.write(clip, np.zeros(4800, dtype="float32"), 48000)
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: str(clip))

    path, label, src, player, analyze, predictions = handler(
        single_file.build_single_analysis_tab, "select_and_load_audio_file"
    )()

    assert path == label == src == str(clip)
    assert player["visible"] is True
    assert player["value"] == {"src": str(clip)}
    assert analyze["interactive"] is True
    assert predictions is None, "a new file drops the last results"


@pytest.mark.parametrize(
    ("name", "subtype"),
    [("clip.aiff", None), ("clip.ogg", None), ("clip.wav", "FLOAT")],
)
def test_a_format_the_browser_cannot_play_is_decoded_for_it(tmp_path, name, subtype):
    clip = tmp_path / name
    sf.write(clip, np.full(4800, 0.5, dtype="float32"), 48000, subtype=subtype)

    preview = single_file.audio_preview(str(clip))
    data, rate = sf.read(preview, dtype="float32")

    assert preview.endswith(".wav")
    assert sf.info(preview).subtype == "PCM_16"
    assert rate == 48000
    assert data.shape == (4800,)
    assert data.max() == pytest.approx(0.5, abs=0.05)
    assert single_file.audio_preview(str(clip)) == preview


def test_detections_become_player_labels():
    labels = single_file.player_detections(_detections())

    assert labels[0] == {
        "id": "det-0",
        "start": 3.0,
        "end": 6.0,
        "label": "Eurasian Blackbird",
        "scientificName": "Turdus merula",
        "commonName": "Eurasian Blackbird",
        "confidence": pytest.approx(0.912, abs=1e-3),
        "origin": "BirdNET",
    }
    assert labels[1]["label"] == "Noise"


def test_a_table_row_finds_its_detection():
    detections = _detections()

    assert single_file.detection_id(detections, 3.0, 6.0, "Turdus merula") == "det-0"
    assert single_file.detection_id(detections, 3.0, 6.0, "Parus major") is None


def test_the_analysis_marks_its_detections_in_the_player(appdir, handler):
    value = handler(single_file.build_single_analysis_tab, "show_detections")(
        {"table": _detections()}, "clip.wav"
    )

    assert value["src"] == "clip.wav"
    assert [d["start"] for d in value["detections"]] == [3.0, 9.0]


@pytest.mark.parametrize(("column", "plays"), [(0, True), (3, False)])
def test_a_clicked_detection_is_shown_in_the_player(appdir, handler, column, plays):
    evt = gr.SelectData(
        None,
        {
            "index": [1, column],
            "value": "",
            "row_value": ["▶", "0:00:09", "0:00:12", "Noise", "", "0.500"],
        },
    )

    value = handler(single_file.build_single_analysis_tab, "play_selected_detection")(
        evt, {"table": _detections()}, "clip.wav"
    )

    assert value["select"]["start"] == 9.0
    assert value["select"]["end"] == 12.0
    assert value["select"]["play"] is plays
    assert value["select"]["id"] == "det-1"
    assert len(value["detections"]) == 2


def test_a_failed_export_is_logged_and_shown(
    appdir, handler, monkeypatch, logged, tmp_path
):
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: str(tmp_path / "out.csv"))
    monkeypatch.setattr(single_file, "save_as_csv", _fail(OSError("disk")))

    with pytest.raises(gr.Error, match="disk"):
        handler(single_file.build_single_analysis_tab, "download_csv")({"table": None})

    assert len(logged) == 1
