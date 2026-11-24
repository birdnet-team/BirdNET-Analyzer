import os
import time

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

gr = pytest.importorskip("gradio")
pytest.importorskip("signavis")

from signavis.gradio import SignaVisPlayer  # noqa: E402

from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import single_file  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.fixture
def logged(monkeypatch):
    records = []
    monkeypatch.setattr(gu.utils, "write_error_log", records.append)
    return records


@pytest.fixture(autouse=True)
def preview_dir(tmp_path, monkeypatch):
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(single_file.tempfile, "gettempdir", lambda: str(temp))
    return temp / "birdnet-analyzer-preview"


def _old_preview(directory, name, size, age_days):
    directory.mkdir(exist_ok=True)
    path = directory / f"{name * 40}.wav"
    path.write_bytes(b"\0" * size)
    then = time.time() - age_days * 86400
    os.utime(path, (then, then))
    return path


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
        handler(single_file.build_single_analysis_tab, "load_player")(
            "clip.wav", False, False, True
        )

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


def test_a_row_clicked_in_the_table_does_not_dim_the_player(appdir):
    with gr.Blocks() as demo:
        single_file.build_single_analysis_tab()

    events = {getattr(e.fn, "__name__", None): e for e in demo.fns.values()}

    assert events["play_selected_detection"].show_progress == "hidden"


def test_the_player_leaves_the_keyboard_to_the_rest_of_the_gui(appdir):
    with gr.Blocks() as demo:
        single_file.build_single_analysis_tab()

    player = next(c for c in demo.blocks.values() if isinstance(c, SignaVisPlayer))

    assert '"keyboardShortcuts": false' in player.js_on_load


def test_the_player_starts_with_the_lighter_classic_preset(appdir):
    with gr.Blocks() as demo:
        single_file.build_single_analysis_tab()

    player = next(c for c in demo.blocks.values() if isinstance(c, SignaVisPlayer))

    assert '"defaultPreset": "classicSmall"' in player.js_on_load


def test_a_detection_clicked_in_the_player_plays(appdir):
    with gr.Blocks() as demo:
        single_file.build_single_analysis_tab()

    player = next(c for c in demo.blocks.values() if isinstance(c, SignaVisPlayer))

    assert '"playOnLabelClick": true' in player.js_on_load


def test_the_player_ui_speaks_the_gui_language(appdir, monkeypatch):
    from birdnet_analyzer.gui import localization as loc

    monkeypatch.setattr(loc, "TARGET_LANGUAGE", "de")
    with gr.Blocks() as demo:
        single_file.build_single_analysis_tab()

    player = next(c for c in demo.blocks.values() if isinstance(c, SignaVisPlayer))

    assert '"locale": "de"' in player.js_on_load


def test_the_confidence_choice_is_remembered(appdir):
    from birdnet_analyzer import settings
    from birdnet_analyzer.gui import localization as loc
    from birdnet_analyzer.gui import state as gs

    settings.set_tab_setting("single", "show_confidence", True)

    with gr.Blocks():
        single_file.build_single_analysis_tab()

    label = loc.localize("single-tab-show-confidence-label")
    checkbox = next(c for c in gs.persisted_components() if c.label == label)

    assert checkbox.value is True


def test_cancelling_the_dialog_changes_nothing(appdir, handler, monkeypatch):
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: None)

    result = handler(
        single_file.build_single_analysis_tab, "select_and_load_audio_file"
    )()

    assert list(result) == [gr.skip()] * 10 + [None]


def test_a_cancelled_dialog_stops_the_chained_steps(appdir, handler):
    tab = single_file.build_single_analysis_tab

    assert handler(tab, "lock_analyze")(None) == gr.skip()
    assert handler(tab, "load_player")(None, False, False, True) == (gr.skip(),) * 4


def test_the_player_gets_the_file_itself(appdir, handler, monkeypatch, tmp_path):
    clip = tmp_path / "clip.wav"
    sf.write(clip, np.zeros(4800, dtype="float32"), 48000)
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: str(clip))

    (
        path,
        label,
        src,
        player,
        checkbox,
        spectrogram_checkbox,
        freq_checkbox,
        predictions,
        table,
        actions,
        selection,
    ) = handler(single_file.build_single_analysis_tab, "select_and_load_audio_file")()

    assert path == label == str(clip)
    assert src is None
    assert player["visible"] is True, "the player shows up before the audio is ready"
    assert checkbox["visible"] is True
    assert spectrogram_checkbox["visible"] is True, "it can be turned off for any file"
    assert freq_checkbox["visible"] is True
    assert predictions is None, "a new file drops the last results"
    assert table["visible"] is False
    assert table["value"] is None
    assert actions["visible"] is False
    assert selection == str(clip)

    src, player, analyze, long_recording = handler(
        single_file.build_single_analysis_tab, "load_player"
    )(path, True, False, False)

    assert src == str(clip)
    assert player == {
        "src": str(clip),
        "showConfidence": True,
        "viewMode": "spectrogram",
        "autoFreqView": False,
        "spectrogram": {
            "colorScheme": "viridis",
            "windowSize": 1024,
            "overlapLevel": 2,
            "oversamplingLevel": 0,
            "scale": "linear",
            "contrastRangeDb": 80,
        },
    }
    assert analyze["interactive"] is True
    assert long_recording is False


def test_the_player_draws_with_the_spectrogram_settings(appdir, handler, tmp_path):
    clip = tmp_path / "clip.wav"
    sf.write(clip, np.zeros(4800, dtype="float32"), 48000)
    for key, value in {
        "spectrogram_colormap_dropdown": "Greys",
        "spectrogram_fft_size_dropdown": 2048,
        "spectrogram_overlap_slider": 90,
        "spectrogram_dynamic_range_slider": 60,
        "spectrogram_freq_scale_radio": "log",
    }.items():
        settings.set_tab_setting("settings", key, value)

    _, player, *_ = handler(single_file.build_single_analysis_tab, "load_player")(
        str(clip), False, False, True
    )

    assert player["spectrogram"] == {
        "colorScheme": "grayscale",
        "windowSize": 2048,
        "overlapLevel": 4,
        "oversamplingLevel": 0,
        "scale": "cqt",
        "contrastRangeDb": 60,
    }


def test_the_frequency_range_fit_can_be_switched_off(appdir, handler):
    apply = handler(single_file.build_single_analysis_tab, "apply_auto_freq")

    assert apply("clip.wav", False) == {"src": "clip.wav", "autoFreqView": False}
    assert apply(None, False) == gr.skip()


def test_returning_to_the_tab_brings_changed_spectrogram_settings(appdir, handler):
    apply = handler(single_file.build_single_analysis_tab, "apply_spectrogram_settings")
    settings.set_tab_setting("settings", "spectrogram_colormap_dropdown", "jet")

    value = apply("clip.wav")

    assert value["src"] == "clip.wav"
    assert value["spectrogram"]["colorScheme"] == "jet"
    assert "detections" not in value, "the player keeps its detections"
    assert apply(None) == gr.skip()


@pytest.mark.parametrize(("show", "view"), [(False, "waveform"), (True, "spectrogram")])
def test_a_long_recording_shows_its_spectrogram_on_request(
    appdir, handler, monkeypatch, tmp_path, show, view
):
    clip = tmp_path / "clip.wav"
    sf.write(clip, np.zeros(4800, dtype="float32"), 48000)
    real_info = sf.info

    def long_info(path):
        info = real_info(path)
        info.duration = single_file.LONG_RECORDING_SECONDS
        return info

    monkeypatch.setattr(sf, "info", long_info)

    _, player, _, long_recording = handler(
        single_file.build_single_analysis_tab, "load_player"
    )(str(clip), False, show, True)

    assert player["viewMode"] == view
    assert long_recording is True


def test_turning_on_the_spectrogram_keeps_the_detections(appdir, handler):
    value = handler(single_file.build_single_analysis_tab, "show_detections")(
        {"table": _detections(), "shown": _detections()}, "clip.wav", False, True, True
    )

    assert value["viewMode"] == "spectrogram"
    assert len(value["detections"]) == 2


def test_the_spectrogram_choice_is_remembered(appdir):
    from birdnet_analyzer import settings
    from birdnet_analyzer.gui import localization as loc
    from birdnet_analyzer.gui import state as gs

    settings.set_tab_setting("single", "show_long_spectrogram", True)

    with gr.Blocks():
        single_file.build_single_analysis_tab()

    label = loc.localize("single-tab-long-spectrogram-label")
    checkbox = next(c for c in gs.persisted_components() if c.label == label)

    assert checkbox.value is True


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


def test_a_failed_conversion_leaves_no_preview_behind(
    tmp_path, monkeypatch, preview_dir
):
    clip = tmp_path / "clip.ogg"
    sf.write(clip, np.zeros(4800, dtype="float32"), 48000)
    monkeypatch.setattr(sf, "write", _fail(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        single_file.audio_preview(str(clip))

    assert list(preview_dir.iterdir()) == []


def test_old_previews_are_removed(tmp_path, preview_dir):
    clip = tmp_path / "clip.ogg"
    sf.write(clip, np.zeros(4800, dtype="float32"), 48000)
    stale = _old_preview(preview_dir, "a", 10, single_file.PREVIEW_MAX_AGE_DAYS + 1)
    recent = _old_preview(preview_dir, "b", 10, 1)

    preview = single_file.audio_preview(str(clip))

    assert not stale.exists()
    assert recent.exists()
    assert os.path.exists(preview)


def test_the_least_recently_used_previews_go_over_the_size_limit(
    tmp_path, monkeypatch, preview_dir
):
    monkeypatch.setattr(single_file, "PREVIEW_MAX_BYTES", 250_000)
    clip = tmp_path / "clip.ogg"
    sf.write(clip, np.zeros(48000, dtype="float32"), 48000)
    oldest = _old_preview(preview_dir, "a", 100_000, 3)
    newer = _old_preview(preview_dir, "b", 100_000, 2)

    preview = single_file.audio_preview(str(clip))

    assert not oldest.exists()
    assert newer.exists()
    assert os.path.exists(preview)


def test_a_reused_preview_counts_as_recently_used(tmp_path, preview_dir):
    clip = tmp_path / "clip.ogg"
    sf.write(clip, np.zeros(4800, dtype="float32"), 48000)
    preview = single_file.audio_preview(str(clip))
    then = time.time() - 3 * 86400
    os.utime(preview, (then, then))

    single_file.audio_preview(str(clip))

    assert os.path.getmtime(preview) > then + 86400


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


def test_a_table_row_finds_its_detection_at_a_small_hop():
    detections = pd.DataFrame(
        {
            "start_time": [0.2, 0.3],
            "end_time": [3.2, 3.3],
            "species_name": ["Turdus merula_Eurasian Blackbird"] * 2,
            "confidence": [0.9, 0.8],
        }
    )

    assert single_file.detection_id(detections, 0.3, 3.3, "Turdus merula") == "det-1"


def test_the_analysis_marks_its_detections_in_the_player(appdir, handler):
    value = handler(single_file.build_single_analysis_tab, "show_detections")(
        {"table": _detections(), "shown": _detections()},
        "clip.wav",
        False,
        False,
        False,
    )

    assert value["src"] == "clip.wav"
    assert value["showConfidence"] is False
    assert [d["start"] for d in value["detections"]] == [3.0, 9.0]


def test_the_confidence_can_be_shown_before_an_analysis(appdir, handler):
    value = handler(single_file.build_single_analysis_tab, "show_detections")(
        None, "clip.wav", True, False, False
    )

    assert value == {
        "src": "clip.wav",
        "detections": [],
        "showConfidence": True,
        "viewMode": "spectrogram",
    }


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
        evt, {"table": _detections(), "shown": _detections()}, "clip.wav", True
    )

    assert value["select"]["start"] == 9.0
    assert value["select"]["end"] == 12.0
    assert value["select"]["play"] is plays
    assert value["select"]["id"] == "det-1"
    assert "detections" not in value, "the player keeps the labels it has"
    assert value["showConfidence"] is True


def test_a_failed_export_is_logged_and_shown(
    appdir, handler, monkeypatch, logged, tmp_path
):
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: str(tmp_path / "out.csv"))
    monkeypatch.setattr(single_file, "save_as_csv", _fail(OSError("disk")))

    with pytest.raises(gr.Error, match="disk"):
        handler(single_file.build_single_analysis_tab, "download_csv")({"table": None})

    assert len(logged) == 1


def test_info_text_names_the_select_button_as_labelled():
    import json
    from pathlib import Path

    lang_dir = Path(__file__).parents[2] / "birdnet_analyzer" / "lang"

    for lang_file in sorted(lang_dir.glob("*.json")):
        texts = json.loads(lang_file.read_text(encoding="utf-8"))
        label = texts["single-tab-select-file-button-label"]

        assert label in texts["single-tab-info-text"], lang_file.name


@pytest.mark.parametrize(
    "key", ["single-tab-show-confidence-label", "single-tab-long-spectrogram-label"]
)
def test_the_player_labels_are_translated(key):
    import json
    from pathlib import Path

    from birdnet_analyzer.settings import LANG_DIR

    def label(file):
        return json.loads(file.read_text(encoding="utf-8"))[key]

    english = label(Path(LANG_DIR) / "en.json")
    untranslated = [
        f.stem
        for f in Path(LANG_DIR).glob("*.json")
        if f.stem != "en" and label(f) == english
    ]

    assert untranslated == []
