import pytest

gr = pytest.importorskip("gradio")

from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import segments as gui_segments  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.mark.parametrize(("slider", "speed"), [(-2, 0.5), (0, 1.0), (3, 3.0)])
def test_the_speed_slider_reaches_the_extraction_as_a_factor(
    monkeypatch, tmp_path, slider, speed
):
    seen = {}

    def fake_segments(*args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(gui_segments, "segments", fake_segments)

    gui_segments._extract_segments(
        str(tmp_path),
        str(tmp_path),
        str(tmp_path),
        0.25,
        1.0,
        100,
        slider,
        3.0,
        1,
        "random",
        10,
        progress=None,
    )

    assert seen["audio_speed"] == speed


def test_a_failing_folder_pick_is_logged_and_shown(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "APPDIR", tmp_path)
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(gs, "_PERSISTED", [])
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(gu, "select_directory", fail)
    with gr.Blocks() as demo:
        gui_segments.build_segments_tab()
    select = next(
        event.fn.func
        for event in demo.fns.values()
        if getattr(getattr(event.fn, "func", None), "__name__", "")
        == "select_directory_to_state_and_tb"
    )

    with pytest.raises(gr.Error, match="disk"):
        select("segments-audio-dir")

    assert len(logged) == 1
