import pytest

gr = pytest.importorskip("gradio")

from birdnet_analyzer.gui import segments as gui_segments  # noqa: E402


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
