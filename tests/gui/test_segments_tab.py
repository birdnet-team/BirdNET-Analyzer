import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer.gui import segments as gui_segments  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.mark.parametrize(("slider", "speed"), [(-2, 0.5), (0, 1.0), (3, 3.0)])
def test_the_speed_slider_reaches_the_extraction_as_a_factor(
    monkeypatch, tmp_path, slider, speed
):
    seen = {}

    def fake_segments(*args, **kwargs):
        seen.update(kwargs)
        return [(str(tmp_path / "a.wav"), True)]

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


def test_a_failing_folder_pick_is_logged_and_shown(appdir, handler, monkeypatch):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(gu, "select_directory", fail)
    select = handler(
        gui_segments.build_segments_tab, "select_directory_to_state_and_tb"
    )

    with pytest.raises(gr.Error, match="disk"):
        select("segments-audio-dir")

    assert len(logged) == 1


def test_an_extraction_that_found_nothing_is_reported_instead_of_called_a_success(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(gui_segments, "segments", lambda *args, **kwargs: [])

    with pytest.raises(gr.Error) as raised:
        gui_segments._extract_segments(
            str(tmp_path),
            str(tmp_path),
            str(tmp_path),
            0.25,
            1.0,
            100,
            0,
            3.0,
            1,
            "random",
            10,
            progress=None,
        )

    assert raised.value.message == loc.LANGUAGE_LOOKUP["segments-tab-no-segments-error"]


def _extract(audio, results):
    return gui_segments._extract_segments(
        audio,
        results,
        None,
        0.25,
        1.0,
        100,
        0,
        3.0,
        1,
        "random",
        10,
        progress=None,
    )


@pytest.mark.parametrize(
    ("audio", "results"),
    [("recordings", "May & June"), ("May & June", None)],
    ids=["chosen results folder", "results next to the recordings"],
)
def test_a_results_folder_that_is_gone_is_reported_instead_of_searched(
    monkeypatch, tmp_path, audio, results
):
    def fail(*args, **kwargs):
        raise AssertionError("the extraction started")

    monkeypatch.setattr(gui_segments, "segments", fail)
    (tmp_path / "recordings").mkdir()

    with pytest.raises(gr.Error) as raised:
        _extract(str(tmp_path / audio), results and str(tmp_path / results))

    assert raised.value.message == loc.LANGUAGE_LOOKUP["error-path-not-found"].format(
        path=str(tmp_path / "May & June").replace("&", "&amp;")
    )


@pytest.mark.parametrize(
    ("audio", "results"),
    [("recordings", None), ("gone", "results")],
    ids=["results next to the recordings", "recordings named by the result table"],
)
def test_an_extraction_starts_when_its_results_folder_is_there(
    monkeypatch, tmp_path, audio, results
):
    started = []
    (tmp_path / "recordings").mkdir()
    (tmp_path / "results").mkdir()

    def fake_segments(*args, **kwargs):
        started.append(args)
        return [(str(tmp_path / "a.wav"), True)]

    monkeypatch.setattr(gui_segments, "segments", fake_segments)

    _extract(str(tmp_path / audio), results and str(tmp_path / results))

    assert len(started) == 1


def test_nothing_found_because_the_recordings_are_gone_is_reported_as_such(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(gui_segments, "segments", lambda *args, **kwargs: [])
    (tmp_path / "results").mkdir()

    with pytest.raises(gr.Error) as raised:
        _extract(str(tmp_path / "May & June"), str(tmp_path / "results"))

    assert raised.value.message == loc.LANGUAGE_LOOKUP["error-path-not-found"].format(
        path=str(tmp_path / "May & June").replace("&", "&amp;")
    )
