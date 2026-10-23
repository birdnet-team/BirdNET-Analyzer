import os

import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import review  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.fixture
def appdir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "APPDIR", tmp_path)
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr(gs, "_PERSISTED", [])
    return tmp_path


def _handler(name):
    with gr.Blocks() as demo:
        review.build_review_tab()

    for event in demo.fns.values():
        fn = getattr(event.fn, "func", event.fn)
        if getattr(fn, "__name__", "") == name:
            return fn

    raise LookupError(name)


def _review_state(directory, files):
    return {
        "input_directory": str(directory),
        "species_list": [],
        "current_species": "",
        "files": files,
        review.POSITIVE_LABEL_DIR: [],
        review.NEGATIVE_LABEL_DIR: [],
        "skipped": [],
        "history": [],
    }


def test_out_of_range_confidences_are_left_out_of_the_regression(
    appdir, monkeypatch, tmp_path
):
    from sklearn import linear_model

    folder = tmp_path / "segments"
    for label, names in (
        (review.POSITIVE_LABEL_DIR, ["0.900_a.wav", "1.500_b.wav"]),
        (review.NEGATIVE_LABEL_DIR, ["0.200_c.wav", "0.100_d.wav"]),
    ):
        (folder / label).mkdir(parents=True)
        for name in names:
            (folder / label / name).touch()

    fitted = []

    class RecordingRegression(linear_model.LogisticRegression):
        def fit(self, X, y, *args, **kwargs):
            fitted.append(sorted(zip((x for (x,) in X), y, strict=True)))
            return super().fit(X, y, *args, **kwargs)

    monkeypatch.setattr(linear_model, "LogisticRegression", RecordingRegression)
    monkeypatch.setattr(gu, "select_folder", lambda *a, **k: str(folder))
    updates = _handler("start_review")(_review_state("", []))

    label = loc.localize("review-tab-regression-plot-label")
    plot = next(
        update
        for component, update in updates.items()
        if isinstance(component, gr.Plot) and component.label == label
    )
    (axes,) = plot.constructor_args["value"].axes
    points = sorted(map(tuple, axes.collections[-1].get_offsets().tolist()))
    expected = [(0.1, 0), (0.2, 0), (0.9, 1)]

    assert points == pytest.approx(expected)
    assert fitted == [expected]


def test_a_failed_move_is_logged_and_shown(appdir, monkeypatch, tmp_path):
    clip = tmp_path / "0.900_a.wav"
    clip.touch()
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(os, "rename", fail)

    with pytest.raises(gr.Error, match="disk"):
        _handler("next_review")(
            _review_state(tmp_path, [str(clip)]), target_dir=review.POSITIVE_LABEL_DIR
        )

    assert len(logged) == 1


def test_running_out_of_files_is_reported_as_is(appdir, monkeypatch, tmp_path):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    with pytest.raises(gr.Error) as raised:
        _handler("next_review")(_review_state(tmp_path, []))

    assert raised.value.message == loc.localize("review-tab-no-files-error")
    assert logged == []


def test_the_download_buttons_skip_an_empty_plot(appdir, monkeypatch):
    downloads = []
    monkeypatch.setattr(gu, "download_plot", lambda *args: downloads.append(args))
    download = _handler("download_plot")

    assert download(None, "spectrogram") is None
    assert downloads == []

    download("plot", "regression")

    assert downloads == [("plot", "regression")]
