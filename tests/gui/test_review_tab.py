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

    return next(
        event.fn
        for event in demo.fns.values()
        if getattr(event.fn, "__name__", "") == name
    )


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
    empty = {
        "input_directory": "",
        "species_list": [],
        "current_species": "",
        "files": [],
        review.POSITIVE_LABEL_DIR: [],
        review.NEGATIVE_LABEL_DIR: [],
        "skipped": [],
        "history": [],
    }

    updates = _handler("start_review")(empty)

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
