import re

import pandas as pd
import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.evaluation.preprocessing import data_processor  # noqa: E402
from birdnet_analyzer.gui import evaluation  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.fixture
def appdir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "APPDIR", tmp_path)
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr(gs, "_PERSISTED", [])
    return tmp_path


@pytest.fixture
def demo(appdir):
    with gr.Blocks() as blocks:
        evaluation.build_evaluation_tab()

    return blocks


def _event(demo, name):
    return next(
        event
        for event in demo.fns.values()
        if getattr(event.fn, "__name__", "") == name
    )


class FakeProcessor:
    classes = ("species",)
    samples_df = pd.DataFrame({"filename": ["recording.wav"]})

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.recording_duration = kwargs.get("recording_duration")


def _duration_textbox():
    label = loc.localize("eval-tab-recording-duration-textbox-label")

    return next(
        c
        for c in gs.persisted_components()
        if isinstance(c, gr.Textbox) and c.label == label
    )


def _call(event, values):
    return event.fn(*(values.get(c) for c in event.inputs))


def _refresh(demo, recording_duration):
    event = _event(demo, "refresh_processor")
    columns = [""] * (len(event.inputs) - 9)

    return event.fn(
        "annotations",
        "predictions",
        None,
        3,
        0.5,
        recording_duration,
        False,
        None,
        None,
        *columns,
    )


def _processor_state(recording_duration=None):
    return evaluation.ProcessorState(
        FakeProcessor(recording_duration=recording_duration),
        "annotations",
        "predictions",
    )


def test_the_overall_row_follows_programmatic_averaging_changes(demo):
    event = _event(demo, "recompute_overall")
    averaging_radio = event.inputs[-1]

    assert event.targets == [(averaging_radio._id, "change")]
    assert event.fn(None, None, None, "micro") == gr.update()


@pytest.mark.parametrize(
    ("text", "duration"),
    [("12.5", 12.5), (" 60 ", 60.0), ("3", 3.0), ("", None), ("  ", None)],
)
def test_a_recording_duration_is_parsed(demo, monkeypatch, text, duration):
    monkeypatch.setattr(data_processor, "DataProcessor", FakeProcessor)

    *_, proc_state = _refresh(demo, text)

    assert proc_state.processor.kwargs["recording_duration"] == duration


@pytest.mark.parametrize("text", ["12o", "1", "0", "-5"])
def test_a_duration_the_processor_cannot_use_does_not_interrupt_the_refresh(
    demo, monkeypatch, text
):
    monkeypatch.setattr(data_processor, "DataProcessor", FakeProcessor)

    *_, proc_state = _refresh(demo, text)

    assert proc_state.processor.kwargs["recording_duration"] is None


@pytest.mark.parametrize("handler", ["calculate_metrics", "download_data_table"])
@pytest.mark.parametrize("text", ["12o", "1", "15", "", "nan"])
def test_a_duration_the_processor_was_not_built_with_is_rejected(
    demo, monkeypatch, handler, text
):
    event = _event(demo, handler)
    dialogs = []
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: dialogs.append(k))
    message = loc.localize("eval-tab-error-no-valid-recording-duration")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _call(
            event, {event.inputs[0]: _processor_state(12.0), _duration_textbox(): text}
        )

    assert dialogs == []


@pytest.mark.parametrize("text", ["12", " 12.0 "])
def test_the_duration_the_processor_was_built_with_is_accepted(demo, monkeypatch, text):
    event = _event(demo, "download_data_table")
    dialogs = []
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: dialogs.append(k))

    _call(event, {event.inputs[0]: _processor_state(12.0), _duration_textbox(): text})

    assert len(dialogs) == 1


def test_calculating_without_a_metric_is_rejected(demo):
    event = _event(demo, "calculate_metrics")
    message = loc.localize("eval-tab-error-no-metric-selected")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _call(event, {event.inputs[0]: _processor_state()})


def test_a_failed_data_table_export_is_logged_with_its_step(
    demo, monkeypatch, tmp_path
):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: str(tmp_path / "data.csv"))
    state = _processor_state()

    def fail():
        raise OSError("disk")

    state.processor.get_sample_data = fail

    with pytest.raises(gr.Error) as raised:
        _event(demo, "download_data_table").fn(state, None)

    assert raised.value.message.startswith(
        loc.localize("eval-tab-error-saving-data-table")
    )
    assert "disk" in raised.value.message
    assert len(logged) == 1


def test_exporting_before_calculating_is_reported_as_is(demo, monkeypatch):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    with pytest.raises(gr.Error) as raised:
        _event(demo, "download_results_table").fn(None, None, None, False, "macro")

    assert raised.value.message == loc.localize("eval-tab-error-calc-metrics-first")
    assert logged == []
