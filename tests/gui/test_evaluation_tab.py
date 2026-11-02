import asyncio
import csv
import re
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

gr = pytest.importorskip("gradio")

from gradio.state_holder import SessionState  # noqa: E402

import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer.evaluation.assessment import performance_assessor  # noqa: E402
from birdnet_analyzer.evaluation.assessment.performance_assessor import (  # noqa: E402
    PerformanceAssessor,
)
from birdnet_analyzer.evaluation.preprocessing import data_processor  # noqa: E402
from birdnet_analyzer.gui import evaluation  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402

RECORDINGS = ("rec1", "rec2", "rec3")
DEFAULT_METRICS = [default for *_, default in evaluation._METRIC_CONTROLS.values()]
PROCESSOR_HANDLERS = {
    "refresh_processor",
    "refresh_after_pick",
    "check_parameters",
    "calculate_metrics",
    "download_data_table",
}


@pytest.fixture
def demo(appdir):
    with gr.Blocks() as blocks:
        evaluation.build_evaluation_tab()

    return blocks


def _write_table(path, header, rows):
    lines = ["\t".join(header), *("\t".join(str(v) for v in row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recording(folders, recording, class_column="Common Name", annotated=True):
    duration = "File Duration (s)"
    if annotated:
        _write_table(
            folders["annotation"] / f"{recording}.txt",
            ["Begin Time (s)", "End Time (s)", "Class", duration],
            [[0, 3, "Robin", 30], [3, 6, "Wren", 30]],
        )
    _write_table(
        folders["prediction"] / f"{recording}.txt",
        ["Begin Time (s)", "End Time (s)", class_column, "Confidence", duration],
        [[0, 3, "Robin", 0.9, 30], [3, 6, "Wren", 0.6, 30], [6, 9, "Robin", 0.3, 30]],
    )


def _folders(tmp_path, class_column):
    folders = {
        "annotation": tmp_path / "annotations",
        "prediction": tmp_path / "predictions",
    }
    for folder in folders.values():
        folder.mkdir()
    for recording in RECORDINGS:
        _write_recording(folders, recording, class_column)

    return folders


@pytest.fixture
def folders(tmp_path):
    return _folders(tmp_path, "Common Name")


@pytest.fixture
def unmapped_folders(tmp_path):
    return _folders(tmp_path, "Species")


def _values(
    folders,
    recording_duration="",
    min_overlap=0.5,
    score_unannotated=False,
    prediction_class=None,
):
    """The processor inputs as the tab passes them, columns left at their defaults."""
    annotation_columns = [None] * 5
    prediction_columns = [None, None, prediction_class, None, None, None]

    return (
        str(folders["annotation"]),
        str(folders["prediction"]),
        None,
        3,
        min_overlap,
        recording_duration,
        score_unannotated,
        *annotation_columns,
        *prediction_columns,
    )


def _refresh(proc_state, values, classes=None, recordings=None, listed=None):
    listed = evaluation._Listed() if listed is None else listed

    return evaluation.refresh_processor(
        listed, proc_state, classes, recordings, *values
    )


def _calculate(proc_state, values, classes=None, recordings=None, metrics=None):
    outputs = evaluation.calculate_metrics(
        proc_state,
        0.5,
        False,
        "macro",
        classes,
        recordings,
        *(DEFAULT_METRICS if metrics is None else metrics),
        *values,
    )

    return SimpleNamespace(
        notes=outputs[3]["value"], labels=outputs[9], proc_state=outputs[10]
    )


@pytest.fixture
def built(monkeypatch):
    """The recording duration of every DataProcessor built."""
    durations = []

    class CountingProcessor(data_processor.DataProcessor):
        def __init__(self, **kwargs):
            durations.append(kwargs["recording_duration"])
            super().__init__(**kwargs)

    monkeypatch.setattr(data_processor, "DataProcessor", CountingProcessor)
    return durations


def _event(demo, name, trigger=None):
    return next(
        event
        for event in demo.fns.values()
        if getattr(event.fn, "__name__", "") == name
        and (trigger is None or trigger in event.targets)
    )


def _persisted(component_type, label_key):
    label = loc.localize(label_key)

    return next(
        c
        for c in gs.persisted_components()
        if isinstance(c, component_type) and c.label == label
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
def test_a_recording_duration_is_parsed(folders, text, duration):
    *_, proc_state = _refresh(None, _values(folders, text))

    assert proc_state.processor.recording_duration == duration


@pytest.mark.parametrize("text", ["12o", "1", "0", "-5"])
def test_a_duration_the_processor_cannot_use_does_not_interrupt_the_refresh(
    folders, text
):
    *_, proc_state = _refresh(None, _values(folders, text))

    assert proc_state.processor.recording_duration is None


def test_calculating_with_a_changed_duration_uses_a_processor_built_with_it(folders):
    *_, built_with_30 = _refresh(None, _values(folders, "30"))

    result = _calculate(built_with_30, _values(folders, "60"))

    assert built_with_30.processor.recording_duration == 30.0
    assert result.proc_state.processor.recording_duration == 60.0
    assert result.labels.shape == (len(RECORDINGS) * 20, 2)


@pytest.mark.parametrize("text", ["12o", "1", "nan"])
@pytest.mark.parametrize(
    "handler",
    [
        lambda values: _calculate(None, values),
        lambda values: evaluation.download_data_table(None, *values),
        lambda values: evaluation.check_parameters(
            evaluation._Checked(), None, *values
        ),
    ],
    ids=["calculate", "download", "check"],
)
def test_a_duration_the_processor_cannot_use_is_rejected_before_reading(
    folders, monkeypatch, built, handler, text
):
    dialogs = []
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: dialogs.append(k))
    message = loc.localize("eval-tab-error-no-valid-recording-duration")

    with pytest.raises(gr.Error, match=re.escape(message)):
        handler(_values(folders, text))

    assert built == []
    assert dialogs == []


def test_the_processor_is_built_once_per_change_of_its_inputs(folders, built):
    checked = evaluation._Checked()
    *_, proc_state = _refresh(None, _values(folders))
    proc_state = evaluation.check_parameters(checked, proc_state, *_values(folders))
    proc_state = _calculate(proc_state, _values(folders)).proc_state

    proc_state = evaluation.check_parameters(
        checked, proc_state, *_values(folders, "60")
    )
    _calculate(proc_state, _values(folders, "60"))

    assert built == [None, 60.0]


def test_a_folder_picked_again_is_read_again_and_its_new_recordings_selected(
    folders, built
):
    classes, recordings, proc_state = _refresh(None, _values(folders))
    _write_recording(folders, "rec4")

    *_, offered, picked = evaluation.refresh_after_pick(
        True,
        evaluation._Listed(),
        proc_state,
        classes["value"],
        recordings["value"],
        *_values(folders),
    )

    assert len(built) == 2
    assert offered["choices"] == offered["value"] == [*RECORDINGS, "rec4"]
    assert picked.processor is not proc_state.processor


def test_a_cancelled_folder_dialog_reads_nothing(folders, built):
    classes, recordings, proc_state = _refresh(None, _values(folders))

    *_, kept = evaluation.refresh_after_pick(
        False,
        evaluation._Listed(),
        proc_state,
        classes["value"],
        recordings["value"],
        *_values(folders),
    )

    assert kept is proc_state
    assert len(built) == 1


def test_a_partial_selection_survives_a_folder_picked_again(folders):
    classes, _, proc_state = _refresh(None, _values(folders))
    _write_recording(folders, "rec4")

    _, recordings, _ = evaluation.refresh_after_pick(
        True,
        evaluation._Listed(),
        proc_state,
        classes["value"],
        ["rec1", "rec3"],
        *_values(folders),
    )

    assert recordings["value"] == ["rec1", "rec3"]


def test_an_unmapped_column_leaves_the_lists_alone_until_calculating(
    unmapped_folders,
):
    updates = _refresh(None, _values(unmapped_folders))

    assert updates == (gr.update(), gr.update(), None)
    with pytest.raises(gr.Error) as raised:
        _calculate(None, _values(unmapped_folders))
    assert raised.value.message.startswith(loc.localize("eval-tab-error-missing-col"))
    assert "Common Name" in raised.value.message


def test_a_partial_selection_survives_a_column_unmapped_and_mapped_again(
    unmapped_folders,
):
    mapped = _values(unmapped_folders, prediction_class="Species")
    classes, _, proc_state = _refresh(None, mapped)

    *_, still = _refresh(
        proc_state, _values(unmapped_folders), classes["value"], ["rec1", "rec3"]
    )
    _, recordings, _ = _refresh(still, mapped, classes["value"], ["rec1", "rec3"])

    assert still is proc_state
    assert recordings["value"] == ["rec1", "rec3"]


def test_mapping_the_missing_column_offers_the_classes_and_recordings(
    unmapped_folders,
):
    mapped = _values(unmapped_folders, prediction_class="Species")

    classes, recordings, proc_state = _refresh(None, mapped)

    assert classes["value"] == ["Robin", "Wren"]
    assert recordings["value"] == list(RECORDINGS)
    assert _calculate(proc_state, mapped).labels.shape == (len(RECORDINGS) * 10, 2)


@pytest.mark.parametrize(
    ("selection", "rows"), [(["rec1", "rec3"], 2 * 10), ([], len(RECORDINGS) * 10)]
)
def test_the_recording_selection_limits_the_tensors_and_none_means_all(
    folders, selection, rows
):
    result = _calculate(None, _values(folders), recordings=selection)

    assert result.labels.shape == (rows, 2)


def test_the_data_table_is_exported_for_the_current_values(
    folders, monkeypatch, tmp_path
):
    target = tmp_path / "data.csv"
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: str(target))
    *_, stale = _refresh(None, _values(folders))

    evaluation.download_data_table(stale, *_values(folders, "60"))

    with open(target, newline="") as f:
        assert len(list(csv.DictReader(f))) == len(RECORDINGS) * 20


@pytest.mark.parametrize(
    ("score_unannotated", "key"),
    [
        (False, "eval-tab-note-unmatched-dropped"),
        (True, "eval-tab-note-unmatched-empty"),
    ],
)
def test_the_notes_say_how_recordings_without_annotations_were_scored(
    folders, score_unannotated, key
):
    _write_recording(folders, "rec9", annotated=False)

    result = _calculate(None, _values(folders, score_unannotated=score_unannotated))

    assert f"{loc.localize(key)} rec9" in result.notes


def test_calculating_without_a_metric_is_rejected(folders):
    message = loc.localize("eval-tab-error-no-metric-selected")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _calculate(None, _values(folders), metrics=[False] * len(DEFAULT_METRICS))


def test_a_failed_data_table_export_is_logged_with_its_step(
    folders, monkeypatch, tmp_path
):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: str(tmp_path / "data.csv"))
    *_, state = _refresh(None, _values(folders))

    def fail():
        raise OSError("disk")

    state.processor.get_sample_data = fail

    with pytest.raises(gr.Error) as raised:
        evaluation.download_data_table(state, *_values(folders))

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


def test_the_parameter_fields_are_checked_when_left_not_per_keystroke(demo):
    fields = [
        _persisted(gr.Number, "eval-tab-sample-duration-number-label"),
        _persisted(gr.Textbox, "eval-tab-recording-duration-textbox-label"),
        _persisted(gr.Number, "eval-tab-min-overlap-number-label"),
    ]
    targets = {
        target
        for event in demo.fns.values()
        if getattr(event.fn, "__name__", "") in PROCESSOR_HANDLERS
        for target in event.targets
    }

    for field in fields:
        assert (field._id, "blur") in targets
        assert (field._id, "submit") in targets
        assert not {(field._id, "change"), (field._id, "input")} & targets


def test_everything_that_builds_the_processor_shares_one_queue(demo):
    queues = {
        event.concurrency_id
        for event in demo.fns.values()
        if getattr(event.fn, "__name__", "") in PROCESSOR_HANDLERS
    }

    assert len(queues) == 1


def test_calculating_stays_locked_until_the_lists_follow_the_inputs(demo, folders):
    events = demo.fns

    def chain_start(event):
        while event.trigger_after is not None:
            event = events[event.trigger_after]
        return event

    refreshes = [
        event
        for event in events.values()
        if getattr(event.fn, "__name__", "")
        in {"refresh_processor", "refresh_after_pick"}
    ]

    assert refreshes
    for refresh in refreshes:
        unlock = next(
            e
            for e in events.values()
            if e.trigger_after == refresh._id and not e.trigger_only_on_failure
        )
        assert chain_start(refresh).fn() == gr.update(interactive=False)
        assert unlock.fn is evaluation._unlock_calculation
        assert unlock.inputs[0] in refresh.inputs
        assert chain_start(refresh).outputs == unlock.outputs
        assert refresh.trigger_mode == "always_last"
        assert unlock.trigger_mode == "multiple"

    listed = evaluation._Listed()
    unlock = evaluation._unlock_calculation
    unmapped = _values(folders, prediction_class="Species")
    evaluation.refresh_after_pick(True, listed, None, None, None, *_values(folders))

    assert unlock(listed, *_values(folders)) == gr.update(interactive=True)
    assert unlock(listed, *_values(folders, min_overlap=0.2)) == gr.update(
        interactive=True
    )
    assert unlock(listed, *unmapped) == gr.update()

    _refresh(None, unmapped, listed=listed)

    assert unlock(listed, *unmapped) == gr.update(interactive=True)


def test_a_failed_step_empties_the_lists_and_unlocks_calculating(demo):
    events = demo.fns
    names = {"refresh_processor", "refresh_after_pick", "select_directory"}
    steps = [e for e in events.values() if getattr(e.fn, "__name__", "") in names]
    refresh = next(e for e in steps if e.fn.__name__ == "refresh_processor")

    assert len(steps) == 5
    for step in steps:
        recover = next(
            e
            for e in events.values()
            if e.trigger_after == step._id and e.trigger_only_on_failure
        )
        classes, recordings, button = asyncio.run(
            demo.process_api(recover, [], SessionState(demo))
        )["data"]

        assert recover.outputs[:2] == refresh.outputs[:2]
        assert classes["choices"] == classes["value"] == []
        assert recordings["choices"] == recordings["value"] == []
        assert button["interactive"] is True


def test_an_invalid_parameter_is_reported_once_when_entered_and_left(folders):
    checked = evaluation._Checked()
    kept = object()
    message = re.escape(loc.localize("eval-tab-error-no-valid-recording-duration"))

    with pytest.raises(gr.Error, match=message):
        evaluation.check_parameters(checked, None, *_values(folders, "12o"))
    assert evaluation.check_parameters(checked, kept, *_values(folders, "12o")) is kept
    with pytest.raises(gr.Error, match=message):
        evaluation.check_parameters(checked, None, *_values(folders, "1o"))


def test_recordings_are_chosen_in_a_multiselect_dropdown(demo):
    label = loc.localize("eval-tab-select-recordings-checkboxgroup-label")
    recordings = next(
        c for c in demo.blocks.values() if getattr(c, "label", None) == label
    )

    assert isinstance(recordings, gr.Dropdown)
    assert recordings.multiselect


def test_the_tab_offers_every_registered_metric():
    assert set(evaluation._METRIC_CONTROLS) == set(performance_assessor.METRICS)


@pytest.mark.parametrize(
    ("score_unannotated", "key"),
    [
        (False, "eval-tab-note-unmatched-dropped"),
        (True, "eval-tab-note-unmatched-empty"),
    ],
)
def test_the_notes_name_unmatched_recordings_and_empty_classes(score_unannotated, key):
    notes = evaluation._build_notes({"rec2", "rec1"}, ("Wren",), score_unannotated)

    assert notes == (
        f"⚠️ {loc.localize(key)} rec1, rec2\n\n"
        f"📊 {loc.localize('eval-tab-note-empty-classes')} Wren"
    )


def test_there_are_no_notes_when_everything_was_scored():
    assert evaluation._build_notes(set(), (), False) == ""


def test_the_metrics_tables_have_a_row_per_class_and_an_overall_footer():
    rng = np.random.default_rng(0)
    predictions = rng.random((50, 2))
    labels = rng.integers(0, 2, (50, 2))
    pa = PerformanceAssessor(
        num_classes=2, threshold=0.4, classes=("a", "b"), metrics_list=("f1", "recall")
    )

    class_table, overall_row, _ = evaluation._build_metrics_tables(
        pa, predictions, labels, "micro"
    )

    per_class = pa.calculate_metrics(predictions, labels, per_class_metrics=True)
    overall = pa.calculate_metrics(predictions, labels, averaging="micro")
    assert list(class_table.columns) == ["", "F1", "Recall", "Support"]
    assert class_table[""].tolist() == ["a", "b"]
    assert class_table["F1"].tolist() == per_class.loc["F1"].round(3).tolist()
    assert class_table["Support"].tolist() == labels.sum(axis=0).tolist()
    assert overall_row[""].tolist() == [loc.localize("eval-tab-overall-row-label")]
    assert overall_row["Recall"].tolist() == overall.loc["Recall"].round(3).tolist()
    assert overall_row["Support"].tolist() == [labels.sum()]


def _processor(classes, recordings):
    return SimpleNamespace(
        classes=tuple(classes), samples_df=pd.DataFrame({"filename": recordings})
    )


@pytest.mark.parametrize(
    ("previous", "current_classes", "current_recordings", "classes", "recordings"),
    [
        (None, None, None, ["a", "b", "c"], ["r1", "r2"]),
        (None, ["c", "x"], ["r2"], ["c"], ["r2"]),
        (None, ["x"], ["r9"], ["a", "b", "c"], ["r1", "r2"]),
        (_processor("ab", ["r1"]), ["a", "b"], ["r1"], ["a", "b", "c"], ["r1", "r2"]),
        (_processor("ab", ["r1"]), ["a"], ["r1"], ["a"], ["r1", "r2"]),
    ],
)
def test_a_refresh_keeps_a_partial_selection_and_extends_a_full_one(
    previous, current_classes, current_recordings, classes, recordings
):
    processor = _processor("abc", ["r1", "r1", "r2"])

    assert evaluation._selection_choices(
        processor, previous, current_classes, current_recordings
    ) == ((["a", "b", "c"], classes), (["r1", "r2"], recordings))


def test_the_tab_draws_its_figures_without_a_gui_backend():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    pa = PerformanceAssessor(num_classes=2, classes=("a", "b"))
    plt.switch_backend("template")

    try:
        figures = evaluation._build_figures(
            pa, rng.random((50, 2)), rng.integers(0, 2, (50, 2)), False
        )
        backend = matplotlib.get_backend()
    finally:
        plt.switch_backend("agg")

    assert backend == "agg"
    assert all(figure is not None for figure in figures)
