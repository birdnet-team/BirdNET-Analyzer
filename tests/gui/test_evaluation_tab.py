import csv
import json
import re
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

gr = pytest.importorskip("gradio")

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
PREDICTION_HEADER = [
    "Begin Time (s)",
    "End Time (s)",
    "Common Name",
    "Scientific Name",
    "Confidence",
    "File Duration (s)",
]


@pytest.fixture
def demo(appdir):
    with gr.Blocks() as blocks:
        evaluation.build_evaluation_tab()

    return blocks


def _write_table(path, header, rows):
    lines = ["\t".join(header), *("\t".join(str(v) for v in row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_params(folder, rows):
    (folder / "birdnet.analyze-params.csv").write_text(
        "Parameter,Value\nSegment length,5.0\nSegment overlap,0.0\n"
        f"Merge consecutive detections,1\nAudio speed,1.0\n{rows}",
        encoding="utf-8",
    )


def _write_recording(
    folders, recording, class_column="Common Name", annotated=True, names=None
):
    """Robin and Wren annotated; Robin detected twice, Wren once."""
    robin, wren = names or ("Robin", "Wren")
    duration = "File Duration (s)"
    if annotated:
        _write_table(
            folders["annotation"] / f"{recording}.txt",
            ["Begin Time (s)", "End Time (s)", "Class", duration],
            [[0, 3, robin, 30], [3, 6, wren, 30]],
        )
    _write_table(
        folders["prediction"] / f"{recording}.txt",
        ["Begin Time (s)", "End Time (s)", class_column, "Confidence", duration],
        [[0, 3, "Robin", 0.9, 30], [3, 6, "Wren", 0.6, 30], [6, 9, "Robin", 0.3, 30]],
    )


def _folders(tmp_path, class_column="Common Name", names=None):
    folders = {
        "annotation": tmp_path / "annotations",
        "prediction": tmp_path / "predictions",
    }
    for folder in folders.values():
        folder.mkdir()
    for recording in RECORDINGS:
        _write_recording(folders, recording, class_column, names=names)

    return folders


@pytest.fixture
def folders(tmp_path):
    return _folders(tmp_path)


@pytest.fixture
def unmapped_folders(tmp_path):
    return _folders(tmp_path, "Species")


def _values(
    folders,
    recording_duration="",
    min_overlap=0.5,
    score_unannotated=False,
    prediction_class=None,
    mapping=None,
    sample_duration=3,
    annotation_class=None,
    prediction_recording=None,
):
    """The processor inputs as the tab passes them after a pick of the folders."""
    duration = "File Duration (s)"
    annotation_columns = [None, None, annotation_class, None, duration]
    prediction_columns = [
        None,
        None,
        prediction_class,
        None,
        prediction_recording,
        duration,
    ]

    return (
        str(folders["annotation"]),
        str(folders["prediction"]),
        mapping,
        sample_duration,
        min_overlap,
        recording_duration,
        score_unannotated,
        *annotation_columns,
        *prediction_columns,
    )


_REFRESHED = (
    "classes",
    "recordings",
    "proc_state",
    "status",
    "suggestions",
    "suggestions_group",
    "suggestion_choices",
    "mapping_group",
    "selection_group",
    "class_column",
    "annotation_columns",
    "prediction_columns",
)


def _refresh(proc_state, values, classes=None, recordings=None, picked=None):
    if picked is None:
        outputs = evaluation.refresh_processor(proc_state, classes, recordings, *values)
    else:
        outputs = evaluation.refresh_after_pick(
            picked, proc_state, classes, recordings, *values
        )

    return SimpleNamespace(**dict(zip(_REFRESHED, outputs, strict=True)))


def _calculate(
    proc_state,
    values,
    classes=None,
    recordings=None,
    metrics=None,
    threshold=0.5,
    averaging="macro",
    class_wise=False,
):
    outputs = evaluation.calculate_metrics(
        proc_state,
        threshold,
        class_wise,
        averaging,
        classes,
        recordings,
        *(DEFAULT_METRICS if metrics is None else metrics),
        *values,
    )

    return SimpleNamespace(
        header=outputs[1],
        table=outputs[2]["value"],
        overall=outputs[3]["value"],
        notes=outputs[4]["value"],
        pa=outputs[8],
        predictions=outputs[9],
        labels=outputs[10],
        proc_state=outputs[11],
        class_wise=outputs[12],
        widths=outputs[2]["column_widths"],
    )


def _pick(monkeypatch, kind, folder, sample_duration=3, threshold=0.1):
    monkeypatch.setattr(gu, "select_folder", lambda **_: str(folder))
    outputs = evaluation.select_directory(kind, None, sample_duration, threshold)
    n_fields = len(evaluation._DEFAULT_COLUMNS[kind])

    return SimpleNamespace(
        columns=[update["value"] for update in outputs[4 : 4 + n_fields]],
        sample_duration=outputs[4 + n_fields],
        threshold=outputs[5 + n_fields],
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
    averaging_radio = event.inputs[3]

    assert event.targets == [(averaging_radio._id, "change")]
    assert event.fn(None, None, None, "micro", False) == (gr.update(),) * 3


@pytest.mark.parametrize("class_wise", [False, True], ids=["overall", "per class"])
def test_switching_the_averaging_after_a_calculation_renews_the_overall_scores(
    folders, class_wise
):
    result = _calculate(None, _values(folders), class_wise=class_wise)

    overall, metrics_plot, thresholds_plot = evaluation.recompute_overall(
        result.pa, result.predictions, result.labels, "micro", result.class_wise
    )

    micro = result.pa.calculate_metrics(
        result.predictions, result.labels, averaging="micro"
    )
    assert overall["value"][""].tolist() == [
        loc.localize("eval-tab-overall-micro-label").format(n=2)
    ]
    assert overall["value"]["Recall"].tolist() == micro.loc["Recall"].round(3).tolist()
    assert overall["column_widths"] == result.widths
    assert len(result.widths) == len(result.table.columns)
    if class_wise:
        assert metrics_plot == thresholds_plot == gr.update()
    else:
        assert metrics_plot is not None
        assert thresholds_plot is not None
        assert not isinstance(metrics_plot, dict)


def test_an_averaging_change_waits_for_a_running_calculation_and_follows_its_plots(
    demo,
):
    recompute = _event(demo, "recompute_overall")
    calculate = _event(demo, "calculate_metrics")

    assert recompute.concurrency_id == calculate.concurrency_id
    assert recompute.inputs[:3] == calculate.outputs[8:11]
    assert recompute.inputs[4] in calculate.outputs
    assert recompute.inputs[4] not in calculate.inputs


@pytest.mark.parametrize(
    ("text", "duration"),
    [("12.5", 12.5), (" 60 ", 60.0), ("3", 3.0), ("90,5", 90.5), ("", None)],
)
def test_a_recording_duration_is_parsed(folders, text, duration):
    refreshed = _refresh(None, _values(folders, text))

    assert refreshed.proc_state.processor.recording_duration == duration


@pytest.mark.parametrize("text", ["12o", "1", "0", "-5"])
def test_a_duration_the_processor_cannot_use_does_not_interrupt_the_refresh(
    folders, text
):
    refreshed = _refresh(None, _values(folders, text))

    assert refreshed.proc_state.processor.recording_duration is None


def test_calculating_with_a_changed_duration_uses_a_processor_built_with_it(folders):
    built_with_30 = _refresh(None, _values(folders, "30")).proc_state

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
        lambda values: evaluation.check_parameters(None, *values),
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


@pytest.mark.parametrize(
    ("sample_duration", "min_overlap", "key"),
    [
        (None, 0.5, "eval-tab-error-window-length"),
        (0, 0.5, "eval-tab-error-window-length"),
        (3, None, "eval-tab-error-min-overlap"),
        (3, 4, "eval-tab-error-min-overlap"),
    ],
)
def test_an_unusable_window_or_overlap_is_explained_before_reading(
    folders, built, sample_duration, min_overlap, key
):
    values = _values(folders, min_overlap=min_overlap, sample_duration=sample_duration)

    with pytest.raises(gr.Error, match=re.escape(loc.localize(key))):
        evaluation.check_parameters(None, *values)

    assert loc.localize(key) in _refresh(None, values).status["value"]
    assert built == []


def test_the_processor_is_built_once_per_change_of_its_inputs(folders, built):
    proc_state = _refresh(None, _values(folders)).proc_state
    proc_state, *_ = evaluation.check_parameters(proc_state, *_values(folders))
    proc_state = _calculate(proc_state, _values(folders)).proc_state

    proc_state, *_ = evaluation.check_parameters(proc_state, *_values(folders, "60"))
    _calculate(proc_state, _values(folders, "60"))

    assert built == [None, 60.0]


def test_tables_changed_on_disk_are_read_again_before_anything_is_calculated(folders):
    proc_state = _refresh(None, _values(folders)).proc_state
    _write_recording(folders, "rec4")
    changed = loc.localize("eval-tab-error-tables-changed")

    with pytest.raises(gr.Error) as raised:
        _calculate(proc_state, _values(folders))
    checked = evaluation.check_parameters(proc_state, *_values(folders, "60"))
    picked = _refresh(proc_state, _values(folders), picked=True)
    result = _calculate(picked.proc_state, _values(folders))

    assert raised.value.message == changed
    assert checked[1]["value"] == f"⚠️ {changed}"
    assert picked.recordings["choices"] == [*RECORDINGS, "rec4"]
    assert result.labels.shape == ((len(RECORDINGS) + 1) * 10, 2)


def test_a_folder_picked_again_is_read_again_and_its_new_recordings_selected(
    folders, built
):
    first = _refresh(None, _values(folders))
    _write_recording(folders, "rec4")

    again = _refresh(
        first.proc_state,
        _values(folders),
        first.classes["value"],
        first.recordings["value"],
        picked=True,
    )

    assert len(built) == 2
    assert again.recordings["choices"] == again.recordings["value"]
    assert again.recordings["value"] == [*RECORDINGS, "rec4"]
    assert again.proc_state.processor is not first.proc_state.processor


def test_a_cancelled_folder_dialog_reads_nothing(folders, built):
    first = _refresh(None, _values(folders))

    kept = _refresh(
        first.proc_state,
        _values(folders),
        first.classes["value"],
        first.recordings["value"],
        picked=False,
    )

    assert kept.proc_state is first.proc_state
    assert len(built) == 1


def test_a_partial_selection_survives_a_folder_picked_again(folders):
    first = _refresh(None, _values(folders))
    _write_recording(folders, "rec4")

    again = _refresh(
        first.proc_state,
        _values(folders),
        first.classes["value"],
        ["rec1", "rec3"],
        picked=True,
    )

    assert again.recordings["value"] == ["rec1", "rec3"]


def test_an_unmapped_column_is_explained_and_its_column_box_opened(unmapped_folders):
    refreshed = _refresh(None, _values(unmapped_folders))

    assert refreshed.classes == gr.update()
    assert refreshed.proc_state is None
    assert '"Common Name"' in refreshed.status["value"]
    assert "Species" in refreshed.status["value"]
    assert refreshed.prediction_columns == gr.update(open=True)
    assert refreshed.annotation_columns == gr.update()
    with pytest.raises(gr.Error) as raised:
        _calculate(None, _values(unmapped_folders))
    assert raised.value.message == refreshed.status["value"].removeprefix("⚠️ ")


def test_a_partial_selection_survives_a_column_unmapped_and_mapped_again(
    unmapped_folders,
):
    mapped = _values(unmapped_folders, prediction_class="Species")
    first = _refresh(None, mapped)

    still = _refresh(
        first.proc_state,
        _values(unmapped_folders),
        first.classes["value"],
        ["rec1", "rec3"],
    ).proc_state
    again = _refresh(still, mapped, first.classes["value"], ["rec1", "rec3"])

    assert still is first.proc_state
    assert again.recordings["value"] == ["rec1", "rec3"]


def test_mapping_the_missing_column_offers_the_classes_and_recordings(
    unmapped_folders,
):
    mapped = _values(unmapped_folders, prediction_class="Species")

    refreshed = _refresh(None, mapped)

    assert refreshed.classes["value"] == ["Robin", "Wren"]
    assert refreshed.recordings["value"] == list(RECORDINGS)
    assert _calculate(refreshed.proc_state, mapped).labels.shape == (
        len(RECORDINGS) * 10,
        2,
    )


def test_the_status_reports_what_was_paired(folders):
    status = _refresh(None, _values(folders)).status["value"]

    assert loc.localize("eval-tab-status-recordings-all").format(n=3) in status
    assert (
        loc.localize("eval-tab-status-classes-all").format(n=2, column="Common Name")
        in status
    )


@pytest.mark.parametrize(
    ("score_unannotated", "key"),
    [
        (False, "eval-tab-status-prediction-only-dropped"),
        (True, "eval-tab-status-prediction-only-empty"),
    ],
)
def test_the_status_names_recordings_that_could_not_be_paired(
    folders, score_unannotated, key
):
    _write_recording(folders, "rec9", annotated=False)
    _write_table(
        folders["annotation"] / "rec8.txt",
        ["Begin Time (s)", "End Time (s)", "Class"],
        [[0, 3, "Robin"]],
    )

    status = _refresh(
        None, _values(folders, score_unannotated=score_unannotated)
    ).status["value"]

    assert (
        loc.localize("eval-tab-status-recordings-some").format(matched=3, total=5)
        in status
    )
    assert (
        loc.localize("eval-tab-status-annotation-only").format(names="rec8") in status
    )
    assert loc.localize(key).format(names="rec9") in status
    assert loc.localize("eval-tab-status-pairing-rule") in status


def test_the_status_says_what_was_read_left_out_and_how_the_analysis_ran(folders):
    (folders["annotation"] / "README.txt").write_text("By J. Doe\n", encoding="utf-8")
    (folders["annotation"] / "notes.txt").write_bytes(b"\x00\x01")
    _write_recording(folders, "rec4", annotated=False)
    _write_table(
        folders["annotation"] / "rec4.txt",
        ["Begin Time (s)", "End Time (s)", "Class"],
        [],
    )
    for table in sorted(folders["prediction"].glob("*.txt")):
        table.rename(table.with_name(f"{table.stem}.BirdNET.selection.table.txt"))
    _write_table(folders["prediction"] / "BirdNET_SelectionTable.txt", ["x"], [])
    _write_params(folders["prediction"], "Model,perch\nMinimum confidence,0.25\n")

    status = _refresh(None, _values(folders, sample_duration=5)).status["value"]

    settings = ", ".join(
        [
            "Perch",
            loc.localize("eval-tab-status-segment").format(segment=5),
            loc.localize("eval-tab-status-min-conf").format(min_conf=0.25),
        ]
    )
    expected = [
        loc.localize("eval-tab-status-annotations").format(tables=4, rows=6, classes=2),
        loc.localize("eval-tab-status-empty-annotations").format(n=1),
        loc.localize("eval-tab-status-predictions").format(
            tables=4, rows=12, classes=2
        ),
        loc.localize("eval-tab-status-analysis").format(settings=settings),
        loc.localize("eval-tab-status-recordings-all").format(n=4),
        loc.localize("eval-tab-status-skipped-annotation").format(
            n=2, names="README.txt, notes.txt"
        ),
        loc.localize("eval-tab-status-superseded").format(
            n=1, names="BirdNET_SelectionTable.txt"
        ),
    ]
    assert [line for line in status.split("\n") if line in expected] == expected


def _move_predictions(folders, subfolders, columns=None):
    tables = sorted(folders["prediction"].glob("*.txt"))
    for table in tables:
        frame = pd.read_csv(table, sep="\t")
        for name, value in (columns or {}).items():
            frame[name] = value.format(recording=table.stem)
        for subfolder in subfolders:
            (folders["prediction"] / subfolder).mkdir(exist_ok=True)
            frame.to_csv(
                folders["prediction"] / subfolder / table.name, sep="\t", index=False
            )
        table.unlink()


def _two_analyses(folders):
    _move_predictions(
        folders, ["run_24", "run_30"], {"Begin Path": "/audio/{recording}.wav"}
    )


def _two_sites(folders):
    _move_predictions(folders, ["site1", "site2"])


def _decimal_comma(folders):
    _write_table(
        folders["annotation"] / "rec1.txt",
        ["Begin Time (s)", "End Time (s)", "Class"],
        [["0,5", 3, "Robin"]],
    )


def _audacity_labels(folders):
    for table in folders["prediction"].glob("*.txt"):
        table.write_text(
            "0.0\t3.0\tRobin\t0.9\n9.0\t12.0\tWren\t0.6\n", encoding="utf-8"
        )


def _no_annotation_tables(folders):
    for table in folders["annotation"].glob("*.txt"):
        table.unlink()


def _bad_mapping(folders):
    (folders["annotation"].parent / "mapping.json").write_text(
        '{"Robin": "Robin",}', encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("make", "options", "expected"),
    [
        (
            _no_annotation_tables,
            {},
            lambda: loc.localize("eval-tab-status-no-annotation-tables"),
        ),
        (
            _decimal_comma,
            {},
            lambda: (
                loc.localize("eval-tab-error-not-a-number").format(
                    file="rec1.txt", row=1, column="Begin Time (s)", value="0,5"
                )
                + " "
                + loc.localize("eval-tab-error-decimal-comma")
            ),
        ),
        (
            _audacity_labels,
            {},
            lambda: (
                loc.localize("eval-tab-error-missing-prediction-column").format(
                    column="Begin Time (s)",
                    field=loc.localize("eval-tab-column-start-time-label"),
                    available="0.0, 0.9, 3.0, Robin",
                )
                + " "
                + loc.localize("eval-tab-error-no-header")
            ),
        ),
        (
            _two_sites,
            {},
            lambda: loc.localize(
                "eval-tab-error-duplicate-prediction-recording"
            ).format(recording="rec1", places="site1, site2"),
        ),
        (
            _two_analyses,
            {"prediction_recording": "Begin Path"},
            lambda: loc.localize("eval-tab-error-split-prediction-recording").format(
                recording="rec1", places="run_24, run_30"
            ),
        ),
        (
            lambda folders: folders.update(annotation=folders["prediction"]),
            {"annotation_class": "Common Name"},
            lambda: loc.localize("eval-tab-error-same-tables").format(file="rec1.txt"),
        ),
        (
            _bad_mapping,
            {"mapping": "mapping.json"},
            lambda: loc.localize("eval-tab-error-bad-mapping"),
        ),
    ],
    ids=[
        "no annotation tables",
        "decimal comma",
        "no header row",
        "two sites",
        "two analyses",
        "one folder twice",
        "bad class mapping",
    ],
)
def test_a_problem_with_the_tables_is_explained_in_the_status(
    tmp_path, make, options, expected
):
    folders = _folders(tmp_path)
    make(folders)
    if "mapping" in options:
        options = {**options, "mapping": str(tmp_path / options["mapping"])}

    refreshed = _refresh(None, _values(folders, **options))

    assert refreshed.status["value"] == f"⚠️ {expected()}"
    assert refreshed.suggestions == []


@pytest.mark.parametrize("both", [False, True], ids=["annotations only", "both"])
def test_annotation_tables_with_scores_are_pointed_out_as_predictions(folders, both):
    for recording in RECORDINGS:
        table = folders["annotation"] / f"{recording}.txt"
        frame = pd.read_csv(table, sep="\t").assign(Confidence=0.9)
        frame.to_csv(table, sep="\t", index=False)
    values = list(_values(folders))
    if not both:
        values[1] = None

    status = _refresh(None, values).status["value"]

    warning = loc.localize("eval-tab-status-annotations-have-scores").format(
        column="Confidence"
    )
    assert warning in status.split("\n")
    assert (
        loc.localize("eval-tab-status-annotations").format(tables=3, rows=6, classes=2)
        in status
    )


def test_calculating_with_no_recording_paired_is_explained(tmp_path):
    folders = _folders(tmp_path)
    for table in folders["annotation"].iterdir():
        table.rename(table.with_name(f"other_{table.name}"))

    with pytest.raises(gr.Error) as raised:
        _calculate(None, _values(folders))

    assert raised.value.message.startswith(
        loc.localize("eval-tab-error-nothing-paired")
    )


def test_a_pick_switches_to_the_prediction_column_with_the_annotated_names(tmp_path):
    folders = _folders(tmp_path, names=("Turdus merula", "Troglodytes troglodytes"))
    for table in folders["prediction"].iterdir():
        frame = pd.read_csv(table, sep="\t")
        frame["Scientific Name"] = frame["Common Name"].map(
            {"Robin": "Turdus merula", "Wren": "Troglodytes troglodytes"}
        )
        frame.to_csv(table, sep="\t", index=False)

    picked = _refresh(None, _values(folders), picked=True)
    typed = _refresh(None, _values(folders))

    status = picked.status["value"]
    assert picked.class_column == gr.update(value="Scientific Name")
    assert picked.classes["value"] == ["Troglodytes troglodytes", "Turdus merula"]
    assert (
        picked.proc_state.processor.get_column_name("Class", prediction=True)
        == "Scientific Name"
    )
    assert (
        loc.localize("eval-tab-status-column-switched").format(column="Scientific Name")
        in status
    )
    assert (
        loc.localize("eval-tab-status-classes-all").format(
            n=2, column="Scientific Name"
        )
        in status
    )
    assert typed.class_column == gr.update()


def _pair(annotated, predicted, hits, windows):
    return loc.localize("eval-tab-status-suggestion-pair").format(
        annotated=annotated, predicted=predicted, hits=hits, windows=windows
    )


def test_a_renamed_class_is_suggested_and_the_suggestion_scores_it(tmp_path):
    folders = _folders(tmp_path, names=("European Robin", "Wren"))

    refreshed = _refresh(None, _values(folders))
    mapping = evaluation.apply_name_suggestions(
        refreshed.suggestions, refreshed.suggestion_choices["value"], None
    )
    result = _calculate(None, _values(folders, mapping=mapping), threshold=0.5)

    status = refreshed.status["value"]
    assert refreshed.suggestions == [("European Robin", "Robin", 3, 3)]
    assert refreshed.suggestions_group == gr.update(visible=True)
    assert refreshed.suggestion_choices == gr.update(
        choices=[(_pair("European Robin", "Robin", 3, 3), "Robin")], value=["Robin"]
    )
    assert (
        loc.localize("eval-tab-status-classes-some").format(
            found=1, n=2, column="Common Name", names="European Robin"
        )
        in status
    )
    assert loc.localize("eval-tab-status-suggestions-below").format(n=1) in status
    with open(mapping, encoding="utf-8") as f:
        assert json.load(f) == {"Robin": "European Robin"}
    robin = result.table.set_index("").loc["European Robin"]
    assert robin["Recall"] == 1.0


def test_only_the_checked_name_matches_are_used(tmp_path):
    folders = _folders(tmp_path, names=("European Robin", "Winter Wren"))

    refreshed = _refresh(None, _values(folders))
    mapping = evaluation.apply_name_suggestions(refreshed.suggestions, ["Wren"], None)
    applied = _refresh(None, _values(folders, mapping=mapping))
    unchanged = evaluation.apply_name_suggestions(refreshed.suggestions, [], None)

    assert refreshed.suggestion_choices["value"] == ["Robin", "Wren"]
    with open(mapping, encoding="utf-8") as f:
        assert json.load(f) == {"Wren": "Winter Wren"}
    assert applied.suggestions == [("European Robin", "Robin", 3, 3)]
    assert unchanged == gr.update()


def test_the_name_matches_join_a_loaded_mapping_and_take_effect(tmp_path):
    folders = _folders(tmp_path, names=("European Robin", "Winter Wren"))
    loaded = tmp_path / "mapping.json"
    loaded.write_text(
        json.dumps({"Robin": "Robin (BirdNET)", "Dunnock": "Hedge Accentor"}),
        encoding="utf-8",
    )

    refreshed = _refresh(None, _values(folders, mapping=str(loaded)))
    mapping = evaluation.apply_name_suggestions(
        refreshed.suggestions, refreshed.suggestion_choices["value"], str(loaded)
    )
    applied = _refresh(None, _values(folders, mapping=mapping))

    assert refreshed.suggestions == [
        ("European Robin", "Robin (BirdNET)", 3, 3),
        ("Winter Wren", "Wren", 3, 3),
    ]
    with open(mapping, encoding="utf-8") as f:
        assert json.load(f) == {
            "Robin": "European Robin",
            "Dunnock": "Hedge Accentor",
            "Robin (BirdNET)": "European Robin",
            "Wren": "Winter Wren",
        }
    assert applied.suggestions == []
    assert (
        loc.localize("eval-tab-status-classes-all").format(n=2, column="Common Name")
        in applied.status["value"]
    )


def test_a_changed_parameter_renews_the_name_matches_with_the_summary(tmp_path):
    folders = _folders(tmp_path, names=("European Robin", "Wren"))
    proc_state = _refresh(None, _values(folders)).proc_state

    built, status, suggestions, group, choices = evaluation.check_parameters(
        proc_state, *_values(folders, sample_duration=1.5)
    )

    assert suggestions == [("European Robin", "Robin", 6, 6)]
    assert group == gr.update(visible=True)
    assert choices == gr.update(
        choices=[(_pair("European Robin", "Robin", 6, 6), "Robin")], value=["Robin"]
    )
    assert (
        loc.localize("eval-tab-status-suggestions-below").format(n=1)
        in (status["value"])
    )
    assert built.processor.sample_duration == 1.5


def test_a_class_detected_everywhere_is_not_suggested_for_a_renamed_one(tmp_path):
    folders = _folders(tmp_path, names=("European Robin", "Wren"))
    for table in folders["prediction"].iterdir():
        frame = pd.read_csv(table, sep="\t")
        for start in range(0, 30, 3):
            frame.loc[len(frame)] = [start, start + 3, "Noise", 0.95, 30]
        frame.to_csv(table, sep="\t", index=False)

    refreshed = _refresh(None, _values(folders))

    assert refreshed.suggestions == [("European Robin", "Robin", 3, 3)]


@pytest.mark.parametrize(
    ("rows", "threshold"),
    [
        ("Minimum confidence,0.25\n", 0.25),
        ("Minimum confidence,0.25\nTop N,3\n", 0.25),
        ("Minimum confidence,0\nTop N,3\n", None),
    ],
    ids=["minimum confidence", "top N as well", "top N alone"],
)
def test_the_analysis_settings_of_the_prediction_folder_are_applied_on_a_pick(
    folders, monkeypatch, rows, threshold
):
    _write_params(folders["prediction"], f"Model,perch\n{rows}")

    picked = _pick(monkeypatch, "prediction", folders["prediction"], 3, 0.1)
    higher = _pick(monkeypatch, "prediction", folders["prediction"], 3, 0.5)
    annotations = _pick(monkeypatch, "annotation", folders["annotation"], 3, 0.1)

    assert picked.sample_duration == gr.update(value=5.0)
    assert picked.threshold == (
        gr.update() if threshold is None else gr.update(value=threshold)
    )
    assert higher.threshold == gr.update()
    assert annotations.sample_duration == annotations.threshold == gr.update()
    status = _refresh(None, _values(folders)).status["value"]
    assert (
        loc.localize("eval-tab-status-window-differs").format(window=3, segment=5)
        in status
    )


def test_the_annotated_classes_are_selected_and_offered_first(folders):
    for recording in RECORDINGS:
        table = folders["prediction"] / f"{recording}.txt"
        frame = pd.read_csv(table, sep="\t")
        frame.loc[len(frame)] = [9, 12, "Blackbird", 0.8, 30]
        frame.to_csv(table, sep="\t", index=False)

    refreshed = _refresh(None, _values(folders))
    result = _calculate(refreshed.proc_state, _values(folders))

    assert refreshed.classes["choices"] == ["Robin", "Wren", "Blackbird"]
    assert refreshed.classes["value"] == ["Robin", "Wren"]
    assert result.table[""].tolist() == ["Robin", "Wren"]
    assert "Blackbird" in result.notes


def test_a_selected_class_without_annotations_is_not_available_and_not_averaged(
    folders,
):
    for recording in RECORDINGS:
        table = folders["prediction"] / f"{recording}.txt"
        frame = pd.read_csv(table, sep="\t")
        frame.loc[len(frame)] = [9, 12, "Blackbird", 0.8, 30]
        frame.to_csv(table, sep="\t", index=False)

    with_blackbird = _calculate(
        None, _values(folders), classes=["Robin", "Wren", "Blackbird"]
    )
    without = _calculate(None, _values(folders), classes=["Robin", "Wren"])

    blackbird = with_blackbird.table.set_index("").loc["Blackbird"]
    assert set(blackbird.drop(loc.localize("eval-tab-support-column-label"))) == {
        loc.localize("eval-tab-not-available")
    }
    assert (
        with_blackbird.overall.iloc[:, 1:].to_numpy().tolist()
        == without.overall.iloc[:, 1:].to_numpy().tolist()
    )


def test_a_combined_result_table_is_matched_to_its_recordings_by_default(
    tmp_path, monkeypatch
):
    """As BirdNET writes it: one running timeline, the in-file time in File Offset."""
    folders = {kind: tmp_path / kind for kind in ("annotation", "prediction")}
    for folder in folders.values():
        folder.mkdir()
    detections = []
    for i, recording in enumerate(RECORDINGS):
        _write_table(
            folders["annotation"] / f"{recording}.txt",
            ["Begin Time (s)", "End Time (s)", "Class"],
            [[0, 3, "Robin"], [3, 6, "Wren"]],
        )
        audio = tmp_path / f"{recording}.wav"
        offset = 30 * i
        detections += [
            [offset, offset + 3, "Robin", 0.9, 0, audio],
            [offset + 3, offset + 6, "Wren", 0.6, 3, audio],
        ]
    _write_table(
        folders["prediction"] / "BirdNET_SelectionTable.txt",
        [
            "Begin Time (s)",
            "End Time (s)",
            "Common Name",
            "Confidence",
            "File Offset (s)",
            "Begin Path",
        ],
        detections,
    )
    preselected = []
    for kind, folder in folders.items():
        preselected += _pick(monkeypatch, kind, folder).columns

    result = _calculate(
        None,
        (*map(str, folders.values()), None, 3, 0.5, "", False, *preselected),
    )

    assert "⚠️" not in result.notes
    assert result.overall["Recall"].tolist() == [1.0]
    assert result.overall["Precision"].tolist() == [1.0]


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
    stale = _refresh(None, _values(folders)).proc_state

    evaluation.download_data_table(stale, *_values(folders, "60"))

    with open(target, newline="") as f:
        assert len(list(csv.DictReader(f))) == len(RECORDINGS) * 20


@pytest.mark.parametrize(("name", "separator"), [("results.csv", ","), ("r.TSV", "\t")])
def test_the_results_table_export_is_the_table_on_screen(
    folders, monkeypatch, tmp_path, name, separator
):
    for recording in RECORDINGS:
        table = folders["prediction"] / f"{recording}.txt"
        frame = pd.read_csv(table, sep="\t")
        frame.loc[len(frame)] = [9, 12, "Blackbird", 0.8, 30]
        frame.to_csv(table, sep="\t", index=False)
    target = tmp_path / name
    monkeypatch.setattr(gu, "save_file_dialog", lambda **k: str(target))
    result = _calculate(None, _values(folders), classes=["Blackbird", "Robin", "Wren"])

    evaluation.download_results_table(
        result.pa, result.predictions, result.labels, "macro"
    )

    exported = pd.read_csv(target, sep=separator)
    assert exported["Class"].tolist() == [
        "Robin",
        "Wren",
        "Blackbird",
        *result.overall[""],
    ]
    assert exported["Class"].tolist() == [*result.table[""], *result.overall[""]]
    assert exported["Recall"].tolist()[:2] == result.table["Recall"].tolist()[:2]
    assert exported["Recall"].tolist()[3:] == result.overall["Recall"].tolist()
    with open(target, newline="", encoding="utf-8") as f:
        rows = {row["Class"]: row for row in csv.DictReader(f, delimiter=separator)}
    blackbird = rows["Blackbird"]
    assert {blackbird[c] for c in blackbird if c != "Class"} == {"", "0"}
    assert blackbird["Annotated windows"] == "0"


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

    assert loc.localize(key).format(n=1, names="rec9") in result.notes


def test_the_notes_name_recordings_whose_annotations_all_count_as_missed(folders):
    _write_table(
        folders["annotation"] / "rec8.txt",
        ["Begin Time (s)", "End Time (s)", "Class"],
        [[0, 3, "Robin"]],
    )

    result = _calculate(None, _values(folders))

    assert (
        loc.localize("eval-tab-note-annotation-only").format(n=1, names="rec8")
        in result.notes
    )


@pytest.mark.parametrize(
    ("threshold", "expected"), [(0.2, True), (0.3, False)], ids=["below", "above"]
)
def test_a_threshold_below_every_score_is_pointed_out(folders, threshold, expected):
    result = _calculate(None, _values(folders), threshold=threshold)

    note = loc.localize("eval-tab-note-threshold-below-scores").format(min_conf=0.3)
    assert (note in result.notes) is expected


def test_the_header_says_what_was_compared(folders):
    result = _calculate(None, _values(folders), recordings=["rec1", "rec3"])

    assert result.header == loc.localize("eval-tab-results-header").format(
        recordings=2, windows=20, window=3, classes=2, threshold=0.5
    )


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
    state = _refresh(None, _values(folders)).proc_state

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
        _event(demo, "download_results_table").fn(None, None, None, "macro")

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


def test_calculating_is_locked_while_the_lists_change(demo):
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
        unlock = next(e for e in events.values() if e.trigger_after == refresh._id)
        assert chain_start(refresh).fn()["interactive"] is False
        assert unlock.fn()["interactive"] is True
        assert chain_start(refresh).outputs == unlock.outputs


def test_calculate_is_unlocked_only_behind_the_builds_queued_before(demo):
    queue = _event(demo, "refresh_after_pick").concurrency_id
    unlocks = [
        event
        for event in demo.fns.values()
        if getattr(event.fn, "__name__", "") == "_unlock_calculation"
    ]

    assert len(unlocks) == len(evaluation._DEFAULT_COLUMNS) + 2
    assert {event.concurrency_id for event in unlocks} == {queue}


def test_a_new_mapping_file_or_column_refreshes_the_lists(demo):
    events = demo.fns
    refresh = _event(demo, "refresh_processor")
    while refresh.trigger_after is not None:
        refresh = events[refresh.trigger_after]
    triggers = set(refresh.targets)
    apply = _event(demo, "apply_name_suggestions")
    mapping_file = next(
        c
        for c in demo.blocks.values()
        if isinstance(c, gr.File)
        and c.label == loc.localize("eval-tab-upload-mapping-file-label")
    )
    score_unannotated = _persisted(
        gr.Checkbox, "eval-tab-score-unannotated-checkbox-label"
    )
    column_dropdowns = [
        c
        for c in demo.blocks.values()
        if isinstance(c, gr.Dropdown) and not c.multiselect
    ]

    assert apply.outputs == [mapping_file]
    assert (mapping_file._id, "change") in triggers
    assert (score_unannotated._id, "change") in triggers
    assert len(column_dropdowns) == sum(map(len, evaluation._DEFAULT_COLUMNS.values()))
    for dropdown in column_dropdowns:
        assert (dropdown._id, "input") in triggers
        assert (dropdown._id, "change") not in triggers


def test_the_name_matches_used_are_the_ones_listed_with_the_summary(demo):
    apply = _event(demo, "apply_name_suggestions")
    stored, checked = apply.inputs[:2]

    assert isinstance(checked, gr.CheckboxGroup)
    for name in ("refresh_processor", "refresh_after_pick", "check_parameters"):
        outputs = _event(demo, name).outputs
        assert stored in outputs
        assert checked in outputs


def test_recordings_are_chosen_in_a_multiselect_dropdown(demo):
    label = loc.localize("eval-tab-select-recordings-checkboxgroup-label")
    recordings = next(
        c for c in demo.blocks.values() if getattr(c, "label", None) == label
    )

    assert isinstance(recordings, gr.Dropdown)
    assert recordings.multiselect


def test_the_tab_offers_every_registered_metric():
    assert set(evaluation._METRIC_CONTROLS) == set(performance_assessor.METRICS)


def test_the_overall_row_counts_the_classes_it_combines():
    labels = np.array([[1, 0], [0, 0], [1, 0]])
    predictions = np.array([[0.9, 0.8], [0.1, 0.2], [0.7, 0.1]])
    pa = PerformanceAssessor(
        num_classes=2, threshold=0.5, classes=("a", "b"), metrics_list=("recall",)
    )

    _, overall_row, _ = evaluation._build_metrics_tables(
        pa, predictions, labels, "macro"
    )

    assert overall_row[""].tolist() == [
        loc.localize("eval-tab-overall-macro-label").format(n=1)
    ]


def test_more_classes_than_the_plots_can_show_are_noted_and_the_best_supported_drawn(
    tmp_path,
):
    folders = {kind: tmp_path / kind for kind in ("annotation", "prediction")}
    names = [f"Species {i:02d}" for i in range(evaluation._PLOTTED_CLASSES + 1)]
    for folder in folders.values():
        folder.mkdir()
    _write_table(
        folders["annotation"] / "rec1.txt",
        ["Begin Time (s)", "End Time (s)", "Class"],
        [[3 * i, 3 * i + 3, name] for i, name in enumerate(names)]
        + [[0, 3, names[-1]]],
    )
    _write_table(
        folders["prediction"] / "rec1.txt",
        ["Begin Time (s)", "End Time (s)", "Common Name", "Confidence"],
        [[3 * i, 3 * i + 3, name, 0.9] for i, name in enumerate(names)],
    )
    values = (*map(str, folders.values()), None, 3, 0.5, "", False, *[None] * 11)

    result = _calculate(None, values)
    panels, *_ = evaluation._plotted(result.pa, result.predictions, result.labels)

    note = loc.localize("eval-tab-note-plot-limit").format(
        limit=evaluation._PLOTTED_CLASSES
    )
    assert note in result.notes
    assert panels.classes == (*names[: evaluation._PLOTTED_CLASSES - 1], names[-1])
    rng = np.random.default_rng(0)
    predictions = rng.random((50, 2))
    labels = rng.integers(0, 2, (50, 2))
    pa = PerformanceAssessor(
        num_classes=2, threshold=0.4, classes=("a", "b"), metrics_list=("f1", "recall")
    )

    class_table, overall_row, _ = evaluation._build_metrics_tables(
        pa, predictions, labels, "micro"
    )

    support = loc.localize("eval-tab-support-column-label")
    per_class = pa.calculate_metrics(predictions, labels, per_class_metrics=True)
    overall = pa.calculate_metrics(predictions, labels, averaging="micro")
    assert list(class_table.columns) == ["", "F1", "Recall", support]
    assert class_table[""].tolist() == ["a", "b"]
    assert class_table["F1"].tolist() == per_class.loc["F1"].round(3).tolist()
    assert class_table[support].tolist() == labels.sum(axis=0).tolist()
    assert overall_row[""].tolist() == [
        loc.localize("eval-tab-overall-micro-label").format(n=2)
    ]
    assert overall_row["Recall"].tolist() == overall.loc["Recall"].round(3).tolist()
    assert overall_row[support].tolist() == [labels.sum()]


def _processor(classes, recordings, annotated=None):
    return SimpleNamespace(
        classes=tuple(classes),
        annotated_classes=tuple(classes if annotated is None else annotated),
        samples_df=pd.DataFrame({"filename": recordings}),
    )


@pytest.mark.parametrize(
    ("previous", "current_classes", "current_recordings", "classes", "recordings"),
    [
        (None, None, None, ["a", "b"], ["r1", "r2"]),
        (None, ["c", "x"], ["r2"], ["c"], ["r2"]),
        (None, ["x"], ["r9"], ["a", "b"], ["r1", "r2"]),
        (_processor("ab", ["r1"]), ["a", "b"], ["r1"], ["a", "b"], ["r1", "r2"]),
        (_processor("ab", ["r1"]), ["a"], ["r1"], ["a"], ["r1", "r2"]),
        (_processor("abc", ["r1"], "ab"), ["a", "b", "c"], None, ["a", "b", "c"], None),
        (_processor("ac", ["r1"], "a"), ["a", "c"], None, ["a", "b", "c"], None),
    ],
    ids=[
        "first",
        "kept",
        "invalid",
        "default-again",
        "partial",
        "everything",
        "everything-grows",
    ],
)
def test_a_refresh_keeps_a_narrowed_selection_and_renews_a_default_one(
    previous, current_classes, current_recordings, classes, recordings
):
    processor = _processor("abc", ["r1", "r1", "r2"], annotated="ab")

    (offered, selected), (_, selected_recordings) = evaluation._selection_choices(
        processor, previous, current_classes, current_recordings
    )

    assert offered == ["a", "b", "c"]
    assert selected == classes
    if recordings is not None:
        assert selected_recordings == recordings


def test_the_tab_draws_its_figures_without_a_gui_backend():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    pa = PerformanceAssessor(num_classes=2, classes=("a", "b"))
    plt.switch_backend("template")

    try:
        figures = evaluation._build_figures(
            pa, rng.random((50, 2)), rng.integers(0, 2, (50, 2)), False, "macro"
        )
        backend = matplotlib.get_backend()
    finally:
        plt.switch_backend("agg")

    assert backend == "agg"
    assert all(figure is not None for figure in figures)
