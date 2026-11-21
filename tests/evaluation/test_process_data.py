import codecs
import json
import logging
import os
import sys
import tracemalloc
import warnings
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import pytest

from birdnet_analyzer import evaluation, logs, utils
from birdnet_analyzer.evaluation import process_data
from birdnet_analyzer.evaluation.errors import EvaluationInputError
from birdnet_analyzer.evaluation.preprocessing.data_processor import DataProcessor

matplotlib.use("Agg")

ANN_HEADER = ["Start Time", "End Time", "Class"]
PRED_HEADER = ["Start Time", "End Time", "Class", "Confidence"]
RAVEN_COLUMNS = {
    "Start Time": "Begin Time (s)",
    "End Time": "End Time (s)",
    "Class": "Class",
    "Recording": "Begin File",
}
BIRDNET_COLUMNS = {
    "Start Time": "Begin Time (s)",
    "End Time": "End Time (s)",
    "Class": "Common Name",
    "Confidence": "Confidence",
    "Recording": "Begin Path",
}


def _write_table(path, header, rows, sep="\t"):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [sep.join(header)]
    lines += [sep.join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dirs(tmp_path):
    ann = tmp_path / "annotations"
    pred = tmp_path / "predictions"
    ann.mkdir()
    pred.mkdir()

    return ann, pred


@pytest.fixture
def cli(monkeypatch):
    def run(*args):
        monkeypatch.setattr(sys, "argv", ["birdnet-evaluate", *args])

        with mock.patch.dict(os.environ):
            try:
                evaluation.main()
            finally:
                logs._remove_installed_handlers()

    return run


def test_unmatched_prediction_file_dropped_by_default(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=DataProcessor.__module__)
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    _write_table(pred / "rec2.txt", PRED_HEADER, [[0, 3, "X", 0.8], [3, 6, "X", 0.7]])

    result = process_data(
        annotation_path=str(ann),
        prediction_path=str(pred),
        recording_duration=30,
        metrics_list=("precision", "recall"),
    )

    assert result.unmatched_recordings == ("rec2",)
    assert result.predictions.shape[0] == 10
    assert result.metrics_df.loc["Precision", "Overall"] == 1.0
    assert "rec2" in caplog.text
    assert "dropped" in caplog.text


def test_unmatched_prediction_file_scored_as_empty_when_opted_in(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=DataProcessor.__module__)
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    _write_table(pred / "rec2.txt", PRED_HEADER, [[0, 3, "X", 0.8], [3, 6, "X", 0.7]])

    result = process_data(
        annotation_path=str(ann),
        prediction_path=str(pred),
        recording_duration=30,
        metrics_list=("precision", "recall"),
        score_unannotated_as_empty=True,
    )

    assert result.unmatched_recordings == ("rec2",)
    assert result.predictions.shape[0] == 20
    assert result.metrics_df.loc["Precision", "Overall"] == 1 / 3
    assert "rec2" in caplog.text
    assert "false positive" in caplog.text


def test_dotted_recording_names_do_not_collide(tmp_path):
    ann, pred = _dirs(tmp_path)
    for name in ("2023.05.01", "2023.06.01"):
        _write_table(ann / f"{name}.txt", ANN_HEADER, [[0, 3, "X"]])
        _write_table(pred / f"{name}.txt", PRED_HEADER, [[0, 3, "X", 0.9]])

    processor = DataProcessor(
        prediction_directory_path=str(pred),
        annotation_directory_path=str(ann),
        recording_duration=30,
    )

    assert set(processor.samples_df["filename"].unique()) == {
        "2023.05.01",
        "2023.06.01",
    }
    assert processor.unmatched_prediction_files == set()


def test_empty_class_excluded_from_overall(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9], [0, 3, "Y", 0.9]])

    result = process_data(
        annotation_path=str(ann),
        prediction_path=str(pred),
        recording_duration=30,
        metrics_list=("precision", "recall", "f1"),
    )

    assert result.empty_classes == ("Y",)
    assert result.metrics_df.loc["Precision", "Overall"] == 1.0


def test_support_counts_reported_per_class(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"], [3, 6, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])

    result = process_data(
        annotation_path=str(ann),
        prediction_path=str(pred),
        recording_duration=30,
        metrics_list=("precision",),
        class_wise=True,
    )

    assert "Support" in result.metrics_df.index
    assert result.metrics_df.loc["Support", "X"] == 2


def test_averaging_methods_differ_end_to_end(tmp_path):
    ann, pred = _dirs(tmp_path)
    # A: three annotated windows, one of them found. B: one annotated window, found.
    _write_table(
        ann / "rec1.txt",
        ANN_HEADER,
        [[0, 3, "A"], [3, 6, "A"], [6, 9, "A"], [0, 3, "B"]],
    )
    _write_table(
        pred / "rec1.txt",
        PRED_HEADER,
        [[0, 3, "A", 0.9], [0, 3, "B", 0.9]],
    )

    def overall_recall(averaging):
        return process_data(
            annotation_path=str(ann),
            prediction_path=str(pred),
            recording_duration=30,
            metrics_list=("recall",),
            averaging=averaging,
        ).metrics_df.loc["Recall", "Overall"]

    assert overall_recall("macro") == pytest.approx(2 / 3)
    assert overall_recall("micro") == pytest.approx(0.5)
    assert overall_recall("weighted") == pytest.approx(0.5)


def test_raven_annotation_matches_birdnet_prediction_table(tmp_path):
    # As a real dataset has it: the annotation is a Raven table with dots in its name,
    # the class in a "Call type" column and the audio in "Begin File"; the prediction
    # is a BirdNET selection table with only a "Begin Path" column.
    ann, pred = _dirs(tmp_path)
    audio = "S19_20180214_080003_resampled.wav"
    _write_table(
        ann / "S19_20180214_080003.Table.1.selections.txt",
        ["Begin Time (s)", "End Time (s)", "Begin File", "Call type"],
        [[0, 3, audio, "gibbon.female"], [6, 9, audio, "noise"]],
    )
    _write_table(
        pred / "S19_20180214_080003_resampled.BirdNET.selection.table.txt",
        ["Begin Time (s)", "End Time (s)", "Common Name", "Confidence", "Begin Path"],
        [[0, 3, "gibbon.female", 0.9, f"/data/{audio}"]],
    )

    result = process_data(
        annotation_path=str(ann),
        prediction_path=str(pred),
        recording_duration=9,
        columns_annotations={
            "Start Time": "Begin Time (s)",
            "End Time": "End Time (s)",
            "Class": "Call type",
            "Recording": "Begin File",
        },
        columns_predictions={
            "Start Time": "Begin Time (s)",
            "End Time": "End Time (s)",
            "Class": "Common Name",
            "Confidence": "Confidence",
        },
        metrics_list=("precision", "recall"),
        class_wise=True,
    )

    assert result.unmatched_recordings == ()
    assert set(result.classes) == {"gibbon.female", "noise"}
    assert result.empty_classes == ()
    assert result.metrics_df.loc["Precision", "gibbon.female"] == 1.0
    assert result.metrics_df.loc["Recall", "noise"] == 0.0


def test_samples_take_the_max_confidence_of_overlapping_predictions(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "A"], [4.0, 4.3, "B"]])
    _write_table(
        pred / "rec1.txt",
        PRED_HEADER,
        [[0, 9, "Z", 0.99], [1, 4, "A", 0.9], [0, 3, "A", 0.4], [6, 9, "B", 0.2]],
    )

    processor = DataProcessor(
        prediction_directory_path=str(pred),
        annotation_directory_path=str(ann),
        sample_duration=3,
        min_overlap=0.5,
        recording_duration=9,
    )
    processor.classes = ("A", "B")
    processor.process_data()

    samples = processor.samples_df
    assert list(samples.columns) == [
        "filename",
        "sample_index",
        "start_time",
        "end_time",
        "A_confidence",
        "A_annotation",
        "B_confidence",
        "B_annotation",
    ]
    assert samples["filename"].tolist() == ["rec1"] * 3
    assert samples["A_confidence"].tolist() == [0.9, 0.9, 0.0]
    assert samples["B_confidence"].tolist() == [0.0, 0.0, 0.2]
    assert samples["A_annotation"].tolist() == [1, 0, 0]
    assert samples["B_annotation"].tolist() == [0, 1, 0]
    assert "Z_confidence" not in samples.columns
    assert samples["A_confidence"].dtype == np.float64
    assert samples["A_annotation"].dtype == np.int64


def test_samples_are_ordered_by_recording_name(tmp_path):
    ann, pred = _dirs(tmp_path)
    names = [f"rec_{letter}" for letter in "hcfadgbe"]
    _write_table(
        ann / "all.txt", [*ANN_HEADER, "Recording"], [[0, 3, "X", n] for n in names]
    )
    _write_table(
        pred / "all.txt",
        [*PRED_HEADER, "Recording"],
        [[0, 3, "X", 0.9, n] for n in names],
    )

    processor = DataProcessor(
        prediction_directory_path=str(pred),
        annotation_directory_path=str(ann),
        recording_duration=3,
    )

    assert processor.samples_df["filename"].tolist() == sorted(names)


def test_predictions_without_the_confidence_column_are_rejected(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])

    with pytest.raises(EvaluationInputError, match="Score") as error:
        DataProcessor(
            prediction_directory_path=str(pred),
            annotation_directory_path=str(ann),
            recording_duration=3,
            columns_predictions={
                **DataProcessor.DEFAULT_COLUMNS_PREDICTIONS,
                "Confidence": "Score",
            },
        )

    assert error.value.code == "missing-column"
    assert error.value.details == {
        "side": "prediction",
        "field": "Confidence",
        "column": "Score",
        "available": ["Class", "Confidence", "End Time", "Start Time"],
    }


def test_a_confidence_column_without_any_value_is_rejected(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", ""], [3, 6, "X", ""]])

    with pytest.raises(ValueError, match="no values"):
        DataProcessor(str(pred), str(ann), recording_duration=6)


def test_annotation_only_recording_is_kept_without_recording_duration(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=DataProcessor.__module__)
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    _write_table(ann / "rec2.txt", ANN_HEADER, [[3, 6, "X"]])

    processor = DataProcessor(
        prediction_directory_path=str(pred), annotation_directory_path=str(ann)
    )

    rec2 = processor.samples_df[processor.samples_df["filename"] == "rec2"]
    assert rec2["X_annotation"].tolist() == [0, 1]
    assert rec2["X_confidence"].tolist() == [0.0, 0.0]
    assert processor.unmatched_annotation_files == {"rec2"}
    assert "rec2" in caplog.text
    assert "false negative" in caplog.text


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"sample_duration": 0}, "Sample duration must be positive"),
        ({"min_overlap": 0}, "Min overlap must be greater than 0"),
        ({"min_overlap": 4}, "Min overlap cannot exceed the sample duration"),
        ({"recording_duration": 0}, "Recording duration must be greater than 0"),
        (
            {"sample_duration": 15, "recording_duration": 10},
            "Sample duration cannot exceed the recording duration",
        ),
    ],
    ids=[
        "window",
        "margin",
        "margin-above-window",
        "duration",
        "window-above-duration",
    ],
)
def test_unusable_settings_are_rejected_before_any_file_is_read(
    tmp_path, settings, message
):
    with pytest.raises(ValueError, match=message):
        process_data(str(tmp_path / "missing"), str(tmp_path / "missing"), **settings)


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (
            {"columns_predictions": {"Start Time": "Begin", "End Time": "End"}},
            "prediction columns: Class",
        ),
        (
            {"columns_annotations": {"Start Time": None, "End Time": "End"}},
            "annotation columns: Start Time, Class",
        ),
    ],
    ids=["predictions", "annotations"],
)
def test_a_required_column_left_unmapped_is_rejected_by_name(
    tmp_path, columns, message
):
    with pytest.raises(ValueError, match=message):
        process_data(str(tmp_path / "missing"), str(tmp_path / "missing"), **columns)


@pytest.mark.parametrize(
    ("annotation_header", "annotation_rows", "prediction_rows", "end_times"),
    [
        ([*ANN_HEADER, "Duration"], [[0, 3, "X", 7]], [[3, 6, "X", 0.9]], [3, 6, 7]),
        (ANN_HEADER, [[0, 3, "X"]], [[7, 10, "X", 0.9]], [3, 6, 9, 10]),
    ],
    ids=["duration-column", "last-end-time"],
)
def test_the_windows_cover_the_recording_up_to_its_duration_or_last_event(
    tmp_path, annotation_header, annotation_rows, prediction_rows, end_times
):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", annotation_header, annotation_rows)
    _write_table(pred / "rec1.txt", PRED_HEADER, prediction_rows)

    processor = DataProcessor(str(pred), str(ann))

    assert processor.samples_df["start_time"].tolist() == [0, 3, 6, 9][: len(end_times)]
    assert processor.samples_df["end_time"].tolist() == end_times


@pytest.mark.parametrize(
    ("min_overlap", "event", "marked"),
    [
        (0.5, (4.6, 5.6), [0, 1, 0]),
        (0.5, (4.6, 5.1), [0, 0, 0]),
        (0.05, (4.6, 5.1), [1, 1, 0]),
        (0.5, (1.0, 1.2), [1, 0, 0]),
    ],
    ids=[
        "long enough on one side only",
        "too short for either window",
        "a smaller margin marks both",
        "inside a window although shorter than the margin",
    ],
)
def test_an_event_marks_the_windows_it_overlaps_by_at_least_the_margin(
    tmp_path, min_overlap, event, marked
):
    ann, pred = _dirs(tmp_path)
    start, end = event
    _write_table(ann / "rec1.txt", ANN_HEADER, [[start, end, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[start, end, "X", 0.9]])

    processor = DataProcessor(
        str(pred),
        str(ann),
        sample_duration=5,
        min_overlap=min_overlap,
        recording_duration=15,
    )

    assert processor.samples_df["X_annotation"].tolist() == marked
    assert processor.samples_df["X_confidence"].tolist() == [0.9 * m for m in marked]


def test_rows_without_a_class_a_time_or_a_score_are_ignored(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(
        ann / "rec1.txt",
        ANN_HEADER,
        [[0, 3, "X"], [6, 9, ""], ["", 3, "Y"], [" ", 3, "Y"]],
    )
    _write_table(
        pred / "rec1.txt",
        PRED_HEADER,
        [[0, 3, "X", 0.9], [0, "", "Y", 0.95], [3, 6, "X", ""], [6, 9, "", 0.9]],
    )

    processor = DataProcessor(str(pred), str(ann), recording_duration=9)
    samples = processor.samples_df

    assert processor.classes == ("X", "Y")
    assert samples["X_annotation"].tolist() == [1, 0, 0]
    assert samples["X_confidence"].tolist() == [0.9, 0.0, 0.0]
    assert (samples[["Y_annotation", "Y_confidence"]] == 0).all().all()


def test_a_class_mapping_renames_the_predicted_classes(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "Robin"]])
    _write_table(
        pred / "rec1.txt", PRED_HEADER, [[0, 3, "amerob", 0.9], [3, 6, "x", 0.2]]
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_bytes(codecs.BOM_UTF8 + json.dumps({"amerob": "Robin"}).encode())

    result = process_data(
        str(ann),
        str(pred),
        mapping_path=str(mapping),
        recording_duration=6,
        metrics_list=("recall",),
    )

    assert result.classes == ("Robin", "x")
    assert result.metrics_df.loc["Recall", "Overall"] == 1.0


def test_only_the_selected_classes_and_recordings_are_scored(tmp_path):
    ann, pred = _dirs(tmp_path)
    for name in ("rec1", "rec2"):
        _write_table(ann / f"{name}.txt", ANN_HEADER, [[0, 3, "X"], [3, 6, "Y"]])
        _write_table(
            pred / f"{name}.txt", PRED_HEADER, [[0, 3, "X", 0.9], [3, 6, "Y", 0.9]]
        )

    result = process_data(
        str(ann),
        str(pred),
        recording_duration=6,
        metrics_list=("recall",),
        selected_classes=["Y", "Z"],
        selected_recordings=["rec2"],
    )

    assert result.classes == ("Y",)
    assert result.predictions.shape == result.labels.shape == (2, 1)


def test_selecting_only_unknown_classes_is_an_error(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])

    with pytest.raises(ValueError, match="No valid classes"):
        process_data(str(ann), str(pred), recording_duration=3, selected_classes=["Z"])


def test_a_named_prediction_file_is_scored_against_the_named_annotation_file(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "site_a.txt", ANN_HEADER, [[0, 3, "X"], [3, 6, "X"]])
    _write_table(pred / "site_b.BirdNET.results.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    _write_table(pred / "other.txt", PRED_HEADER, [[0, 3, "X", 0.9]])

    result = process_data(
        str(ann / "site_a.txt"),
        str(pred / "site_b.BirdNET.results.txt"),
        recording_duration=6,
        metrics_list=("recall",),
    )

    assert result.sample_data["filename"].tolist() == ["site_a", "site_a"]
    assert result.metrics_df.loc["Recall", "Overall"] == 0.5


def test_a_day_of_predictions_is_binned_without_a_pairwise_matrix(tmp_path):
    ann, pred = _dirs(tmp_path)
    rng = np.random.default_rng(0)
    start = rng.integers(0, 86_400 // 3, 5_000) * 3.0
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "A"]])
    _write_table(
        pred / "rec1.txt",
        PRED_HEADER,
        [[s, s + 3, "A", c] for s, c in zip(start, rng.random(5_000), strict=True)],
    )

    tracemalloc.start()
    processor = DataProcessor(str(pred), str(ann), recording_duration=86_400)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 5000 events against 28 800 windows as a boolean matrix alone would be 144 MB.
    assert peak < 50_000_000
    assert (processor.samples_df["A_confidence"] > 0).sum() == len(np.unique(start))


def test_cli_writes_uncaught_errors_to_the_error_log(tmp_path, monkeypatch, cli):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    logged = []
    monkeypatch.setattr(utils, "write_error_log", logged.append)

    with pytest.raises(EvaluationInputError, match="Confidence"):
        cli(
            "--annotation_path",
            str(ann),
            "--prediction_path",
            str(pred),
            "--recording_duration",
            "3",
        )

    assert len(logged) == 1
    assert isinstance(logged[0], EvaluationInputError)


def test_the_cli_writes_the_tables_and_plots_into_the_output_folder(tmp_path, cli):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"], [3, 6, "Y"]])
    _write_table(
        pred / "rec1.txt",
        PRED_HEADER,
        [[0, 3, "X", 0.9], [3, 6, "Y", 0.9], [6, 9, "X", 0.9]],
    )
    out = tmp_path / "out"

    cli(
        "--annotation_path",
        str(ann),
        "--prediction_path",
        str(pred),
        "--recording_duration",
        "9",
        "--metrics",
        "precision",
        "recall",
        "--class_wise",
        "--output_dir",
        str(out),
        "--plot_metrics",
        "--plot_confusion_matrix",
        "--plot_metrics_all_thresholds",
    )

    results = pd.read_csv(out / "results_table.csv", index_col=0)
    assert results.loc["Precision", "X"] == 0.5
    assert results.loc["Recall", "Y"] == 1.0
    assert results.loc["Support", "X"] == 1
    assert len(pd.read_csv(out / "data_table.csv")) == 3
    assert sorted(p.name for p in out.glob("*.png")) == [
        "confusion_matrix.png",
        "metrics_all_thresholds.png",
        "metrics_plot.png",
    ]


def test_an_assessor_for_one_class_is_binary_and_for_more_multilabel(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"], [3, 6, "Y"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9], [3, 6, "Y", 0.4]])
    processor = DataProcessor(str(pred), str(ann), recording_duration=30)

    single = evaluation.build_assessor(processor, ["Y"], None, 0.2, ("f1",))
    every = evaluation.build_assessor(processor, None, None, 0.3, ("ap", "f1"))

    pa, predictions, labels, classes = single
    assert classes == pa.classes == ("Y",)
    assert (pa.task, pa.num_classes, pa.threshold, pa.metrics_list) == (
        "binary",
        1,
        0.2,
        ("f1",),
    )
    assert predictions.shape == labels.shape == (10, 1)
    pa, predictions, _, classes = every
    assert classes == pa.classes == ("X", "Y")
    assert (pa.task, pa.num_classes, pa.metrics_list) == ("multilabel", 2, ("ap", "f1"))
    assert predictions.shape == (10, 2)


def test_the_tensors_are_handed_to_scikit_learn_in_row_major_order(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"], [3, 6, "Y"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9], [3, 6, "Y", 0.4]])
    processor = DataProcessor(str(pred), str(ann), recording_duration=9)

    predictions, labels, _ = processor.get_filtered_tensors()

    assert predictions.flags.c_contiguous
    assert labels.flags.c_contiguous


def test_tables_are_found_in_subfolders_but_hidden_and_parameter_files_are_not(
    tmp_path,
):
    ann, pred = _dirs(tmp_path)
    for name in (
        "rec1.txt",
        "site/rec2.TXT",
        ".backup/rec3.txt",
        "._rec4.txt",
        "birdnet.analyze-params.csv",
        "BirdNET_analysis_params.CSV",
    ):
        _write_table(ann / name, ANN_HEADER, [[0, 3, "X"]])
    _write_table(ann / "rec5.csv", ANN_HEADER, [[0, 3, "X"]], sep=",")
    for name in ("rec1", "rec2", "rec5"):
        _write_table(pred / f"{name}.txt", PRED_HEADER, [[0, 3, "X", 0.9]])

    processor = DataProcessor(str(pred), str(ann), recording_duration=3)

    tables = processor.tables["annotation"]
    assert tables.files == ("rec1.txt", "rec5.csv", "site/rec2.TXT")
    assert sorted(set(tables.table["source_file"])) == list(tables.files)
    assert tables.skipped == ()
    assert processor.matched_recordings == {"rec1", "rec2", "rec5"}
    assert processor.unmatched_annotation_files == set()


@pytest.mark.parametrize(
    ("split_table", "stale_table", "sep"),
    [
        ("rec1.BirdNET.selection.table.txt", "BirdNET_SelectionTable.txt", "\t"),
        ("rec1.BirdNET.selection.table.txt", "rec1.BirdNET.results.txt", "\t"),
        ("rec1.BirdNET.selection.table.txt", "BirdNET_AudacityLabels.txt", "\t"),
        ("rec1.BirdNET.results.csv", "BirdNET_CombinedTable.csv", ","),
    ],
)
def test_a_combined_table_next_to_per_recording_tables_is_skipped(
    tmp_path, split_table, stale_table, sep
):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / split_table, PRED_HEADER, [[0, 3, "X", 0.9]], sep=sep)
    _write_table(
        pred / stale_table,
        [*PRED_HEADER, "Recording"],
        [[0, 3, "X", 0.9, "rec1"], [3, 6, "X", 0.9, "rec1"]],
        sep=sep,
    )

    result = process_data(
        str(ann), str(pred), recording_duration=6, metrics_list=("precision",)
    )

    assert result.skipped_files == (("prediction", stale_table, "superseded"),)
    assert result.metrics_df.loc["Precision", "Overall"] == 1.0


def test_files_without_the_mapped_columns_are_skipped(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=DataProcessor.__module__)
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(ann / "rec2.txt", [*ANN_HEADER, "Notes"], [[3, 6, "X", "faint"]])
    (ann / "README.txt").write_text("Annotated by J. Doe.\nTimes in seconds.\n")
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    _write_table(pred / "rec2.txt", PRED_HEADER, [[3, 6, "X", 0.8]])
    _write_table(pred / "rec3.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "species_list.txt", ["Parus major_Great Tit"], [["Jay"]])

    result = process_data(
        str(ann), str(pred), recording_duration=6, metrics_list=("precision", "recall")
    )

    assert result.skipped_files == (
        ("annotation", "README.txt", "missing-columns"),
        ("prediction", "rec3.txt", "missing-columns"),
        ("prediction", "species_list.txt", "missing-columns"),
    )
    assert result.metrics_df["Overall"].tolist() == [1.0, 1.0]
    assert "README.txt" in caplog.text


@pytest.mark.parametrize(
    ("encode", "label"),
    [
        pytest.param(lambda text: text.encode(), "Mönchsgrasmücke", id="utf-8"),
        pytest.param(
            lambda text: (
                codecs.BOM_UTF8
                + text.replace("\tEnd", "\t" + chr(0xFEFF) + "End")
                .replace("\n", "\r\n")
                .encode()
            ),
            "Mönchsgrasmücke",
            id="utf-8, a byte order mark in the header, crlf",
        ),
        pytest.param(
            lambda text: text.encode("utf-16"), "Mönchsgrasmücke", id="utf-16"
        ),
        pytest.param(
            lambda text: text.replace("\t", ";").encode("cp1252"),
            "Mönchsgrasmücke €",
            id="cp1252, semicolons",
        ),
        pytest.param(
            lambda text: text.replace("\t", ",").encode("latin-1"),
            "Grünfink" + chr(0x81),
            id="latin-1, commas",
        ),
        pytest.param(
            lambda text: (text.removesuffix("\n") + "\t\n").encode(),
            "Mönchsgrasmücke",
            id="a separator after the last value of a row",
            marks=pytest.mark.filterwarnings("ignore::pandas.errors.ParserWarning"),
        ),
    ],
)
def test_tables_are_read_in_the_common_encodings_and_separators(
    tmp_path, encode, label
):
    ann, pred = _dirs(tmp_path)
    (ann / "rec1.txt").write_bytes(
        encode(f" Start Time \tEnd Time\tClass\n0\t3\t{label}\n")
    )
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, label, 0.9]])

    processor = DataProcessor(str(pred), str(ann), recording_duration=6)

    assert processor.classes == (label,)
    assert processor.samples_df[f"{label}_annotation"].tolist() == [1, 0]


@pytest.mark.parametrize("zero_bytes", [False, True], ids=["header only", "0 bytes"])
def test_a_table_without_rows_still_counts_its_recording(tmp_path, zero_bytes):
    ann, pred = _dirs(tmp_path)

    def write_empty(path, header):
        if zero_bytes:
            path.write_bytes(b"")
        else:
            _write_table(path, header, [])

    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    write_empty(ann / "rec2.txt", ANN_HEADER)
    _write_table(pred / "rec2.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    _write_table(ann / "rec3.txt", ANN_HEADER, [[0, 3, "X"]])
    write_empty(pred / "rec3.txt", PRED_HEADER)
    write_empty(ann / "rec4.txt", ANN_HEADER)
    write_empty(pred / "rec4.txt", PRED_HEADER)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = process_data(
            str(ann),
            str(pred),
            recording_duration=6,
            metrics_list=("precision", "recall"),
        )

    assert result.metrics_df["Overall"].tolist() == [0.5, 0.5]
    assert result.unmatched_recordings == ()
    assert result.annotation_only_recordings == ()
    assert result.sample_data["filename"].tolist() == [
        name for name in ("rec1", "rec2", "rec3", "rec4") for _ in range(2)
    ]
    windowed = set(DataProcessor(str(pred), str(ann)).samples_df["filename"])
    assert windowed == {"rec1", "rec2", "rec3"}


@pytest.mark.parametrize(
    ("annotations", "predictions", "code", "details"),
    [
        pytest.param(
            {},
            {"rec1.txt": PRED_HEADER},
            "no-tables",
            {"side": "annotation", "folder": "annotations"},
            id="no annotation table",
        ),
        pytest.param(
            {"rec1.txt": ANN_HEADER},
            {".rec1.txt": PRED_HEADER, "birdnet.analyze-params.csv": PRED_HEADER},
            "no-tables",
            {"side": "prediction", "folder": "predictions"},
            id="only hidden and parameter files",
        ),
        pytest.param(
            {"README.txt": ["Annotated by J. Doe"]},
            {"rec1.txt": PRED_HEADER},
            "missing-column",
            {
                "side": "annotation",
                "field": "Start Time",
                "column": "Start Time",
                "available": ["Annotated by J. Doe"],
            },
            id="no annotation table with the columns",
        ),
        pytest.param(
            {"rec1.txt": ANN_HEADER},
            {"rec1.txt": ANN_HEADER, "rec2.txt": b""},
            "missing-column",
            {
                "side": "prediction",
                "field": "Confidence",
                "column": "Confidence",
                "available": ["Class", "End Time", "Start Time"],
            },
            id="annotation tables picked as predictions",
        ),
    ],
)
def test_folders_without_usable_tables_are_rejected_with_a_reason(
    tmp_path, annotations, predictions, code, details
):
    ann, pred = _dirs(tmp_path)
    for folder, files in ((ann, annotations), (pred, predictions)):
        for name, header in files.items():
            if isinstance(header, bytes):
                (folder / name).write_bytes(header)
            else:
                _write_table(folder / name, header, [[0, 3, "X", 0.9][: len(header)]])

    with pytest.raises(EvaluationInputError) as error:
        DataProcessor(str(pred), str(ann), recording_duration=3)

    assert error.value.code == code
    if "folder" in details:
        details = {**details, "folder": str(tmp_path / details["folder"])}
    assert error.value.details == details


@pytest.mark.parametrize(
    ("side", "name", "header", "rows", "details"),
    [
        pytest.param(
            "annotation",
            "rec1.txt",
            ANN_HEADER,
            [[0, 3, "X"], ["0:17.0", 20, "X"]],
            {"file": "rec1.txt", "column": "Start Time", "row": 2, "value": "0:17.0"},
            id="a clock time",
        ),
        pytest.param(
            "prediction",
            "site/rec2.txt",
            PRED_HEADER,
            [[0, 3, "X", 0.9], [3, 6, "X", "high"]],
            {
                "file": "site/rec2.txt",
                "column": "Confidence",
                "row": 2,
                "value": "high",
            },
            id="a word for a score",
        ),
        pytest.param(
            "annotation",
            "rec1.txt",
            [*ANN_HEADER, "Duration"],
            [[0, 3, "X", 90], [3, 6, "X", "90 s"]],
            {"file": "rec1.txt", "column": "Duration", "row": 2, "value": "90 s"},
            id="a unit after the duration",
        ),
    ],
)
def test_a_time_or_score_that_is_not_a_number_is_reported_with_its_place(
    tmp_path, side, name, header, rows, details
):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    _write_table((ann if side == "annotation" else pred) / name, header, rows)

    with pytest.raises(EvaluationInputError, match="not a number") as error:
        DataProcessor(str(pred), str(ann))

    assert error.value.code == "not-a-number"
    assert error.value.details == {"side": side, **details}


def test_a_decimal_comma_is_reported_as_not_a_number(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [["1,5", "3,0", "X"]], sep=";")
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])

    with pytest.raises(EvaluationInputError) as error:
        DataProcessor(str(pred), str(ann))

    assert error.value.code == "not-a-number"
    assert error.value.details["value"] == "1,5"


def test_a_file_that_cannot_be_read_as_a_table_is_skipped(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    (ann / "rec2.txt").write_text("Start Time\tEnd Time\tClass\n0\t3\tX\n3\t6\tX\tY\n")
    (ann / "rec3.txt").write_bytes(b"\x00\x01binary")
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])

    processor = DataProcessor(str(pred), str(ann), recording_duration=3)

    assert processor.tables["annotation"].skipped == (
        ("rec2.txt", "unreadable"),
        ("rec3.txt", "unreadable"),
    )
    assert processor.matched_recordings == {"rec1"}


def test_recordings_pair_whatever_the_case_of_their_names(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "REC1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(
        pred / "rec1.BirdNET.selection.table.txt", PRED_HEADER, [[0, 3, "X", 0.9]]
    )
    _write_table(ann / "Rec2.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec2.BirdNET.selection.table.txt", PRED_HEADER, [])

    processor = DataProcessor(str(pred), str(ann), recording_duration=3)

    assert processor.matched_recordings == {"REC1", "Rec2"}
    assert processor.empty_prediction_recordings == {"Rec2"}
    assert processor.samples_df["filename"].tolist() == ["REC1", "Rec2"]
    assert processor.samples_df["X_confidence"].tolist() == [0.9, 0.0]


def test_rows_without_a_recording_name_belong_to_their_table_file(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(
        ann / "all.txt",
        [*ANN_HEADER, "Recording"],
        [[0, 3, "X", "rec1"], [0, 3, "X", ""]],
    )
    _write_table(ann / "rec2.txt", ANN_HEADER, [[0, 3, "X"]])
    for name in ("rec1", "rec2", "all"):
        _write_table(pred / f"{name}.txt", PRED_HEADER, [[0, 3, "X", 0.9]])

    processor = DataProcessor(str(pred), str(ann), recording_duration=3)

    assert processor.matched_recordings == {"all", "rec1", "rec2"}
    assert processor.samples_df["X_annotation"].tolist() == [1, 1, 1]


def test_an_optional_column_mapped_to_none_is_not_read(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(
        ann / "rec1.txt",
        [*ANN_HEADER, "Duration", "Recording"],
        [[0, 3, "X", 1.5, "tape 7"], [6, 9, "X", 2.0, "tape 7"]],
    )
    _write_table(pred / "rec1.txt", PRED_HEADER, [[6, 9, "X", 0.9]])

    processor = DataProcessor(
        str(pred),
        str(ann),
        columns_annotations={
            **DataProcessor.DEFAULT_COLUMNS_ANNOTATIONS,
            "Recording": None,
            "Duration": None,
        },
    )

    assert processor.matched_recordings == {"rec1"}
    assert processor.samples_df["end_time"].tolist() == [3, 6, 9]


@pytest.mark.parametrize("start_column", ["Begin Time (s)", "File Offset (s)"])
def test_the_times_of_a_file_sequence_table_count_from_each_recording(
    tmp_path, start_column
):
    ann_header = ["Begin Time (s)", "End Time (s)", "Class", "Begin File"]
    pred_header = ["Begin Time (s)", "End Time (s)", "Common Name", "Confidence"]
    annotations = [(0, 3, "X", "rec1"), (2.7, 5, "X", "rec2"), (6.5, 8, "Y", "rec2")]
    predictions = [(0, 3, "X", 0.9, "rec1"), (3, 6, "X", 0.8, "rec2")]
    # In the sequence, the second recording starts 90 s after the first.
    offsets = {"rec1": 0, "rec2": 90}
    sequence, split = tmp_path / "sequence", tmp_path / "split"
    _write_table(
        sequence / "ann" / "rec1.Table.1.selections.txt",
        [*ann_header, "File Offset (s)"],
        [
            [begin + offsets[rec], end + offsets[rec], label, f"{rec}.wav", begin]
            for begin, end, label, rec in annotations
        ],
    )
    _write_table(
        sequence / "pred" / "BirdNET_SelectionTable.txt",
        [*pred_header, "File Offset (s)", "Begin Path"],
        [
            [begin + offsets[rec], end + offsets[rec], label, score, begin, rec]
            for begin, end, label, score, rec in predictions
        ],
    )
    for begin, end, label, rec in annotations:
        _write_table(
            split / "ann" / f"{rec}_{label}.txt",
            [*ann_header, "File Offset (s)"],
            [[begin, end, label, f"{rec}.wav", begin]],
        )
    for begin, end, label, score, rec in predictions:
        _write_table(
            split / "pred" / f"{rec}.BirdNET.selection.table.txt",
            [*pred_header, "File Offset (s)", "Begin Path"],
            [[begin, end, label, score, begin, rec]],
        )

    def samples(folder):
        return DataProcessor(
            str(folder / "pred"),
            str(folder / "ann"),
            sample_duration=3,
            min_overlap=0.3,
            recording_duration=9,
            columns_annotations={**RAVEN_COLUMNS, "Start Time": start_column},
            columns_predictions=BIRDNET_COLUMNS,
        ).samples_df

    pd.testing.assert_frame_equal(samples(sequence), samples(split))
    assert samples(split)["X_annotation"].tolist() == [1, 0, 0, 1, 1, 0]
    assert samples(split)["Y_annotation"].tolist() == [0, 0, 0, 0, 0, 1]


@pytest.mark.parametrize(
    ("side", "tables", "recording", "places"),
    [
        pytest.param(
            "prediction",
            {"site1/rec.txt": None, "site2/rec.txt": None},
            "rec",
            ["site1", "site2"],
            id="split tables in two site folders",
        ),
        pytest.param(
            "prediction",
            {
                "BirdNET_SelectionTable.txt": [
                    "/audio/site1/rec.wav",
                    "/audio/b/rec.wav",
                ]
            },
            "rec",
            ["/audio/b", "/audio/site1"],
            id="a combined table of two folders",
        ),
        pytest.param(
            "prediction",
            {"combined.txt": ["D:\\site1\\rec.wav", "D:\\Site2\\REC.WAV"]},
            "REC",
            ["D:\\site1", "D:\\Site2"],
            id="windows paths",
        ),
        pytest.param(
            "annotation",
            {"site1/rec.txt": None, "site2/REC.txt": None},
            "REC",
            ["site1", "site2"],
            id="annotations in two site folders",
        ),
    ],
)
def test_a_recording_name_in_two_folders_is_rejected(
    tmp_path, side, tables, recording, places
):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    folder = ann if side == "annotation" else pred
    header = ANN_HEADER if side == "annotation" else PRED_HEADER
    (folder / "rec.txt").unlink()
    for name, paths in tables.items():
        if paths is None:
            _write_table(folder / name, header, [[0, 3, "X", 0.9][: len(header)]])
        else:
            _write_table(
                folder / name,
                [*header, "Recording"],
                [[0, 3, "X", 0.9, path] for path in paths],
            )

    with pytest.raises(EvaluationInputError, match="two folders") as error:
        DataProcessor(str(pred), str(ann), recording_duration=3)

    assert error.value.code == "duplicate-recording"
    assert error.value.details == {
        "side": side,
        "recording": recording,
        "places": places,
    }


def test_the_tables_of_one_recording_in_one_folder_are_merged(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.Table.1.selections.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(ann / "rec1.Table.2.selections.txt", ANN_HEADER, [[3, 6, "Y"]])
    _write_table(
        pred / "rec1.txt",
        [*PRED_HEADER, "Begin Path"],
        [
            [0, 3, "X", 0.9, "C:\\Audio\\rec1.wav"],
            [3, 6, "Y", 0.8, "c:/audio/rec1.wav"],
        ],
    )

    processor = DataProcessor(
        str(pred),
        str(ann),
        recording_duration=6,
        columns_predictions={
            **DataProcessor.DEFAULT_COLUMNS_PREDICTIONS,
            "Recording": "Begin Path",
        },
    )

    assert processor.matched_recordings == {"rec1"}
    assert processor.samples_df["X_annotation"].tolist() == [1, 0]
    assert processor.samples_df["Y_annotation"].tolist() == [0, 1]
    assert processor.samples_df["Y_confidence"].tolist() == [0.0, 0.8]


def test_recordings_in_subfolders_with_unique_names_match(tmp_path):
    ann, pred = _dirs(tmp_path)
    for site, name in (("site1", "a"), ("site2", "b")):
        _write_table(ann / f"{name}.txt", ANN_HEADER, [[0, 3, "X"]])
        _write_table(
            pred / site / f"{name}.BirdNET.selection.table.txt",
            PRED_HEADER,
            [[0, 3, "X", 0.9]],
        )

    processor = DataProcessor(str(pred), str(ann), recording_duration=3)

    assert processor.matched_recordings == {"a", "b"}
    assert processor.samples_df["X_confidence"].tolist() == [0.9, 0.9]


def test_the_processor_reports_the_classes_and_recordings_of_each_side(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "matched.txt", ANN_HEADER, [[0, 3, "Wren"]])
    _write_table(
        pred / "matched.txt",
        PRED_HEADER,
        [[0, 3, "wren_code", 0.9], [3, 6, "Robin", 0.4]],
    )
    _write_table(ann / "annotated_only.txt", ANN_HEADER, [[0, 3, "Tit"]])
    _write_table(pred / "predicted_only.txt", PRED_HEADER, [[0, 3, "Jay", 0.9]])
    _write_table(ann / "no_calls.txt", ANN_HEADER, [])
    _write_table(pred / "no_calls.txt", PRED_HEADER, [[0, 3, "Robin", 0.5]])
    _write_table(ann / "no_detections.txt", ANN_HEADER, [[0, 3, "Wren"]])
    _write_table(pred / "no_detections.txt", PRED_HEADER, [])

    processor = DataProcessor(
        str(pred),
        str(ann),
        class_mapping={"wren_code": "Wren"},
        recording_duration=6,
    )

    assert processor.annotated_classes == ("Tit", "Wren")
    assert processor.predicted_classes == ("Robin", "Wren")
    assert processor.classes == ("Robin", "Tit", "Wren")
    assert processor.matched_recordings == {"matched", "no_calls", "no_detections"}
    assert processor.unmatched_prediction_files == {"predicted_only"}
    assert processor.unmatched_annotation_files == {"annotated_only"}
    assert processor.empty_annotation_recordings == {"no_calls"}
    assert processor.empty_prediction_recordings == {"no_detections"}
    assert processor.tables["annotation"].empty_files == ("no_calls.txt",)
    assert processor.tables["prediction"].files == (
        "matched.txt",
        "no_calls.txt",
        "no_detections.txt",
        "predicted_only.txt",
    )


def test_class_labels_match_whatever_their_whitespace_and_case(tmp_path):
    ann, pred = _dirs(tmp_path)
    _write_table(
        ann / "rec1.txt",
        ANN_HEADER,
        [[0, 3, " great  tit "], [3, 6, "Great Tit"], [6, 9, "Great Tit"]],
    )
    _write_table(
        pred / "rec1.txt",
        PRED_HEADER,
        [[0, 3, "GREAT TIT", 0.9], [3, 6, "grti", 0.8], [6, 9, "Blue tit", 0.7]],
    )

    processor = DataProcessor(
        str(pred),
        str(ann),
        class_mapping={"grti": "great tit "},
        recording_duration=9,
    )

    assert processor.classes == ("Blue tit", "Great Tit")
    assert processor.samples_df["Great Tit_annotation"].tolist() == [1, 1, 1]
    assert processor.samples_df["Great Tit_confidence"].tolist() == [0.9, 0.8, 0.0]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ('{"Eurasian Chaffinch": "Common Chaffinch",}', "not-json"),
        ('[["Eurasian Chaffinch", "Common Chaffinch"]]', "not-an-object"),
        ('{"Eurasian Chaffinch": 1}', "not-an-object"),
    ],
    ids=["trailing comma", "list", "number"],
)
def test_a_class_mapping_that_is_not_an_object_of_names_is_rejected(
    tmp_path, content, reason
):
    mapping = tmp_path / "mapping.json"
    mapping.write_text(content, encoding="utf-8")

    with pytest.raises(EvaluationInputError) as error:
        process_data(str(tmp_path), str(tmp_path), mapping_path=str(mapping))

    assert error.value.code == "bad-mapping"
    assert error.value.details["file"] == str(mapping)
    assert error.value.details["reason"] == reason
    assert ("message" in error.value.details) is (reason == "not-json")


def test_the_cli_detects_the_columns_of_birdnet_tables(tmp_path, cli, caplog):
    caplog.set_level(logging.WARNING)
    ann, pred = _dirs(tmp_path)
    selections = ["Begin Time (s)", "End Time (s)", "Class"]
    _write_table(ann / "rec1.txt", selections, [[0, 3, "Wren"], [3, 6, "Robin"]])
    _write_table(ann / "rec2.txt", selections, [[0, 3, "Wren"]])
    (ann / "README.txt").write_text("Annotated by J. Doe\n", encoding="utf-8")
    _write_table(
        pred / "rec1.BirdNET.selection.table.txt",
        [
            "Selection",
            "Begin Time (s)",
            "End Time (s)",
            "Common Name",
            "Confidence",
            "File Offset (s)",
            "Begin Path",
        ],
        [
            [1, 0, 3, "Wren", 0.9, 0, "/audio/rec1.wav"],
            [2, 6, 9, "Robin", 0.8, 6, "/audio/rec1.wav"],
        ],
    )
    out = tmp_path / "out"

    cli(
        "--annotation_path",
        str(ann),
        "--prediction_path",
        str(pred),
        "--recording_duration",
        "9",
        "--metrics",
        "precision",
        "recall",
        "--class_wise",
        "--output_dir",
        str(out),
    )

    results = pd.read_csv(out / "results_table.csv", index_col=0)
    assert results.loc["Precision", "Wren"] == 1.0
    assert results.loc["Recall", "Wren"] == 0.5
    assert results.loc["Precision", "Robin"] == 0.0
    assert "README.txt" in caplog.text
    assert "rec2" in caplog.text
