import json
import logging
import os
import sys
import tracemalloc
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import pytest

from birdnet_analyzer import evaluation, logs, utils
from birdnet_analyzer.evaluation import process_data
from birdnet_analyzer.evaluation.preprocessing.data_processor import DataProcessor

matplotlib.use("Agg")

ANN_HEADER = ["Start Time", "End Time", "Class"]
PRED_HEADER = ["Start Time", "End Time", "Class", "Confidence"]


def _write_table(path, header, rows):
    lines = ["\t".join(header)]
    lines += ["\t".join(str(v) for v in row) for row in rows]
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

    with pytest.raises(KeyError, match="Score"):
        DataProcessor(
            prediction_directory_path=str(pred),
            annotation_directory_path=str(ann),
            recording_duration=3,
            columns_predictions={
                **DataProcessor.DEFAULT_COLUMNS_PREDICTIONS,
                "Confidence": "Score",
            },
        )


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
    _write_table(pred / "rec2.txt", PRED_HEADER, [])

    processor = DataProcessor(
        prediction_directory_path=str(pred), annotation_directory_path=str(ann)
    )

    rec2 = processor.samples_df[processor.samples_df["filename"] == "rec2"]
    assert rec2["X_annotation"].tolist() == [0, 1]
    assert rec2["X_confidence"].tolist() == [0.0, 0.0]
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
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"], [6, 9, ""], ["", 3, "Y"]])
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
    mapping.write_text(json.dumps({"amerob": "Robin"}), encoding="utf-8")

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

    with pytest.raises(KeyError, match="Confidence"):
        cli(
            "--annotation_path",
            str(ann),
            "--prediction_path",
            str(pred),
            "--recording_duration",
            "3",
        )

    assert len(logged) == 1
    assert isinstance(logged[0], KeyError)


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
