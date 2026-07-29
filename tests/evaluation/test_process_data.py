"""End-to-end tests for the evaluation pipeline (``process_data`` + ``DataProcessor``).

These exercise the behaviour that used to produce "weird results": prediction files
without a matching annotation, recording names containing dots, and empty classes
dragging down the aggregate score.
"""

import pytest

from birdnet_analyzer.evaluation import process_data
from birdnet_analyzer.evaluation.preprocessing.data_processor import DataProcessor

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


def test_unmatched_prediction_file_dropped_by_default(tmp_path):
    """A recording with predictions but no annotation is dropped, not scored."""
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    # rec2 has no annotation file at all.
    _write_table(
        pred / "rec2.txt", PRED_HEADER, [[0, 3, "X", 0.8], [3, 6, "X", 0.7]]
    )

    result = process_data(
        annotation_path=str(ann),
        prediction_path=str(pred),
        recording_duration=30,
        metrics_list=("precision", "recall"),
    )

    assert result.unmatched_recordings == ("rec2",)
    # Only rec1 (30 s / 3 s = 10 samples) is scored.
    assert result.predictions.shape[0] == 10
    # rec2's predictions are not counted as false positives.
    assert result.metrics_df.loc["Precision", "Overall"] == 1.0


def test_unmatched_prediction_file_scored_as_empty_when_opted_in(tmp_path):
    """Opting in keeps unmatched recordings and scores their predictions as FPs."""
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    _write_table(pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9]])
    _write_table(
        pred / "rec2.txt", PRED_HEADER, [[0, 3, "X", 0.8], [3, 6, "X", 0.7]]
    )

    result = process_data(
        annotation_path=str(ann),
        prediction_path=str(pred),
        recording_duration=30,
        metrics_list=("precision", "recall"),
        score_unannotated_as_empty=True,
    )

    assert result.unmatched_recordings == ("rec2",)
    # Both recordings are scored: 20 samples.
    assert result.predictions.shape[0] == 20
    # rec2's two predictions become false positives -> precision 1 / (1 + 2).
    assert result.metrics_df.loc["Precision", "Overall"] == 1 / 3


def test_dotted_recording_names_do_not_collide(tmp_path):
    """Date-style names with dots stay distinct instead of merging on the first dot."""
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
    """A class with predictions but no annotations is left out of the average."""
    ann, pred = _dirs(tmp_path)
    _write_table(ann / "rec1.txt", ANN_HEADER, [[0, 3, "X"]])
    # Y is predicted but never annotated -> an empty class.
    _write_table(
        pred / "rec1.txt", PRED_HEADER, [[0, 3, "X", 0.9], [0, 3, "Y", 0.9]]
    )

    result = process_data(
        annotation_path=str(ann),
        prediction_path=str(pred),
        recording_duration=30,
        metrics_list=("precision", "recall", "f1"),
    )

    assert result.empty_classes == ("Y",)
    # X is perfect; Y (empty) does not pull the overall precision down.
    assert result.metrics_df.loc["Precision", "Overall"] == 1.0


def test_support_counts_reported_per_class(tmp_path):
    """Per-class output carries a Support row with the positive-sample counts."""
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
    """macro / micro / weighted give different overall numbers on imbalanced data."""
    ann, pred = _dirs(tmp_path)
    # Class A: three annotated windows, model gets one of them (recall 1/3).
    # Class B: one annotated window, model gets it (recall 1/1).
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

    macro = overall_recall("macro")
    micro = overall_recall("micro")
    weighted = overall_recall("weighted")

    # macro = mean(1/3, 1) = 2/3; micro pools TP/FN = 2/4 = 0.5; weighted by support
    # = (3*(1/3) + 1*1) / 4 = 0.5.
    assert macro == pytest.approx(2 / 3, rel=1e-6)
    assert micro == pytest.approx(0.5, rel=1e-6)
    assert weighted == pytest.approx(0.5, rel=1e-6)
    assert macro != micro


def test_raven_annotation_matches_birdnet_prediction_table(tmp_path):
    """Real-world Raven/BirdNET naming and columns still line up.

    Mirrors a real dataset: the annotation is a Raven table whose name contains dots
    (``.Table.1.selections``) and whose class lives in a ``Call type`` column, matched
    via its ``Begin File`` column. The prediction is a BirdNET selection table whose
    name carries the ``.BirdNET.selection.table`` suffix and which only has a
    ``Begin Path`` column (no ``Begin File``), so it is matched via its file name.
    Both must collapse to the same recording key.
    """
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

    # The dotted annotation name and the BirdNET suffix resolve to the same recording.
    assert result.unmatched_recordings == ()
    assert set(result.classes) == {"gibbon.female", "noise"}
    # "noise" is annotated (so not empty) but never predicted.
    assert result.empty_classes == ()
    assert result.metrics_df.loc["Precision", "gibbon.female"] == 1.0
    assert result.metrics_df.loc["Recall", "noise"] == 0.0
