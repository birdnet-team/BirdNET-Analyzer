"""
Core script for assessing performance of prediction models against annotated data.

This script uses the `DataProcessor` and `PerformanceAssessor` classes to process
prediction and annotation data, compute metrics, and optionally generate plots. It
supports flexible configurations for columns, class mappings, and filtering based on
selected classes or recordings.
"""

import argparse
import json
import logging
import os
import typing
from collections.abc import Sequence
from typing import Literal, NamedTuple

from birdnet_analyzer.evaluation.assessment.performance_assessor import (
    PerformanceAssessor,
)
from birdnet_analyzer.evaluation.preprocessing.data_processor import DataProcessor

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationResult(NamedTuple):
    """The outcome of :func:`process_data`.

    Besides the metrics and the tensors the assessment ran on, it carries the context
    a caller needs to explain the numbers: which recordings had predictions but no
    annotations, and which classes were empty (and therefore excluded from any
    aggregate score). ``sample_data`` is the per-window matrix the metrics were derived
    from -- the same table the GUI offers as the "data table" download.
    """

    metrics_df: "pd.DataFrame"
    pa: PerformanceAssessor
    predictions: "np.ndarray"
    labels: "np.ndarray"
    classes: tuple[str, ...]
    unmatched_recordings: tuple[str, ...]
    empty_classes: tuple[str, ...]
    sample_data: "pd.DataFrame"


def process_data(
    annotation_path: str,
    prediction_path: str,
    mapping_path: str | None = None,
    sample_duration: float = 3.0,
    min_overlap: float = 0.5,
    recording_duration: float | None = None,
    columns_annotations: dict[str, str] | None = None,
    columns_predictions: dict[str, str] | None = None,
    selected_classes: Sequence[str] | None = None,
    selected_recordings: list[str] | None = None,
    metrics_list: tuple[str, ...] = ("auroc", "precision", "recall", "f1", "ap"),
    threshold: float = 0.1,
    class_wise: bool = False,
    averaging: Literal["macro", "micro", "weighted"] = "macro",
    score_unannotated_as_empty: bool = False,
) -> EvaluationResult:
    """
    Processes data, computes metrics, and prepares the performance assessment pipeline.

    Args:
        annotation_path (str): Path to the annotation file or folder.
        prediction_path (str): Path to the prediction file or folder.
        mapping_path (Optional[str]): Path to the class mapping JSON file,
            if applicable.
        sample_duration (float): Duration of each sample interval in seconds.
        min_overlap (float): Minimum overlap required between predictions and
            annotations.
        recording_duration (Optional[float]): Total duration of the recordings, if
            known.
        columns_annotations (Optional[Dict[str, str]]): Custom column mappings for
            annotations.
        columns_predictions (Optional[Dict[str, str]]): Custom column mappings for
            predictions.
        selected_classes (Optional[List[str]]): List of classes to include in the
            analysis.
        selected_recordings (Optional[List[str]]): List of recordings to include in the
            analysis.
        metrics_list (Tuple[str, ...]): Metrics to compute for performance assessment.
        threshold (float): Confidence threshold for predictions.
        class_wise (bool): Whether to calculate metrics on a per-class basis.
        averaging (Literal["macro", "micro", "weighted"]): How to aggregate the
            per-class metrics into the overall score. Ignored when ``class_wise`` is
            True.
        score_unannotated_as_empty (bool): Whether recordings that have predictions but
            no annotation file are kept and scored as all-negative (True) or dropped as
            unscoreable (False, the default).

    Returns:
        EvaluationResult: The metrics DataFrame, the ``PerformanceAssessor``, the
            prediction and label tensors, and the context (classes, unmatched
            recordings, empty classes).
    """
    if mapping_path:
        with open(mapping_path) as f:
            class_mapping = json.load(f)
    else:
        class_mapping = None

    annotation_dir, annotation_file = (
        (os.path.dirname(annotation_path), os.path.basename(annotation_path))
        if os.path.isfile(annotation_path)
        else (annotation_path, None)
    )
    prediction_dir, prediction_file = (
        (os.path.dirname(prediction_path), os.path.basename(prediction_path))
        if os.path.isfile(prediction_path)
        else (prediction_path, None)
    )

    processor = DataProcessor(
        prediction_directory_path=prediction_dir,
        prediction_file_name=prediction_file,
        annotation_directory_path=annotation_dir,
        annotation_file_name=annotation_file,
        class_mapping=class_mapping,
        sample_duration=sample_duration,
        min_overlap=min_overlap,
        columns_predictions=columns_predictions,
        columns_annotations=columns_annotations,
        recording_duration=recording_duration,
        score_unannotated_as_empty=score_unannotated_as_empty,
    )

    available_classes = processor.classes
    available_recordings = processor.samples_df["filename"].unique().tolist()

    if selected_classes is None:
        selected_classes = available_classes
    if selected_recordings is None:
        selected_recordings = available_recordings

    predictions, labels, classes = processor.get_filtered_tensors(
        selected_classes, selected_recordings
    )

    num_classes = len(classes)
    task = "binary" if num_classes == 1 else "multilabel"

    pa = PerformanceAssessor(
        num_classes=num_classes,
        threshold=threshold,
        classes=classes,
        task=task,
        metrics_list=metrics_list,
    )

    metrics_df = pa.calculate_metrics(
        predictions,
        labels,
        per_class_metrics=class_wise,
        averaging=averaging,
        include_support=class_wise,
    )

    return EvaluationResult(
        metrics_df=metrics_df,
        pa=pa,
        predictions=predictions,
        labels=labels,
        classes=classes,
        unmatched_recordings=tuple(sorted(processor.unmatched_prediction_files)),
        empty_classes=pa.empty_classes(labels),
        sample_data=processor.get_sample_data(),
    )


# The metrics the PerformanceAssessor understands; also the CLI's allowed choices.
VALID_METRICS = ("accuracy", "recall", "precision", "f1", "ap", "auroc")


def _threshold_arg(value: str) -> float:
    """argparse type for ``--threshold``: a float strictly between 0 and 1.

    Fails at the parser with a clear message instead of deep inside the
    PerformanceAssessor, which requires ``0 < threshold < 1``.
    """
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"threshold must be a number between 0 and 1 (exclusive), got {value!r}"
        ) from None
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError(
            f"threshold must be between 0 and 1 (exclusive), got {value}"
        )
    return number


def main():
    """
    Entry point for the script. Parses command-line arguments and orchestrates the
    performance assessment pipeline.
    """
    import matplotlib.pyplot as plt

    from birdnet_analyzer.cli import verbosity_args
    from birdnet_analyzer.logs import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Performance Assessor Core Script", parents=[verbosity_args()]
    )
    parser.add_argument(
        "--annotation_path", required=True, help="Path to annotation file or folder"
    )
    parser.add_argument(
        "--prediction_path", required=True, help="Path to prediction file or folder"
    )
    parser.add_argument(
        "--mapping_path", help="Path to class mapping JSON file (optional)"
    )
    parser.add_argument(
        "--sample_duration", type=float, default=3.0, help="Sample duration in seconds"
    )
    parser.add_argument(
        "--min_overlap", type=float, default=0.5, help="Minimum overlap in seconds"
    )
    parser.add_argument(
        "--recording_duration", type=float, help="Recording duration in seconds"
    )
    parser.add_argument(
        "--columns_annotations",
        type=json.loads,
        help="JSON string for columns_annotations",
    )
    parser.add_argument(
        "--columns_predictions",
        type=json.loads,
        help="JSON string for columns_predictions",
    )
    parser.add_argument(
        "--selected_classes", nargs="+", help="List of selected classes"
    )
    parser.add_argument(
        "--selected_recordings", nargs="+", help="List of selected recordings"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=VALID_METRICS,
        default=["auroc", "precision", "recall", "f1", "ap"],
        help="List of metrics (accuracy is excluded by default; in this sample-based "
        "multilabel setting it is dominated by true negatives and misleadingly high)",
    )
    parser.add_argument(
        "--threshold",
        type=_threshold_arg,
        default=0.1,
        help="Threshold value, strictly between 0 and 1",
    )
    parser.add_argument(
        "--class_wise", action="store_true", help="Calculate class-wise metrics"
    )
    parser.add_argument(
        "--averaging",
        choices=["macro", "micro", "weighted"],
        default="macro",
        help="How to aggregate per-class metrics into the overall score",
    )
    parser.add_argument(
        "--score_unannotated_as_empty",
        action="store_true",
        help=(
            "Keep recordings that have predictions but no annotation file and score "
            "them as all-negative, instead of dropping them (default)"
        ),
    )
    parser.add_argument("--plot_metrics", action="store_true", help="Plot metrics")
    parser.add_argument(
        "--plot_confusion_matrix", action="store_true", help="Plot confusion matrix"
    )
    parser.add_argument(
        "--plot_metrics_all_thresholds",
        action="store_true",
        help="Plot metrics for all thresholds",
    )
    parser.add_argument("--output_dir", help="Directory to save plots")

    args = parser.parse_args()

    result = process_data(
        annotation_path=args.annotation_path,
        prediction_path=args.prediction_path,
        mapping_path=args.mapping_path,
        sample_duration=args.sample_duration,
        min_overlap=args.min_overlap,
        recording_duration=args.recording_duration,
        columns_annotations=args.columns_annotations,
        columns_predictions=args.columns_predictions,
        selected_classes=args.selected_classes,
        selected_recordings=args.selected_recordings,
        metrics_list=args.metrics,
        threshold=args.threshold,
        class_wise=args.class_wise,
        averaging=args.averaging,
        score_unannotated_as_empty=args.score_unannotated_as_empty,
    )
    pa = result.pa
    predictions = result.predictions
    labels = result.labels

    logger.info(result.metrics_df)

    if result.empty_classes:
        logger.warning(
            "Classes with no positive annotations were excluded from the overall "
            "score: %s",
            ", ".join(result.empty_classes),
        )

    # Mirror the GUI's "results table" and "data table" downloads.
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        results_table_path = os.path.join(args.output_dir, "results_table.csv")
        result.metrics_df.to_csv(results_table_path, index=True)
        logger.info("Saved results table to %s", results_table_path)

        data_table_path = os.path.join(args.output_dir, "data_table.csv")
        result.sample_data.to_csv(data_table_path, index=False)
        logger.info("Saved data table to %s", data_table_path)

    if args.plot_metrics:
        pa.plot_metrics(predictions, labels, per_class_metrics=args.class_wise)
        if args.output_dir:
            plt.savefig(os.path.join(args.output_dir, "metrics_plot.png"))
        else:
            plt.show()

    if args.plot_confusion_matrix:
        pa.plot_confusion_matrix(predictions, labels)
        if args.output_dir:
            plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
        else:
            plt.show()

    if args.plot_metrics_all_thresholds:
        pa.plot_metrics_all_thresholds(
            predictions, labels, per_class_metrics=args.class_wise
        )
        if args.output_dir:
            plt.savefig(os.path.join(args.output_dir, "metrics_all_thresholds.png"))
        else:
            plt.show()
