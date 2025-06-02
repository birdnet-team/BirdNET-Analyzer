"""
Core script for assessing performance of prediction models against annotated data.

This script uses the `DataProcessor` and `PerformanceAssessor` classes to process prediction and
annotation data, compute metrics, and optionally generate plots. It supports flexible configurations
for columns, class mappings, and filtering based on selected classes or recordings.
"""

import argparse
import json
import os
from collections.abc import Sequence

from birdnet_analyzer.evaluation.assessment.performance_assessor import (
    PerformanceAssessor,
)
from birdnet_analyzer.evaluation.preprocessing.data_processor import DataProcessor


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
    metrics_list: tuple[str, ...] = ("accuracy", "precision", "recall"),
    threshold: float = 0.1,
    class_wise: bool = False,
):
    """
    Processes data, computes metrics, and prepares the performance assessment pipeline.

    Args:
        annotation_path (str): Path to the annotation file or folder.
        prediction_path (str): Path to the prediction file or folder.
        mapping_path (Optional[str]): Path to the class mapping JSON file, if applicable.
        sample_duration (float): Duration of each sample interval in seconds.
        min_overlap (float): Minimum overlap required between predictions and annotations.
        recording_duration (Optional[float]): Total duration of the recordings, if known.
        columns_annotations (Optional[Dict[str, str]]): Custom column mappings for annotations.
        columns_predictions (Optional[Dict[str, str]]): Custom column mappings for predictions.
        selected_classes (Optional[List[str]]): List of classes to include in the analysis.
        selected_recordings (Optional[List[str]]): List of recordings to include in the analysis.
        metrics_list (Tuple[str, ...]): Metrics to compute for performance assessment.
        threshold (float): Confidence threshold for predictions.
        class_wise (bool): Whether to calculate metrics on a per-class basis.

    Returns:
        Tuple: Metrics DataFrame, `PerformanceAssessor` object, predictions tensor, labels tensor.
    """
    # Load class mapping if provided
    if mapping_path:
        with open(mapping_path) as f:
            class_mapping = json.load(f)
    else:
        class_mapping = None

    # Determine directory and file paths for annotations and predictions
    annotation_dir, annotation_file = (
        (os.path.dirname(annotation_path), os.path.basename(annotation_path)) if os.path.isfile(annotation_path) else (annotation_path, None)
    )
    prediction_dir, prediction_file = (
        (os.path.dirname(prediction_path), os.path.basename(prediction_path)) if os.path.isfile(prediction_path) else (prediction_path, None)
    )

    # Initialize the DataProcessor to handle and prepare data
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
    )

    # Get the available classes and recordings
    available_classes = processor.classes
    available_recordings = processor.samples_df["filename"].unique().tolist()

    # Default to all classes or recordings if none are specified
    if selected_classes is None:
        selected_classes = available_classes
    if selected_recordings is None:
        selected_recordings = available_recordings

    # Retrieve predictions and labels tensors for the selected classes and recordings
    predictions, labels, classes = processor.get_filtered_tensors(selected_classes, selected_recordings)

    num_classes = len(classes)
    task = "binary" if num_classes == 1 else "multilabel"

    # Initialize the PerformanceAssessor for computing metrics
    pa = PerformanceAssessor(
        num_classes=num_classes,
        threshold=threshold,
        classes=classes,
        task=task,
        metrics_list=metrics_list,
    )

    # Compute performance metrics
    metrics_df = pa.calculate_metrics(predictions, labels, per_class_metrics=class_wise)

    return metrics_df, pa, predictions, labels


def main():
    """
    Entry point for the script. Parses command-line arguments and orchestrates the performance assessment pipeline.
    """
    import matplotlib.pyplot as plt

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Performance Assessor Core Script")
    parser.add_argument("--annotation_path", required=True, help="Path to annotation file or folder")
    parser.add_argument("--prediction_path", required=True, help="Path to prediction file or folder")
    parser.add_argument("--mapping_path", help="Path to class mapping JSON file (optional)")
    parser.add_argument("--sample_duration", type=float, default=3.0, help="Sample duration in seconds")
    parser.add_argument("--min_overlap", type=float, default=0.5, help="Minimum overlap in seconds")
    parser.add_argument("--recording_duration", type=float, help="Recording duration in seconds")
    parser.add_argument("--columns_annotations", type=json.loads, help="JSON string for columns_annotations")
    parser.add_argument("--columns_predictions", type=json.loads, help="JSON string for columns_predictions")
    parser.add_argument("--selected_classes", nargs="+", help="List of selected classes")
    parser.add_argument("--selected_recordings", nargs="+", help="List of selected recordings")
    parser.add_argument("--metrics", nargs="+", default=["accuracy", "precision", "recall"], help="List of metrics")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold value (0-1)")
    parser.add_argument("--class_wise", action="store_true", help="Calculate class-wise metrics")
    parser.add_argument("--plot_metrics", action="store_true", help="Plot metrics")
    parser.add_argument("--plot_confusion_matrix", action="store_true", help="Plot confusion matrix")
    parser.add_argument("--plot_metrics_all_thresholds", action="store_true", help="Plot metrics for all thresholds")
    parser.add_argument("--output_dir", help="Directory to save plots")

    # Parse arguments
    args = parser.parse_args()

    # Process data and compute metrics
    metrics_df, pa, predictions, labels = process_data(
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
    )

    # Display the computed metrics
    print(metrics_df)

    # Create output directory if needed
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate plots if specified
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
        pa.plot_metrics_all_thresholds(predictions, labels, per_class_metrics=args.class_wise)
        if args.output_dir:
            plt.savefig(os.path.join(args.output_dir, "metrics_all_thresholds.png"))
        else:
            plt.show()
