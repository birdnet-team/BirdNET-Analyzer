"""
DataProcessor class for handling and transforming sample data with annotations and
predictions.

This module defines the DataProcessor class, which processes prediction and annotation
data, aligns them with sampled time intervals, and generates tensors for further model
training or evaluation.
"""

import logging
import ntpath
import posixpath
from collections import Counter
from collections.abc import Callable, Sequence
from typing import ClassVar

import numpy as np
import pandas as pd

from birdnet_analyzer.evaluation.errors import EvaluationInputError
from birdnet_analyzer.evaluation.preprocessing.utils import (
    Tables,
    read_tables,
    recording_key,
)

logger = logging.getLogger(__name__)

_SKIP_REASONS = {
    "missing-columns": "without the mapped columns",
    "superseded": "that the per-recording BirdNET tables next to them supersede",
    "unreadable": "that cannot be read as tables",
}


def _by_recording(table: pd.DataFrame) -> dict:
    return dict(list(table.groupby("recording_filename", sort=False)))


def _is_sorted(values: np.ndarray) -> bool:
    return bool(np.all(values[1:] >= values[:-1]))


def _recording(value, source_file: str) -> tuple[str, str]:
    """The recording of a row and the folder it is in.

    A row without a value in the recording column belongs to its table file.
    """
    if not isinstance(value, str):
        value = "" if pd.isna(value) else str(value)
    value = value.strip()
    folder = posixpath.dirname(source_file)

    if not value:
        return recording_key(source_file), folder

    return recording_key(value), ntpath.dirname(value) or folder


def _check_unique(side: str, places: set[tuple[str, str]]) -> None:
    """Rejects a recording name that one side has in two folders."""
    folders: dict[str, tuple[str, dict[str, str]]] = {}

    for key, folder in sorted(places):
        same = folder.replace("\\", "/").rstrip("/").casefold()
        folders.setdefault(key.casefold(), (key, {}))[1].setdefault(same, folder)

    for key, found in folders.values():
        if len(found) > 1:
            first, second = (found[same] or "." for same in sorted(found)[:2])
            raise EvaluationInputError(
                "duplicate-recording",
                f"The {side} tables have the recording {key!r} in two folders, "
                f"{first!r} and {second!r}. Recordings are matched by name, so a "
                "name must not occur in more than one folder.",
                side=side,
                recording=key,
                places=[first, second],
            )


def _annotation_spelling(annotation_recordings: set[str]) -> Callable:
    """Gives a recording name the spelling of the annotations' name, if one fits."""
    spellings: dict[str, str] = {}
    for key in sorted(annotation_recordings):
        spellings.setdefault(key.casefold(), key)

    def spelled(key):
        if key in annotation_recordings:
            return key

        return spellings.get(key.casefold(), key)

    return spelled


def _normalized_label(label):
    if not isinstance(label, str):
        return label

    return " ".join(label.split()) or np.nan


def _label_spellings(labels: pd.Series) -> dict[str, str]:
    """The most frequent spelling of each label, by its casefold."""
    spellings: dict[str, str] = {}
    counts = Counter(label for label in labels if isinstance(label, str))
    for label, _ in counts.most_common():
        spellings.setdefault(label.casefold(), label)

    return spellings


def _classes(labels: pd.Series) -> tuple[str, ...]:
    return tuple(sorted(label for label in labels.unique() if pd.notna(label)))


class DataProcessor:
    """
    Processor for handling and transforming sample data with annotations and
    predictions.

    This class processes prediction and annotation data, aligning them with sampled time
    intervals, and generates tensors for further model training or evaluation.

    Attributes:
        tables: The tables read, by side (``"annotation"``, ``"prediction"``).
        classes: The classes of both sides, sorted.
        annotated_classes: The classes in the annotation rows, sorted.
        predicted_classes: The classes in the prediction rows that are scored, after
            the class mapping, sorted.
        matched_recordings: The recordings with both annotation and prediction tables;
            every recording when both file names are given.
        unmatched_prediction_files: The recordings with prediction tables only; none
            when both file names are given.
        unmatched_annotation_files: The recordings with annotation tables only; all
            their annotations count as missed. None when both file names are given.
        empty_annotation_recordings: The recordings whose annotation tables have no
            rows: annotated, without calls.
        empty_prediction_recordings: The recordings whose prediction tables have no
            rows: analysed, without detections.
    """

    DEFAULT_COLUMNS_PREDICTIONS: ClassVar[dict[str, str]] = {
        "Start Time": "Start Time",
        "End Time": "End Time",
        "Class": "Class",
        "Recording": "Recording",
        "Duration": "Duration",
        "Confidence": "Confidence",
    }

    DEFAULT_COLUMNS_ANNOTATIONS: ClassVar[dict[str, str]] = {
        "Start Time": "Start Time",
        "End Time": "End Time",
        "Class": "Class",
        "Recording": "Recording",
        "Duration": "Duration",
    }

    def __init__(
        self,
        prediction_directory_path: str,
        annotation_directory_path: str,
        prediction_file_name: str | None = None,
        annotation_file_name: str | None = None,
        class_mapping: dict[str, str] | None = None,
        sample_duration: float = 3.0,
        min_overlap: float = 0.5,
        columns_predictions: dict[str, str] | None = None,
        columns_annotations: dict[str, str] | None = None,
        recording_duration: float | None = None,
        score_unannotated_as_empty: bool = False,
    ) -> None:
        """
        Initializes the DataProcessor by loading prediction and annotation data.

        Args:
            prediction_directory_path (str): Path to the folder containing prediction
                files.
            annotation_directory_path (str): Path to the folder containing annotation
                files.
            prediction_file_name (Optional[str]): Name of the prediction file to
                process.
            annotation_file_name (Optional[str]): Name of the annotation file to
                process.
            class_mapping (Optional[Dict[str, str]]): Optional dictionary mapping raw
                class names to standardized class names.
            sample_duration (float, optional): Length of each data sample in seconds.
                Defaults to 3.0.
            min_overlap (float, optional): Minimum overlap required between prediction
                and annotation to consider a match.
            columns_predictions (Optional[Dict[str, str]], optional): Column name
                mappings for prediction files.
            columns_annotations (Optional[Dict[str, str]], optional): Column name
                mappings for annotation files.
            recording_duration (Optional[float], optional): User-specified recording
                duration in seconds. Defaults to None.
            score_unannotated_as_empty (bool, optional): How to treat recordings that
                have predictions but no matching annotation file. When ``False``
                (default) those recordings are dropped, because without a ground truth
                their predictions cannot be scored. When ``True`` they are kept and
                treated as if fully annotated with no events, so every prediction on
                them counts as a false positive (only correct when the annotations are
                exhaustive). Defaults to False.

        Raises:
            ValueError: If any parameter is invalid (e.g., negative sample duration).
        """
        self.sample_duration: float = sample_duration
        self.min_overlap: float = min_overlap
        self.class_mapping: dict[str, str] | None = class_mapping

        self.columns_predictions: dict[str, str] = (
            columns_predictions
            if columns_predictions is not None
            else self.DEFAULT_COLUMNS_PREDICTIONS.copy()
        )
        self.columns_annotations: dict[str, str] = (
            columns_annotations
            if columns_annotations is not None
            else self.DEFAULT_COLUMNS_ANNOTATIONS.copy()
        )

        self.recording_duration: float | None = recording_duration
        self.score_unannotated_as_empty: bool = score_unannotated_as_empty

        self.prediction_directory_path: str = prediction_directory_path
        self.prediction_file_name: str | None = prediction_file_name
        self.annotation_directory_path: str = annotation_directory_path
        self.annotation_file_name: str | None = annotation_file_name

        self.predictions_df: pd.DataFrame = pd.DataFrame()
        self.annotations_df: pd.DataFrame = pd.DataFrame()
        self.tables: dict[str, Tables] = {}

        self.matched_recordings: set[str] = set()
        self.unmatched_prediction_files: set[str] = set()
        self.unmatched_annotation_files: set[str] = set()
        self.empty_annotation_recordings: set[str] = set()
        self.empty_prediction_recordings: set[str] = set()
        self._recordings: set[str] = set()

        self.classes: tuple[str, ...] = ()
        self.annotated_classes: tuple[str, ...] = ()
        self.predicted_classes: tuple[str, ...] = ()

        self.samples_df: pd.DataFrame = pd.DataFrame()
        self.prediction_tensors: np.ndarray = np.array([])
        self.label_tensors: np.ndarray = np.array([])

        self._validate_columns()
        self._validate_parameters()

        self.load_data()
        self.process_data()
        self.create_tensors()

    def _validate_parameters(self) -> None:
        """
        Validates the input parameters for correctness.

        Raises:
            ValueError: If sample duration, minimum overlap, or recording duration is
            invalid.
        """
        if self.sample_duration <= 0:
            raise ValueError("Sample duration must be positive.")

        if self.recording_duration is not None:
            if self.recording_duration <= 0:
                raise ValueError("Recording duration must be greater than 0.")
            if self.sample_duration > self.recording_duration:
                raise ValueError(
                    "Sample duration cannot exceed the recording duration."
                )

        if self.min_overlap <= 0:
            raise ValueError("Min overlap must be greater than 0.")
        if self.min_overlap > self.sample_duration:
            raise ValueError("Min overlap cannot exceed the sample duration.")

    def _validate_columns(self) -> None:
        """
        Validates that essential columns are provided in the column mappings.

        Raises:
            ValueError: If required columns are missing or have None values.
        """
        required_columns = ["Start Time", "End Time", "Class"]

        missing_pred_columns = [
            col
            for col in required_columns
            if col not in self.columns_predictions
            or self.columns_predictions[col] is None
        ]

        missing_annot_columns = [
            col
            for col in required_columns
            if col not in self.columns_annotations
            or self.columns_annotations[col] is None
        ]

        if missing_pred_columns:
            raise ValueError(
                f"Missing or None prediction columns: {', '.join(missing_pred_columns)}"
            )
        if missing_annot_columns:
            raise ValueError(
                "Missing or None annotation columns: "
                f"{', '.join(missing_annot_columns)}"
            )

    def load_data(self) -> None:
        """
        Loads the prediction and annotation data into DataFrames.

        Depending on whether specific files are provided, this method either reads the
        tables in the given directories and their subdirectories or reads the
        specified files. It applies any specified class mapping, cleans the class
        labels, and keys every row by the recording it belongs to.

        Raises:
            EvaluationInputError: If a side has no usable table, a time or score is
                not a number, or a recording name comes from two folders.
            ValueError: If the predictions have rows but no confidence value.
        """
        directory_mode = (
            self.prediction_file_name is None or self.annotation_file_name is None
        )
        self.tables = {
            side: self._read_tables(side, directory, None if directory_mode else name)
            for side, directory, name in (
                (
                    "annotation",
                    self.annotation_directory_path,
                    self.annotation_file_name,
                ),
                (
                    "prediction",
                    self.prediction_directory_path,
                    self.prediction_file_name,
                ),
            )
        }
        self._log_skipped_files()

        annotations = self._prepared(self.tables["annotation"].table, prediction=False)
        predictions = self._prepared(self.tables["prediction"].table, prediction=True)

        confidence_col = self.get_column_name("Confidence", prediction=True)
        if not predictions.empty and predictions[confidence_col].isna().all():
            raise ValueError(f"Confidence column {confidence_col!r} has no values.")

        class_col_annot = self.get_column_name("Class", prediction=False)
        class_col_pred = self.get_column_name("Class", prediction=True)
        spellings = _label_spellings(annotations[class_col_annot])
        for table, column in (
            (annotations, class_col_annot),
            (predictions, class_col_pred),
        ):
            table[column] = [
                spellings.get(label.casefold(), label)
                if isinstance(label, str)
                else label
                for label in table[column]
            ]

        named_pair = not directory_mode and not any(
            self.get_column_name("Recording", prediction=prediction) in table.columns
            for table, prediction in ((predictions, True), (annotations, False))
        )
        # A named pair is one recording, whatever the two files are called.
        recording = recording_key(self.annotation_file_name) if named_pair else None
        annotations["recording_filename"], annotation_empty = self._recordings_of(
            "annotation", annotations, recording
        )
        predictions["recording_filename"], prediction_empty = self._recordings_of(
            "prediction", predictions, recording
        )

        annotation_recordings = set(annotations["recording_filename"])
        self.empty_annotation_recordings = annotation_empty - annotation_recordings
        annotation_recordings |= annotation_empty

        spelled = _annotation_spelling(annotation_recordings)
        predictions["recording_filename"] = predictions["recording_filename"].map(
            spelled
        )
        prediction_empty = {spelled(key) for key in prediction_empty}
        prediction_recordings = set(predictions["recording_filename"])
        self.empty_prediction_recordings = prediction_empty - prediction_recordings
        prediction_recordings |= prediction_empty

        # A recording that only has predictions has no ground truth, so its predictions
        # cannot be scored. By default those recordings are dropped; only when the
        # caller opts in (exhaustive annotations) are they kept and their predictions
        # counted as false positives. This only applies when matching whole
        # directories -- an explicitly named single file pair is always a deliberate
        # 1:1 comparison.
        if directory_mode:
            self.matched_recordings = prediction_recordings & annotation_recordings
            self.unmatched_prediction_files = (
                prediction_recordings - annotation_recordings
            )
            self.unmatched_annotation_files = (
                annotation_recordings - prediction_recordings
            )
        else:
            self.matched_recordings = prediction_recordings | annotation_recordings
            self.unmatched_prediction_files = set()
            self.unmatched_annotation_files = set()

        if self.unmatched_prediction_files:
            names = ", ".join(sorted(self.unmatched_prediction_files))
            if self.score_unannotated_as_empty:
                logger.warning(
                    "Prediction files without matching annotation file: %s. Their "
                    "annotations are assumed empty, so every prediction on them "
                    "counts as a false positive.",
                    names,
                )
            else:
                logger.warning(
                    "Prediction files without matching annotation file: %s. These "
                    "recordings are dropped and not scored.",
                    names,
                )
                predictions = predictions[
                    predictions["recording_filename"].isin(annotation_recordings)
                ]
                prediction_recordings &= annotation_recordings

        if self.unmatched_annotation_files:
            logger.warning(
                "Annotation files without matching prediction file: %s. Their "
                "predictions are assumed empty, so every annotation on them counts "
                "as a false negative.",
                ", ".join(sorted(self.unmatched_annotation_files)),
            )

        self.annotations_df = annotations
        self.predictions_df = predictions
        self._recordings = annotation_recordings | prediction_recordings

        self.annotated_classes = _classes(annotations[class_col_annot])
        self.predicted_classes = _classes(predictions[class_col_pred])
        self.classes = tuple(sorted({*self.annotated_classes, *self.predicted_classes}))

    def _read_tables(self, side: str, directory: str, file_name: str | None) -> Tables:
        prediction = side == "prediction"
        fields = ["Start Time", "End Time", "Class", "Recording", "Duration"]
        if prediction:
            fields.append("Confidence")

        return read_tables(
            directory,
            side,
            {field: self.get_column_name(field, prediction) for field in fields},
            file_name,
        )

    def _log_skipped_files(self) -> None:
        for side, tables in self.tables.items():
            for reason, explanation in _SKIP_REASONS.items():
                files = [path for path, why in tables.skipped if why == reason]
                if files:
                    logger.warning(
                        "Skipped %s files %s: %s", side, explanation, ", ".join(files)
                    )

    def _prepared(self, table: pd.DataFrame, prediction: bool) -> pd.DataFrame:
        """A side's rows with per-recording times and mapped, cleaned class labels."""
        table = table.copy()
        self._shift_sequence_times(table, prediction)

        column = self.get_column_name("Class", prediction=prediction)
        mapping = self.class_mapping or {}
        if prediction and mapping:
            table[column] = table[column].map(lambda label: mapping.get(label, label))
        table[column] = table[column].map(_normalized_label)

        return table

    def _shift_sequence_times(self, table: pd.DataFrame, prediction: bool) -> None:
        """Makes the times of a file-sequence table count from each recording's start.

        BirdNET's combined table and a Raven file sequence count ``Begin Time (s)``
        from the start of the first file; ``File Offset (s)`` is the time in the file.
        """
        if not {"Begin Time (s)", "File Offset (s)"} <= set(table.columns):
            return

        shift = (
            (
                pd.to_numeric(table["Begin Time (s)"], errors="coerce")
                - pd.to_numeric(table["File Offset (s)"], errors="coerce")
            )
            .fillna(0)
            .to_numpy()
        )
        shifted = shift != 0

        if not shifted.any():
            return

        for field in ("Start Time", "End Time"):
            column = self.get_column_name(field, prediction=prediction)
            if column != "File Offset (s)" and column in table.columns:
                times = table[column].to_numpy(dtype=float, copy=True)
                # Rounded so the times equal those in the per-recording tables, not
                # just up to the float error of the subtraction.
                times[shifted] = np.round(times[shifted] - shift[shifted], 9)
                table[column] = times

    def _recordings_of(
        self, side: str, table: pd.DataFrame, recording: str | None
    ) -> tuple[pd.Series, set[str]]:
        """The recording of each row, and of each of the side's tables without rows.

        Raises:
            EvaluationInputError: If a recording name comes from two folders.
        """
        empty_files = self.tables[side].empty_files

        if recording is not None:
            return (
                pd.Series(recording, index=table.index, dtype=object),
                {recording} if empty_files else set(),
            )

        column = self.get_column_name("Recording", prediction=side == "prediction")
        values = table[column] if column in table.columns else [None] * len(table)
        found: dict[tuple, tuple[str, str]] = {}
        keys = []
        for row in zip(values, table["source_file"], strict=True):
            place = found.get(row)
            if place is None:
                place = found[row] = _recording(*row)
            keys.append(place[0])

        empty = {_recording(None, name) for name in empty_files}
        _check_unique(side, {*found.values(), *empty})

        return pd.Series(keys, index=table.index, dtype=object), {
            key for key, _ in empty
        }

    def process_data(self) -> None:
        """
        Processes the loaded data, aligns predictions and annotations with sample
        intervals, and updates the samples DataFrame.

        Every recording of either side, including those whose tables have no rows, is
        processed once, in sorted order, and the per-recording results are concatenated
        into the `samples_df` attribute.
        """
        predictions = _by_recording(self.predictions_df)
        annotations = _by_recording(self.annotations_df)
        no_predictions = self.predictions_df.iloc[:0]
        no_annotations = self.annotations_df.iloc[:0]

        recordings = self._recordings | predictions.keys() | annotations.keys()
        frames = [
            self.process_recording(
                recording,
                predictions.get(recording, no_predictions),
                annotations.get(recording, no_annotations),
            )
            for recording in sorted(recordings, key=str)
        ]
        frames = [frame for frame in frames if not frame.empty]
        self.samples_df = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    def process_recording(
        self, recording_filename: str, pred_df: pd.DataFrame, annot_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes a single recording by determining its duration, initializing sample
        intervals, and updating the intervals with predictions and annotations.

        Args:
            recording_filename (str): The name of the recording.
            pred_df (pd.DataFrame): Predictions DataFrame specific to the recording.
            annot_df (pd.DataFrame): Annotations DataFrame specific to the recording.

        Returns:
            pd.DataFrame: A DataFrame containing sample intervals with prediction and
            annotation data.
        """
        file_duration = self.determine_file_duration(pred_df, annot_df)

        if file_duration <= 0:
            return pd.DataFrame()

        samples_df = self.initialize_samples(
            recording_filename=recording_filename, file_duration=file_duration
        )

        self.update_samples_with_predictions(pred_df, samples_df)

        self.update_samples_with_annotations(annot_df, samples_df)

        return samples_df

    def determine_file_duration(
        self, pred_df: pd.DataFrame, annot_df: pd.DataFrame
    ) -> float:
        """
        Determines the duration of the recording based on available dataframes or the
        specified recording duration.

        This method prioritizes the explicitly set `recording_duration` if available.
        Otherwise, it computes the duration from the `Duration` or `End Time` columns in
        the predictions and annotations DataFrames. Handles edge cases where data may be
        incomplete or missing.

        Args:
            pred_df (pd.DataFrame): Predictions DataFrame containing duration or end
                time information.
            annot_df (pd.DataFrame): Annotations DataFrame containing duration or end
                time information.

        Returns:
            float: The determined duration of the recording. Defaults to 0 if no valid
                duration is found.
        """
        if self.recording_duration is not None:
            return self.recording_duration

        duration = 0.0

        file_duration_col_pred = self.get_column_name("Duration", prediction=True)

        file_duration_col_annot = self.get_column_name("Duration", prediction=False)

        if (
            file_duration_col_pred in pred_df.columns
            and pred_df[file_duration_col_pred].notna().any()
        ):
            duration = max(duration, pred_df[file_duration_col_pred].dropna().max())

        if (
            file_duration_col_annot in annot_df.columns
            and annot_df[file_duration_col_annot].notna().any()
        ):
            duration = max(duration, annot_df[file_duration_col_annot].dropna().max())

        if duration == 0.0:
            end_time_col_pred = self.get_column_name("End Time", prediction=True)
            end_time_col_annot = self.get_column_name("End Time", prediction=False)

            max_end_pred = (
                pred_df[end_time_col_pred].max()
                if end_time_col_pred in pred_df.columns
                else 0.0
            )
            max_end_annot = (
                annot_df[end_time_col_annot].max()
                if end_time_col_annot in annot_df.columns
                else 0.0
            )
            duration = max(filter(pd.notna, (max_end_pred, max_end_annot)), default=0.0)

            if duration < 0:
                duration = 0.0

        return duration

    def initialize_samples(
        self, recording_filename: str, file_duration: float
    ) -> pd.DataFrame:
        """
        Initializes a DataFrame of time-based sample intervals for the specified
        recording.

        Samples are evenly spaced time intervals of length `sample_duration` that cover
        the entire recording duration. Each sample is initialized with confidence scores
        and annotation values for all classes.

        Args:
            recording_filename (str): The name of the recording.
            file_duration (float): The total duration of the recording in seconds.

        Returns:
            pd.DataFrame: A DataFrame containing initialized sample intervals,
                confidence scores, and annotations. Returns an empty DataFrame if the
                file duration is less than or equal to 0.
        """
        if file_duration <= 0:
            return pd.DataFrame()

        intervals = np.arange(0, file_duration, self.sample_duration)
        if len(intervals) == 0:
            intervals = np.array([0])

        samples = {
            "filename": recording_filename,
            "sample_index": np.arange(len(intervals)),
            "start_time": intervals,
            "end_time": np.minimum(intervals + self.sample_duration, file_duration),
        }

        for label in self.classes:
            samples[f"{label}_confidence"] = np.zeros(len(intervals))
            samples[f"{label}_annotation"] = np.zeros(len(intervals), dtype=np.int64)

        return pd.DataFrame(samples)

    def update_samples_with_predictions(
        self, pred_df: pd.DataFrame, samples_df: pd.DataFrame
    ) -> None:
        """
        Updates the samples DataFrame with prediction confidence scores.

        For each prediction in the predictions DataFrame, this method identifies
        overlapping samples based on the specified `min_overlap`. It then updates the
        confidence scores for those samples, retaining the maximum confidence value if
        multiple predictions overlap.

        Note:
            `min_overlap` is applied as a margin on both sides of the sample window, not
            as a guaranteed intersection length: a prediction shorter than `min_overlap`
            that falls entirely inside a window still marks it. This keeps short events
            from being dropped, but means the effective overlap requirement is looser
            than `min_overlap` seconds for events shorter than the sample duration.

        Args:
            pred_df (pd.DataFrame): DataFrame containing prediction information.
            samples_df (pd.DataFrame): DataFrame of samples to be updated with
                confidence scores.

        Raises:
            KeyError: If `pred_df` has rows but no confidence column.
        """
        if pred_df.empty:
            return

        sample_rows, class_cols, event_rows = self._overlapping_pairs(
            pred_df, samples_df, prediction=True
        )
        confidence_col = self.get_column_name("Confidence", prediction=True)
        confidence = pred_df[confidence_col].to_numpy()[event_rows].astype(float)
        if sample_rows.size == 0:
            return

        hit_classes, hit_cols = np.unique(class_cols, return_inverse=True)
        columns = [f"{self.classes[cls]}_confidence" for cls in hit_classes]
        block = samples_df[columns].to_numpy(dtype=float, copy=True)
        np.fmax.at(block, (sample_rows, hit_cols), confidence)
        samples_df[columns] = block

    def update_samples_with_annotations(
        self, annot_df: pd.DataFrame, samples_df: pd.DataFrame
    ) -> None:
        """
        Updates the samples DataFrame with annotations.

        For each annotation in the annotations DataFrame, this method identifies
        overlapping samples based on the specified `min_overlap`. It sets the annotation
        value to 1 for the overlapping samples.

        Args:
            annot_df (pd.DataFrame): DataFrame containing annotation information.
            samples_df (pd.DataFrame): DataFrame of samples to be updated with
                annotations.
        """
        if annot_df.empty:
            return

        sample_rows, class_cols, _ = self._overlapping_pairs(
            annot_df, samples_df, prediction=False
        )
        if sample_rows.size == 0:
            return

        hit_classes, hit_cols = np.unique(class_cols, return_inverse=True)
        columns = [f"{self.classes[cls]}_annotation" for cls in hit_classes]
        block = samples_df[columns].to_numpy(copy=True)
        block[sample_rows, hit_cols] = 1
        samples_df[columns] = block

    def _overlapping_pairs(
        self, events: pd.DataFrame, samples_df: pd.DataFrame, prediction: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pairs every event of a known class with each sample it overlaps.

        On a grid of consecutive samples the samples an event overlaps form one index
        range, found with two binary searches; irregular sample intervals fall back to
        comparing every event with every sample.

        Returns:
            Three equally long arrays: the sample positions, the class positions (into
            `self.classes`) and the event positions (into `events`) of the pairs.
        """
        class_col = self.get_column_name("Class", prediction=prediction)
        start_time_col = self.get_column_name("Start Time", prediction=prediction)
        end_time_col = self.get_column_name("End Time", prediction=prediction)

        class_cols = pd.Index(self.classes).get_indexer(events[class_col])
        known = np.flatnonzero(class_cols >= 0)
        if known.size == 0:
            return known, known, known

        latest_start = (
            events[end_time_col].to_numpy()[known].astype(float) - self.min_overlap
        )
        earliest_end = (
            events[start_time_col].to_numpy()[known].astype(float) + self.min_overlap
        )
        start_time = samples_df["start_time"].to_numpy(dtype=float)
        end_time = samples_df["end_time"].to_numpy(dtype=float)

        order = np.argsort(start_time, kind="stable")
        if _is_sorted(end_time[order]):
            first = np.searchsorted(end_time[order], earliest_end, side="left")
            stop = np.searchsorted(start_time[order], latest_start, side="right")
            counts = np.where(
                np.isnan(latest_start) | np.isnan(earliest_end),
                0,
                np.maximum(stop - first, 0),
            )
            pair = np.repeat(np.arange(known.size), counts)
            offsets = np.arange(counts.sum()) - np.repeat(
                np.cumsum(counts) - counts, counts
            )
            sample_rows = order[np.repeat(first, counts) + offsets]
        else:
            hit = (start_time <= latest_start[:, None]) & (
                end_time >= earliest_end[:, None]
            )
            pair, sample_rows = np.nonzero(hit)

        event_rows = known[pair]
        return sample_rows, class_cols[event_rows], event_rows

    def create_tensors(self) -> None:
        """
        Creates prediction and label tensors from the samples DataFrame.

        This method converts confidence scores and annotations for each class into
        numpy arrays (tensors).
        """
        if self.samples_df.empty:
            self.prediction_tensors = np.empty((0, len(self.classes)), dtype=np.float32)
            self.label_tensors = np.empty((0, len(self.classes)), dtype=np.int64)
            return

        annotation_columns = [f"{cls}_annotation" for cls in self.classes]
        confidence_columns = [f"{cls}_confidence" for cls in self.classes]

        self.prediction_tensors = self.samples_df[confidence_columns].to_numpy(
            dtype=np.float32
        )
        self.label_tensors = self.samples_df[annotation_columns].to_numpy(
            dtype=np.int64
        )

    def get_column_name(self, field_name: str, prediction: bool = True) -> str | None:
        """
        Retrieves the appropriate column name for the specified field.

        This method checks the column mapping (for predictions or annotations) and
        returns the corresponding column name. If the field is not mapped, it returns
        the field name directly. The optional fields, Recording and Duration, have no
        column when mapped to None.

        Args:
            field_name (str): The name of the field (e.g., "Class", "Start Time").
            prediction (bool): Whether to fetch the name from the predictions mapping
                (True) or annotations mapping (False).

        Returns:
            Optional[str]: The column name corresponding to the field, None for an
                optional field without a column.
        """
        mapping = self.columns_predictions if prediction else self.columns_annotations
        column = mapping.get(field_name, field_name)

        if column is None and field_name not in ("Recording", "Duration"):
            return field_name

        return column

    def get_sample_data(self) -> pd.DataFrame:
        """
        Retrieves the DataFrame containing all sample intervals, prediction scores, and
        annotations.

        This method provides a copy of the `samples_df` DataFrame, ensuring that the
        original data is not modified when accessed externally.

        Returns:
            pd.DataFrame: A copy of the `samples_df` DataFrame, which contains the
                sampled data.
        """
        return self.samples_df.copy()

    def get_filtered_tensors(
        self,
        selected_classes: Sequence[str] | None = None,
        selected_recordings: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """
        Filters the prediction and label tensors based on selected classes and
        recordings.

        This method extracts subsets of the prediction and label tensors for specific
        classes and/or recordings. It ensures that the filtered tensors correspond to
        valid classes and recordings present in the sampled data.

        Args:
            selected_classes (List[str], optional): A list of class names to filter by.
            If None, all classes are included.
            selected_recordings (List[str], optional): A list of recording filenames to
                filter by. If None, all recordings are included.

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[str]]: A tuple containing:
                - Filtered prediction tensors (numpy.ndarray)
                - Filtered label tensors (numpy.ndarray)
                - Tuple of selected class names (Tuple[str])

        Raises:
            ValueError: If the `samples_df` is empty or missing required columns.
            KeyError: If required confidence or annotation columns are missing in the
                DataFrame.
        """
        if self.samples_df.empty:
            raise ValueError("samples_df is empty.")

        if "filename" not in self.samples_df.columns:
            raise ValueError("samples_df must contain a 'filename' column.")

        classes = (
            self.classes
            if selected_classes is None
            else tuple(cls for cls in selected_classes if cls in self.classes)
        )

        if not classes:
            raise ValueError("No valid classes selected.")

        mask = pd.Series(True, index=self.samples_df.index)

        if selected_recordings is not None:
            if selected_recordings:
                mask &= self.samples_df["filename"].isin(selected_recordings)
            else:
                # If `selected_recordings` is an empty list, select no samples
                mask = pd.Series(False, index=self.samples_df.index)

        filtered_samples = self.samples_df.loc[mask]

        confidence_columns = [f"{cls}_confidence" for cls in classes]
        annotation_columns = [f"{cls}_annotation" for cls in classes]

        if not all(
            col in filtered_samples.columns
            for col in confidence_columns + annotation_columns
        ):
            raise KeyError("Required confidence or annotation columns are missing.")

        predictions = np.ascontiguousarray(
            filtered_samples[confidence_columns].to_numpy(dtype=np.float32)
        )
        labels = np.ascontiguousarray(
            filtered_samples[annotation_columns].to_numpy(dtype=np.int64)
        )

        return predictions, labels, classes
