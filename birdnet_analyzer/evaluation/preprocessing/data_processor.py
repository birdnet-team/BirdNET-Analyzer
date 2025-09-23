"""
DataProcessor class for handling and transforming sample data with annotations and predictions.

This module defines the DataProcessor class, which processes prediction and annotation data,
aligns them with sampled time intervals, and generates tensors for further model training or evaluation.
"""

import os
import warnings
from typing import ClassVar

import numpy as np
import pandas as pd

from birdnet_analyzer.evaluation.preprocessing.utils import (
    extract_recording_filename,
    extract_recording_filename_from_filename,
    read_and_concatenate_files_in_directory,
)


class DataProcessor:
    """
    Processor for handling and transforming sample data with annotations and predictions.

    This class processes prediction and annotation data, aligning them with sampled time intervals,
    and generates tensors for further model training or evaluation.
    """

    # Default column mappings for predictions and annotations
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
        sample_duration: int = 3,
        min_overlap: float = 0.5,
        columns_predictions: dict[str, str] | None = None,
        columns_annotations: dict[str, str] | None = None,
        recording_duration: float | None = None,
    ) -> None:
        """
        Initializes the DataProcessor by loading prediction and annotation data.

        Args:
            prediction_directory_path (str): Path to the folder containing prediction files.
            annotation_directory_path (str): Path to the folder containing annotation files.
            prediction_file_name (Optional[str]): Name of the prediction file to process.
            annotation_file_name (Optional[str]): Name of the annotation file to process.
            class_mapping (Optional[Dict[str, str]]): Optional dictionary mapping raw class
                                                      names to standardized class names.
            sample_duration (int, optional): Length of each data sample in seconds. Defaults to 3.
            min_overlap (float, optional): Minimum overlap required between prediction and
                                           annotation to consider a match.
            columns_predictions (Optional[Dict[str, str]], optional): Column name mappings for prediction files.
            columns_annotations (Optional[Dict[str, str]], optional): Column name mappings for annotation files.
            recording_duration (Optional[float], optional): User-specified recording duration in seconds.
                                                            Defaults to None.

        Raises:
            ValueError: If any parameter is invalid (e.g., negative sample duration).
        """
        # Initialize instance variables
        self.sample_duration: int = sample_duration
        self.min_overlap: float = min_overlap
        self.class_mapping: dict[str, str] | None = class_mapping

        # Use provided column mappings or defaults
        self.columns_predictions: dict[str, str] = columns_predictions if columns_predictions is not None else self.DEFAULT_COLUMNS_PREDICTIONS.copy()
        self.columns_annotations: dict[str, str] = columns_annotations if columns_annotations is not None else self.DEFAULT_COLUMNS_ANNOTATIONS.copy()

        self.recording_duration: float | None = recording_duration

        # Paths and filenames
        self.prediction_directory_path: str = prediction_directory_path
        self.prediction_file_name: str | None = prediction_file_name
        self.annotation_directory_path: str = annotation_directory_path
        self.annotation_file_name: str | None = annotation_file_name

        # DataFrames for predictions and annotations
        self.predictions_df: pd.DataFrame = pd.DataFrame()
        self.annotations_df: pd.DataFrame = pd.DataFrame()

        # Placeholder for unique classes across predictions and annotations
        self.classes: tuple[str, ...] = ()

        # Placeholder for samples DataFrame and tensors
        self.samples_df: pd.DataFrame = pd.DataFrame()
        self.prediction_tensors: np.ndarray = np.array([])
        self.label_tensors: np.ndarray = np.array([])

        # Validate column mappings and parameters
        self._validate_columns()
        self._validate_parameters()

        # Load and process data
        self.load_data()
        self.process_data()
        self.create_tensors()

    def _validate_parameters(self) -> None:
        """
        Validates the input parameters for correctness.

        Raises:
            ValueError: If sample duration, minimum overlap, or recording duration is invalid.
        """
        # Validate sample duration
        if self.sample_duration <= 0:
            raise ValueError("Sample duration must be positive.")

        # Validate recording duration
        if self.recording_duration is not None:
            if self.recording_duration <= 0:
                raise ValueError("Recording duration must be greater than 0.")
            if self.sample_duration > self.recording_duration:
                raise ValueError("Sample duration cannot exceed the recording duration.")

        # Validate minimum overlap
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
        # Required columns for predictions and annotations
        required_columns = ["Start Time", "End Time", "Class"]

        # Check for missing or None columns in predictions
        missing_pred_columns = [col for col in required_columns if col not in self.columns_predictions or self.columns_predictions[col] is None]

        # Check for missing or None columns in annotations
        missing_annot_columns = [col for col in required_columns if col not in self.columns_annotations or self.columns_annotations[col] is None]

        if missing_pred_columns:
            raise ValueError(f"Missing or None prediction columns: {', '.join(missing_pred_columns)}")
        if missing_annot_columns:
            raise ValueError(f"Missing or None annotation columns: {', '.join(missing_annot_columns)}")

    def load_data(self) -> None:
        """
        Loads the prediction and annotation data into DataFrames.

        Depending on whether specific files are provided, this method either reads all files
        in the given directories or reads the specified files. The method also applies any
        specified class mapping and prepares the data for further processing.

        Raises:
            ValueError: If file reading fails or data preparation encounters issues.
        """
        if self.prediction_file_name is None or self.annotation_file_name is None:
            # Case: No specific files provided; load all files in directories.
            self.predictions_df = read_and_concatenate_files_in_directory(self.prediction_directory_path)
            self.annotations_df = read_and_concatenate_files_in_directory(self.annotation_directory_path)

            # Ensure 'source_file' column exists for traceability
            if "source_file" not in self.predictions_df.columns:
                self.predictions_df["source_file"] = ""

            if "source_file" not in self.annotations_df.columns:
                self.annotations_df["source_file"] = ""

            # Prepare DataFrames
            self.predictions_df = self._prepare_dataframe(self.predictions_df, prediction=True)
            self.annotations_df = self._prepare_dataframe(self.annotations_df, prediction=False)

            # Apply class mapping to predictions if provided
            if self.class_mapping:
                class_col_pred = self.get_column_name("Class", prediction=True)
                self.predictions_df[class_col_pred] = self.predictions_df[class_col_pred].apply(lambda x: self.class_mapping.get(x, x))
        else:
            # Case: Specific files are provided for predictions and annotations.
            # Ensure filenames correspond to the same recording (heuristic check).
            if not self.prediction_file_name.startswith(os.path.splitext(self.annotation_file_name)[0]):
                warnings.warn(
                    "Prediction file name and annotation file name do not fully match, but proceeding anyway.",
                    stacklevel=2,
                )

            # Construct full file paths
            prediction_file = os.path.join(self.prediction_directory_path, self.prediction_file_name)
            annotation_file = os.path.join(self.annotation_directory_path, self.annotation_file_name)

            # Load files into DataFrames
            self.predictions_df = pd.read_csv(prediction_file, sep="\t")
            self.annotations_df = pd.read_csv(annotation_file, sep="\t")

            # Add 'source_file' column to identify origins
            self.predictions_df["source_file"] = self.prediction_file_name
            self.annotations_df["source_file"] = self.annotation_file_name

            # Prepare DataFrames
            self.predictions_df = self._prepare_dataframe(self.predictions_df, prediction=True)
            self.annotations_df = self._prepare_dataframe(self.annotations_df, prediction=False)

            # Apply class mapping to predictions if provided
            if self.class_mapping:
                class_col_pred = self.get_column_name("Class", prediction=True)
                self.predictions_df[class_col_pred] = self.predictions_df[class_col_pred].apply(lambda x: self.class_mapping.get(x, x))

        # Consolidate all unique classes from predictions and annotations
        class_col_pred = self.get_column_name("Class", prediction=True)
        class_col_annot = self.get_column_name("Class", prediction=False)

        pred_classes = set(self.predictions_df[class_col_pred].unique())
        annot_classes = set(self.annotations_df[class_col_annot].unique())

        # Remove any NaN values from the union
        all_classes = {cls for cls in pred_classes.union(annot_classes) if pd.notna(cls)}
        self.classes = tuple(sorted(all_classes))

    def _prepare_dataframe(self, df: pd.DataFrame, prediction: bool) -> pd.DataFrame:
        """
        Prepares a DataFrame by adding a 'recording_filename' column.

        This method extracts the recording filename from either a specified 'Recording' column
        or from the 'source_file' column to ensure traceability.

        Args:
            df (pd.DataFrame): The DataFrame to prepare.
            prediction (bool): Whether the DataFrame is for predictions or annotations.

        Returns:
            pd.DataFrame: The prepared DataFrame with the added 'recording_filename' column.
        """
        # Determine the relevant column for extracting recording filenames
        recording_col = self.get_column_name("Recording", prediction=prediction)

        if recording_col in df.columns:
            # Extract recording filename using the 'Recording' column
            df["recording_filename"] = extract_recording_filename(df[recording_col])
        elif "source_file" in df.columns:
            # Fall back to extracting from the 'source_file' column
            df["recording_filename"] = extract_recording_filename_from_filename(df["source_file"])
        else:
            # Assign a default empty string if no relevant columns exist
            df["recording_filename"] = ""

        return df

    def process_data(self) -> None:
        """
        Processes the loaded data, aligns predictions and annotations with sample intervals,
        and updates the samples DataFrame.

        This method iterates through all recording filenames, processes each recording,
        and aggregates the results into the `samples_df` attribute.
        """
        self.samples_df = pd.DataFrame()  # Initialize the samples DataFrame

        # Get the unique set of recording filenames from both predictions and annotations
        recording_filenames = set(self.predictions_df["recording_filename"].unique()).union(set(self.annotations_df["recording_filename"].unique()))

        # Process each recording
        for recording_filename in recording_filenames:
            # Filter predictions and annotations for the current recording
            pred_df = self.predictions_df[self.predictions_df["recording_filename"] == recording_filename]
            annot_df = self.annotations_df[self.annotations_df["recording_filename"] == recording_filename]

            # Generate sample intervals and annotations for the recording
            samples_df = self.process_recording(recording_filename, pred_df, annot_df)

            # Append the processed DataFrame to the overall samples DataFrame
            self.samples_df = pd.concat([self.samples_df, samples_df], ignore_index=True)

    def process_recording(self, recording_filename: str, pred_df: pd.DataFrame, annot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a single recording by determining its duration, initializing sample intervals,
        and updating the intervals with predictions and annotations.

        Args:
            recording_filename (str): The name of the recording.
            pred_df (pd.DataFrame): Predictions DataFrame specific to the recording.
            annot_df (pd.DataFrame): Annotations DataFrame specific to the recording.

        Returns:
            pd.DataFrame: A DataFrame containing sample intervals with prediction and annotation data.
        """
        # Determine the duration of the recording
        file_duration = self.determine_file_duration(pred_df, annot_df)

        if file_duration <= 0:
            # Return an empty DataFrame if the duration is invalid
            return pd.DataFrame()

        # Initialize sample intervals for the recording
        samples_df = self.initialize_samples(recording_filename=recording_filename, file_duration=file_duration)

        # Update the samples DataFrame with prediction data
        self.update_samples_with_predictions(pred_df, samples_df)

        # Update the samples DataFrame with annotation data
        self.update_samples_with_annotations(annot_df, samples_df)

        return samples_df

    def determine_file_duration(self, pred_df: pd.DataFrame, annot_df: pd.DataFrame) -> float:
        """
        Determines the duration of the recording based on available dataframes or the specified recording duration.

        This method prioritizes the explicitly set `recording_duration` if available.
        Otherwise, it computes the duration from the `Duration` or `End Time` columns in the
        predictions and annotations DataFrames. Handles edge cases where data may be incomplete
        or missing.

        Args:
            pred_df (pd.DataFrame): Predictions DataFrame containing duration or end time information.
            annot_df (pd.DataFrame): Annotations DataFrame containing duration or end time information.

        Returns:
            float: The determined duration of the recording. Defaults to 0 if no valid duration is found.
        """
        if self.recording_duration is not None:
            # Use the explicitly provided recording duration
            return self.recording_duration

        duration = 0.0

        # Extract the 'Duration' column from predictions if available
        file_duration_col_pred = self.get_column_name("Duration", prediction=True)

        file_duration_col_annot = self.get_column_name("Duration", prediction=False)

        # Try to get duration from 'Duration' column in pred_df
        if file_duration_col_pred in pred_df.columns and pred_df[file_duration_col_pred].notna().any():
            duration = max(duration, pred_df[file_duration_col_pred].dropna().max())

        # Try to get duration from 'Duration' column in annot_df
        if file_duration_col_annot in annot_df.columns and annot_df[file_duration_col_annot].notna().any():
            duration = max(duration, annot_df[file_duration_col_annot].dropna().max())

        # If no duration is found, use the maximum 'End Time' value
        if duration == 0.0:
            end_time_col_pred = self.get_column_name("End Time", prediction=True)
            end_time_col_annot = self.get_column_name("End Time", prediction=False)

            max_end_pred = pred_df[end_time_col_pred].max() if end_time_col_pred in pred_df.columns else 0.0
            max_end_annot = annot_df[end_time_col_annot].max() if end_time_col_annot in annot_df.columns else 0.0
            duration = max(max_end_pred, max_end_annot)

            # Handle invalid values (NaN or negative duration)
            if pd.isna(duration) or duration < 0:
                duration = 0.0

        return duration

    def initialize_samples(self, recording_filename: str, file_duration: float) -> pd.DataFrame:
        """
        Initializes a DataFrame of time-based sample intervals for the specified recording.

        Samples are evenly spaced time intervals of length `sample_duration` that cover the
        entire recording duration. Each sample is initialized with confidence scores and
        annotation values for all classes.

        Args:
            recording_filename (str): The name of the recording.
            file_duration (float): The total duration of the recording in seconds.

        Returns:
            pd.DataFrame: A DataFrame containing initialized sample intervals, confidence scores, and annotations.
                         Returns an empty DataFrame if the file duration is less than or equal to 0.
        """
        if file_duration <= 0:
            # Return an empty DataFrame if file duration is invalid
            return pd.DataFrame()

        # Generate start times for each sample interval
        intervals = np.arange(0, file_duration, self.sample_duration)
        if len(intervals) == 0:
            intervals = np.array([0])

        # Prepare sample structure
        samples = {
            "filename": recording_filename,
            "sample_index": [],
            "start_time": [],
            "end_time": [],
        }

        for idx, start in enumerate(intervals):
            samples["sample_index"].append(idx)
            samples["start_time"].append(start)
            samples["end_time"].append(min(start + self.sample_duration, file_duration))

        # Initialize confidence scores and annotations for each class
        for label in self.classes:
            samples[f"{label}_confidence"] = [0.0] * len(samples["sample_index"])  # Float values
            samples[f"{label}_annotation"] = [0] * len(samples["sample_index"])  # Integer values

        return pd.DataFrame(samples)

    def update_samples_with_predictions(self, pred_df: pd.DataFrame, samples_df: pd.DataFrame) -> None:
        """
        Updates the samples DataFrame with prediction confidence scores.

        For each prediction in the predictions DataFrame, this method identifies overlapping
        samples based on the specified `min_overlap`. It then updates the confidence scores
        for those samples, retaining the maximum confidence value if multiple predictions overlap.

        Args:
            pred_df (pd.DataFrame): DataFrame containing prediction information.
            samples_df (pd.DataFrame): DataFrame of samples to be updated with confidence scores.
        """
        # Retrieve the column names for predictions
        class_col = self.get_column_name("Class", prediction=True)
        start_time_col = self.get_column_name("Start Time", prediction=True)
        end_time_col = self.get_column_name("End Time", prediction=True)
        confidence_col = self.get_column_name("Confidence", prediction=True)

        # Iterate through each prediction row
        for _, row in pred_df.iterrows():
            class_name = row[class_col]
            if class_name not in self.classes:
                continue  # Skip predictions for classes not included in the predefined list

            # Extract start and end times, and confidence score
            begin_time = row[start_time_col]
            end_time = row[end_time_col]
            confidence = row.get(confidence_col, 0.0)

            # Identify samples that overlap with the prediction based on min_overlap
            sample_indices = samples_df[(samples_df["start_time"] <= end_time - self.min_overlap) & (samples_df["end_time"] >= begin_time + self.min_overlap)].index

            # Update the confidence scores for the overlapping samples
            for i in sample_indices:
                current_confidence = samples_df.loc[i, f"{class_name}_confidence"]
                samples_df.loc[i, f"{class_name}_confidence"] = max(current_confidence, confidence)

    def update_samples_with_annotations(self, annot_df: pd.DataFrame, samples_df: pd.DataFrame) -> None:
        """
        Updates the samples DataFrame with annotations.

        For each annotation in the annotations DataFrame, this method identifies overlapping
        samples based on the specified `min_overlap`. It sets the annotation value to 1
        for the overlapping samples.

        Args:
            annot_df (pd.DataFrame): DataFrame containing annotation information.
            samples_df (pd.DataFrame): DataFrame of samples to be updated with annotations.
        """
        # Retrieve the column names for annotations
        class_col = self.get_column_name("Class", prediction=False)
        start_time_col = self.get_column_name("Start Time", prediction=False)
        end_time_col = self.get_column_name("End Time", prediction=False)

        # Iterate through each annotation row
        for _, row in annot_df.iterrows():
            class_name = row[class_col]
            if class_name not in self.classes:
                continue  # Skip annotations for classes not included in the predefined list

            # Extract start and end times
            begin_time = row[start_time_col]
            end_time = row[end_time_col]

            # Identify samples that overlap with the annotation based on min_overlap
            sample_indices = samples_df[(samples_df["start_time"] <= end_time - self.min_overlap) & (samples_df["end_time"] >= begin_time + self.min_overlap)].index

            # Set annotation value to 1 for the overlapping samples
            for i in sample_indices:
                samples_df.loc[i, f"{class_name}_annotation"] = 1

    def create_tensors(self) -> None:
        """
        Creates prediction and label tensors from the samples DataFrame.

        This method converts confidence scores and annotations for each class into
        numpy arrays (tensors). It ensures that there are no NaN values in the DataFrame
        before creating the tensors.

        Raises:
            ValueError: If NaN values are found in confidence or annotation columns.
        """
        if self.samples_df.empty:
            # Initialize empty tensors if samples DataFrame is empty
            self.prediction_tensors = np.empty((0, len(self.classes)), dtype=np.float32)
            self.label_tensors = np.empty((0, len(self.classes)), dtype=np.int64)
            return

        # Check for NaN values in annotation columns
        annotation_columns = [f"{cls}_annotation" for cls in self.classes]
        if self.samples_df[annotation_columns].isna().to_numpy().any():
            raise ValueError("NaN values found in annotation columns.")

        # Check for NaN values in confidence columns
        confidence_columns = [f"{cls}_confidence" for cls in self.classes]
        if self.samples_df[confidence_columns].isna().to_numpy().any():
            raise ValueError("NaN values found in confidence columns.")

        # Convert confidence scores and annotations into numpy arrays (tensors)
        self.prediction_tensors = self.samples_df[confidence_columns].to_numpy(dtype=np.float32)
        self.label_tensors = self.samples_df[annotation_columns].to_numpy(dtype=np.int64)

    def get_column_name(self, field_name: str, prediction: bool = True) -> str:
        """
        Retrieves the appropriate column name for the specified field.

        This method checks the column mapping (for predictions or annotations) and
        returns the corresponding column name. If the field is not mapped, it returns
        the field name directly.

        Args:
            field_name (str): The name of the field (e.g., "Class", "Start Time").
            prediction (bool): Whether to fetch the name from the predictions mapping (True)
                               or annotations mapping (False).

        Returns:
            str: The column name corresponding to the field.

        Raises:
            TypeError: If `field_name` or `prediction` is None.
        """
        if field_name is None:
            raise TypeError("field_name cannot be None.")
        if prediction is None:
            raise TypeError("prediction parameter cannot be None.")

        # Select the appropriate mapping based on the `prediction` flag
        mapping = self.columns_predictions if prediction else self.columns_annotations

        if field_name in mapping and mapping[field_name] is not None:
            return mapping[field_name]

        return field_name

    def get_sample_data(self) -> pd.DataFrame:
        """
        Retrieves the DataFrame containing all sample intervals, prediction scores, and annotations.

        This method provides a copy of the `samples_df` DataFrame, ensuring that the original
        data is not modified when accessed externally.

        Returns:
            pd.DataFrame: A copy of the `samples_df` DataFrame, which contains the sampled data.
        """
        # Return a copy of the samples DataFrame to preserve data integrity
        return self.samples_df.copy()

    def get_filtered_tensors(
        self,
        selected_classes: list[str] | None = None,
        selected_recordings: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """
        Filters the prediction and label tensors based on selected classes and recordings.

        This method extracts subsets of the prediction and label tensors for specific classes
        and/or recordings. It ensures that the filtered tensors correspond to valid classes
        and recordings present in the sampled data.

        Args:
            selected_classes (List[str], optional): A list of class names to filter by. If None,
                all classes are included.
            selected_recordings (List[str], optional): A list of recording filenames to filter by. If None,
                all recordings are included.

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[str]]: A tuple containing:
                - Filtered prediction tensors (numpy.ndarray)
                - Filtered label tensors (numpy.ndarray)
                - Tuple of selected class names (Tuple[str])

        Raises:
            ValueError: If the `samples_df` is empty or missing required columns.
            KeyError: If required confidence or annotation columns are missing in the DataFrame.
        """
        if self.samples_df.empty:
            raise ValueError("samples_df is empty.")

        if "filename" not in self.samples_df.columns:
            raise ValueError("samples_df must contain a 'filename' column.")

        # Determine the classes to filter by
        classes = self.classes if selected_classes is None else tuple(cls for cls in selected_classes if cls in self.classes)

        if not classes:
            raise ValueError("No valid classes selected.")

        # Create a mask for filtering samples
        mask = pd.Series(True, index=self.samples_df.index)

        # Apply recording-based filtering if specified
        if selected_recordings is not None:
            if selected_recordings:
                mask &= self.samples_df["filename"].isin(selected_recordings)
            else:
                # If `selected_recordings` is an empty list, select no samples
                mask = pd.Series(False, index=self.samples_df.index)

        # Filter the samples DataFrame using the mask
        filtered_samples = self.samples_df.loc[mask]

        # Prepare column names for confidence and annotation data
        confidence_columns = [f"{cls}_confidence" for cls in classes]
        annotation_columns = [f"{cls}_annotation" for cls in classes]

        # Ensure all required columns are present in the filtered DataFrame
        if not all(col in filtered_samples.columns for col in confidence_columns + annotation_columns):
            raise KeyError("Required confidence or annotation columns are missing.")

        # Convert filtered data into numpy arrays
        predictions = filtered_samples[confidence_columns].to_numpy(dtype=np.float32)
        labels = filtered_samples[annotation_columns].to_numpy(dtype=np.int64)

        # Return the tensors and the list of filtered classes
        return predictions, labels, classes
