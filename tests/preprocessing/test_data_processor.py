from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from birdnet_analyzer.evaluation.preprocessing.data_processor import DataProcessor


class TestDataProcessorInit:

    @patch("pandas.read_csv")
    def test_init_with_all_parameters(self, mock_read_csv):
        """Test initializing DataProcessor with all parameters."""
        # Mock the dataframes returned by pd.read_csv
        mock_predictions_df = pd.DataFrame(
            {
                "start_time": [],
                "end_time": [],
                "class_col": [],
                "source_file": [],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "start_time_annot": [],
                "end_time_annot": [],
                "class_col_annot": [],
                "source_file": [],
            }
        )
        # Set side effect for the mock
        mock_read_csv.side_effect = [mock_predictions_df, mock_annotations_df]

        DataProcessor(
            prediction_directory_path="pred_path",
            prediction_file_name="pred_file.txt",
            annotation_directory_path="annot_path",
            annotation_file_name="annot_file.txt",
            class_mapping={"A": "ClassA"},
            sample_duration=5,
            min_overlap=0.7,
            columns_predictions={
                "Start Time": "start_time",
                "End Time": "end_time",
                "Class": "class_col",
            },
            columns_annotations={
                "Start Time": "start_time_annot",
                "End Time": "end_time_annot",
                "Class": "class_col_annot",
            },
            recording_duration=30,
        )
        # Your assertions here

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_init_with_minimal_parameters(self, mock_read_concat):
        """Test initializing DataProcessor with minimal parameters."""
        # Mock the dataframes returned by the read function
        mock_predictions_df = pd.DataFrame(
            {
                "Start Time": [],
                "End Time": [],
                "Class": [],
                "source_file": [],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "Start Time": [],
                "End Time": [],
                "Class": [],
                "source_file": [],
            }
        )
        # Set side effect for the mock
        mock_read_concat.side_effect = [mock_predictions_df, mock_annotations_df]

        dp = DataProcessor(prediction_directory_path="", annotation_directory_path="")
        assert dp.sample_duration == 3
        assert dp.min_overlap == 0.5
        assert dp.columns_predictions == dp.DEFAULT_COLUMNS_PREDICTIONS
        assert dp.columns_annotations == dp.DEFAULT_COLUMNS_ANNOTATIONS
        assert dp.recording_duration is None

    @patch.object(DataProcessor, "load_data")
    def test_init_with_invalid_sample_duration(self, mock_load_data):
        """Test initializing with invalid sample_duration."""
        with pytest.raises(ValueError, match="Sample duration must be positive"):
            DataProcessor(
                prediction_directory_path="",
                annotation_directory_path="",
                sample_duration=-5,
            )

        with pytest.raises(ValueError, match="Sample duration must be positive"):
            DataProcessor(
                prediction_directory_path="",
                annotation_directory_path="",
                sample_duration=0,
            )

        with pytest.raises(
            ValueError, match="Sample duration cannot exceed the recording duration."
        ):
            DataProcessor(
                prediction_directory_path="",
                annotation_directory_path="",
                sample_duration=15,
                recording_duration=10,  # sample_duration > recording_duration
            )

    @patch.object(DataProcessor, "load_data")
    def test_init_with_invalid_min_overlap(self, mock_load_data):
        """Test initializing with invalid min_overlap values."""
        with pytest.raises(ValueError, match="Min overlap must be greater than 0."):
            DataProcessor(
                prediction_directory_path="",
                annotation_directory_path="",
                min_overlap=-5,
            )

        with pytest.raises(ValueError, match="Min overlap must be greater than 0."):
            DataProcessor(
                prediction_directory_path="",
                annotation_directory_path="",
                min_overlap=0,
            )

        with pytest.raises(
            ValueError, match="Min overlap cannot exceed the sample duration."
        ):
            DataProcessor(
                prediction_directory_path="",
                annotation_directory_path="",
                min_overlap=6,  # Greater than default sample_duration=3
            )

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_init_with_nonexistent_paths(self, mock_read_concat):
        """Test initializing with paths that do not exist."""
        # Mock the dataframes to be empty but with required columns
        mock_predictions_df = pd.DataFrame(
            {
                "Class": [],
                "Start Time": [],
                "End Time": [],
                "source_file": [],
                "recording_filename": [],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "Class": [],
                "Start Time": [],
                "End Time": [],
                "source_file": [],
                "recording_filename": [],
            }
        )
        mock_read_concat.side_effect = [mock_predictions_df, mock_annotations_df]

        dp = DataProcessor(
            prediction_directory_path="nonexistent_path",
            annotation_directory_path="nonexistent_path",
        )
        # Ensure that predictions_df and annotations_df are set correctly
        pd.testing.assert_frame_equal(dp.predictions_df, mock_predictions_df)
        pd.testing.assert_frame_equal(dp.annotations_df, mock_annotations_df)

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_init_with_none_columns(self, mock_read_concat):
        """Test initializing with None columns mappings."""
        # Mock the dataframes returned by the read function
        mock_predictions_df = pd.DataFrame(
            {
                "Start Time": [],
                "End Time": [],
                "Class": [],
                "source_file": [],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "Start Time": [],
                "End Time": [],
                "Class": [],
                "source_file": [],
            }
        )
        # Set side effect for the mock
        mock_read_concat.side_effect = [mock_predictions_df, mock_annotations_df]

        dp = DataProcessor(
            prediction_directory_path="",
            annotation_directory_path="",
            columns_predictions=None,
            columns_annotations=None,
        )
        assert dp.columns_predictions == dp.DEFAULT_COLUMNS_PREDICTIONS
        assert dp.columns_annotations == dp.DEFAULT_COLUMNS_ANNOTATIONS

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_init_with_empty_class_mapping(self, mock_read_concat):
        """Test initializing with empty class_mapping."""
        # Mock the dataframes returned by the read function
        mock_predictions_df = pd.DataFrame(
            {
                "Start Time": [],
                "End Time": [],
                "Class": [],
                "source_file": [],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "Start Time": [],
                "End Time": [],
                "Class": [],
                "source_file": [],
            }
        )
        # Set side effect for the mock
        mock_read_concat.side_effect = [mock_predictions_df, mock_annotations_df]

        dp = DataProcessor(
            prediction_directory_path="", annotation_directory_path="", class_mapping={}
        )
        assert dp.class_mapping == {}

    @patch.object(DataProcessor, "load_data")
    def test_init_with_invalid_recording_duration(self, mock_load_data):
        """Test initializing with negative recording_duration."""
        with pytest.raises(
            ValueError, match="Recording duration must be greater than 0."
        ):
            DataProcessor(
                prediction_directory_path="",
                annotation_directory_path="",
                recording_duration=-10,
            )

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_init_with_large_sample_duration(self, mock_read_concat):
        """Test initializing with large sample_duration."""
        # Mock the dataframes returned by the read function
        mock_predictions_df = pd.DataFrame(
            {
                "Start Time": [],
                "End Time": [],
                "Class": [],
                "source_file": [],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "Start Time": [],
                "End Time": [],
                "Class": [],
                "source_file": [],
            }
        )
        # Set side effect for the mock
        mock_read_concat.side_effect = [mock_predictions_df, mock_annotations_df]

        dp = DataProcessor(
            prediction_directory_path="",
            annotation_directory_path="",
            sample_duration=1000,
        )
        assert dp.sample_duration == 1000

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_init_with_non_default_columns(self, mock_read_concat):
        """Test initializing with custom columns mappings."""
        # Mock the dataframes returned by the read function
        mock_predictions_df = pd.DataFrame(
            {
                "start": [],
                "end_time_pred": [],
                "class_pred": [],
                "source_file": [],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "start_time_annot": [],
                "end": [],
                "class_annot": [],
                "source_file": [],
            }
        )
        # Set side effect for the mock
        mock_read_concat.side_effect = [mock_predictions_df, mock_annotations_df]

        dp = DataProcessor(
            prediction_directory_path="",
            annotation_directory_path="",
            columns_predictions={
                "Start Time": "start",
                "End Time": "end_time_pred",
                "Class": "class_pred",
            },
            columns_annotations={
                "Start Time": "start_time_annot",
                "End Time": "end",
                "Class": "class_annot",
            },
        )
        assert dp.columns_predictions == {
            "Start Time": "start",
            "End Time": "end_time_pred",
            "Class": "class_pred",
        }
        assert dp.columns_annotations == {
            "Start Time": "start_time_annot",
            "End Time": "end",
            "Class": "class_annot",
        }

        # Check if default columns are used for missing mappings
        # Since all required columns are provided, this isn't necessary,
        # but if you want to check optional columns:
        optional_col = "Confidence"
        assert dp.get_column_name(
            optional_col, prediction=True
        ) == dp.DEFAULT_COLUMNS_PREDICTIONS.get(optional_col, optional_col)
        optional_col = "Recording"
        assert dp.get_column_name(
            optional_col, prediction=False
        ) == dp.DEFAULT_COLUMNS_ANNOTATIONS.get(optional_col, optional_col)


class TestDataProcessorLoadData:

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_load_data_with_none_filenames(self, mock_read_concat):
        """Test load_data when prediction_file_name and annotation_file_name are None."""
        # Mocking the DataFrames returned by the utility function
        mock_predictions_df = pd.DataFrame(
            {
                "Class": ["A", "B"],
                "Start Time": [0, 1],
                "End Time": [1, 2],
                "source_file": ["file1.txt", "file2.txt"],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "Class": ["A", "C"],
                "Start Time": [0.5, 1.5],
                "End Time": [1.5, 2.5],
                "source_file": ["file3.txt", "file4.txt"],
            }
        )
        mock_read_concat.side_effect = [mock_predictions_df, mock_annotations_df]

        dp = DataProcessor(
            prediction_directory_path="dummy_pred_path",
            prediction_file_name=None,
            annotation_directory_path="dummy_annot_path",
            annotation_file_name=None,
            columns_predictions={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            columns_annotations={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            recording_duration=10,
        )

        # Ensure that predictions_df and annotations_df are set correctly
        pd.testing.assert_frame_equal(dp.predictions_df, mock_predictions_df)
        pd.testing.assert_frame_equal(dp.annotations_df, mock_annotations_df)

    @patch("pandas.read_csv")
    def test_load_data_with_specific_filenames(self, mock_read_csv):
        """Test load_data when specific filenames are provided."""
        # Mocking the DataFrames returned by pd.read_csv
        mock_predictions_df = pd.DataFrame(
            {
                "Class": ["A", "B"],
                "Start Time": [0, 1],
                "End Time": [1, 2],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "Class": ["A", "C"],
                "Start Time": [0.5, 1.5],
                "End Time": [1.5, 2.5],
            }
        )
        mock_read_csv.side_effect = [mock_predictions_df, mock_annotations_df]

        dp = DataProcessor(
            prediction_directory_path="dummy_pred_path",
            prediction_file_name="predictions.txt",
            annotation_directory_path="dummy_annot_path",
            annotation_file_name="annotations.txt",
            columns_predictions={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            columns_annotations={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            recording_duration=10,
        )

        # Ensure that predictions_df and annotations_df are set correctly
        pd.testing.assert_frame_equal(
            dp.predictions_df, mock_predictions_df.assign(source_file="predictions.txt")
        )
        pd.testing.assert_frame_equal(
            dp.annotations_df, mock_annotations_df.assign(source_file="annotations.txt")
        )

    @patch("pandas.read_csv")
    def test_load_data_missing_prediction_file(self, mock_read_csv):
        """Test load_data when prediction file is missing."""

        # Simulate FileNotFoundError for predictions
        def side_effect(*args, **kwargs):
            if "predictions.txt" in args[0]:
                raise FileNotFoundError("File not found")
            else:
                return pd.DataFrame(
                    {
                        "Class": ["A", "C"],
                        "Start Time": [0.5, 1.5],
                        "End Time": [1.5, 2.5],
                    }
                )

        mock_read_csv.side_effect = side_effect

        with pytest.raises(FileNotFoundError):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                prediction_file_name="predictions.txt",
                annotation_directory_path="dummy_annot_path",
                annotation_file_name="annotations.txt",
                columns_predictions={
                    "Class": "Class",
                    "Start Time": "Start Time",
                    "End Time": "End Time",
                },
                columns_annotations={
                    "Class": "Class",
                    "Start Time": "Start Time",
                    "End Time": "End Time",
                },
                recording_duration=10,
            )

    @patch("pandas.read_csv")
    def test_load_data_missing_annotation_file(self, mock_read_csv):
        """Test load_data when annotation file is missing."""

        # Simulate FileNotFoundError for annotations
        def side_effect(*args, **kwargs):
            if "annotations.txt" in args[0]:
                raise FileNotFoundError("File not found")
            else:
                return pd.DataFrame(
                    {
                        "Class": ["A", "B"],
                        "Start Time": [0, 1],
                        "End Time": [1, 2],
                    }
                )

        mock_read_csv.side_effect = side_effect

        with pytest.raises(FileNotFoundError):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                prediction_file_name="predictions.txt",
                annotation_directory_path="dummy_annot_path",
                annotation_file_name="annotations.txt",
                columns_predictions={
                    "Class": "Class",
                    "Start Time": "Start Time",
                    "End Time": "End Time",
                },
                columns_annotations={
                    "Class": "Class",
                    "Start Time": "Start Time",
                    "End Time": "End Time",
                },
                recording_duration=10,
            )

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_load_data_with_empty_directories(self, mock_read_concat):
        """Test load_data when directories are empty."""
        mock_read_concat.return_value = pd.DataFrame(
            {
                "Class": [],
                "Start Time": [],
                "End Time": [],
                "source_file": [],
            }
        )
        dp = DataProcessor(
            prediction_directory_path="empty_pred_path",
            prediction_file_name=None,
            annotation_directory_path="empty_annot_path",
            annotation_file_name=None,
            columns_predictions={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            columns_annotations={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            recording_duration=10,
        )

        assert dp.predictions_df.empty
        assert dp.annotations_df.empty

    @patch("pandas.read_csv")
    def test_load_data_filenames_do_not_match(self, mock_read_csv):
        """Test load_data with mismatched filenames."""
        mock_predictions_df = pd.DataFrame(
            {
                "Class": ["A"],
                "Start Time": [0],
                "End Time": [1],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "Class": ["A"],
                "Start Time": [0],
                "End Time": [1],
            }
        )
        mock_read_csv.side_effect = [mock_predictions_df, mock_annotations_df]

        with pytest.warns(UserWarning, match="do not fully match"):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                prediction_file_name="prediction_file.txt",
                annotation_directory_path="dummy_annot_path",
                annotation_file_name="different_annotation.txt",
                columns_predictions={
                    "Class": "Class",
                    "Start Time": "Start Time",
                    "End Time": "End Time",
                },
                columns_annotations={
                    "Class": "Class",
                    "Start Time": "Start Time",
                    "End Time": "End Time",
                },
                recording_duration=10,
            )

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_load_data_inconsistent_columns(self, mock_read_concat):
        """Test load_data when files have inconsistent columns."""
        mock_read_concat.side_effect = ValueError(
            "File has different columns than previous files."
        )

        with pytest.raises(ValueError, match="different columns than previous files"):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                prediction_file_name=None,
                annotation_directory_path="dummy_annot_path",
                annotation_file_name=None,
                columns_predictions={
                    "Class": "Class",
                    "Start Time": "Start Time",
                    "End Time": "End Time",
                },
                columns_annotations={
                    "Class": "Class",
                    "Start Time": "Start Time",
                    "End Time": "End Time",
                },
                recording_duration=10,
            )

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_load_data_handle_different_encodings(self, mock_read_concat):
        """Test load_data handling different file encodings."""
        mock_read_concat.return_value = pd.DataFrame(
            {
                "Class": ["A"],
                "Start Time": [0],
                "End Time": [1],
                "source_file": ["file.txt"],
            }
        )
        DataProcessor(
            prediction_directory_path="dummy_pred_path",
            prediction_file_name=None,
            annotation_directory_path="dummy_annot_path",
            annotation_file_name=None,
            columns_predictions={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            columns_annotations={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            recording_duration=10,
        )
        # Should proceed without errors

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_load_data_with_class_mapping(self, mock_read_concat):
        """Test load_data applying class mapping."""
        mock_predictions_df = pd.DataFrame(
            {
                "Class": ["A", "B", "C"],
                "Start Time": [0, 1, 2],
                "End Time": [1, 2, 3],
                "source_file": ["file1.txt", "file2.txt", "file3.txt"],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "Class": ["ClassA", "ClassB", "ClassC"],
                "Start Time": [0, 1, 2],
                "End Time": [1, 2, 3],
                "source_file": ["file1.txt", "file2.txt", "file3.txt"],
            }
        )
        mock_read_concat.side_effect = [mock_predictions_df, mock_annotations_df]

        class_mapping = {"A": "ClassA", "B": "ClassB"}
        dp = DataProcessor(
            prediction_directory_path="dummy_pred_path",
            prediction_file_name=None,
            annotation_directory_path="dummy_annot_path",
            annotation_file_name=None,
            class_mapping=class_mapping,
            columns_predictions={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            columns_annotations={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            recording_duration=10,
        )

        expected_classes = ("C", "ClassA", "ClassB", "ClassC")
        assert dp.classes == expected_classes

    @patch("bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory")
    def test_load_data_without_class_mapping(self, mock_read_concat):
        """Test load_data without class mapping."""
        mock_predictions_df = pd.DataFrame(
            {
                "Class": ["A", "B", "C"],
                "Start Time": [0, 1, 2],
                "End Time": [1, 2, 3],
                "source_file": ["file1.txt", "file2.txt", "file3.txt"],
            }
        )
        mock_annotations_df = pd.DataFrame(
            {
                "Class": ["A", "B", "C"],
                "Start Time": [0, 1, 2],
                "End Time": [1, 2, 3],
                "source_file": ["file1.txt", "file2.txt", "file3.txt"],
            }
        )
        mock_read_concat.side_effect = [mock_predictions_df, mock_annotations_df]

        dp = DataProcessor(
            prediction_directory_path="dummy_pred_path",
            prediction_file_name=None,
            annotation_directory_path="dummy_annot_path",
            annotation_file_name=None,
            class_mapping=None,
            columns_predictions={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            columns_annotations={
                "Class": "Class",
                "Start Time": "Start Time",
                "End Time": "End Time",
            },
            recording_duration=10,
        )

        expected_classes = ("A", "B", "C")
        assert dp.classes == expected_classes


class TestDataProcessorValidateParameters:

    @patch.object(DataProcessor, "load_data")
    def test_sample_duration_zero(self, mock_load_data):
        """Test sample_duration=0 raises ValueError."""
        with pytest.raises(ValueError, match="Sample duration must be positive"):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                sample_duration=0,
            )

    @patch.object(DataProcessor, "load_data")
    def test_sample_duration_negative(self, mock_load_data):
        """Test negative sample_duration raises ValueError."""
        with pytest.raises(ValueError, match="Sample duration must be positive"):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                sample_duration=-1,
            )

    @patch.object(DataProcessor, "process_data")
    @patch.object(DataProcessor, "load_data")
    def test_sample_duration_positive(self, mock_load_data, mock_process_data):
        """Test positive sample_duration does not raise."""
        try:
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                sample_duration=1,
            )
        except ValueError:
            pytest.fail("Unexpected ValueError raised with valid sample_duration")

    @patch.object(DataProcessor, "process_data")
    @patch.object(DataProcessor, "load_data")
    def test_min_overlap_negative(self, mock_load_data, mock_process_data):
        """Test negative min_overlap raises ValueError."""
        with pytest.raises(ValueError, match="Min overlap must be greater than 0."):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                min_overlap=-0.1,
            )

    @patch.object(DataProcessor, "process_data")
    @patch.object(DataProcessor, "load_data")
    def test_min_overlap_greater_than_sample_duration(
        self, mock_load_data, mock_process_data
    ):
        """Test min_overlap > sample_duration raises ValueError."""
        with pytest.raises(
            ValueError, match="Min overlap cannot exceed the sample duration."
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                min_overlap=4.0,  # Greater than sample_duration
                sample_duration=3.0,
            )

    @patch.object(DataProcessor, "process_data")
    @patch.object(DataProcessor, "load_data")
    def test_min_overlap_zero(self, mock_load_data, mock_process_data):
        """Test min_overlap=0 raises ValueError."""
        with pytest.raises(ValueError, match="Min overlap must be greater than 0."):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                min_overlap=0,
            )

    @patch.object(DataProcessor, "process_data")
    @patch.object(DataProcessor, "load_data")
    def test_min_overlap_one(self, mock_load_data, mock_process_data):
        """Test min_overlap=1 does not raise."""
        try:
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                min_overlap=1,
            )
        except ValueError:
            pytest.fail("Unexpected ValueError raised with min_overlap=1")

    @patch.object(DataProcessor, "process_data")
    @patch.object(DataProcessor, "load_data")
    def test_recording_duration_none(self, mock_load_data, mock_process_data):
        """Test recording_duration=None does not raise."""
        try:
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                recording_duration=None,
            )
        except ValueError:
            pytest.fail("Unexpected ValueError raised with recording_duration=None")

    @patch.object(DataProcessor, "load_data")
    def test_recording_duration_zero(self, mock_load_data):
        """Test recording_duration=0 raises ValueError."""
        with pytest.raises(
            ValueError, match="Recording duration must be greater than 0."
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                recording_duration=0,
            )

    @patch.object(DataProcessor, "load_data")
    def test_recording_duration_negative(self, mock_load_data):
        """Test negative recording_duration raises ValueError."""
        with pytest.raises(
            ValueError, match="Recording duration must be greater than 0."
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                recording_duration=-5,
            )


class TestDataProcessorValidateColumns:

    @patch.object(DataProcessor, "process_data")
    @patch.object(DataProcessor, "load_data")
    def test_columns_all_required_present(self, mock_load_data, mock_process_data):
        """Test when all required columns are present and not None."""
        try:
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={
                    "Start Time": "start_time_pred",
                    "End Time": "end_time_pred",
                    "Class": "class_pred",
                },
                columns_annotations={
                    "Start Time": "start_time_annot",
                    "End Time": "end_time_annot",
                    "Class": "class_annot",
                },
            )
        except ValueError:
            pytest.fail(
                "Unexpected ValueError raised with all required columns present"
            )

    @patch.object(DataProcessor, "load_data")
    def test_columns_predictions_missing_start_time(self, mock_load_data):
        """Test missing 'Start Time' in columns_predictions raises ValueError."""
        with pytest.raises(
            ValueError, match="Missing or None prediction columns: Start Time"
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={
                    # "Start Time": "start_time_pred",  # Missing
                    "End Time": "end_time_pred",
                    "Class": "class_pred",
                },
                columns_annotations={
                    "Start Time": "start_time_annot",
                    "End Time": "end_time_annot",
                    "Class": "class_annot",
                },
            )

    @patch.object(DataProcessor, "load_data")
    def test_columns_predictions_missing_end_time(self, mock_load_data):
        """Test missing 'End Time' in columns_predictions raises ValueError."""
        with pytest.raises(
            ValueError, match="Missing or None prediction columns: End Time"
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={
                    "Start Time": "start_time_pred",
                    # "End Time": "end_time_pred",  # Missing
                    "Class": "class_pred",
                },
                columns_annotations={
                    "Start Time": "start_time_annot",
                    "End Time": "end_time_annot",
                    "Class": "class_annot",
                },
            )

    @patch.object(DataProcessor, "load_data")
    def test_columns_predictions_missing_class(self, mock_load_data):
        """Test missing 'Class' in columns_predictions raises ValueError."""
        with pytest.raises(
            ValueError, match="Missing or None prediction columns: Class"
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={
                    "Start Time": "start_time_pred",
                    "End Time": "end_time_pred",
                    # "Class": "class_pred",  # Missing
                },
                columns_annotations={
                    "Start Time": "start_time_annot",
                    "End Time": "end_time_annot",
                    "Class": "class_annot",
                },
            )

    @patch.object(DataProcessor, "load_data")
    def test_columns_annotations_missing_start_time(self, mock_load_data):
        """Test missing 'Start Time' in columns_annotations raises ValueError."""
        with pytest.raises(
            ValueError, match="Missing or None annotation columns: Start Time"
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={
                    "Start Time": "start_time_pred",
                    "End Time": "end_time_pred",
                    "Class": "class_pred",
                },
                columns_annotations={
                    # "Start Time": "start_time_annot",  # Missing
                    "End Time": "end_time_annot",
                    "Class": "class_annot",
                },
            )

    @patch.object(DataProcessor, "load_data")
    def test_columns_annotations_missing_end_time(self, mock_load_data):
        """Test missing 'End Time' in columns_annotations raises ValueError."""
        with pytest.raises(
            ValueError, match="Missing or None annotation columns: End Time"
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={
                    "Start Time": "start_time_pred",
                    "End Time": "end_time_pred",
                    "Class": "class_pred",
                },
                columns_annotations={
                    "Start Time": "start_time_annot",
                    # "End Time": "end_time_annot",  # Missing
                    "Class": "class_annot",
                },
            )

    @patch.object(DataProcessor, "load_data")
    def test_columns_annotations_missing_class(self, mock_load_data):
        """Test missing 'Class' in columns_annotations raises ValueError."""
        with pytest.raises(
            ValueError, match="Missing or None annotation columns: Class"
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={
                    "Start Time": "start_time_pred",
                    "End Time": "end_time_pred",
                    "Class": "class_pred",
                },
                columns_annotations={
                    "Start Time": "start_time_annot",
                    "End Time": "end_time_annot",
                    # "Class": "class_annot",  # Missing
                },
            )

    @patch.object(DataProcessor, "load_data")
    def test_columns_predictions_start_time_none(self, mock_load_data):
        """Test 'Start Time' in columns_predictions set to None raises ValueError."""
        with pytest.raises(
            ValueError, match="Missing or None prediction columns: Start Time"
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={
                    "Start Time": None,  # Set to None
                    "End Time": "end_time_pred",
                    "Class": "class_pred",
                },
                columns_annotations={
                    "Start Time": "start_time_annot",
                    "End Time": "end_time_annot",
                    "Class": "class_annot",
                },
            )

    @patch.object(DataProcessor, "load_data")
    def test_columns_annotations_end_time_none(self, mock_load_data):
        """Test 'End Time' in columns_annotations set to None raises ValueError."""
        with pytest.raises(
            ValueError, match="Missing or None annotation columns: End Time"
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={
                    "Start Time": "start_time_pred",
                    "End Time": "end_time_pred",
                    "Class": "class_pred",
                },
                columns_annotations={
                    "Start Time": "start_time_annot",
                    "End Time": None,  # Set to None
                    "Class": "class_annot",
                },
            )

    @patch.object(DataProcessor, "load_data")
    def test_columns_predictions_empty_dict(self, mock_load_data):
        """Test empty columns_predictions raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Missing or None prediction columns: Start Time, End Time, Class",
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={},
                columns_annotations={
                    "Start Time": "start_time_annot",
                    "End Time": "end_time_annot",
                    "Class": "class_annot",
                },
            )

    @patch.object(DataProcessor, "load_data")
    def test_columns_annotations_empty_dict(self, mock_load_data):
        """Test empty columns_annotations raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Missing or None annotation columns: Start Time, End Time, Class",
        ):
            DataProcessor(
                prediction_directory_path="dummy_pred_path",
                annotation_directory_path="dummy_annot_path",
                columns_predictions={
                    "Start Time": "start_time_pred",
                    "End Time": "end_time_pred",
                    "Class": "class_pred",
                },
                columns_annotations={},
            )


class TestDataProcessorPrepareDataFrame:

    def setup_method(self):
        """Set up a DataProcessor instance for testing."""
        # Start patching
        self.patcher = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory"
        )
        self.mock_read_concat = self.patcher.start()
        # Mock empty DataFrames for predictions and annotations
        self.mock_read_concat.return_value = pd.DataFrame(
            {
                "Class": [],
                "Start Time": [],
                "End Time": [],
                "source_file": [],
                "recording_filename": [],
            }
        )

        self.dp = DataProcessor(
            prediction_directory_path="dummy_pred_path",
            annotation_directory_path="dummy_annot_path",
        )

    def teardown_method(self):
        """Stop patching."""
        self.patcher.stop()

    def test_with_recording_column(self):
        """Test DataFrame with 'Recording' column."""
        df = pd.DataFrame(
            {
                "Recording": ["path/to/recording1.wav", "path/to/recording2.wav"],
                "OtherColumn": [1, 2],
            }
        )
        result_df = self.dp._prepare_dataframe(df.copy(), prediction=True)
        expected_filenames = ["recording1", "recording2"]
        assert result_df["recording_filename"].tolist() == expected_filenames

    def test_with_source_file_column(self):
        """Test DataFrame without 'Recording' but with 'source_file'."""
        df = pd.DataFrame(
            {"source_file": ["file1.txt", "file2.txt"], "OtherColumn": [1, 2]}
        )
        result_df = self.dp._prepare_dataframe(df.copy(), prediction=False)
        expected_filenames = ["file1", "file2"]
        assert result_df["recording_filename"].tolist() == expected_filenames

    def test_without_recording_or_source_file(self):
        """Test DataFrame without 'Recording' and 'source_file'."""
        df = pd.DataFrame({"SomeColumn": [1, 2]})
        result_df = self.dp._prepare_dataframe(df.copy(), prediction=True)
        expected_filenames = ["", ""]
        assert result_df["recording_filename"].tolist() == expected_filenames

    def test_recording_column_with_none_values(self):
        """Test 'Recording' column containing None or NaN."""
        df = pd.DataFrame({"Recording": [None, float("nan"), "path/to/recording.wav"]})
        result_df = self.dp._prepare_dataframe(df.copy(), prediction=False)
        expected_filenames = [None, float("nan"), "recording"]
        pd.testing.assert_series_equal(
            result_df["recording_filename"],
            pd.Series(expected_filenames),
            check_names=False,
        )

    def test_custom_recording_column_mapping(self):
        """Test custom column mapping for 'Recording'."""
        df = pd.DataFrame(
            {
                "CustomRecording": ["path/to/rec1.wav", "path/to/rec2.wav"],
                "OtherColumn": [1, 2],
            }
        )
        self.dp.columns_predictions["Recording"] = "CustomRecording"
        result_df = self.dp._prepare_dataframe(df.copy(), prediction=True)
        expected_filenames = ["rec1", "rec2"]
        assert result_df["recording_filename"].tolist() == expected_filenames

    def test_recording_column_with_non_string_values(self):
        """Test 'Recording' column with non-string values."""
        df = pd.DataFrame({"Recording": [123, 456, "path/to/recording.wav"]})
        result_df = self.dp._prepare_dataframe(df.copy(), prediction=False)
        expected_filenames = [123, 456, "recording"]
        assert result_df["recording_filename"].tolist() == expected_filenames

    def test_source_file_with_none_values(self):
        """Test 'source_file' column containing None or NaN."""
        df = pd.DataFrame(
            {"source_file": [None, float("nan"), "file.txt"], "OtherColumn": [1, 2, 3]}
        )
        result_df = self.dp._prepare_dataframe(df.copy(), prediction=True)
        expected_filenames = [None, float("nan"), "file"]
        pd.testing.assert_series_equal(
            result_df["recording_filename"],
            pd.Series(expected_filenames),
            check_names=False,
        )

    def test_recording_column_with_empty_strings(self):
        """Test 'Recording' column containing empty strings."""
        df = pd.DataFrame({"Recording": ["", "", "path/to/recording.wav"]})
        result_df = self.dp._prepare_dataframe(df.copy(), prediction=False)
        expected_filenames = ["", "", "recording"]
        assert result_df["recording_filename"].tolist() == expected_filenames

    def test_complex_paths_in_recording_column(self):
        """Test 'Recording' column with complex paths."""
        df = pd.DataFrame(
            {"Recording": ["/a/b/c/d/e.wav", "C:\\folder\\subfolder\\file.wav"]}
        )
        result_df = self.dp._prepare_dataframe(df.copy(), prediction=True)
        expected_filenames = ["e", "file"]
        assert result_df["recording_filename"].tolist() == expected_filenames

    def test_both_recording_and_source_file_columns(self):
        """Test DataFrame with both 'Recording' and 'source_file' columns."""
        df = pd.DataFrame(
            {
                "Recording": ["path/to/recording1.wav", "path/to/recording2.wav"],
                "source_file": ["file1.txt", "file2.txt"],
                "OtherColumn": [1, 2],
            }
        )
        result_df = self.dp._prepare_dataframe(df.copy(), prediction=False)
        expected_filenames = ["recording1", "recording2"]
        assert result_df["recording_filename"].tolist() == expected_filenames


class TestDataProcessorProcessData:

    def setup_method(self):
        """Set up a DataProcessor instance for testing."""
        # Start patching
        self.patcher = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory"
        )
        self.mock_read_concat = self.patcher.start()
        # Mock empty DataFrames for predictions and annotations
        self.mock_read_concat.return_value = pd.DataFrame(
            {
                "Class": [],
                "Start Time": [],
                "End Time": [],
                "source_file": [],
                "recording_filename": [],
            }
        )

        self.dp = DataProcessor(
            prediction_directory_path="dummy_pred_path",
            annotation_directory_path="dummy_annot_path",
        )
        self.dp.process_recording = MagicMock(
            return_value=pd.DataFrame({"sample_data": [1, 2, 3]})
        )

    def teardown_method(self):
        """Stop patching."""
        self.patcher.stop()

    def test_empty_predictions_and_annotations(self):
        """Test with empty predictions and annotations DataFrames."""
        self.dp.predictions_df = pd.DataFrame(columns=["recording_filename"])
        self.dp.annotations_df = pd.DataFrame(columns=["recording_filename"])
        self.dp.process_data()
        assert self.dp.samples_df.empty

    def test_single_recording(self):
        """Test with single recording in predictions and annotations."""
        self.dp.predictions_df = pd.DataFrame(
            {"recording_filename": ["rec1"], "OtherColumn": [1]}
        )
        self.dp.annotations_df = pd.DataFrame(
            {"recording_filename": ["rec1"], "OtherColumn": [2]}
        )
        self.dp.process_data()
        assert not self.dp.samples_df.empty
        assert self.dp.process_recording.call_count == 1

    def test_multiple_recordings(self):
        """Test with multiple recordings."""
        self.dp.predictions_df = pd.DataFrame(
            {"recording_filename": ["rec1", "rec2"], "OtherColumn": [1, 2]}
        )
        self.dp.annotations_df = pd.DataFrame(
            {"recording_filename": ["rec1", "rec2"], "OtherColumn": [3, 4]}
        )
        self.dp.process_data()
        assert not self.dp.samples_df.empty
        assert self.dp.process_recording.call_count == 2

    def test_predictions_extra_recordings(self):
        """Test with recordings in predictions not in annotations."""
        self.dp.predictions_df = pd.DataFrame(
            {"recording_filename": ["rec1", "rec2"], "OtherColumn": [1, 2]}
        )
        self.dp.annotations_df = pd.DataFrame(
            {"recording_filename": ["rec1"], "OtherColumn": [3]}
        )
        self.dp.process_data()
        assert self.dp.process_recording.call_count == 2

    def test_annotations_extra_recordings(self):
        """Test with recordings in annotations not in predictions."""
        self.dp.predictions_df = pd.DataFrame(
            {"recording_filename": ["rec1"], "OtherColumn": [1]}
        )
        self.dp.annotations_df = pd.DataFrame(
            {"recording_filename": ["rec1", "rec2"], "OtherColumn": [3, 4]}
        )
        self.dp.process_data()
        assert self.dp.process_recording.call_count == 2

    def test_duplicate_recording_entries(self):
        """Test with duplicate recording filenames."""
        self.dp.predictions_df = pd.DataFrame(
            {"recording_filename": ["rec1", "rec1"], "OtherColumn": [1, 2]}
        )
        self.dp.annotations_df = pd.DataFrame(
            {"recording_filename": ["rec1"], "OtherColumn": [3]}
        )
        self.dp.process_data()
        assert self.dp.process_recording.call_count == 1

    def test_missing_recording_filename_in_predictions(self):
        """Test missing 'recording_filename' in predictions."""
        self.dp.predictions_df = pd.DataFrame({"OtherColumn": [1]})
        self.dp.annotations_df = pd.DataFrame(
            {"recording_filename": ["rec1"], "OtherColumn": [2]}
        )
        with pytest.raises(KeyError):
            self.dp.process_data()

    def test_missing_recording_filename_in_annotations(self):
        """Test missing 'recording_filename' in annotations."""
        self.dp.predictions_df = pd.DataFrame(
            {"recording_filename": ["rec1"], "OtherColumn": [1]}
        )
        self.dp.annotations_df = pd.DataFrame({"OtherColumn": [2]})
        with pytest.raises(KeyError):
            self.dp.process_data()

    def test_no_overlapping_recordings(self):
        """Test with no overlapping recording filenames."""
        self.dp.predictions_df = pd.DataFrame(
            {"recording_filename": ["rec1"], "OtherColumn": [1]}
        )
        self.dp.annotations_df = pd.DataFrame(
            {"recording_filename": ["rec2"], "OtherColumn": [2]}
        )
        self.dp.process_data()
        assert self.dp.process_recording.call_count == 2

    def test_large_number_of_recordings(self):
        """Test with a large number of recordings."""
        num_recordings = 1000
        self.dp.predictions_df = pd.DataFrame(
            {
                "recording_filename": [f"rec{i}" for i in range(num_recordings)],
                "OtherColumn": range(num_recordings),
            }
        )
        self.dp.annotations_df = pd.DataFrame(
            {
                "recording_filename": [f"rec{i}" for i in range(num_recordings)],
                "OtherColumn": range(num_recordings),
            }
        )
        self.dp.process_data()
        assert self.dp.process_recording.call_count == num_recordings


class TestDataProcessorProcessRecording:

    def setup_method(self):
        """Set up a DataProcessor instance for testing."""
        # Start patching
        self.patcher = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory"
        )
        self.mock_read_concat = self.patcher.start()
        # Mock empty DataFrames for predictions and annotations
        self.mock_read_concat.return_value = pd.DataFrame(
            {
                "Class": [],
                "Start Time": [],
                "End Time": [],
                "source_file": [],
                "recording_filename": [],
            }
        )

        self.dp = DataProcessor(
            prediction_directory_path="dummy_pred_path",
            annotation_directory_path="dummy_annot_path",
        )
        self.dp.classes = ("A", "B", "C")
        self.dp.sample_duration = 5
        self.dp.min_overlap = 0.5
        self.dp.recording_duration = 15  # For testing purposes

    def teardown_method(self):
        """Stop patching."""
        self.patcher.stop()

    def test_valid_predictions_and_annotations(self):
        """Test with valid predictions and annotations."""
        pred_df = pd.DataFrame(
            {"Class": ["A", "B"], "Start Time": [0, 5], "End Time": [5, 10]}
        )
        annot_df = pd.DataFrame(
            {"Class": ["A", "C"], "Start Time": [2, 7], "End Time": [7, 12]}
        )
        samples_df = self.dp.process_recording("rec1", pred_df, annot_df)
        assert not samples_df.empty
        assert len(samples_df) == 3  # 15 / 5 = 3 samples

    def test_empty_predictions_and_annotations(self):
        """Test with empty predictions and annotations."""
        pred_df = pd.DataFrame()
        annot_df = pd.DataFrame()
        samples_df = self.dp.process_recording("rec1", pred_df, annot_df)
        assert not samples_df.empty
        assert (
            (samples_df[[f"{cls}_confidence" for cls in self.dp.classes]] == 0)
            .all()
            .all()
        )
        assert (
            (samples_df[[f"{cls}_annotation" for cls in self.dp.classes]] == 0)
            .all()
            .all()
        )

    def test_only_predictions_present(self):
        """Test with only predictions present."""
        pred_df = pd.DataFrame({"Class": ["A"], "Start Time": [0], "End Time": [5]})
        annot_df = pd.DataFrame()
        samples_df = self.dp.process_recording("rec1", pred_df, annot_df)
        assert (
            samples_df["A_confidence"].iloc[0] == 0.0
        )  # Default confidence since 'Confidence' column missing
        assert samples_df["A_annotation"].iloc[0] == 0  # No annotations

    def test_only_annotations_present(self):
        """Test with only annotations present."""
        pred_df = pd.DataFrame()
        annot_df = pd.DataFrame({"Class": ["B"], "Start Time": [5], "End Time": [10]})
        samples_df = self.dp.process_recording("rec1", pred_df, annot_df)
        assert samples_df["B_annotation"].iloc[1] == 1
        assert samples_df["B_confidence"].iloc[1] == 0.0  # No predictions

    def test_no_overlap_between_pred_and_annot(self):
        """Test with predictions and annotations having no overlap."""
        pred_df = pd.DataFrame({"Class": ["A"], "Start Time": [0], "End Time": [5]})
        annot_df = pd.DataFrame({"Class": ["A"], "Start Time": [10], "End Time": [15]})
        samples_df = self.dp.process_recording("rec1", pred_df, annot_df)
        assert samples_df["A_confidence"].iloc[0] == 0.0  # Default confidence
        assert samples_df["A_annotation"].iloc[0] == 0  # No overlap in first sample
        assert samples_df["A_annotation"].iloc[1] == 0  # No overlap in second sample
        assert samples_df["A_annotation"].iloc[2] == 1  # Overlap in third sample

    def test_invalid_times_in_predictions(self):
        """Test with invalid times in predictions."""
        pred_df = pd.DataFrame({"Class": ["A"], "Start Time": [-5], "End Time": [-1]})
        annot_df = pd.DataFrame()
        samples_df = self.dp.process_recording("rec1", pred_df, annot_df)
        # Should handle gracefully without exceptions
        assert not samples_df.empty

    def test_negative_recording_duration(self):
        """Test with negative recording duration."""
        self.dp.recording_duration = -10
        pred_df = pd.DataFrame()
        annot_df = pd.DataFrame()
        samples_df = self.dp.process_recording("rec1", pred_df, annot_df)
        assert samples_df.empty

    def test_custom_sample_duration_and_min_overlap(self):
        """Test with custom sample_duration and min_overlap."""
        self.dp.sample_duration = 3
        self.dp.min_overlap = 0.1
        pred_df = pd.DataFrame({"Class": ["A"], "Start Time": [1], "End Time": [2]})
        annot_df = pd.DataFrame(
            {"Class": ["A"], "Start Time": [1.5], "End Time": [2.5]}
        )
        samples_df = self.dp.process_recording("rec1", pred_df, annot_df)
        assert len(samples_df) == 5  # 15 / 3 = 5 samples
        # Check if overlaps are correctly calculated

    def test_classes_not_in_self_classes(self):
        """Test with classes not in self.classes."""
        pred_df = pd.DataFrame({"Class": ["D"], "Start Time": [0], "End Time": [5]})
        annot_df = pd.DataFrame({"Class": ["D"], "Start Time": [0], "End Time": [5]})
        samples_df = self.dp.process_recording("rec1", pred_df, annot_df)
        # Since 'D' is not in self.classes, it should be skipped
        assert (
            (samples_df[[f"{cls}_confidence" for cls in self.dp.classes]] == 0)
            .all()
            .all()
        )
        assert (
            (samples_df[[f"{cls}_annotation" for cls in self.dp.classes]] == 0)
            .all()
            .all()
        )

    def test_zero_recording_duration(self):
        """Test with zero recording duration."""
        self.dp.recording_duration = 0
        pred_df = pd.DataFrame()
        annot_df = pd.DataFrame()
        samples_df = self.dp.process_recording("rec1", pred_df, annot_df)
        assert samples_df.empty


class TestDetermineFileDuration:
    def setup_method(self):
        """Set up a DataProcessor instance for testing."""
        # Start patching
        self.patcher = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory"
        )
        self.mock_read_concat = self.patcher.start()
        # Mock the function to return DataFrames with expected columns
        self.mock_read_concat.return_value = pd.DataFrame(
            columns=[
                "Class",
                "Start Time",
                "End Time",
                "Confidence",
                "source_file",
                "recording_filename",
            ]
        )
        self.dp = DataProcessor(
            prediction_directory_path="dummy_path",
            annotation_directory_path="dummy_path",
        )

    def teardown_method(self):
        """Stop patching."""
        self.patcher.stop()

    def test_recording_duration_set(self):
        """Test when recording_duration is set."""
        self.dp.recording_duration = 120.0  # 2 minutes
        pred_df = pd.DataFrame()
        annot_df = pd.DataFrame()
        duration = self.dp.determine_file_duration(pred_df, annot_df)
        assert duration == 120.0

    def test_recording_duration_set_with_data(self):
        """Test when recording_duration is set and DataFrames have data."""
        self.dp.recording_duration = 120.0
        pred_df = pd.DataFrame({"Start Time": [10], "End Time": [20]})
        annot_df = pd.DataFrame({"Start Time": [30], "End Time": [40]})
        duration = self.dp.determine_file_duration(pred_df, annot_df)
        assert duration == 120.0

    def test_duration_column_in_predictions(self):
        """Test when 'Duration' column is in predictions DataFrame."""
        self.dp.recording_duration = None
        pred_df = pd.DataFrame(
            {"Start Time": [0], "End Time": [50], "Duration": [100.0]}
        )
        annot_df = pd.DataFrame()
        duration = self.dp.determine_file_duration(pred_df, annot_df)
        assert duration == 100.0

    def test_duration_column_in_annotations(self):
        """Test when 'Duration' column is in annotations DataFrame."""
        self.dp.recording_duration = None
        pred_df = pd.DataFrame()
        annot_df = pd.DataFrame(
            {"Start Time": [0], "End Time": [50], "Duration": [90.0]}
        )
        duration = self.dp.determine_file_duration(pred_df, annot_df)
        assert duration == 90.0

    def test_max_end_time(self):
        """Test when 'Duration' columns are missing, use max of 'End Time'."""
        self.dp.recording_duration = None
        pred_df = pd.DataFrame({"Start Time": [10], "End Time": [80]})
        annot_df = pd.DataFrame({"Start Time": [20], "End Time": [100]})
        duration = self.dp.determine_file_duration(pred_df, annot_df)
        assert duration == 100.0

    def test_empty_dataframes(self):
        """Test when pred_df and annot_df are empty; duration should be zero."""
        self.dp.recording_duration = None
        pred_df = pd.DataFrame()
        annot_df = pd.DataFrame()
        duration = self.dp.determine_file_duration(pred_df, annot_df)
        assert duration == 0

    def test_null_duration_columns(self):
        """Test when 'Duration' columns are present but all null."""
        self.dp.recording_duration = None
        pred_df = pd.DataFrame(
            {"Start Time": [10], "End Time": [20], "Duration": [None]}
        )
        annot_df = pd.DataFrame(
            {"Start Time": [30], "End Time": [40], "Duration": [None]}
        )
        duration = self.dp.determine_file_duration(pred_df, annot_df)
        assert duration == 40.0

    def test_invalid_times(self):
        """Test when pred_df has negative times."""
        self.dp.recording_duration = None
        pred_df = pd.DataFrame({"Start Time": [-10], "End Time": [-5]})
        annot_df = pd.DataFrame()
        duration = self.dp.determine_file_duration(pred_df, annot_df)
        # Since negative durations are not meaningful, expect duration to be 0
        assert duration == 0

    def test_missing_end_time_column(self):
        """Test when 'End Time' column is missing."""
        self.dp.recording_duration = None
        pred_df = pd.DataFrame({"Start Time": [10]})
        annot_df = pd.DataFrame()
        duration = self.dp.determine_file_duration(pred_df, annot_df)
        assert duration == 0

    def test_mixed_null_and_non_null_duration_columns(self):
        """Test with mixed null and non-null 'Duration' values."""
        self.dp.recording_duration = None
        pred_df = pd.DataFrame(
            {"Start Time": [10], "End Time": [20], "Duration": [None]}
        )
        annot_df = pd.DataFrame(
            {"Start Time": [30], "End Time": [40], "Duration": [50.0]}
        )
        duration = self.dp.determine_file_duration(pred_df, annot_df)
        assert duration == 50.0


class TestInitializeSamples:
    def setup_method(self):
        """Set up a DataProcessor instance for testing."""
        # Start patching
        self.patcher = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory"
        )
        self.mock_read_concat = self.patcher.start()
        # Mock the function to return DataFrames with expected columns
        self.mock_read_concat.return_value = pd.DataFrame(
            columns=[
                "Class",
                "Start Time",
                "End Time",
                "Confidence",
                "source_file",
                "recording_filename",
            ]
        )
        self.dp = DataProcessor(
            prediction_directory_path="dummy_path",
            annotation_directory_path="dummy_path",
        )
        self.dp.classes = ("A", "B", "C")
        self.dp.sample_duration = 5

    def teardown_method(self):
        """Stop patching."""
        self.patcher.stop()

    def test_file_duration_zero(self):
        """Test when file_duration is zero."""
        samples_df = self.dp.initialize_samples("rec1", 0)
        assert samples_df.empty

    def test_file_duration_negative(self):
        """Test when file_duration is negative."""
        samples_df = self.dp.initialize_samples("rec1", -10)
        assert samples_df.empty

    def test_sample_duration_larger_than_file_duration(self):
        """Test when sample_duration is larger than file_duration."""
        self.dp.sample_duration = 10
        samples_df = self.dp.initialize_samples("rec1", 5)
        assert len(samples_df) == 1
        assert samples_df.iloc[0]["start_time"] == 0
        assert samples_df.iloc[0]["end_time"] == 5

    def test_file_duration_exact_multiple(self):
        """Test when file_duration is an exact multiple of sample_duration."""
        self.dp.sample_duration = 5
        samples_df = self.dp.initialize_samples("rec1", 15)
        assert len(samples_df) == 3
        assert samples_df.iloc[2]["start_time"] == 10
        assert samples_df.iloc[2]["end_time"] == 15

    def test_file_duration_not_exact_multiple(self):
        """Test when file_duration is not an exact multiple."""
        self.dp.sample_duration = 4
        samples_df = self.dp.initialize_samples("rec1", 10)
        assert len(samples_df) == 3
        assert samples_df.iloc[2]["start_time"] == 8
        assert samples_df.iloc[2]["end_time"] == 10

    def test_empty_classes(self):
        """Test when self.classes is empty."""
        self.dp.classes = ()
        samples_df = self.dp.initialize_samples("rec1", 10)
        assert "A_confidence" not in samples_df.columns
        assert "A_annotation" not in samples_df.columns

    def test_multiple_classes(self):
        """Test when self.classes has multiple classes."""
        self.dp.classes = ("A", "B")
        samples_df = self.dp.initialize_samples("rec1", 5)
        assert all(
            col in samples_df.columns
            for col in ["A_confidence", "B_confidence", "A_annotation", "B_annotation"]
        )

    def test_empty_recording_filename(self):
        """Test when recording_filename is an empty string."""
        samples_df = self.dp.initialize_samples("", 10)
        assert (samples_df["filename"] == "").all()

    def test_sample_duration_float(self):
        """Test when sample_duration is a float value."""
        self.dp.sample_duration = 2.5
        samples_df = self.dp.initialize_samples("rec1", 10)
        expected_start_times = [0, 2.5, 5.0, 7.5]
        expected_end_times = [2.5, 5.0, 7.5, 10.0]
        assert samples_df["start_time"].tolist() == expected_start_times
        assert samples_df["end_time"].tolist() == expected_end_times

    def test_correct_intervals(self):
        """Test that start_time and end_time are correctly calculated."""
        self.dp.sample_duration = 3
        samples_df = self.dp.initialize_samples("rec1", 10)
        expected_start_times = [0, 3, 6, 9]
        expected_end_times = [3, 6, 9, 10]
        assert samples_df["start_time"].tolist() == expected_start_times
        assert samples_df["end_time"].tolist() == expected_end_times


class TestUpdateSamplesWithPredictions:
    def setup_method(self):
        """Set up a DataProcessor instance and samples DataFrame for testing."""
        # Start patching
        self.patcher = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory"
        )
        self.mock_read_concat = self.patcher.start()
        # Mock the function to return DataFrames with expected columns
        self.mock_read_concat.return_value = pd.DataFrame(
            columns=[
                "Class",
                "Start Time",
                "End Time",
                "Confidence",
                "source_file",
                "recording_filename",
            ]
        )
        self.dp = DataProcessor(
            prediction_directory_path="dummy_path",
            annotation_directory_path="dummy_path",
        )
        self.dp.classes = ("A", "B", "C")
        self.dp.sample_duration = 5
        self.dp.min_overlap = 0.5

        # Initialize samples_df
        self.samples_df = pd.DataFrame(
            {
                "filename": ["rec1"] * 3,
                "sample_index": [0, 1, 2],
                "start_time": [0, 5, 10],
                "end_time": [5, 10, 15],
                "A_confidence": [0.0, 0.0, 0.0],
                "B_confidence": [0.0, 0.0, 0.0],
                "C_confidence": [0.0, 0.0, 0.0],
                "A_annotation": [0, 0, 0],
                "B_annotation": [0, 0, 0],
                "C_annotation": [0, 0, 0],
            }
        )

    def teardown_method(self):
        """Stop patching."""
        self.patcher.stop()

    def test_single_prediction_overlapping_one_sample(self):
        """Test single prediction overlapping one sample."""
        pred_df = pd.DataFrame(
            {"Class": ["A"], "Start Time": [1], "End Time": [4], "Confidence": [0.8]}
        )
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert self.samples_df.loc[0, "A_confidence"] == 0.8

    def test_single_prediction_overlapping_multiple_samples(self):
        """Test single prediction overlapping multiple samples."""
        pred_df = pd.DataFrame(
            {"Class": ["A"], "Start Time": [3], "End Time": [8], "Confidence": [0.6]}
        )
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert self.samples_df.loc[0, "A_confidence"] == 0.6
        assert self.samples_df.loc[1, "A_confidence"] == 0.6

    def test_multiple_predictions_same_sample(self):
        """Test multiple predictions overlapping the same sample for the same class."""
        pred_df = pd.DataFrame(
            {
                "Class": ["A", "A"],
                "Start Time": [1, 1],
                "End Time": [4, 4],
                "Confidence": [0.5, 0.7],
            }
        )
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert self.samples_df.loc[0, "A_confidence"] == 0.7  # Max confidence

    def test_predictions_classes_not_in_self_classes(self):
        """Test predictions with classes not in self.classes."""
        pred_df = pd.DataFrame(
            {"Class": ["D"], "Start Time": [0], "End Time": [5], "Confidence": [0.9]}
        )
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert (
            (self.samples_df[["A_confidence", "B_confidence", "C_confidence"]] == 0.0)
            .all()
            .all()
        )

    def test_predictions_missing_confidence(self):
        """Test predictions missing 'Confidence' column."""
        pred_df = pd.DataFrame(
            {
                "Class": ["A"],
                "Start Time": [1],
                "End Time": [4],
                # 'Confidence' missing
            }
        )
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert self.samples_df.loc[0, "A_confidence"] == 0.0  # Default confidence

    def test_predictions_with_negative_times(self):
        """Test predictions with negative times."""
        pred_df = pd.DataFrame(
            {"Class": ["A"], "Start Time": [-3], "End Time": [2], "Confidence": [0.5]}
        )
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert self.samples_df.loc[0, "A_confidence"] == 0.5

    def test_predictions_no_overlap(self):
        """Test predictions that do not overlap any samples."""
        pred_df = pd.DataFrame(
            {"Class": ["A"], "Start Time": [15], "End Time": [20], "Confidence": [0.9]}
        )
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert (self.samples_df["A_confidence"] == 0.0).all()

    def test_predictions_with_different_min_overlap(self):
        """Test predictions with different min_overlap values."""
        pred_df = pd.DataFrame(
            {
                "Class": ["A"],
                "Start Time": [4.6],
                "End Time": [5.1],
                "Confidence": [0.8],
            }
        )
        # With min_overlap 0.5, should not overlap
        self.dp.min_overlap = 0.5
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert self.samples_df.loc[0, "A_confidence"] == 0.0

        # With min_overlap 0.0, should overlap
        self.dp.min_overlap = 0.0
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert self.samples_df.loc[0, "A_confidence"] == 0.8

    def test_predictions_overlapping_different_classes(self):
        """Test predictions overlapping different classes."""
        pred_df = pd.DataFrame(
            {
                "Class": ["A", "B"],
                "Start Time": [1, 6],
                "End Time": [4, 9],
                "Confidence": [0.7, 0.9],
            }
        )
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert self.samples_df.loc[0, "A_confidence"] == 0.7
        assert self.samples_df.loc[1, "B_confidence"] == 0.9

    def test_multiple_predictions_overlapping_multiple_samples(self):
        """Test multiple predictions overlapping multiple samples."""
        pred_df = pd.DataFrame(
            {
                "Class": ["A", "A"],
                "Start Time": [2, 7],
                "End Time": [6, 12],
                "Confidence": [0.5, 0.6],
            }
        )
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert self.samples_df.loc[0, "A_confidence"] == 0.5
        assert self.samples_df.loc[1, "A_confidence"] == 0.6
        assert self.samples_df.loc[2, "A_confidence"] == 0.6

    def test_empty_predictions_dataframe(self):
        """Test when pred_df is empty."""
        pred_df = pd.DataFrame(
            columns=["Class", "Start Time", "End Time", "Confidence"]
        )
        self.dp.update_samples_with_predictions(pred_df, self.samples_df)
        assert (
            (self.samples_df[["A_confidence", "B_confidence", "C_confidence"]] == 0.0)
            .all()
            .all()
        )


class TestUpdateSamplesWithAnnotations:
    def setup_method(self):
        """Set up a DataProcessor instance and samples DataFrame for testing."""
        # Start patching
        self.patcher = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory"
        )
        self.mock_read_concat = self.patcher.start()
        # Mock the function to return DataFrames with expected columns
        self.mock_read_concat.return_value = pd.DataFrame(
            columns=[
                "Class",
                "Start Time",
                "End Time",
                "Confidence",
                "source_file",
                "recording_filename",
            ]
        )
        self.dp = DataProcessor(
            prediction_directory_path="dummy_path",
            annotation_directory_path="dummy_path",
        )
        self.dp.classes = ("A", "B", "C")
        self.dp.sample_duration = 5
        self.dp.min_overlap = 0.5

        # Initialize samples_df
        self.samples_df = pd.DataFrame(
            {
                "filename": ["rec1"] * 3,
                "sample_index": [0, 1, 2],
                "start_time": [0, 5, 10],
                "end_time": [5, 10, 15],
                "A_confidence": [0.0, 0.0, 0.0],
                "B_confidence": [0.0, 0.0, 0.0],
                "C_confidence": [0.0, 0.0, 0.0],
                "A_annotation": [0, 0, 0],
                "B_annotation": [0, 0, 0],
                "C_annotation": [0, 0, 0],
            }
        )

    def teardown_method(self):
        """Stop patching."""
        self.patcher.stop()

    def test_single_annotation_overlapping_one_sample(self):
        """Test single annotation overlapping one sample."""
        annot_df = pd.DataFrame({"Class": ["A"], "Start Time": [1], "End Time": [4]})
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert self.samples_df.loc[0, "A_annotation"] == 1

    def test_single_annotation_overlapping_multiple_samples(self):
        """Test single annotation overlapping multiple samples."""
        annot_df = pd.DataFrame({"Class": ["A"], "Start Time": [3], "End Time": [8]})
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert self.samples_df.loc[0, "A_annotation"] == 1
        assert self.samples_df.loc[1, "A_annotation"] == 1

    def test_annotations_classes_not_in_self_classes(self):
        """Test annotations with classes not in self.classes."""
        annot_df = pd.DataFrame({"Class": ["D"], "Start Time": [0], "End Time": [5]})
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert (
            (self.samples_df[["A_annotation", "B_annotation", "C_annotation"]] == 0)
            .all()
            .all()
        )

    def test_annotations_with_negative_times(self):
        """Test annotations with negative times."""
        annot_df = pd.DataFrame({"Class": ["A"], "Start Time": [-3], "End Time": [2]})
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert self.samples_df.loc[0, "A_annotation"] == 1

    def test_annotations_no_overlap(self):
        """Test annotations that do not overlap any samples."""
        annot_df = pd.DataFrame({"Class": ["A"], "Start Time": [15], "End Time": [20]})
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert (self.samples_df["A_annotation"] == 0).all()

    def test_annotations_with_different_min_overlap(self):
        """Test annotations with different min_overlap values."""
        annot_df = pd.DataFrame(
            {"Class": ["A"], "Start Time": [4.6], "End Time": [5.1]}
        )
        # With min_overlap 0.5, should not overlap
        self.dp.min_overlap = 0.5
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert self.samples_df.loc[0, "A_annotation"] == 0

        # With min_overlap 0.0, should overlap
        self.dp.min_overlap = 0.0
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert self.samples_df.loc[0, "A_annotation"] == 1

    def test_multiple_annotations_same_sample(self):
        """Test multiple annotations overlapping the same sample for the same class."""
        annot_df = pd.DataFrame(
            {"Class": ["A", "A"], "Start Time": [1, 2], "End Time": [4, 5]}
        )
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert self.samples_df.loc[0, "A_annotation"] == 1  # Should be set to 1

    def test_multiple_annotations_overlapping_multiple_samples(self):
        """Test multiple annotations overlapping multiple samples."""
        annot_df = pd.DataFrame(
            {"Class": ["A", "A"], "Start Time": [2, 7], "End Time": [6, 12]}
        )
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert self.samples_df.loc[0, "A_annotation"] == 1
        assert self.samples_df.loc[1, "A_annotation"] == 1
        assert self.samples_df.loc[2, "A_annotation"] == 1

    def test_empty_annotations_dataframe(self):
        """Test when annot_df is empty."""
        annot_df = pd.DataFrame(columns=["Class", "Start Time", "End Time"])
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert (
            (self.samples_df[["A_annotation", "B_annotation", "C_annotation"]] == 0)
            .all()
            .all()
        )

    def test_annotations_overlapping_different_classes(self):
        """Test annotations overlapping different classes."""
        annot_df = pd.DataFrame(
            {"Class": ["A", "B"], "Start Time": [1, 6], "End Time": [4, 9]}
        )
        self.dp.update_samples_with_annotations(annot_df, self.samples_df)
        assert self.samples_df.loc[0, "A_annotation"] == 1
        assert self.samples_df.loc[1, "B_annotation"] == 1


class TestCreateTensors:
    def setup_method(self):
        """Set up a DataProcessor instance for testing."""
        # Mock the file reading functions to prevent actual file I/O
        self.patcher_pred = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory",
            return_value=pd.DataFrame(
                {
                    "Class": [],  # Required column
                    "Start Time": [],
                    "End Time": [],
                    "Confidence": [],
                    "source_file": [],
                    "recording_filename": [],
                }
            ),
        )
        self.mock_read_concat_pred = self.patcher_pred.start()

        self.patcher_annot = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory",
            return_value=pd.DataFrame(
                {
                    "Class": [],  # Required column
                    "Start Time": [],
                    "End Time": [],
                    "source_file": [],
                    "recording_filename": [],
                }
            ),
        )
        self.mock_read_concat_annot = self.patcher_annot.start()

        self.dp = DataProcessor(
            prediction_directory_path="dummy_path",
            annotation_directory_path="dummy_path",
        )

    def teardown_method(self):
        """Stop patching."""
        self.patcher_pred.stop()
        self.patcher_annot.stop()

    def test_empty_samples_df(self):
        """Test when samples_df is empty."""
        self.dp.classes = ("A", "B")
        self.dp.samples_df = pd.DataFrame()
        self.dp.create_tensors()
        assert self.dp.prediction_tensors.shape == (0, 2)
        assert self.dp.label_tensors.shape == (0, 2)

    def test_single_sample_single_class(self):
        """Test with one sample and one class."""
        self.dp.classes = ("A",)
        self.dp.samples_df = pd.DataFrame(
            {
                "A_confidence": [0.8],
                "A_annotation": [1],
            }
        )
        self.dp.create_tensors()
        assert self.dp.prediction_tensors.shape == (1, 1)
        assert self.dp.label_tensors.shape == (1, 1)
        assert self.dp.prediction_tensors[0, 0] == 0.8
        assert self.dp.label_tensors[0, 0] == 1

    def test_multiple_samples_multiple_classes(self):
        """Test with multiple samples and classes."""
        self.dp.classes = ("A", "B")
        self.dp.samples_df = pd.DataFrame(
            {
                "A_confidence": [0.8, 0.5],
                "B_confidence": [0.2, 0.7],
                "A_annotation": [1, 0],
                "B_annotation": [0, 1],
            }
        )
        self.dp.create_tensors()
        assert self.dp.prediction_tensors.shape == (2, 2)
        assert self.dp.label_tensors.shape == (2, 2)
        expected_pred = np.array([[0.8, 0.2], [0.5, 0.7]], dtype=np.float32)
        expected_label = np.array([[1, 0], [0, 1]], dtype=np.int64)
        np.testing.assert_array_equal(self.dp.prediction_tensors, expected_pred)
        np.testing.assert_array_equal(self.dp.label_tensors, expected_label)

    def test_missing_confidence_columns(self):
        """Test when samples_df is missing _confidence columns."""
        self.dp.classes = ("A",)
        self.dp.samples_df = pd.DataFrame(
            {
                "A_annotation": [1],
            }
        )
        with pytest.raises(KeyError):
            self.dp.create_tensors()

    def test_missing_annotation_columns(self):
        """Test when samples_df is missing _annotation columns."""
        self.dp.classes = ("A",)
        self.dp.samples_df = pd.DataFrame(
            {
                "A_confidence": [0.8],
            }
        )
        with pytest.raises(KeyError):
            self.dp.create_tensors()

    def test_nan_in_confidence(self):
        """Test when confidence columns contain NaN values."""
        self.dp.classes = ("A",)
        self.dp.samples_df = pd.DataFrame(
            {
                "A_confidence": [np.nan],
                "A_annotation": [1],
            }
        )
        with pytest.raises(ValueError, match="NaN values found in confidence columns."):
            self.dp.create_tensors()

    def test_nan_in_annotations(self):
        """Test when annotation columns contain NaN values."""
        self.dp.classes = ("A",)
        self.dp.samples_df = pd.DataFrame(
            {
                "A_confidence": [0.8],
                "A_annotation": [np.nan],
            }
        )
        with pytest.raises(ValueError, match="NaN values found in annotation columns"):
            self.dp.create_tensors()

    def test_empty_classes(self):
        """Test when self.classes is empty."""
        self.dp.classes = ()
        self.dp.samples_df = pd.DataFrame()
        self.dp.create_tensors()
        assert self.dp.prediction_tensors.shape == (0, 0)
        assert self.dp.label_tensors.shape == (0, 0)

    def test_non_numeric_confidence(self):
        """Test when confidence columns contain non-numeric data."""
        self.dp.classes = ("A",)
        self.dp.samples_df = pd.DataFrame(
            {
                "A_confidence": ["high"],
                "A_annotation": [1],
            }
        )
        with pytest.raises(ValueError):
            self.dp.create_tensors()

    def test_large_number_of_samples_and_classes(self):
        """Test with a large samples_df."""
        num_samples = 1000
        num_classes = 10
        classes = tuple(f"Class_{i}" for i in range(num_classes))
        self.dp.classes = classes
        data = {}
        for cls in classes:
            data[f"{cls}_confidence"] = np.random.rand(num_samples)
            data[f"{cls}_annotation"] = np.random.randint(0, 2, size=num_samples)
        self.dp.samples_df = pd.DataFrame(data)
        self.dp.create_tensors()
        assert self.dp.prediction_tensors.shape == (num_samples, num_classes)
        assert self.dp.label_tensors.shape == (num_samples, num_classes)


class TestGetColumnName:
    def setup_method(self):
        """Set up a DataProcessor instance for testing."""
        # Mock the file reading functions to prevent actual file I/O
        self.patcher_pred = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory",
            return_value=pd.DataFrame(
                {
                    "Class": [],  # Required column
                    "Start Time": [],
                    "End Time": [],
                    "Confidence": [],
                    "source_file": [],
                    "recording_filename": [],
                }
            ),
        )
        self.mock_read_concat_pred = self.patcher_pred.start()

        self.patcher_annot = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory",
            return_value=pd.DataFrame(
                {
                    "Class": [],  # Required column
                    "Start Time": [],
                    "End Time": [],
                    "source_file": [],
                    "recording_filename": [],
                }
            ),
        )
        self.mock_read_concat_annot = self.patcher_annot.start()

        self.dp = DataProcessor(
            prediction_directory_path="dummy_path",
            annotation_directory_path="dummy_path",
        )

    def teardown_method(self):
        """Stop patching."""
        self.patcher_pred.stop()
        self.patcher_annot.stop()

    def test_default_mapping_prediction(self):
        """Test default mapping for predictions."""
        assert self.dp.get_column_name("Class", prediction=True) == "Class"
        assert self.dp.get_column_name("Start Time", prediction=True) == "Start Time"
        assert self.dp.get_column_name("End Time", prediction=True) == "End Time"

    def test_default_mapping_annotation(self):
        """Test default mapping for annotations."""
        assert self.dp.get_column_name("Class", prediction=False) == "Class"
        assert self.dp.get_column_name("Start Time", prediction=False) == "Start Time"
        assert self.dp.get_column_name("End Time", prediction=False) == "End Time"

    def test_custom_columns_predictions(self):
        """Test custom columns_predictions mapping."""
        self.dp.columns_predictions = {"Class": "Pred_Class"}
        assert self.dp.get_column_name("Class", prediction=True) == "Pred_Class"

    def test_custom_columns_annotations(self):
        """Test custom columns_annotations mapping."""
        self.dp.columns_annotations = {"Class": "Annot_Class"}
        assert self.dp.get_column_name("Class", prediction=False) == "Annot_Class"

    def test_field_not_in_mappings(self):
        """Test field name not in mappings."""
        field_name = "Unknown Field"
        assert self.dp.get_column_name(field_name, prediction=True) == "Unknown Field"
        assert self.dp.get_column_name(field_name, prediction=False) == "Unknown Field"

    def test_empty_columns_predictions(self):
        """Test empty columns_predictions dict."""
        self.dp.columns_predictions = {}
        assert self.dp.get_column_name("Class", prediction=True) == "Class"

    def test_empty_columns_annotations(self):
        """Test empty columns_annotations dict."""
        self.dp.columns_annotations = {}
        assert self.dp.get_column_name("Class", prediction=False) == "Class"

    def test_field_name_is_none(self):
        """Test when field_name is None."""
        with pytest.raises(TypeError, match="field_name cannot be None."):
            self.dp.get_column_name(None, prediction=True)

    def test_prediction_is_none(self):
        """Test when prediction is None."""
        with pytest.raises(TypeError, match="prediction parameter cannot be None."):
            self.dp.get_column_name("Class", prediction=None)

    def test_field_name_empty_string(self):
        """Test when field_name is an empty string."""
        assert self.dp.get_column_name("", prediction=True) == ""


class TestGetSampleData:
    def setup_method(self):
        """Set up a DataProcessor instance for testing."""
        # Mock the file reading functions to prevent actual file I/O
        self.patcher_pred = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory",
            return_value=pd.DataFrame(
                {
                    "Class": [],  # Required column
                    "Start Time": [],
                    "End Time": [],
                    "Confidence": [],
                    "source_file": [],
                    "recording_filename": [],
                }
            ),
        )
        self.mock_read_concat_pred = self.patcher_pred.start()

        self.patcher_annot = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory",
            return_value=pd.DataFrame(
                {
                    "Class": [],  # Required column
                    "Start Time": [],
                    "End Time": [],
                    "source_file": [],
                    "recording_filename": [],
                }
            ),
        )
        self.mock_read_concat_annot = self.patcher_annot.start()

        self.dp = DataProcessor(
            prediction_directory_path="dummy_path",
            annotation_directory_path="dummy_path",
        )

    def teardown_method(self):
        """Stop patching."""
        self.patcher_pred.stop()
        self.patcher_annot.stop()

    def test_empty_samples_df(self):
        """Test when samples_df is empty."""
        self.dp.samples_df = pd.DataFrame()
        sample_data = self.dp.get_sample_data()
        assert sample_data.empty

    def test_samples_df_with_data(self):
        """Test when samples_df has data."""
        self.dp.samples_df = pd.DataFrame(
            {
                "filename": ["rec1"],
                "start_time": [0],
                "end_time": [5],
                "A_confidence": [0.8],
                "A_annotation": [1],
            }
        )
        sample_data = self.dp.get_sample_data()
        pd.testing.assert_frame_equal(sample_data, self.dp.samples_df)

    def test_modifying_returned_df_does_not_affect_samples_df(self):
        """Test that modifying returned DataFrame does not affect samples_df."""
        self.dp.samples_df = pd.DataFrame({"A_confidence": [0.8]})
        sample_data = self.dp.get_sample_data()
        sample_data["A_confidence"] = [0.5]
        assert self.dp.samples_df["A_confidence"][0] == 0.8

    def test_samples_df_with_nan_values(self):
        """Test when samples_df contains NaN values."""
        self.dp.samples_df = pd.DataFrame({"A_confidence": [np.nan]})
        sample_data = self.dp.get_sample_data()
        assert pd.isna(sample_data["A_confidence"][0])

    def test_samples_df_columns(self):
        """Test that columns are preserved."""
        columns = ["filename", "start_time", "end_time", "A_confidence", "A_annotation"]
        self.dp.samples_df = pd.DataFrame(columns=columns)
        sample_data = self.dp.get_sample_data()
        assert list(sample_data.columns) == columns

    def test_samples_df_large_data(self):
        """Test with a large samples_df."""
        num_samples = 1000
        self.dp.samples_df = pd.DataFrame({"A_confidence": np.random.rand(num_samples)})
        sample_data = self.dp.get_sample_data()
        pd.testing.assert_frame_equal(sample_data, self.dp.samples_df)

    def test_samples_df_with_custom_index(self):
        """Test that index is preserved."""
        self.dp.samples_df = pd.DataFrame({"A_confidence": [0.8]}, index=[10])
        sample_data = self.dp.get_sample_data()
        assert sample_data.index[0] == 10

    def test_samples_df_with_different_dtypes(self):
        """Test that data types are preserved."""
        self.dp.samples_df = pd.DataFrame(
            {
                "A_confidence": [0.8],
                "A_annotation": [1],
                "filename": ["rec1"],
                "start_time": [0.0],
            }
        )
        sample_data = self.dp.get_sample_data()
        assert sample_data.dtypes.equals(self.dp.samples_df.dtypes)

    def test_modifications_after_get_sample_data(self):
        """Test that modifications to samples_df after get_sample_data do not affect returned DataFrame."""
        self.dp.samples_df = pd.DataFrame({"A_confidence": [0.8]})
        sample_data = self.dp.get_sample_data()
        self.dp.samples_df["A_confidence"] = [0.5]
        assert sample_data["A_confidence"][0] == 0.8

    def test_samples_df_with_multiindex(self):
        """Test when samples_df has a MultiIndex."""
        index = pd.MultiIndex.from_tuples([("rec1", 0)])
        self.dp.samples_df = pd.DataFrame({"A_confidence": [0.8]}, index=index)
        sample_data = self.dp.get_sample_data()
        assert sample_data.index.equals(index)


class TestGetFilteredTensors:
    def setup_method(self):
        """Set up a DataProcessor instance for testing."""
        # Mock the file reading functions to prevent actual file I/O
        self.patcher_pred = patch(
            "bapat.preprocessing.data_processor.read_and_concatenate_files_in_directory",
            side_effect=[
                pd.DataFrame(
                    {
                        "Class": ["A", "B", "A", "B"],
                        "Start Time": [0, 1, 2, 3],
                        "End Time": [1, 2, 3, 4],
                        "Confidence": [0.8, 0.5, 0.6, 0.7],
                        "source_file": ["file1", "file2", "file1", "file2"],
                        "recording_filename": ["rec1", "rec2", "rec1", "rec2"],
                    }
                ),
                pd.DataFrame(
                    {
                        "Class": ["A", "B", "A", "B"],
                        "Start Time": [0, 1, 2, 3],
                        "End Time": [1, 2, 3, 4],
                        "source_file": ["file1", "file2", "file1", "file2"],
                        "recording_filename": ["rec1", "rec2", "rec1", "rec2"],
                    }
                ),
            ],
        )
        self.mock_read_concat_pred = self.patcher_pred.start()

        # Note: No need to patch annotations separately as side_effect handles both calls.

        self.dp = DataProcessor(
            prediction_directory_path="dummy_path",
            annotation_directory_path="dummy_path",
        )
        # Create a samples_df for testing
        self.dp.samples_df = pd.DataFrame(
            {
                "filename": ["rec1", "rec2", "rec1", "rec2"],
                "A_confidence": [0.8, 0.5, 0.6, 0.7],
                "B_confidence": [0.2, 0.3, 0.4, 0.1],
                "A_annotation": [1, 0, 1, 0],
                "B_annotation": [0, 1, 0, 1],
            }
        )
        self.dp.classes = ("A", "B")
        # Create tensors for the DataProcessor
        self.dp.create_tensors()

    def teardown_method(self):
        """Stop patching."""
        self.patcher_pred.stop()

    def test_valid_classes_and_recordings(self):
        """Test with valid classes and recordings."""
        predictions, labels, classes = self.dp.get_filtered_tensors(
            selected_classes=["A", "B"], selected_recordings=["rec1"]
        )
        assert predictions.shape == (2, 2)
        assert labels.shape == (2, 2)
        assert classes == ("A", "B")

    def test_selected_classes_not_in_data(self):
        """Test when selected classes are not in data."""
        with pytest.raises(ValueError):
            self.dp.get_filtered_tensors(
                selected_classes=["C"], selected_recordings=["rec1"]
            )

    def test_selected_recordings_not_in_data(self):
        """Test when selected recordings are not in data."""
        predictions, labels, classes = self.dp.get_filtered_tensors(
            selected_classes=["A"], selected_recordings=["rec3"]
        )
        assert predictions.shape == (0, 1)
        assert labels.shape == (0, 1)
        assert classes == ("A",)

    def test_empty_selected_classes(self):
        """Test when selected_classes is empty."""
        with pytest.raises(ValueError):
            self.dp.get_filtered_tensors(
                selected_classes=[], selected_recordings=["rec1"]
            )

    def test_empty_selected_recordings(self):
        """Test when selected_recordings is empty."""
        predictions, labels, classes = self.dp.get_filtered_tensors(
            selected_classes=["A"], selected_recordings=[]
        )
        assert predictions.shape == (0, 1)
        assert labels.shape == (0, 1)
        assert classes == ("A",)

    def test_samples_df_is_empty(self):
        """Test when samples_df is empty."""
        self.dp.samples_df = pd.DataFrame()
        with pytest.raises(ValueError, match="samples_df is empty."):
            self.dp.get_filtered_tensors(
                selected_classes=["A"], selected_recordings=["rec1"]
            )

    def test_missing_confidence_or_annotation_columns(self):
        """Test when required columns are missing."""
        self.dp.samples_df = pd.DataFrame(
            {
                "filename": ["rec1"],
                "A_confidence": [0.8],
                # 'A_annotation' is missing
            }
        )
        with pytest.raises(KeyError):
            self.dp.get_filtered_tensors(
                selected_classes=["A"], selected_recordings=["rec1"]
            )

    def test_nan_values_in_data(self):
        """Test when data contains NaN values."""
        self.dp.samples_df = pd.DataFrame(
            {
                "filename": ["rec1"],
                "A_confidence": [np.nan],
                "A_annotation": [1],
            }
        )
        predictions, labels, classes = self.dp.get_filtered_tensors(
            selected_classes=["A"], selected_recordings=["rec1"]
        )
        assert np.isnan(predictions[0, 0])
        assert labels[0, 0] == 1
        assert classes == ("A",)

    def test_partially_available_classes(self):
        """Test when some selected classes are present."""
        predictions, labels, classes = self.dp.get_filtered_tensors(
            selected_classes=["A", "C"], selected_recordings=["rec1"]
        )
        assert predictions.shape == (2, 1)
        assert labels.shape == (2, 1)
        assert classes == ("A",)

    def test_large_data(self):
        """Test with a large samples_df."""
        num_samples = 1000
        self.dp.samples_df = pd.DataFrame(
            {
                "filename": ["rec1"] * num_samples,
                "A_confidence": np.random.rand(num_samples),
                "A_annotation": np.random.randint(0, 2, size=num_samples),
            }
        )
        predictions, labels, classes = self.dp.get_filtered_tensors(
            selected_classes=["A"], selected_recordings=["rec1"]
        )
        assert predictions.shape == (num_samples, 1)
        assert labels.shape == (num_samples, 1)
        assert classes == ("A",)
