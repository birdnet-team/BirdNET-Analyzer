import numpy as np
import pandas as pd
import pytest

from birdnet_analyzer.evaluation.preprocessing.utils import (
    extract_recording_filename,
    extract_recording_filename_from_filename,
    read_and_concatenate_files_in_directory,
)


def test_extract_recording_filename_simple():
    """
    Test extract_recording_filename with simple file paths and extensions.

    Ensures that the function correctly extracts filenames from paths containing extensions.
    """
    input_series = pd.Series(["/path/to/file1.txt", "/path/to/file2.csv"])
    expected_output = pd.Series(["file1", "file2"])
    output_series = extract_recording_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_no_extension():
    """
    Test extract_recording_filename with file paths without extensions.

    Ensures that the function extracts the filenames from paths that do not contain extensions.
    """
    input_series = pd.Series(["/path/to/file1", "/path/to/file2"])
    expected_output = pd.Series(["file1", "file2"])
    output_series = extract_recording_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_multiple_dots():
    """
    Test extract_recording_filename with file paths containing multiple dots.

    Ensures that the function correctly extracts the base filename when there are multiple dots in the filename.
    """
    input_series = pd.Series(
        ["/path/to/file.name.ext", "/path/to/another.file.name.ext"]
    )
    expected_output = pd.Series(["file.name", "another.file.name"])
    output_series = extract_recording_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_leading_trailing_slashes():
    """
    Test extract_recording_filename with paths that have leading/trailing slashes.

    Ensures that the function returns an empty string when the file paths have trailing slashes.
    """
    input_series = pd.Series(["/path/to/file1/", "/path/to/file2/"])
    expected_output = pd.Series(["", ""])
    output_series = extract_recording_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_empty_strings():
    """
    Test extract_recording_filename with empty strings.

    Ensures that the function handles empty strings by returning an empty string in the output.
    """
    input_series = pd.Series(["", ""])
    expected_output = pd.Series(["", ""])
    output_series = extract_recording_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_non_string_elements():
    """
    Test extract_recording_filename with non-string elements such as None.

    Ensures that the function handles None values without errors and returns them as is.
    """
    input_series = pd.Series([None, "/path/to/file1.txt"])
    expected_output = pd.Series([None, "file1"])
    output_series = extract_recording_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_nan_values():
    """
    Test extract_recording_filename with NaN values.

    Ensures that the function handles NaN values and passes them through without modification.
    """
    input_series = pd.Series([np.nan, "/path/to/file1.txt"])
    expected_output = pd.Series([np.nan, "file1"])
    output_series = extract_recording_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_relative_absolute_paths():
    """
    Test extract_recording_filename with both relative and absolute file paths.

    Ensures that filenames are correctly extracted regardless of whether the path is relative or absolute.
    """
    input_series = pd.Series(["file1.txt", "/absolute/path/to/file2.txt"])
    expected_output = pd.Series(["file1", "file2"])
    output_series = extract_recording_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_long_paths():
    """
    Test extract_recording_filename with very long file paths.

    Ensures that the function correctly extracts the base filename even from long paths.
    """
    long_path = "/a" * 255 + "/file.txt"
    input_series = pd.Series([long_path])
    expected_output = pd.Series(["file"])
    output_series = extract_recording_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_unicode_characters():
    """
    Test extract_recording_filename with Unicode characters in filenames.

    Ensures that the function correctly extracts filenames with Unicode characters.
    """
    input_series = pd.Series(["/path/to/файл.txt", "/path/到/文件.txt"])
    expected_output = pd.Series(["файл", "文件"])
    output_series = extract_recording_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_from_filename_simple():
    """
    Test extract_recording_filename_from_filename with filenames containing single dots.

    Ensures that the function extracts the base filename correctly when a single dot is present.
    """
    input_series = pd.Series(["file1.txt", "file2.csv"])
    expected_output = pd.Series(["file1", "file2"])
    output_series = extract_recording_filename_from_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_from_filename_multiple_dots():
    """
    Test extract_recording_filename_from_filename with filenames containing multiple dots.

    Ensures that the function extracts the base filename correctly when multiple dots are present.
    """
    input_series = pd.Series(["file.name.ext", "another.file.name.ext"])
    expected_output = pd.Series(["file", "another"])
    output_series = extract_recording_filename_from_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_from_filename_no_extension():
    """
    Test extract_recording_filename_from_filename with filenames that have no extension.

    Ensures that the function correctly extracts the filename when there is no extension.
    """
    input_series = pd.Series(["file1", "file2"])
    expected_output = pd.Series(["file1", "file2"])
    output_series = extract_recording_filename_from_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_from_filename_empty_strings():
    """
    Test extract_recording_filename_from_filename with empty strings.

    Ensures that the function returns an empty string when an empty filename is provided.
    """
    input_series = pd.Series(["", ""])
    expected_output = pd.Series(["", ""])
    output_series = extract_recording_filename_from_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_from_filename_non_string_elements():
    """
    Test extract_recording_filename_from_filename with non-string elements (e.g., None).

    Ensures that the function handles None values gracefully and returns them as is.
    """
    input_series = pd.Series([None, "file1.txt"])
    expected_output = pd.Series([None, "file1"])
    output_series = extract_recording_filename_from_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_from_filename_nan_values():
    """
    Test extract_recording_filename_from_filename with NaN values.

    Ensures that the function handles NaN values and passes them through without modification.
    """
    input_series = pd.Series([np.nan, "file1.txt"])
    expected_output = pd.Series([np.nan, "file1"])
    output_series = extract_recording_filename_from_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_from_filename_starting_dot():
    """
    Test extract_recording_filename_from_filename with filenames starting with a dot.

    Ensures that the function correctly handles hidden files or filenames that start with a dot.
    """
    input_series = pd.Series([".hiddenfile", ".anotherhiddenfile.txt"])
    expected_output = pd.Series(["", ""])
    output_series = extract_recording_filename_from_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_from_filename_ending_dot():
    """
    Test extract_recording_filename_from_filename with filenames ending with a dot.

    Ensures that the function removes the dot when filenames end with it.
    """
    input_series = pd.Series(["file.", "anotherfile."])
    expected_output = pd.Series(["file", "anotherfile"])
    output_series = extract_recording_filename_from_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_from_filename_only_dots():
    """
    Test extract_recording_filename_from_filename with filenames that contain only dots.

    Ensures that the function handles filenames made entirely of dots.
    """
    input_series = pd.Series(["...", ".."])
    expected_output = pd.Series(["", ""])
    output_series = extract_recording_filename_from_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_extract_recording_filename_from_filename_unicode_characters():
    """
    Test extract_recording_filename_from_filename with Unicode characters in filenames.

    Ensures that the function correctly extracts filenames that contain Unicode characters.
    """
    input_series = pd.Series(["файл.txt", "文件.csv"])
    expected_output = pd.Series(["файл", "文件"])
    output_series = extract_recording_filename_from_filename(input_series)
    pd.testing.assert_series_equal(output_series, expected_output)


def test_read_and_concatenate_files_multiple_txt(tmp_path):
    """
    Test read_and_concatenate_files_in_directory with multiple .txt files.

    Ensures that the function reads multiple .txt files with compatible structures and concatenates them.
    """
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
    df1.to_csv(tmp_path / "file1.txt", sep="\t", index=False)
    df2.to_csv(tmp_path / "file2.txt", sep="\t", index=False)

    result_df = read_and_concatenate_files_in_directory(str(tmp_path))
    expected_df = pd.concat(
        [df1.assign(source_file="file1.txt"), df2.assign(source_file="file2.txt")],
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_read_and_concatenate_files_no_txt(tmp_path):
    """
    Test read_and_concatenate_files_in_directory when there are no .txt files.

    Ensures that the function returns an empty DataFrame when no .txt files are present in the directory.
    """
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df.to_csv(tmp_path / "file1.csv", index=False)

    result_df = read_and_concatenate_files_in_directory(str(tmp_path))
    expected_df = pd.DataFrame()

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_read_and_concatenate_files_different_structures(tmp_path):
    """
    Test read_and_concatenate_files_in_directory with files having different structures.

    Ensures that the function raises a ValueError when attempting to concatenate DataFrames with different structures.
    """
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"C": [5, 6], "D": [7, 8]})
    df1.to_csv(tmp_path / "file1.txt", sep="\t", index=False)
    df2.to_csv(tmp_path / "file2.txt", sep="\t", index=False)

    with pytest.raises(ValueError):
        read_and_concatenate_files_in_directory(str(tmp_path))


def test_read_and_concatenate_files_ignores_non_txt(tmp_path):
    """
    Test read_and_concatenate_files_in_directory with a mix of .txt and non-.txt files.

    Ensures that the function ignores non-.txt files and only processes .txt files.
    """
    df_txt = pd.DataFrame({"A": [1, 2]})
    df_csv = pd.DataFrame({"B": [3, 4]})
    df_txt.to_csv(tmp_path / "file1.txt", sep="\t", index=False)
    df_csv.to_csv(tmp_path / "file2.csv", index=False)

    result_df = read_and_concatenate_files_in_directory(str(tmp_path))
    expected_df = df_txt.assign(source_file="file1.txt")

    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True), expected_df.reset_index(drop=True)
    )


def test_read_and_concatenate_files_nonexistent_directory():
    """
    Test read_and_concatenate_files_in_directory when the directory does not exist.

    Ensures that the function raises a FileNotFoundError when a nonexistent directory is provided.
    """
    with pytest.raises(FileNotFoundError):
        read_and_concatenate_files_in_directory("nonexistent_directory")


def test_read_and_concatenate_files_empty_directory(tmp_path):
    """
    Test read_and_concatenate_files_in_directory with an empty directory.

    Ensures that the function returns an empty DataFrame when the directory is empty.
    """
    result_df = read_and_concatenate_files_in_directory(str(tmp_path))
    expected_df = pd.DataFrame()
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_read_and_concatenate_files_large_files(tmp_path):
    """
    Test read_and_concatenate_files_in_directory with large .txt files.

    Ensures that the function can process and concatenate large .txt files correctly.
    """
    df_large = pd.DataFrame({"A": range(10000), "B": range(10000)})
    df_large.to_csv(tmp_path / "large_file.txt", sep="\t", index=False)

    result_df = read_and_concatenate_files_in_directory(str(tmp_path))
    expected_df = df_large.assign(source_file="large_file.txt")

    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True), expected_df.reset_index(drop=True)
    )


def test_read_and_concatenate_files_invalid_path():
    """
    Test read_and_concatenate_files_in_directory with an invalid directory path.

    Ensures that the function raises a FileNotFoundError when an invalid path is provided.
    """
    with pytest.raises(FileNotFoundError):
        read_and_concatenate_files_in_directory("")


def test_read_and_concatenate_files_different_encodings(tmp_path):
    """
    Test read_and_concatenate_files_in_directory with files in different encodings.

    Ensures that the function can read and concatenate files with different encodings.
    """
    df_utf8 = pd.DataFrame({"A": ["こんにちは", "世界"]})
    df_ascii = pd.DataFrame({"A": ["hello", "world"]})

    # Write files with utf-8 encoding
    df_utf8.to_csv(tmp_path / "utf8_file.txt", sep="\t", index=False, encoding="utf-8")
    df_ascii.to_csv(
        tmp_path / "ascii_file.txt", sep="\t", index=False, encoding="utf-8"
    )

    # Call the function to read and concatenate
    result_df = read_and_concatenate_files_in_directory(str(tmp_path))

    # Create the expected DataFrame and sort both for comparison
    expected_df = pd.concat(
        [
            df_utf8.assign(source_file="utf8_file.txt"),
            df_ascii.assign(source_file="ascii_file.txt"),
        ],
        ignore_index=True,
    )

    # Sort both DataFrames by column A for comparison
    result_df_sorted = result_df.sort_values(by="A").reset_index(drop=True)
    expected_df_sorted = expected_df.sort_values(by="A").reset_index(drop=True)

    pd.testing.assert_frame_equal(result_df_sorted, expected_df_sorted)
