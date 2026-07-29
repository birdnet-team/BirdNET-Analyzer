"""
Utility Functions for Data Processing Tasks

This module provides helper functions to handle common data processing tasks, such as:
- Extracting recording keys from file paths or selection-table filenames.
- Reading and concatenating text files from a specified directory.

It is designed to work seamlessly with pandas and file system operations.
"""

import os

import pandas as pd

# Suffixes BirdNET appends to a recording name when it writes a selection/result table,
# e.g. ``soundscape.BirdNET.selection.table.txt``. Stripped so the table matches the
# annotation file (and audio file) it belongs to.
_TABLE_SUFFIXES = (
    ".BirdNET.selection.table",
    ".BirdNET.results",
    ".BirdNET",
)

# Audio extensions that may still cling to a name after the table suffix is removed,
# e.g. ``soundscape.wav.BirdNET.selection.table.txt`` -> ``soundscape.wav`` -> key.
_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wave")


def recording_key(name):
    """Derives the recording identifier a prediction/annotation entry belongs to.

    This is the single source of truth for matching prediction files to annotation
    files. It takes the base name (dropping any directory and the final extension),
    then removes a BirdNET table suffix and a residual audio extension, so that a
    Raven selection table, a plain annotation file, and the ``Begin File`` column of
    the same recording all collapse to the same key.

    Unlike the previous ``split(".")[0]`` approach, a dot inside the recording name
    (dates such as ``2023.05.01_dawn``) is preserved, so distinct recordings never
    collide.

    Args:
        name: A file path, file name, or recording reference. Non-string values
            (e.g. ``NaN``) are returned unchanged.

    Returns:
        The recording key, or the input unchanged if it is not a string.
    """
    if not isinstance(name, str):
        return name

    key = os.path.splitext(os.path.basename(name))[0]

    for suffix in _TABLE_SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break

    root, ext = os.path.splitext(key)
    if ext.lower() in _AUDIO_EXTENSIONS:
        key = root

    return key


def extract_recording_filename(path_column: pd.Series) -> pd.Series:
    """Extract the recording key from a column of file paths (e.g. ``Begin File``).

    Args:
        path_column (pd.Series): A pandas Series containing file paths.

    Returns:
        pd.Series: The recording key for each entry.
    """
    return path_column.apply(recording_key)


def extract_recording_filename_from_filename(filename_series: pd.Series) -> pd.Series:
    """Extract the recording key from a column of selection-table file names.

    Args:
        filename_series (pd.Series): A pandas Series containing file names.

    Returns:
        pd.Series: The recording key for each entry.
    """
    return filename_series.apply(recording_key)


def read_and_concatenate_files_in_directory(directory_path: str) -> pd.DataFrame:
    """
    Read and concatenate all .txt files in a directory into a single DataFrame.

    This function scans the specified directory for all .txt files, reads each file into
    a DataFrame, appends a 'source_file' column containing the filename, and
    concatenates all DataFrames into one.
    If the files have inconsistent columns, a ValueError is raised.

    Args:
        directory_path (str): Path to the directory containing the .txt files.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing the data from all .txt files,
        or an empty DataFrame if no files are found.

    Raises:
        ValueError: If the columns in the files are inconsistent.
    """
    df_list: list[pd.DataFrame] = []  # List to hold individual DataFrames
    columns_set = None  # To ensure consistency in column names

    # Iterate through each file in the directory
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(
                directory_path, filename
            )  # Construct the full file path

            try:
                # Attempt to read the file as a tab-separated values file with
                # UTF-8 encoding
                df = pd.read_csv(filepath, sep="\t", encoding="utf-8")
            except UnicodeDecodeError:
                # Fallback to 'latin-1' encoding if UTF-8 fails
                df = pd.read_csv(filepath, sep="\t", encoding="latin-1")

            # Check for column consistency across files
            if columns_set is None:
                columns_set = set(
                    df.columns
                )  # Initialize with the first file's columns
            elif set(df.columns) != columns_set:
                raise ValueError(
                    f"File {filename} has different columns than the previous files."
                )

            # Add a column to indicate the source file for traceability
            df["source_file"] = filename

            # Append the DataFrame to the list
            df_list.append(df)

    # Concatenate all DataFrames if any were processed, else return an empty DataFrame
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()  # Return an empty DataFrame if no .txt files were found
