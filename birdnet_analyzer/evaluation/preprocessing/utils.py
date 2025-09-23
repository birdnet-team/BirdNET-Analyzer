"""
Utility Functions for Data Processing Tasks

This module provides helper functions to handle common data processing tasks, such as:
- Extracting recording filenames from file paths or filenames.
- Reading and concatenating text files from a specified directory.

It is designed to work seamlessly with pandas and file system operations.
"""

import os

import pandas as pd


def extract_recording_filename(path_column: pd.Series) -> pd.Series:
    """
    Extract the recording filename from a path column.

    This function processes a pandas Series containing file paths and extracts the base filename
    (without the extension) for each path.

    Args:
        path_column (pd.Series): A pandas Series containing file paths.

    Returns:
        pd.Series: A pandas Series containing the extracted recording filenames.
    """
    # Apply a lambda function to extract the base filename without extension
    return path_column.apply(lambda x: os.path.splitext(os.path.basename(x))[0] if isinstance(x, str) else x)


def extract_recording_filename_from_filename(filename_series: pd.Series) -> pd.Series:
    """
    Extract the recording filename from a filename Series.

    This function processes a pandas Series containing filenames and extracts the base filename
    (without the extension) for each.

    Args:
        filename_series (pd.Series): A pandas Series containing filenames.

    Returns:
        pd.Series: A pandas Series containing the extracted recording filenames.
    """
    # Apply a lambda function to split filenames and remove the extension
    return filename_series.apply(lambda x: x.split(".")[0] if isinstance(x, str) else x)


def read_and_concatenate_files_in_directory(directory_path: str) -> pd.DataFrame:
    """
    Read and concatenate all .txt files in a directory into a single DataFrame.

    This function scans the specified directory for all .txt files, reads each file into a DataFrame,
    appends a 'source_file' column containing the filename, and concatenates all DataFrames into one.
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
            filepath = os.path.join(directory_path, filename)  # Construct the full file path

            try:
                # Attempt to read the file as a tab-separated values file with UTF-8 encoding
                df = pd.read_csv(filepath, sep="\t", encoding="utf-8")
            except UnicodeDecodeError:
                # Fallback to 'latin-1' encoding if UTF-8 fails
                df = pd.read_csv(filepath, sep="\t", encoding="latin-1")

            # Check for column consistency across files
            if columns_set is None:
                columns_set = set(df.columns)  # Initialize with the first file's columns
            elif set(df.columns) != columns_set:
                raise ValueError(f"File {filename} has different columns than the previous files.")

            # Add a column to indicate the source file for traceability
            df["source_file"] = filename

            # Append the DataFrame to the list
            df_list.append(df)

    # Concatenate all DataFrames if any were processed, else return an empty DataFrame
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()  # Return an empty DataFrame if no .txt files were found
