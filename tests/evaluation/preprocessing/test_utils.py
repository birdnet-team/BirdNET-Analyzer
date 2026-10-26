import warnings

import numpy as np
import pandas as pd
import pytest

from birdnet_analyzer.evaluation.preprocessing.utils import (
    extract_recording_filename,
    read_and_concatenate_files_in_directory,
    recording_key,
)


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("/data/site_a/rec.wav", "rec"),
        ("rec.txt", "rec"),
        ("rec.BirdNET.selection.table.txt", "rec"),
        ("rec.BirdNET.results.csv", "rec"),
        ("rec.BirdNET.txt", "rec"),
        ("rec.wav.BirdNET.selection.table.txt", "rec"),
        ("2023.05.01_dawn.txt", "2023.05.01_dawn"),
        ("S19_20180214.Table.1.selections.txt", "S19_20180214.Table.1.selections"),
        ("rec", "rec"),
        (".hidden", ".hidden"),
    ],
)
def test_a_result_table_and_its_recording_share_one_key(name, key):
    assert recording_key(name) == key


def test_missing_names_pass_through_the_key_extraction():
    pd.testing.assert_series_equal(
        extract_recording_filename(pd.Series([None, np.nan, "/data/rec.wav"])),
        pd.Series([None, np.nan, "rec"]),
    )


def test_the_tables_of_a_folder_are_stacked_with_their_file_names(tmp_path):
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(
        tmp_path / "file1.txt", sep="\t", index=False
    )
    pd.DataFrame({"A": [5], "B": [6]}).to_csv(
        tmp_path / "file2.txt", sep="\t", index=False
    )
    (tmp_path / "notes.csv").write_text("A,B\n9,9\n", encoding="utf-8")
    (tmp_path / "empty").mkdir()

    table = read_and_concatenate_files_in_directory(str(tmp_path))

    pd.testing.assert_frame_equal(
        table,
        pd.DataFrame(
            {
                "A": [1, 2, 5],
                "B": [3, 4, 6],
                "source_file": ["file1.txt", "file1.txt", "file2.txt"],
            }
        ),
    )
    assert read_and_concatenate_files_in_directory(str(tmp_path / "empty")).empty


def test_header_only_tables_are_skipped_but_their_columns_are_kept(tmp_path):
    full = pd.DataFrame({"A": [1, 2], "B": [3.5, 4.5]})
    full.to_csv(tmp_path / "file1.txt", sep="\t", index=False)
    full.iloc[:0].to_csv(tmp_path / "file2.txt", sep="\t", index=False)
    only_headers = tmp_path / "headers"
    only_headers.mkdir()
    full.iloc[:0].to_csv(only_headers / "file1.txt", sep="\t", index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = read_and_concatenate_files_in_directory(str(tmp_path))
        empty = read_and_concatenate_files_in_directory(str(only_headers))

    pd.testing.assert_frame_equal(table, full.assign(source_file="file1.txt"))
    assert empty.columns.tolist() == ["A", "B", "source_file"]
    assert empty.empty


def test_a_table_with_other_columns_is_rejected_by_name(tmp_path):
    pd.DataFrame({"A": [1]}).to_csv(tmp_path / "file1.txt", sep="\t", index=False)
    pd.DataFrame({"C": [1]}).to_csv(tmp_path / "file2.txt", sep="\t", index=False)

    with pytest.raises(ValueError, match=r"file2\.txt has different columns"):
        read_and_concatenate_files_in_directory(str(tmp_path))


def test_a_latin_1_table_is_read_next_to_utf_8_ones(tmp_path):
    pd.DataFrame({"A": ["Mönchsgrasmücke"]}).to_csv(
        tmp_path / "latin1.txt", sep="\t", index=False, encoding="latin-1"
    )
    pd.DataFrame({"A": ["Grünfink"]}).to_csv(
        tmp_path / "utf8.txt", sep="\t", index=False, encoding="utf-8"
    )

    table = read_and_concatenate_files_in_directory(str(tmp_path))

    assert table["A"].tolist() == ["Mönchsgrasmücke", "Grünfink"]
