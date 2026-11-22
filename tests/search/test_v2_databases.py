"""Embeddings databases written by BirdNET-Analyzer 2.x."""

import os
from pathlib import Path

import numpy as np
import pytest

from birdnet_analyzer.embeddings import upgrade
from birdnet_analyzer.embeddings.core import SETTINGS_KEY
from birdnet_analyzer.search.core import open_database, read_database_settings
from tests import v2_databases

EMBEDDINGS = np.arange(24, dtype=np.float32).reshape(3, 8) / 8


def _contents(database: str):
    with open_database(database) as db:
        windows = []
        for window_id in sorted(db.match_window_ids()):
            window = db.get_window(window_id)
            recording = db.get_recording(window.recording_id)
            windows.append((window_id, recording.filename, window.offsets))

        embeddings = np.stack(
            [db.get_embedding(window_id) for window_id, *_ in windows]
        )

        return windows, embeddings, db.get_metadata(SETTINGS_KEY).to_dict()


def test_a_v2_database_opens_with_its_windows_embeddings_and_settings(tmp_path):
    first = str(tmp_path / "recordings" / "site 1" / "a.wav")
    second = str(tmp_path / "recordings" / "site 2" / "b.wav")
    database = v2_databases.write(
        tmp_path / "db",
        [
            (first, 0.0, 3.0, EMBEDDINGS[0]),
            (first, 1.5, 4.5, EMBEDDINGS[1]),
            (second, 0.0, 3.0, EMBEDDINGS[2]),
        ],
        {"BANDPASS_FMIN": 500, "BANDPASS_FMAX": 8000, "AUDIO_SPEED": 2.0},
    )

    windows, embeddings, settings = _contents(database)

    assert windows == [
        (1, os.path.join("recordings", "site 1", "a.wav"), [0.0, 3.0]),
        (2, os.path.join("recordings", "site 1", "a.wav"), [1.5, 4.5]),
        (3, os.path.join("recordings", "site 2", "b.wav"), [0.0, 3.0]),
    ]
    np.testing.assert_array_equal(embeddings, EMBEDDINGS)
    assert settings == {
        "BANDPASS_FMIN": 500,
        "BANDPASS_FMAX": 8000,
        "AUDIO_SPEED": 2.0,
        "AUDIO_ROOT": str(tmp_path),
        "MODEL_VERSION": "2.4",
    }


@pytest.mark.parametrize(
    ("recordings", "audio_root", "filenames"),
    [
        (
            ["C:\\data\\rec\\a.wav", "C:/data/rec\\sub\\b.wav"],
            "C:\\data",
            ["rec\\a.wav", "rec\\sub\\b.wav"],
        ),
        (["rec/a.wav", "rec/sub/b.wav"], ".", ["rec/a.wav", "rec/sub/b.wav"]),
        (
            ["C:\\rec\\a.wav", "D:\\rec\\b.wav"],
            "",
            ["C:\\rec\\a.wav", "D:\\rec\\b.wav"],
        ),
    ],
    ids=["written on Windows", "relative paths", "several drives"],
)
def test_the_recordings_of_a_v2_database_are_stored_relative_to_their_audio_root(
    tmp_path, recordings, audio_root, filenames
):
    database = v2_databases.write(
        tmp_path / "db",
        [(path, 0.0, 3.0, EMBEDDINGS[i]) for i, path in enumerate(recordings)],
    )

    windows, _, settings = _contents(database)

    assert [filename for _, filename, _ in windows] == filenames
    assert settings["AUDIO_ROOT"] == audio_root


def test_the_file_of_a_v2_database_is_kept_beside_the_new_one(tmp_path):
    database = v2_databases.write(
        tmp_path / "db", [(str(tmp_path / "rec" / "a.wav"), 0.0, 3.0, EMBEDDINGS[0])]
    )
    written = Path(database, "hoplite.sqlite").read_bytes()

    read_database_settings(database)

    assert Path(database, v2_databases.BACKUP).read_bytes() == written
    assert not upgrade.is_old_format(database)
    assert not [name for name in os.listdir(database) if name.endswith(".part")]


@pytest.mark.parametrize("failing", ["insert_metadata", "insert_recording"])
def test_a_failing_conversion_leaves_the_v2_database_as_it_was(
    tmp_path, monkeypatch, failing
):
    from perch_hoplite.db.sqlite_usearch_impl import SQLiteUSearchDB

    database = v2_databases.write(
        tmp_path / "db", [(str(tmp_path / "rec" / "a.wav"), 0.0, 3.0, EMBEDDINGS[0])]
    )
    written = Path(database, "hoplite.sqlite").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(SQLiteUSearchDB, failing, fail)
        with pytest.raises(OSError, match="disk full"):
            read_database_settings(database)

    assert Path(database, "hoplite.sqlite").read_bytes() == written
    assert sorted(os.listdir(database)) == ["hoplite.sqlite", "usearch.index"]
    assert read_database_settings(database)["MODEL_VERSION"] == "2.4"


def test_a_current_database_is_not_converted(hoplite_database):
    database = hoplite_database(v2_databases.SETTINGS)

    read_database_settings(database)

    assert v2_databases.BACKUP not in os.listdir(database)
