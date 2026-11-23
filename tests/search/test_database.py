import os
import sqlite3

import pytest

from birdnet_analyzer.embeddings.core import get_or_create_database
from birdnet_analyzer.search import core as search_core
from birdnet_analyzer.search import utils as search_utils

SETTINGS = {
    "AUDIO_SPEED": 1.0,
    "BANDPASS_FMIN": 0,
    "BANDPASS_FMAX": 15000,
    "AUDIO_ROOT": "recordings",
}


def test_a_mistyped_database_path_is_not_created(tmp_path):
    typo = tmp_path / "typo"

    with pytest.raises(FileNotFoundError):
        search_core.get_database(str(typo))

    assert not typo.exists()


def test_a_folder_without_a_database_is_left_untouched(tmp_path):
    (tmp_path / "recording.wav").touch()

    with pytest.raises(FileNotFoundError):
        search_core.get_database(str(tmp_path))

    assert [p.name for p in tmp_path.iterdir()] == ["recording.wav"]


def test_a_folder_left_behind_by_a_failed_open_is_not_opened(hoplite_leftover):
    with pytest.raises(FileNotFoundError):
        search_core.get_database(hoplite_leftover)

    assert os.listdir(hoplite_leftover) == ["hoplite.sqlite"]


def test_a_mistyped_database_leaves_no_output_folder(tmp_path):
    output = tmp_path / "out"

    with pytest.raises(FileNotFoundError):
        search_core.search(str(output), str(tmp_path / "typo"), str(tmp_path), "q.wav")

    assert not output.exists()


def test_an_existing_database_opens(hoplite_database):
    db = search_core.get_database(hoplite_database(SETTINGS))

    try:
        assert db.count_embeddings() == 0
    finally:
        db.db.close()


def test_the_settings_are_read_and_the_database_closed(
    hoplite_database, hoplite_connections
):
    path = hoplite_database(SETTINGS)

    assert search_core.read_database_settings(path) == SETTINGS
    assert hoplite_connections.still_open() == []


def test_a_database_without_settings_raises_key_error_and_is_closed(
    hoplite_database, hoplite_connections
):
    path = hoplite_database()

    with pytest.raises(KeyError):
        search_core.read_database_settings(path)

    assert hoplite_connections.still_open() == []


def test_a_failing_search_closes_the_database(
    hoplite_database, hoplite_connections, monkeypatch, tmp_path
):
    def fail(*args, **kwargs):
        raise RuntimeError("search failed")

    monkeypatch.setattr(search_utils, "get_search_results", fail)

    with pytest.raises(RuntimeError, match="search failed"):
        search_core.search(
            str(tmp_path / "out"), hoplite_database(SETTINGS), str(tmp_path), "q.wav"
        )

    assert hoplite_connections.still_open() == []


def _database_of_v2(folder):
    """The tables perch-hoplite 0.x created, as BirdNET-Analyzer 2.x left them."""
    folder.mkdir()
    connection = sqlite3.connect(folder / "hoplite.sqlite")
    connection.execute(
        "CREATE TABLE hoplite_metadata (key STRING PRIMARY KEY, data STRING NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE hoplite_embeddings (id INTEGER PRIMARY KEY, offsets BLOB)"
    )
    connection.commit()
    connection.close()
    (folder / "usearch.index").write_bytes(b"")

    return folder


@pytest.mark.parametrize(
    "open_database",
    [
        search_core.get_database,
        search_core.read_database_settings,
        lambda path: get_or_create_database(str(path), embedding_dim=1024),
    ],
)
def test_a_database_of_v2_is_named_as_one_and_left_as_it_is(tmp_path, open_database):
    database = _database_of_v2(tmp_path / "db")
    before = (database / "hoplite.sqlite").read_bytes()

    with pytest.raises(ValueError, match=r"written by BirdNET-Analyzer 2\.x"):
        open_database(database)

    assert (database / "hoplite.sqlite").read_bytes() == before
