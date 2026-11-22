"""Embedding recordings into a database and searching it, with the real models."""

import shutil
from pathlib import Path

import pytest
import soundfile as sf

from birdnet_analyzer.config import DEFAULT_ACOUSTIC_MODEL_VERSION
from birdnet_analyzer.embeddings.core import SETTINGS_KEY, embeddings
from birdnet_analyzer.search.core import open_database, read_database_settings, search

DATASET = Path("tests/data/embeddings/embeddings-dataset/s1")
WINDOWS_PER_RECORDING = 4


@pytest.fixture
def recordings(tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    shutil.copy(DATASET / "file01.wav", folder)
    shutil.copy(DATASET / "file02.wav", folder)

    return folder


@pytest.fixture
def query(recordings, tmp_path):
    """Seconds 3 to 6 of file02."""
    signal, rate = sf.read(recordings / "file02.wav")
    path = tmp_path / "query.wav"
    sf.write(path, signal[3 * rate : 6 * rate], rate)

    return path


def _best_match(database: str, query, tmp_path) -> tuple[float, str, str, str]:
    output = tmp_path / "found"
    search(str(output), database, str(tmp_path), str(query), n_results=1)
    (found,) = output.iterdir()
    score, recording, start, end = found.stem.split("_")

    return float(score), recording, start, end


def _forget_model_version(database: str):
    """Leaves the database as one written before the version was stored.

    A function of its own, so the database is released on return: Windows cannot
    overwrite the index file while a view of it is mapped.
    """
    from ml_collections import ConfigDict

    with open_database(database) as db:
        settings = db.get_metadata(SETTINGS_KEY).to_dict()
        del settings["MODEL_VERSION"]
        db.insert_metadata(SETTINGS_KEY, ConfigDict(settings))
        db.commit()


def _count_embeddings(database: str) -> int:
    with open_database(database) as db:
        return db.count_embeddings()


def test_a_clip_of_an_embedded_recording_is_its_own_best_match(
    recordings, query, tmp_path
):
    database = str(tmp_path / "db")

    embeddings(str(recordings), database, n_workers=1)

    score, *window = _best_match(database, query, tmp_path)
    assert window == ["file02", "3.0", "6.0"]
    assert score > 0.99
    stored = read_database_settings(database)["MODEL_VERSION"]
    assert stored == DEFAULT_ACOUSTIC_MODEL_VERSION


def test_a_database_without_a_stored_model_stays_on_birdnet_2_4(
    recordings, query, tmp_path
):
    database = str(tmp_path / "db")
    embeddings(str(recordings), database, n_workers=1, birdnet="2.4")
    _forget_model_version(database)
    shutil.copy(DATASET / "file03.wav", recordings)

    embeddings(str(recordings), database, n_workers=1)

    assert _count_embeddings(database) == 3 * WINDOWS_PER_RECORDING
    score, *window = _best_match(database, query, tmp_path)
    assert window == ["file02", "3.0", "6.0"]
    assert score > 0.99
