import os
import sqlite3
import tempfile

import pytest

# Importing the package applies the GUI's persisted model directory to the birdnet
# library; a test run must not depend on this machine's GUI settings.
if "BIRDNET_APP_DATA" not in os.environ:
    from birdnet.utils.local_data import get_app_data_path

    os.environ["BIRDNET_APP_DATA"] = str(get_app_data_path() / "birdnet")


@pytest.fixture(autouse=True, scope="session")
def two_inference_workers():
    """Caps the birdnet library's default worker count at 2.

    The default is one worker per physical core when memory allows, each loading
    its own model.
    """
    from birdnet.acoustic.inference import session

    default_n_workers = session.default_n_workers

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            session,
            "default_n_workers",
            lambda **kwargs: min(2, default_n_workers(**kwargs)),
        )
        yield


@pytest.fixture(autouse=True, scope="session")
def session_tempdir(tmp_path_factory):
    """Keeps the run's temporary files, e.g. the birdnet session logs, out of TEMP.

    The environment variables carry it to spawned worker processes.
    """
    tempdir = str(tmp_path_factory.mktemp("tempdir"))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tempfile, "tempdir", tempdir)
        for name in ("TEMP", "TMP", "TMPDIR"):
            patch.setenv(name, tempdir)
        yield


@pytest.fixture(autouse=True)
def isolated_appdir(monkeypatch, tmp_path_factory):
    """Points the GUI settings, state and error log at a directory of the test."""
    from birdnet_analyzer import settings

    appdir = tmp_path_factory.mktemp("appdir")
    monkeypatch.setattr(settings, "APPDIR", appdir)
    monkeypatch.setattr(
        settings, "GUI_SETTINGS_PATH", str(appdir / "gui-settings.json")
    )
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(appdir / "state.json"))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(appdir / "error_log.txt"))


@pytest.fixture
def hoplite_database(tmp_path):
    """A factory that builds an empty hoplite database in ``tmp_path``.

    The factory returns the database's path.
    """

    def make(settings: dict | None = None) -> str:
        from ml_collections import ConfigDict

        from birdnet_analyzer.embeddings.core import (
            SETTINGS_KEY,
            get_or_create_database,
        )

        path = str(tmp_path / "db")
        db = get_or_create_database(path, embedding_dim=8)
        if settings is not None:
            db.insert_metadata(SETTINGS_KEY, ConfigDict(settings))
        db.commit()
        db.db.close()
        return path

    return make


@pytest.fixture
def hoplite_leftover(tmp_path) -> str:
    """A folder with only the ``hoplite.sqlite`` a failed perch open leaves behind."""
    import gc

    from perch_hoplite.db import sqlite_usearch_impl

    path = tmp_path / "leftover"
    with pytest.raises(ValueError, match="No usearch_cfg"):
        sqlite_usearch_impl.SQLiteUSearchDB.create(str(path))
    # The failed open leaks its connection in a reference cycle.
    gc.collect()
    assert os.listdir(path) == ["hoplite.sqlite"]
    return str(path)


class _ConnectionSpy:
    def __init__(self):
        self.connections: list[sqlite3.Connection] = []

    def still_open(self) -> list[sqlite3.Connection]:
        still_open = []
        for connection in self.connections:
            try:
                connection.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                continue
            still_open.append(connection)
        return still_open


@pytest.fixture
def hoplite_connections(monkeypatch):
    """Records the connection of every hoplite database successfully opened."""
    from perch_hoplite.db import sqlite_usearch_impl

    create = sqlite_usearch_impl.SQLiteUSearchDB.create
    spy = _ConnectionSpy()

    def recording_create(*args, **kwargs):
        db = create(*args, **kwargs)
        spy.connections.append(db.db)
        return db

    monkeypatch.setattr(
        sqlite_usearch_impl.SQLiteUSearchDB, "create", staticmethod(recording_create)
    )
    return spy
