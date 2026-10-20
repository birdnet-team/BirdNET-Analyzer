import os
import sqlite3

import pytest

# Importing the package applies the GUI's persisted model directory to the birdnet
# library; a test run must not depend on this machine's GUI settings.
if "BIRDNET_APP_DATA" not in os.environ:
    from birdnet.utils.local_data import get_app_data_path

    os.environ["BIRDNET_APP_DATA"] = str(get_app_data_path() / "birdnet")


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
