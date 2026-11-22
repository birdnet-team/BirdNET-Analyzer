"""Embeddings databases written by BirdNET-Analyzer 2.x, with perch-hoplite 0.1.

Both formats keep the embedding vectors in the usearch index, under the id of their
row in the SQLite file. Only that file is rewritten; the index is used as it is.
"""

from __future__ import annotations

import json
import logging
import ntpath
import os
import posixpath
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

logger = logging.getLogger(__name__)

BACKUP_SUFFIX = ".v2-backup"
# The only model BirdNET-Analyzer 2.x extracted embeddings with.
MODEL_VERSION = "2.4"


def is_old_format(database_path: str) -> bool:
    """Whether the folder holds a database in the format of BirdNET-Analyzer 2.x."""
    from perch_hoplite.db.sqlite_usearch_impl import HOPLITE_FILENAME

    sqlite_file = Path(database_path, HOPLITE_FILENAME)
    if not sqlite_file.is_file():
        return False

    # immutable: a plain read-only connection leaves a -shm file beside the database.
    uri = f"{sqlite_file.resolve().as_uri()}?mode=ro&immutable=1"

    try:
        with closing(sqlite3.connect(uri, uri=True)) as db:
            columns = [
                row[1] for row in db.execute("PRAGMA table_info(hoplite_metadata)")
            ]
    except sqlite3.Error:
        return False

    return "data" in columns


def upgrade(database_path: str) -> None:
    """Rewrites the SQLite file of an old database in the current format.

    The previous file stays beside it, with ``BACKUP_SUFFIX`` appended, and is
    replaced only once the new one is complete.
    """
    from perch_hoplite.db.sqlite_usearch_impl import HOPLITE_FILENAME

    sqlite_file = os.path.join(database_path, HOPLITE_FILENAME)
    backup = sqlite_file + BACKUP_SUFFIX

    with tempfile.TemporaryDirectory(dir=database_path, prefix=".upgrade-") as scratch:
        with closing(sqlite3.connect(sqlite_file)) as old:
            # A leftover write-ahead log would be applied to the file that replaces
            # this one.
            old.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            _write_current_format(old, scratch)

        if not os.path.exists(backup):
            shutil.copy2(sqlite_file, backup)

        os.replace(os.path.join(scratch, HOPLITE_FILENAME), sqlite_file)

    logger.warning(
        "Converted the embeddings database %s, written by an earlier BirdNET-Analyzer "
        "version, to the current format. The previous file is kept as %s.",
        database_path,
        backup,
    )


def _write_current_format(old: sqlite3.Connection, folder: str) -> None:
    from ml_collections import ConfigDict
    from perch_hoplite.db.sqlite_usearch_impl import SQLiteUSearchDB

    from birdnet_analyzer.embeddings.core import (
        MODEL_VERSION_KEY,
        SETTINGS_KEY,
        _ensure_deployment,
    )

    metadata = {
        key: json.loads(data)
        for key, data in old.execute("SELECT key, data FROM hoplite_metadata")
    }
    sources = old.execute("SELECT id, source FROM hoplite_sources").fetchall()
    audio_root, filenames = _relative_to_audio_root([source for _, source in sources])

    new = SQLiteUSearchDB.create(
        folder, usearch_cfg=ConfigDict(metadata.pop("usearch_config"))
    )
    try:
        deployment_id = _ensure_deployment(new)
        recording_ids = {
            source_id: new.insert_recording(
                filename=filename, deployment_id=deployment_id
            )
            for (source_id, _), filename in zip(sources, filenames, strict=True)
        }
        # The window keeps the id of the embedding: it is its key in the index.
        new.db.executemany(
            "INSERT INTO windows (id, recording_id, offsets) VALUES (?, ?, ?)",
            (
                (embedding_id, recording_ids[source_id], _offsets(blob))
                for embedding_id, source_id, blob in old.execute(
                    "SELECT id, source_idx, offsets FROM hoplite_embeddings"
                )
            ),
        )

        if SETTINGS_KEY in metadata:
            metadata[SETTINGS_KEY] |= {
                "AUDIO_ROOT": audio_root,
                MODEL_VERSION_KEY: MODEL_VERSION,
            }

        for key, value in metadata.items():
            new.insert_metadata(key, ConfigDict(value))

        new.db.commit()
        new.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        new.db.close()


def _offsets(blob: bytes) -> list[float]:
    import numpy as np

    return [round(float(value), 6) for value in np.frombuffer(blob, dtype="<f4")]


def _relative_to_audio_root(sources: list[str]) -> tuple[str, list[str]]:
    """The stored paths as the current format keeps them.

    BirdNET-Analyzer 2.x stored each path as it was given. The current format stores
    it relative to the parent of the embedded folder, the audio root.
    """
    windows_paths = any("\\" in s or ntpath.splitdrive(s)[0] for s in sources)
    paths = ntpath if windows_paths else posixpath

    try:
        folder = paths.commonpath([paths.dirname(source) for source in sources])
    except ValueError:
        # No common folder: several drives, or relative next to absolute paths.
        return "", list(sources)

    audio_root = paths.dirname(folder) or "."

    return audio_root, [paths.relpath(source, audio_root) for source in sources]
