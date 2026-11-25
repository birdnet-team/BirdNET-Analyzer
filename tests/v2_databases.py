"""Embeddings databases as BirdNET-Analyzer 2.x wrote them, with perch-hoplite 0.1."""

import json
import os
import sqlite3
from contextlib import closing

import numpy as np

SETTINGS = {"BANDPASS_FMIN": 0, "BANDPASS_FMAX": 15000, "AUDIO_SPEED": 1.0}
BACKUP = "hoplite.sqlite.v2-backup"

_SCHEMA = """
    PRAGMA journal_mode=WAL;
    CREATE TABLE hoplite_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset STRING NOT NULL,
        source STRING NOT NULL
    );
    CREATE TABLE hoplite_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_idx INTEGER NOT NULL,
        offsets BLOB NOT NULL,
        FOREIGN KEY (source_idx) REFERENCES hoplite_sources(id)
    );
    CREATE TABLE hoplite_metadata (
        key STRING PRIMARY KEY,
        data STRING NOT NULL
    );
    CREATE TABLE hoplite_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding_id INTEGER NOT NULL,
        label STRING NOT NULL,
        type INT NOT NULL,
        provenance STRING NOT NULL,
        FOREIGN KEY (embedding_id) REFERENCES hoplite_embeddings(id)
    );
    CREATE UNIQUE INDEX idx_embedding ON hoplite_embeddings (id);
    CREATE UNIQUE INDEX source_pairs ON hoplite_sources (dataset, source);
    CREATE INDEX embedding_source ON hoplite_embeddings (source_idx);
    CREATE INDEX idx_label ON hoplite_labels (embedding_id, label);
"""


def write(path, windows, settings: dict | None = SETTINGS) -> str:
    """Writes a database of ``(recording path, start, end, embedding)`` windows.

    The windows get the ids 1, 2, ... in the given order.
    """
    from usearch import index as uindex

    os.makedirs(path)
    dim = len(windows[0][3])
    usearch_config = {
        "embedding_dim": dim,
        "dtype": "float16",
        "metric_name": "IP",
        "expansion_add": 256,
        "expansion_search": 128,
    }
    index = uindex.Index(
        ndim=dim,
        metric=uindex.MetricKind.IP,
        dtype="f16",
        expansion_add=256,
        expansion_search=128,
    )

    with closing(sqlite3.connect(os.path.join(path, "hoplite.sqlite"))) as db:
        db.executescript(_SCHEMA)
        metadata = {"usearch_config": usearch_config}
        if settings is not None:
            metadata["birdnet_analyzer_settings"] = settings
        db.executemany(
            "INSERT INTO hoplite_metadata (key, data) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in metadata.items()],
        )

        source_ids: dict[str, int] = {}
        for recording, start, end, embedding in windows:
            if recording not in source_ids:
                source_ids[recording] = db.execute(
                    "INSERT INTO hoplite_sources (dataset, source) VALUES (?, ?)",
                    ("birdnet_analyzer_dataset", recording),
                ).lastrowid
            window_id = db.execute(
                "INSERT INTO hoplite_embeddings (source_idx, offsets) VALUES (?, ?)",
                (source_ids[recording], np.array([start, end], "<f4").tobytes()),
            ).lastrowid
            index.add(window_id, np.asarray(embedding, np.float16))

        db.commit()

    index.save(os.path.join(path, "usearch.index"))

    return str(path)


def rewrite(database: str, path) -> str:
    """Writes the windows of a current database as BirdNET-Analyzer 2.x would have."""
    from birdnet_analyzer.embeddings.core import SETTINGS_KEY
    from birdnet_analyzer.search.core import open_database

    with open_database(database) as db:
        settings = db.get_metadata(SETTINGS_KEY).to_dict()
        windows = []
        for window_id in sorted(db.match_window_ids()):
            window = db.get_window(window_id)
            recording = db.get_recording(window.recording_id)
            windows.append(
                (
                    os.path.join(settings["AUDIO_ROOT"], recording.filename),
                    *window.offsets,
                    db.get_embedding(window_id),
                )
            )

    return write(path, windows, {key: settings[key] for key in SETTINGS})
