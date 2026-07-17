"""Crash-safe resume journal for multi-file analyses.

While an analysis runs, the journal persists every finished file's detections to
``<output>/.birdnet-resume/`` via the birdnet library's ``on_file_complete``
callback. If the run is interrupted (crash, power loss, cancel), a subsequent
run with the same parameters skips the already-completed files and combines
their stored detections with the fresh results. On successful completion the
journal directory is deleted.

Layout of the journal directory:

- ``manifest.json``: parameter fingerprint, a human-readable parameter
  snapshot, total file count and the run metadata needed to write outputs
  when no new inference happens (all files were already completed).
- ``results/<key>.parquet``: one file per completed input file, holding that
  file's detection dataframe. ``<key>`` is a hash of the normalized absolute
  input path. Files the library reported as unprocessable are stored with a
  ``.invalid.parquet`` suffix instead. Partials are written to a temp name and
  moved into place with :func:`os.replace`, so their presence is an atomic
  "this file is done" marker — no separate completed-list is needed.

The completion callback runs on the library's dispatcher thread, off the
inference hot path. It must never raise: a raising callback cancels the whole
analysis, and a persistence problem (e.g. full disk) should not kill an
otherwise working run — that file simply won't be resumable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

logger = logging.getLogger(__name__)

JOURNAL_DIRNAME = ".birdnet-resume"
MANIFEST_FILENAME = "manifest.json"
RESULTS_DIRNAME = "results"
COMPLETED_SUFFIX = ".parquet"
INVALID_SUFFIX = ".invalid.parquet"
MANIFEST_VERSION = 1

# Parquet cannot store float16, which the library uses for time/confidence
# columns; cast on write (downstream code casts to float32 anyway).
_FLOAT16_COLUMNS = ("start_time", "end_time", "confidence")


@dataclass(frozen=True)
class RunMetadata:
    """Result metadata needed to write outputs without a fresh inference run."""

    model_path: str
    model_fmin: int
    model_fmax: int
    model_sr: int
    segment_duration_s: float
    hop_duration_s: float

    @classmethod
    def from_result(cls, result) -> RunMetadata:
        return cls(
            model_path=str(result.model_path),
            model_fmin=int(result.model_fmin),
            model_fmax=int(result.model_fmax),
            model_sr=int(result.model_sr),
            segment_duration_s=float(result.segment_duration_s),
            hop_duration_s=float(result.hop_duration_s),
        )


@dataclass(frozen=True)
class ResumeProgress:
    """Lightweight progress snapshot of an interrupted run (e.g. for the GUI)."""

    n_completed: int
    n_files_total: int
    params: dict


def compute_fingerprint(params: dict) -> str:
    """Hash the analysis parameters that affect detection results.

    Output-formatting parameters (rtype, merge_consecutive, split_tables, ...)
    must not be part of ``params``: the journal stores raw detections, so those
    may change freely between an interrupted run and its resume.
    """
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _file_key(path) -> str:
    normalized = os.path.normcase(os.path.normpath(str(path)))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:24]


class ResumeJournal:
    def __init__(self, directory: Path, fingerprint: str) -> None:
        self._directory = directory
        self._fingerprint = fingerprint
        self._results_dir = directory / RESULTS_DIRNAME
        self._manifest_path = directory / MANIFEST_FILENAME
        self._metadata_written = False

    @classmethod
    def open(cls, output_dir, params: dict, n_files_total: int) -> ResumeJournal:
        """Open the journal for ``output_dir``, resuming or starting fresh.

        An existing journal is kept only if its parameter fingerprint matches;
        otherwise it is wiped so results from different settings never mix.
        """
        directory = Path(output_dir) / JOURNAL_DIRNAME
        fingerprint = compute_fingerprint(params)
        journal = cls(directory, fingerprint)

        manifest = journal._read_manifest()

        if manifest is not None and manifest.get("fingerprint") == fingerprint:
            journal._metadata_written = manifest.get("metadata") is not None
            if manifest.get("n_files_total") != n_files_total:
                manifest["n_files_total"] = n_files_total
                journal._write_manifest(manifest)
            return journal

        shutil.rmtree(directory, ignore_errors=True)
        journal._results_dir.mkdir(parents=True, exist_ok=True)
        journal._write_manifest(
            {
                "version": MANIFEST_VERSION,
                "fingerprint": fingerprint,
                "params": params,
                "n_files_total": n_files_total,
                "metadata": None,
            }
        )
        return journal

    @staticmethod
    def inspect(output_dir) -> ResumeProgress | None:
        """Return progress of an interrupted run in ``output_dir``, if any.

        Intended for the GUI to detect resumable state when the user selects
        an output directory. Returns ``None`` if there is no (readable)
        journal.
        """
        directory = Path(output_dir) / JOURNAL_DIRNAME
        try:
            with open(directory / MANIFEST_FILENAME, encoding="utf-8") as f:
                manifest = json.load(f)
            n_completed = sum(
                1
                for p in (directory / RESULTS_DIRNAME).iterdir()
                if p.name.endswith(COMPLETED_SUFFIX)
            )
        except (OSError, ValueError):
            return None

        return ResumeProgress(
            n_completed=n_completed,
            n_files_total=int(manifest.get("n_files_total", 0)),
            params=manifest.get("params", {}),
        )

    def completed_subset(self, files: Iterable) -> set[Path]:
        """Return the subset of ``files`` that already have a stored result."""
        return {Path(file) for file in files if self._partial_path(file) is not None}

    def invalid_subset(self, files: Iterable) -> set[Path]:
        """Return the subset of ``files`` stored as unprocessable."""
        return {
            Path(file)
            for file in files
            if (self._results_dir / (_file_key(file) + INVALID_SUFFIX)).exists()
        }

    def on_file_complete(self, result) -> None:
        """Persist a single-file result; passed to the library as callback.

        Never raises: a raising callback cancels the whole analysis.
        """
        try:
            # Metadata first: a stored partial must imply stored metadata, so
            # the all-files-completed resume path can always write outputs.
            self._ensure_metadata(result)

            df = result.to_dataframe()
            for col in _FLOAT16_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].astype("float32")

            file_path = str(result.inputs[0])
            invalid = len(result.unprocessable_inputs) > 0
            key = _file_key(file_path)
            suffix = INVALID_SUFFIX if invalid else COMPLETED_SUFFIX

            tmp_path = self._results_dir / (key + ".tmp")
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, self._results_dir / (key + suffix))
        except Exception:
            logger.warning(
                "Failed to persist resume state for %s; the file will be "
                "re-analyzed if the run is resumed.",
                getattr(result, "inputs", ["<unknown>"])[0],
                exc_info=True,
            )

    def combined_dataframe(
        self, current_df: pd.DataFrame | None, input_files: list
    ) -> pd.DataFrame:
        """Combine stored partials with this run's results, in input order.

        ``current_df`` holds the detections of the files analyzed in this run
        (may be ``None`` when everything was already completed); stored
        partials provide the previously completed files. Rows are ordered by
        position in ``input_files``, then by time, so combined outputs (e.g.
        the Raven table's accumulated offsets) stay grouped per file exactly
        like in an uninterrupted run.
        """
        import pandas as pd

        def norm(path) -> str:
            return os.path.normcase(os.path.normpath(str(path)))

        # Compare normalized paths: stored partials may carry path strings from
        # a previous invocation with different casing (e.g. drive letter).
        current_inputs = (
            {norm(p) for p in current_df["input"].unique()}
            if current_df is not None
            else set()
        )
        parts = []

        for file in input_files:
            # This run's in-memory result wins over a stored partial (both can
            # exist for the same file, e.g. after a mid-run persist failure).
            if norm(file) in current_inputs:
                continue

            partial_path = self._partial_path(file)

            if partial_path is not None:
                parts.append(pd.read_parquet(partial_path))

        if current_df is not None:
            parts.append(current_df)

        if not parts:
            raise ValueError("No results to combine.")

        # Empty frames (files without detections) contribute no rows but would
        # degrade dtypes in concat; keep one only if everything is empty.
        non_empty = [p for p in parts if not p.empty]
        df = pd.concat(non_empty, ignore_index=True) if non_empty else parts[0].copy()

        # Stored partials are float32, fresh results float16; unify — pandas
        # cannot sort float16 columns anyway.
        for col in _FLOAT16_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype("float32")

        order = {norm(file): i for i, file in enumerate(input_files)}
        df["_file_order"] = df["input"].map(lambda p: order.get(norm(p)))
        df = df.sort_values(
            by=["_file_order", "start_time", "end_time"], kind="stable"
        ).drop(columns="_file_order")
        return df.reset_index(drop=True)

    def metadata(self) -> RunMetadata | None:
        manifest = self._read_manifest()

        if manifest is None or not manifest.get("metadata"):
            return None

        return RunMetadata(**manifest["metadata"])

    def finalize(self) -> None:
        """Delete the journal after a successful run."""
        shutil.rmtree(self._directory, ignore_errors=True)

    def _partial_path(self, file) -> Path | None:
        key = _file_key(file)

        for suffix in (COMPLETED_SUFFIX, INVALID_SUFFIX):
            path = self._results_dir / (key + suffix)
            if path.exists():
                return path

        return None

    def _ensure_metadata(self, result) -> None:
        if self._metadata_written:
            return

        manifest = self._read_manifest() or {
            "version": MANIFEST_VERSION,
            "fingerprint": self._fingerprint,
            "params": {},
            "n_files_total": 0,
            "metadata": None,
        }
        manifest["metadata"] = asdict(RunMetadata.from_result(result))
        self._write_manifest(manifest)
        self._metadata_written = True

    def _read_manifest(self) -> dict | None:
        try:
            with open(self._manifest_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _write_manifest(self, manifest: dict) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        tmp_path = self._manifest_path.with_suffix(".json.tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

        os.replace(tmp_path, self._manifest_path)
