"""Tests for the crash-safe resume journal and the resumable analyze() flow."""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from birdnet_analyzer.analyze.core import analyze
from birdnet_analyzer.analyze.resume import JOURNAL_DIRNAME, ResumeJournal

RESULT_COLUMNS = ["input", "start_time", "end_time", "species_name", "confidence"]


class FakeResult:
    """Stand-in for the library's AcousticFilePredictionResult."""

    def __init__(self, files, rows, invalid_indices=()):
        self.inputs = np.array([str(f) for f in files])
        self.unprocessable_inputs = list(invalid_indices)
        self._rows = rows
        self.hop_duration_s = 3.0
        self.model_fmin = 0
        self.model_fmax = 15000
        self.model_path = "fake_model.tflite"
        self.model_sr = 48000
        self.segment_duration_s = 3.0

    def to_dataframe(self):
        df = pd.DataFrame(self._rows, columns=RESULT_COLUMNS)

        if not df.empty:
            # The library uses float16 for these columns.
            for col in ("start_time", "end_time", "confidence"):
                df[col] = df[col].astype("float16")

        return df


def detection_rows(file):
    """Two deterministic, file-specific detection rows."""
    name = Path(file).stem
    return [
        (str(file), 0.0, 3.0, f"Sci{name}_Common{name}", 0.8),
        (str(file), 3.0, 6.0, f"Sci{name}_Common{name}", 0.6),
    ]


def make_fake_run_inference(crash_after=None, invalid_files=()):
    """Build a run_inference replacement that drives on_file_complete per file.

    Args:
        crash_after: Simulate a crash (raise) after this many files completed.
            None runs to completion.
        invalid_files: Files to report as unprocessable (empty result).

    Returns:
        The fake function and a list recording the files of each invocation.
    """
    calls = []

    def fake_run_inference(path, on_file_complete=None, **kwargs):
        files = sorted(Path(p) for p in path) if not isinstance(path, str) else []
        calls.append(files)
        all_rows = []

        for i, file in enumerate(files):
            invalid = file in set(invalid_files)
            rows = [] if invalid else detection_rows(file)
            all_rows.extend(rows)

            if on_file_complete is not None:
                on_file_complete(
                    FakeResult([file], rows, invalid_indices=[0] if invalid else [])
                )

            if crash_after is not None and i + 1 >= crash_after:
                raise RuntimeError("simulated crash")

        return FakeResult(files, all_rows)

    return fake_run_inference, calls


@pytest.fixture
def env(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    audio = np.zeros(4800, dtype=np.float32)
    files = []

    for i in range(4):
        file = input_dir / f"rec_{i}.wav"
        sf.write(file, audio, 48000)
        files.append(file.absolute())

    return {"input_dir": input_dir, "output_dir": output_dir, "files": files}


def run_analyze(env, **kwargs):
    return analyze(str(env["input_dir"]), str(env["output_dir"]), rtype="csv", **kwargs)


def read_combined_csv(env):
    return pd.read_csv(env["output_dir"] / "BirdNET_CombinedTable.csv")


def test_journal_persistence_and_inspect(env):
    files = env["files"]
    params = {"min_conf": 0.25, "model": "birdnet"}
    journal = ResumeJournal.open(env["output_dir"], params, n_files_total=4)

    # Persist two files, one of them unprocessable.
    journal.on_file_complete(FakeResult([files[0]], detection_rows(files[0])))
    journal.on_file_complete(FakeResult([files[1]], [], invalid_indices=[0]))

    assert journal.completed_subset(files) == {files[0], files[1]}
    assert journal.invalid_subset(files) == {files[1]}

    progress = ResumeJournal.inspect(env["output_dir"])
    assert progress is not None
    assert progress.n_completed == 2
    assert progress.n_files_total == 4

    # Reopening with identical params keeps the stored partials...
    journal = ResumeJournal.open(env["output_dir"], params, n_files_total=4)
    assert journal.completed_subset(files) == {files[0], files[1]}
    assert journal.metadata() is not None

    # ...while changed params wipe the journal.
    journal = ResumeJournal.open(
        env["output_dir"], {**params, "min_conf": 0.5}, n_files_total=4
    )
    assert journal.completed_subset(files) == set()

    journal.finalize()
    assert not (env["output_dir"] / JOURNAL_DIRNAME).exists()
    assert ResumeJournal.inspect(env["output_dir"]) is None


def test_combined_dataframe_orders_by_input(env):
    files = env["files"]
    journal = ResumeJournal.open(env["output_dir"], {"p": 1}, n_files_total=4)

    # Later files completed first (out of order), plus fresh results for the rest.
    journal.on_file_complete(FakeResult([files[3]], detection_rows(files[3])))
    journal.on_file_complete(FakeResult([files[1]], detection_rows(files[1])))
    current_df = FakeResult(
        [files[0], files[2]],
        detection_rows(files[0]) + detection_rows(files[2]),
    ).to_dataframe()

    df = journal.combined_dataframe(current_df, files)

    assert list(df["input"].unique()) == [str(f) for f in files]
    assert len(df) == 8
    # Rows within a file stay time-ordered.
    assert df.groupby("input", sort=False)["start_time"].is_monotonic_increasing.all()


def test_analyze_resumes_after_crash(env):
    # First run crashes after two files were persisted.
    fake, calls = make_fake_run_inference(crash_after=2)

    with (
        patch("birdnet_analyzer.model_utils.run_inference", fake),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        run_analyze(env)

    journal_dir = env["output_dir"] / JOURNAL_DIRNAME
    assert journal_dir.exists()
    assert ResumeJournal.inspect(env["output_dir"]).n_completed == 2

    # Second run only gets the files the first run did not complete. The fake
    # processes files in sorted order, so the first two were persisted.
    fake, calls = make_fake_run_inference()

    with patch("birdnet_analyzer.model_utils.run_inference", fake):
        run_analyze(env)

    assert len(calls) == 1
    assert set(calls[0]) == set(env["files"][2:])

    df = read_combined_csv(env)
    assert set(df["File"]) == {str(f) for f in env["files"]}
    assert len(df) == 8
    assert not journal_dir.exists(), "journal must be removed after success"


def test_analyze_all_files_already_completed(env):
    # Crash after every file was persisted but before outputs were written.
    fake, _ = make_fake_run_inference(crash_after=4)

    with (
        patch("birdnet_analyzer.model_utils.run_inference", fake),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        run_analyze(env)

    assert ResumeJournal.inspect(env["output_dir"]).n_completed == 4

    # The resume must not run any inference and still produce the outputs.
    def must_not_run(*args, **kwargs):
        raise AssertionError("run_inference must not be called")

    with patch("birdnet_analyzer.model_utils.run_inference", must_not_run):
        result = run_analyze(env)

    assert result is None
    df = read_combined_csv(env)
    assert set(df["File"]) == {str(f) for f in env["files"]}
    assert not (env["output_dir"] / JOURNAL_DIRNAME).exists()


def test_changed_params_invalidate_resume_state(env):
    fake, _ = make_fake_run_inference(crash_after=2)

    with (
        patch("birdnet_analyzer.model_utils.run_inference", fake),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        run_analyze(env, min_conf=0.25)

    # A different detection parameter must trigger a full re-analysis.
    fake, calls = make_fake_run_inference()

    with patch("birdnet_analyzer.model_utils.run_inference", fake):
        run_analyze(env, min_conf=0.5)

    assert len(calls[0]) == 4
    assert not (env["output_dir"] / JOURNAL_DIRNAME).exists()


def test_analyze_with_invalid_file_completes_and_cleans_up(env):
    invalid = env["files"][1]
    fake, _ = make_fake_run_inference(invalid_files=[invalid])

    with patch("birdnet_analyzer.model_utils.run_inference", fake):
        run_analyze(env)

    df = read_combined_csv(env)
    assert str(invalid) not in set(df["File"])
    assert len(df) == 6
    assert not (env["output_dir"] / JOURNAL_DIRNAME).exists()


def test_single_file_input_does_not_create_journal(env):
    single = env["files"][0]

    def fake_single(path, on_file_complete=None, **kwargs):
        assert on_file_complete is None
        return FakeResult([single], detection_rows(single))

    with patch("birdnet_analyzer.model_utils.run_inference", fake_single):
        analyze(str(single), str(env["output_dir"]), rtype="csv")

    assert not (env["output_dir"] / JOURNAL_DIRNAME).exists()
    assert os.path.exists(env["output_dir"] / "BirdNET_CombinedTable.csv")
