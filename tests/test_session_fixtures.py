import os
import tempfile
from pathlib import Path

import psutil
from birdnet.acoustic.inference.configs import ProcessingConfig


def test_the_inference_default_is_at_most_two_workers():
    cores = psutil.cpu_count(logical=False) or 1

    assert ProcessingConfig.validate_n_workers(None) == min(2, cores)
    assert ProcessingConfig.validate_n_workers(1) == 1


def test_temporary_files_stay_in_the_test_session(tmp_path_factory):
    tempdir = Path(tempfile.gettempdir())

    assert tempdir.is_relative_to(tmp_path_factory.getbasetemp())
    assert {Path(os.environ[name]) for name in ("TEMP", "TMP", "TMPDIR")} == {tempdir}
