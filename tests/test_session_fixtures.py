import os
import tempfile
from pathlib import Path

import birdnet

from birdnet_analyzer import settings


def test_the_inference_default_is_at_most_two_workers():
    model = birdnet.load("acoustic", "3.0", "onnx")

    assert 1 <= model.predict_session()._conf.processing_conf.workers <= 2
    assert model.predict_session(n_workers=1)._conf.processing_conf.workers == 1


def test_temporary_files_stay_in_the_test_session(tmp_path_factory):
    tempdir = Path(tempfile.gettempdir())

    assert tempdir.is_relative_to(tmp_path_factory.getbasetemp())
    assert {Path(os.environ[name]) for name in ("TEMP", "TMP", "TMPDIR")} == {tempdir}


def test_every_test_runs_against_a_temporary_appdir(tmp_path_factory):
    basetemp = tmp_path_factory.getbasetemp()

    assert settings.APPDIR.is_relative_to(basetemp)
    for path in (
        settings.GUI_SETTINGS_PATH,
        settings.STATE_SETTINGS_PATH,
        settings.ERROR_LOG_FILE,
    ):
        assert Path(path).is_relative_to(basetemp)
