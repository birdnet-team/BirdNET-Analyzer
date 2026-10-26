import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.config as cfg  # noqa: E402
from birdnet_analyzer.gui import single_file  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


def test_the_kaleidoscope_dialog_offers_the_csv_it_saves(appdir, handler, monkeypatch):
    dialogs = []
    monkeypatch.setattr(
        gu, "save_file_dialog", lambda **kwargs: dialogs.append(kwargs) or None
    )

    handler(single_file.build_single_analysis_tab, "download_kaleidoscope")(
        {"table": None}
    )

    (dialog,) = dialogs
    assert dialog["default_filename"] == cfg.OUTPUT_KALEIDOSCOPE_FILENAME
    assert cfg.OUTPUT_KALEIDOSCOPE_FILENAME.endswith(".csv")
    assert dialog["filetypes"] == ("Kaleidoscope (*.csv)",)
