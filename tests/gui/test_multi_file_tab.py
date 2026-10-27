import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.multi_file as mfa  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


def test_a_failing_folder_pick_is_logged_and_shown(
    appdir, handler, monkeypatch, tmp_path
):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)
    monkeypatch.setattr(gu, "select_folder", lambda *a, **k: str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(gu, "get_audio_files_and_durations", fail)

    with pytest.raises(gr.Error, match="disk"):
        handler(mfa.build_multi_analysis_tab, "select_directory_on_empty")()

    assert len(logged) == 1
