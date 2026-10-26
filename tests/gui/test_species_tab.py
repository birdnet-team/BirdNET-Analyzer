import pytest

gr = pytest.importorskip("gradio")

from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import species  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


def test_a_failing_folder_pick_is_logged_and_shown(
    appdir, handler, monkeypatch, tmp_path
):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)
    monkeypatch.setattr(gu, "select_folder", lambda *a, **k: str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(settings, "set_state", fail)

    with pytest.raises(gr.Error, match="disk"):
        handler(species.build_species_tab, "select_directory_and_update_tb")(
            "species_list.txt"
        )

    assert len(logged) == 1
