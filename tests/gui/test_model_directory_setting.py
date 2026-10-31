import contextlib
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from birdnet.utils.local_data import get_app_data_path, get_model_path  # noqa: E402

from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.fixture
def source_run(appdir, monkeypatch):
    monkeypatch.delenv(settings.MODEL_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(settings, "FROZEN", False)
    monkeypatch.setattr(settings, "MODEL_DIR_STARTUP_WARNING", None)
    monkeypatch.setattr(gr, "Info", lambda *a, **k: None)


def test_the_settings_tab_switches_the_model_directory_live(
    source_run, handler, monkeypatch, tmp_path
):
    target = tmp_path / "models"
    monkeypatch.setattr(gu, "select_folder", lambda **_: str(target))

    shown = handler(gu.build_settings, "on_model_dir_select")()

    assert Path(shown).samefile(target)
    assert get_model_path("acoustic", "3.0", "onnx", "fp32").is_relative_to(shown)

    shown = handler(gu.build_settings, "on_model_dir_reset")()

    assert Path(shown) == get_app_data_path() / "birdnet"
    assert get_model_path("acoustic", "3.0", "onnx", "fp32").is_relative_to(shown)


def test_the_model_directory_stays_while_a_model_is_in_use(
    source_run, handler, monkeypatch, tmp_path
):
    target = tmp_path / "models"
    dialogs = []
    monkeypatch.setattr(
        gu, "select_folder", lambda **_: dialogs.append(target) or str(target)
    )
    select = handler(gu.build_settings, "on_model_dir_select")
    reset = handler(gu.build_settings, "on_model_dir_reset")
    settings.set_model_directory(str(tmp_path / "before"))
    before = gu._effective_model_directory()

    with gu.download_progress():
        for click in (select, reset):
            with pytest.raises(gr.Error) as refused:
                click()

            assert refused.value.message == gu.loc.localize("model-dir-busy-error")

    assert gu._effective_model_directory() == before
    assert dialogs == []

    assert Path(select()).samefile(target)


def test_a_task_started_while_the_dialog_is_open_keeps_the_directory(
    source_run, handler, monkeypatch, tmp_path
):
    with contextlib.ExitStack() as running:

        def dialog(**_):
            running.enter_context(gu.download_progress())
            return str(tmp_path / "models")

        monkeypatch.setattr(gu, "select_folder", dialog)
        before = gu._effective_model_directory()

        with pytest.raises(gr.Error) as refused:
            handler(gu.build_settings, "on_model_dir_select")()

        assert refused.value.message == gu.loc.localize("model-dir-busy-error")
        assert gu._effective_model_directory() == before


def test_a_picked_directory_is_no_longer_reported_unavailable_on_reload(
    source_run, monkeypatch, tmp_path
):
    warnings = []
    monkeypatch.setattr(gr, "Warning", lambda message, **_: warnings.append(message))
    monkeypatch.setattr(settings, "MODEL_DIR_STARTUP_WARNING", "X:/unplugged")

    gu._show_startup_warnings()
    assert len(warnings) == 1
    assert "X:/unplugged" in warnings[0]

    settings.set_model_directory(str(tmp_path / "models"))
    gu._show_startup_warnings()

    assert len(warnings) == 1
