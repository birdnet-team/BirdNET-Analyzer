import importlib
import sys
from pathlib import Path

from birdnet_analyzer import logs, settings


def test_gui_runtime_files_use_user_appdir_when_not_frozen(monkeypatch, tmp_path):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    reloaded_settings = importlib.reload(settings)
    expected_appdir = tmp_path / ".local" / "share" / "BirdNET-Analyzer-GUI"

    assert expected_appdir == reloaded_settings.APPDIR
    assert Path(reloaded_settings.GUI_SETTINGS_PATH) == (
        expected_appdir / "gui-settings.json"
    )
    assert Path(reloaded_settings.STATE_SETTINGS_PATH) == (
        expected_appdir / "state.json"
    )
    assert Path(reloaded_settings.ERROR_LOG_FILE) == expected_appdir / "error_log.txt"

    logs.setup_logging()

    try:
        reloaded_settings.ensure_settings_file()
        reloaded_settings.set_state("train-data-dir", "/tmp/train")
        reloaded_settings.write_error_log(RuntimeError("gui path test"))

        assert (expected_appdir / "gui-settings.json").exists()
        assert (expected_appdir / "state.json").exists()
        assert (expected_appdir / "error_log.txt").exists()
    finally:
        logs._remove_installed_handlers()
