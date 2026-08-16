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


def test_frozen_output_diverted_only_for_windowed_builds(monkeypatch):
    # Frozen builds divert stdout/stderr into logs.txt only when nobody can read them.
    # The console CLI executable (real stream, whether terminal, pipe or file) must
    # keep printing; the windowed GUI must be diverted. On Windows a windowed
    # PyInstaller build has sys.stdout None; on macOS the only frozen artifact is the
    # windowed .app.
    from unittest.mock import MagicMock

    monkeypatch.setattr(settings, "FROZEN", False)
    assert not settings._divert_output_to_log()

    monkeypatch.setattr(settings, "FROZEN", True)
    monkeypatch.setattr(sys, "platform", "win32")

    monkeypatch.setattr(sys, "stdout", MagicMock())
    monkeypatch.setattr(sys, "stderr", MagicMock())
    assert not settings._divert_output_to_log(), "console exe must keep printing"

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert settings._divert_output_to_log(), "windowed exe must be diverted"

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "stdout", MagicMock())
    monkeypatch.setattr(sys, "stderr", MagicMock())
    assert settings._divert_output_to_log(), "the mac .app is always windowed"
