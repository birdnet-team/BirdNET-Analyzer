import codecs
import importlib
import json
import sys
import threading
from pathlib import Path

import pytest

from birdnet_analyzer import logs, settings


@pytest.fixture
def appdir(monkeypatch, tmp_path):
    """Points the GUI settings and state at a not yet existing APPDIR in tmp_path."""
    path = tmp_path / "BirdNET-Analyzer-GUI"
    monkeypatch.setattr(settings, "APPDIR", path)
    monkeypatch.setattr(settings, "GUI_SETTINGS_PATH", str(path / "gui-settings.json"))
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(path / "state.json"))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    logs.setup_logging()

    yield path

    logs._remove_installed_handlers()


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_a_corrupted_settings_file_falls_back_to_the_defaults(
    appdir, tmp_path, content
):
    appdir.mkdir()
    (appdir / "gui-settings.json").write_text(content, encoding="utf-8")

    assert settings.get_setting("language-id", "en") == "en"
    assert settings.theme() == "light"
    assert "gui-settings.json" in (tmp_path / "error_log.txt").read_text(
        encoding="utf-8"
    )


def test_a_settings_file_with_a_byte_order_mark_is_read(appdir):
    appdir.mkdir()
    (appdir / "gui-settings.json").write_bytes(
        codecs.BOM_UTF8 + json.dumps({"theme": "dark"}).encode()
    )

    assert settings.theme() == "dark"

    settings.set_setting("language-id", "de")

    assert json.loads((appdir / "gui-settings.json").read_text(encoding="utf-8")) == {
        "theme": "dark",
        "language-id": "de",
    }


def test_settings_are_only_written_by_set_setting(appdir):
    assert settings.get_setting("theme", "light") == "light"
    assert not appdir.exists()

    settings.set_setting("theme", "dark")

    assert json.loads((appdir / "gui-settings.json").read_text(encoding="utf-8")) == {
        "theme": "dark"
    }


def test_a_failed_write_leaves_the_settings_file_intact(appdir, monkeypatch):
    settings.set_setting("theme", "dark")
    before = (appdir / "gui-settings.json").read_text(encoding="utf-8")

    def fail_midway(obj, fp, **kwargs):
        fp.write('{"lang')
        raise TypeError("not serializable")

    monkeypatch.setattr(json, "dump", fail_midway)

    with pytest.raises(TypeError):
        settings.set_setting("language-id", "de")

    assert (appdir / "gui-settings.json").read_text(encoding="utf-8") == before
    assert [path.name for path in appdir.iterdir()] == ["gui-settings.json"]


def lock_for_reading(monkeypatch, locked: Path):
    def guarded_open(file, mode="r", *args, **kwargs):
        if mode == "r" and Path(file) == locked:
            raise PermissionError(13, "locked", str(file))

        return open(file, mode, *args, **kwargs)

    monkeypatch.setattr(settings, "open", guarded_open, raising=False)


def test_a_settings_file_that_cannot_be_read_is_not_overwritten(appdir, monkeypatch):
    settings.set_setting("theme", "dark")
    settings.set_setting("language-id", "de")
    before = (appdir / "gui-settings.json").read_text(encoding="utf-8")
    lock_for_reading(monkeypatch, appdir / "gui-settings.json")

    assert settings.theme() == "light"

    with pytest.raises(PermissionError):
        settings.set_setting("theme", "light")

    assert (appdir / "gui-settings.json").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    "write",
    [
        lambda: settings.set_state("train-data-dir", "/elsewhere"),
        lambda: settings.update_tab_settings("multi", {"confidence_slider": 0.5}),
    ],
    ids=["set_state", "update_tab_settings"],
)
def test_a_state_file_that_cannot_be_read_is_not_overwritten(
    appdir, tmp_path, monkeypatch, write
):
    settings.set_state("train-data-dir", "/tmp/train")
    settings.set_tab_setting("multi", "confidence_slider", 0.1)
    before = (appdir / "state.json").read_text(encoding="utf-8")
    lock_for_reading(monkeypatch, appdir / "state.json")

    write()

    assert (appdir / "state.json").read_text(encoding="utf-8") == before
    assert "locked" in (tmp_path / "error_log.txt").read_text(encoding="utf-8")


def test_simultaneous_settings_writes_keep_every_value(appdir):
    keys = [f"key_{i}" for i in range(8)]
    readers = 4

    for round_ in range(20):
        barrier = threading.Barrier(len(keys) + readers)

        def write(key, round_=round_, barrier=barrier):
            barrier.wait()
            settings.set_setting(key, round_)

        def read(barrier=barrier):
            barrier.wait()
            for _ in range(10):
                settings.theme()

        threads = [threading.Thread(target=write, args=(key,)) for key in keys]
        threads += [threading.Thread(target=read) for _ in range(readers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert {key: settings.get_setting(key) for key in keys} == dict.fromkeys(
            keys, round_
        )


def test_reading_a_missing_state_file_creates_nothing(appdir):
    assert settings.get_state_dict() == {}
    assert not appdir.exists()

    settings.set_state("train-data-dir", "/tmp/train")

    assert json.loads((appdir / "state.json").read_text(encoding="utf-8")) == {
        "train-data-dir": "/tmp/train"
    }


@pytest.fixture
def reloadable_settings():
    module_state = dict(vars(settings))

    yield settings

    vars(settings).update(module_state)


def test_gui_runtime_files_use_user_appdir_when_not_frozen(
    reloadable_settings, monkeypatch, tmp_path
):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    reloaded_settings = importlib.reload(reloadable_settings)
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
        reloaded_settings.set_setting("theme", "light")
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
