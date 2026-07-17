import importlib
import sys
from pathlib import Path

import pytest

from birdnet_analyzer import settings


@pytest.fixture
def error_log(monkeypatch, tmp_path):
    log_path = tmp_path / "error_log.txt"
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(log_path))

    return log_path


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

    reloaded_settings.ensure_settings_file()
    reloaded_settings.set_state("train-data-dir", "/tmp/train")
    reloaded_settings.write_error_log(RuntimeError("gui path test"))

    assert (expected_appdir / "gui-settings.json").exists()
    assert (expected_appdir / "state.json").exists()
    assert (expected_appdir / "error_log.txt").exists()


def _raised(message: str) -> Exception:
    """Returns an exception that carries a stacktrace, as write_error_log gets it."""
    try:
        raise RuntimeError(message)
    except RuntimeError as e:
        return e


def test_long_stacktraces_are_shortened(monkeypatch, error_log):
    monkeypatch.setattr(settings, "MAX_ERROR_LOG_ENTRY_SIZE", 200)
    settings.write_error_log(_raised("x" * 5000 + "tail"))

    content = error_log.read_text(encoding="utf-8")

    assert len(content) < 500
    assert "characters omitted" in content
    # Both ends of the stacktrace survive, so the exception stays identifiable.
    assert "Traceback (most recent call last)" in content
    assert content.rstrip().endswith("tail")


def test_error_log_is_rotated_when_it_gets_too_large(monkeypatch, error_log):
    monkeypatch.setattr(settings, "MAX_ERROR_LOG_SIZE", 1000)
    error_log.write_text("old entry\n" + "x" * 1000, encoding="utf-8")

    settings.write_error_log(RuntimeError("new entry"))

    backup = error_log.with_name("error_log.txt.1")

    assert backup.read_text(encoding="utf-8").startswith("old entry")

    content = error_log.read_text(encoding="utf-8")

    assert "new entry" in content
    assert "old entry" not in content


def test_error_log_is_not_rotated_while_it_is_small(monkeypatch, error_log):
    monkeypatch.setattr(settings, "MAX_ERROR_LOG_SIZE", 1000)
    settings.write_error_log(RuntimeError("old entry"))
    settings.write_error_log(RuntimeError("new entry"))

    content = error_log.read_text(encoding="utf-8")

    assert not error_log.with_name("error_log.txt.1").exists()
    assert "old entry" in content
    assert "new entry" in content


def test_log_file_is_rotated_when_it_gets_too_large(tmp_path):
    path = tmp_path / "logs.txt"
    log = settings._RotatingLogFile(path, 100)

    log.write("old output\n")
    log.write("x" * 100 + "\n")
    log.write("new output\n")
    log.flush()

    assert "old output" in path.with_name("logs.txt.1").read_text(encoding="utf-8")

    content = path.read_text(encoding="utf-8")

    assert content == "new output\n"

    log.close()


def test_log_file_is_rotated_when_it_is_already_too_large_on_startup(tmp_path):
    path = tmp_path / "logs.txt"
    path.write_text("output of the previous session\n" * 100, encoding="utf-8")

    log = settings._RotatingLogFile(path, 100)
    log.write("new session\n")
    log.flush()

    backup = path.with_name("logs.txt.1")

    assert "previous session" in backup.read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "new session\n"

    log.close()


def test_log_file_keeps_writing_when_it_cannot_be_rotated(monkeypatch, tmp_path):
    # On Windows the rename fails while another process holds the log open. Losing the
    # output of the running app would be worse than an oversized log file.
    def deny_replace(self, target):
        raise PermissionError(target)

    path = tmp_path / "logs.txt"
    log = settings._RotatingLogFile(path, 100)
    monkeypatch.setattr(Path, "replace", deny_replace)

    for _ in range(50):
        log.write("x" * 10 + "\n")

    log.flush()

    assert not path.with_name("logs.txt.1").exists()
    assert len(path.read_text(encoding="utf-8")) == 550

    log.close()


def test_log_file_behaves_like_the_stream_it_replaces(tmp_path):
    log = settings._RotatingLogFile(tmp_path / "logs.txt", 100)

    assert log.writable()
    assert not log.isatty()
    assert log.encoding == "utf-8"
    assert log.fileno() > 0
    assert log.write("written\n") == len("written\n")

    print("through print", file=log)
    log.close()

    assert log.closed
    assert (tmp_path / "logs.txt").read_text(encoding="utf-8") == (
        "written\nthrough print\n"
    )
