import json
from pathlib import Path

import pytest

import birdnet_analyzer.gui.localization as loc
from birdnet_analyzer import logs, settings


@pytest.fixture
def appdir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "APPDIR", tmp_path)
    monkeypatch.setattr(
        settings, "GUI_SETTINGS_PATH", str(tmp_path / "gui-settings.json")
    )
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr(loc, "TARGET_LANGUAGE", "de")
    monkeypatch.setattr(loc, "LANGUAGE_LOOKUP", {})
    logs.setup_logging()

    yield tmp_path

    logs._remove_installed_handlers()


def fallback_lookup() -> dict:
    path = Path(settings.LANG_DIR) / f"{settings.FALLBACK_LANGUAGE}.json"

    return json.loads(path.read_text(encoding="utf-8"))


def test_a_corrupted_settings_file_falls_back_to_the_default_language(appdir):
    (appdir / "gui-settings.json").write_text("{not json", encoding="utf-8")

    loc.load_local_state()

    assert loc.TARGET_LANGUAGE == settings.FALLBACK_LANGUAGE
    assert fallback_lookup() == loc.LANGUAGE_LOOKUP


@pytest.mark.parametrize("language", ["xx", "DE"])
def test_an_unknown_language_falls_back_to_the_default_strings(appdir, language):
    (appdir / "gui-settings.json").write_text(
        json.dumps({"language-id": language}), encoding="utf-8"
    )

    loc.load_local_state()

    assert loc.TARGET_LANGUAGE == settings.FALLBACK_LANGUAGE
    assert fallback_lookup() == loc.LANGUAGE_LOOKUP
