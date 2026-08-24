"""The model-directory resolver and the directory probe.

``apply_model_directory`` runs at package import, before anything can import
``birdnet`` (which freezes its model directory from ``BIRDNET_APP_DATA`` at its
own import time).
"""

import json
import os
import subprocess
import sys

import pytest

from birdnet_analyzer import settings


@pytest.fixture
def clean(monkeypatch, tmp_path):
    monkeypatch.delenv(settings.MODEL_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(settings, "GUI_SETTINGS_PATH", str(tmp_path / "gui.json"))
    monkeypatch.setattr(settings, "FROZEN", False)
    monkeypatch.setattr(settings, "MODEL_DIR_STARTUP_WARNING", None)
    return tmp_path


def store(tmp_path, value):
    (tmp_path / "gui.json").write_text(
        json.dumps({settings.MODEL_DIR_SETTING_KEY: value}), encoding="utf-8"
    )


def test_source_run_with_nothing_configured_leaves_the_library_default(clean):
    settings.apply_model_directory()

    assert settings.MODEL_DIR_ENV_VAR not in os.environ
    assert settings.MODEL_DIR_STARTUP_WARNING is None


def test_an_existing_env_var_wins_over_the_setting(clean, monkeypatch):
    env_dir = clean / "from-env"
    monkeypatch.setenv(settings.MODEL_DIR_ENV_VAR, str(env_dir))
    store(clean, str(clean / "from-setting"))

    settings.apply_model_directory()

    assert os.environ[settings.MODEL_DIR_ENV_VAR] == str(env_dir)


def test_an_unusable_env_var_fails_fast(clean, monkeypatch):
    blocker = clean / "a-file"
    blocker.write_text("")
    monkeypatch.setenv(settings.MODEL_DIR_ENV_VAR, str(blocker / "sub"))

    with pytest.raises(SystemExit, match="BIRDNET_APP_DATA"):
        settings.apply_model_directory()


def test_the_stored_setting_is_applied(clean):
    target = clean / "chosen"
    store(clean, str(target))

    settings.apply_model_directory()

    assert os.environ[settings.MODEL_DIR_ENV_VAR] == str(target)
    assert target.is_dir(), "the configured directory is created"


def test_an_unusable_setting_falls_back_with_a_warning(clean, monkeypatch):
    blocker = clean / "a-file"
    blocker.write_text("")
    bad = str(blocker / "sub")
    store(clean, bad)
    monkeypatch.setattr(settings, "FROZEN", True)

    settings.apply_model_directory()

    assert bad == settings.MODEL_DIR_STARTUP_WARNING
    assert os.environ[settings.MODEL_DIR_ENV_VAR] == str(
        settings.default_model_directory()
    )


def test_frozen_default_is_local_appdata_on_windows(clean, monkeypatch):
    monkeypatch.setattr(settings, "FROZEN", True)

    settings.apply_model_directory()

    applied = os.environ[settings.MODEL_DIR_ENV_VAR]
    assert applied == str(settings.default_model_directory())
    if sys.platform == "win32":
        assert "AppData\\Local" in applied
        assert applied.endswith("models")


def test_probe_accepts_a_writable_directory(clean):
    assert settings.probe_model_directory(clean / "new") in ("ok", "ok-low-space")


def test_probe_rejects_an_uncreatable_path(clean):
    blocker = clean / "a-file"
    blocker.write_text("")

    assert settings.probe_model_directory(blocker / "sub") == "invalid"


def test_probe_flags_a_readonly_directory_by_content(clean, monkeypatch):
    def deny(_directory):
        raise PermissionError("read-only")

    monkeypatch.setattr(settings, "_test_write", deny)

    empty = clean / "empty-readonly"
    empty.mkdir()
    assert settings.probe_model_directory(empty) == "invalid"

    populated = clean / "seeded-readonly"
    (populated / "acoustic-models").mkdir(parents=True)
    assert settings.probe_model_directory(populated) == "readonly"


def test_resolver_runs_at_package_import(clean):
    # A fresh interpreter with a redirected home applies the stored setting during
    # `import birdnet_analyzer`, so it takes effect for every entry point before
    # birdnet can be imported.
    home = clean / "home"
    appdir = home / (
        "AppData/Roaming/BirdNET-Analyzer-GUI"
        if sys.platform == "win32"
        else (
            "Library/Application Support/BirdNET-Analyzer-GUI"
            if sys.platform == "darwin"
            else ".local/share/BirdNET-Analyzer-GUI"
        )
    )
    appdir.mkdir(parents=True)
    target = clean / "import-time-target"
    (appdir / "gui-settings.json").write_text(
        json.dumps({settings.MODEL_DIR_SETTING_KEY: str(target)}), encoding="utf-8"
    )

    env = dict(os.environ)
    env.pop(settings.MODEL_DIR_ENV_VAR, None)
    env["USERPROFILE"] = str(home)
    env["HOME"] = str(home)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import birdnet_analyzer, os; "
            f"print(os.environ.get('{settings.MODEL_DIR_ENV_VAR}', ''))",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=110,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(target)
