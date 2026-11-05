"""Where the frozen app keeps matplotlib's font cache between launches."""

import os
import subprocess
import sys

import pytest

from birdnet_analyzer import settings

HOOK_DIR = "/tmp/matplotlib-from-the-pyinstaller-hook"


def mplconfigdir_after_import(tmp_path, *, frozen: bool) -> str:
    """Imports birdnet_analyzer in a fresh interpreter, started the way the
    PyInstaller hook leaves it, and returns the MPLCONFIGDIR the import left."""
    home = tmp_path / "home"
    home.mkdir()
    out = tmp_path / "mplconfigdir.txt"
    env = dict(os.environ)
    env["HOME"] = env["USERPROFILE"] = str(home)
    env["MPLCONFIGDIR"] = HOOK_DIR
    freeze = "import sys; sys.frozen = True; sys._MEIPASS = '.'; " if frozen else ""

    subprocess.run(
        [
            sys.executable,
            "-c",
            f"{freeze}import os, pathlib, birdnet_analyzer; "
            f"pathlib.Path({str(out)!r}).write_text(os.environ['MPLCONFIGDIR'])",
        ],
        env=env,
        timeout=110,
        check=True,
    )

    return out.read_text()


@pytest.mark.skipif(sys.platform != "darwin", reason="the mac app only")
def test_the_frozen_mac_app_keeps_the_cache_in_its_app_dir(tmp_path):
    configdir = mplconfigdir_after_import(tmp_path, frozen=True)

    assert configdir.startswith(str(tmp_path / "home"))
    assert configdir.endswith(os.path.join(settings.APP_NAME, "matplotlib"))


@pytest.mark.skipif(sys.platform == "darwin", reason="the mac app differs")
def test_other_frozen_builds_keep_the_hooks_directory(tmp_path):
    assert mplconfigdir_after_import(tmp_path, frozen=True) == HOOK_DIR


def test_a_source_run_leaves_matplotlib_alone(tmp_path):
    assert mplconfigdir_after_import(tmp_path, frozen=False) == HOOK_DIR
