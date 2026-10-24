"""Starting the GUI through ``gui.main``, up to the window."""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui  # noqa: E402
import birdnet_analyzer.gui.localization as loc  # noqa: E402
import birdnet_analyzer.gui.single_file as sfa  # noqa: E402
import birdnet_analyzer.gui.utils as gu  # noqa: E402
from birdnet_analyzer import logs, settings  # noqa: E402

LABEL_KEYS = [
    "species-list-radio-option-custom-list",
    "single-tab-output-header-confidence",
]

IMPORT_PROBE = """
import json
import os
import sys
from pathlib import Path

import birdnet_analyzer
from birdnet_analyzer import settings

assert settings.APPDIR.is_relative_to(Path(sys.argv[1])), settings.APPDIR
settings.set_setting("language-id", "de")

appdir = os.path.normcase(os.path.abspath(settings.APPDIR))
reads = []


def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if path.startswith(appdir):
            reads.append(path)


sys.addaudithook(audit)

import birdnet_analyzer.gui.single_file as sfa
import birdnet_analyzer.gui.utils as gu

reads_on_import = list(reads)
on_import = [gu.CUSTOM_SPECIES, sfa.HEADER_CONFIDENCE_LBL]

import birdnet_analyzer.gui.localization as loc

loc.load_local_state()

print(
    json.dumps(
        {
            "reads": reads_on_import,
            "on_import": on_import,
            "loaded": [gu.CUSTOM_SPECIES, sfa.HEADER_CONFIDENCE_LBL],
        }
    )
)
"""


def labels(language: str) -> list[str]:
    path = Path(settings.LANG_DIR) / f"{language}.json"
    lookup = json.loads(path.read_text(encoding="utf-8"))

    return [lookup[key] for key in LABEL_KEYS]


def test_importing_the_gui_reads_no_settings_and_localizes_labels_on_access(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=110,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout.splitlines()[-1])

    assert seen["reads"] == []
    assert seen["on_import"] == labels("en")
    assert seen["loaded"] == labels("de")


@pytest.fixture
def gui_main(monkeypatch):
    """Runs ``gui.main`` and undoes what it configures for the process."""
    monkeypatch.setattr(loc, "TARGET_LANGUAGE", loc.TARGET_LANGUAGE)
    monkeypatch.setattr(loc, "LANGUAGE_LOOKUP", loc.LANGUAGE_LOOKUP)

    def run():
        with mock.patch.dict(os.environ), warnings.catch_warnings():
            birdnet_analyzer.gui.main()

    yield run

    logs._remove_installed_handlers()


def test_the_window_opens_in_the_language_of_the_settings(monkeypatch, gui_main):
    settings.set_setting("language-id", "de")
    seen = []
    monkeypatch.setattr(
        gu,
        "open_window",
        lambda builders: seen.append([gu.CUSTOM_SPECIES, sfa.HEADER_CONFIDENCE_LBL]),
    )

    gui_main()

    assert seen == [labels("de")]


def test_the_window_opens_despite_a_corrupted_settings_file(monkeypatch, gui_main):
    Path(settings.GUI_SETTINGS_PATH).write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(loc, "TARGET_LANGUAGE", "de")
    opened = []
    monkeypatch.setattr(gu, "open_window", opened.append)

    gui_main()

    assert len(opened) == 1
    assert loc.TARGET_LANGUAGE == settings.FALLBACK_LANGUAGE


class WindowBuilt(Exception):
    pass


def test_the_window_is_built_without_a_warning(monkeypatch, gui_main):
    def launch_instead(*args, **kwargs):
        raise WindowBuilt

    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace())
    monkeypatch.setattr(gr.Blocks, "launch", launch_instead)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        with pytest.raises(WindowBuilt):
            gui_main()

    assert not caught, "\n".join(
        f"{w.category.__name__} at {w.filename}:{w.lineno}: {w.message}" for w in caught
    )
