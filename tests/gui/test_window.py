"""The desktop window the GUI opens."""

import ast
import os
import string
import sys
from pathlib import Path

import pytest

import birdnet_analyzer
from birdnet_analyzer import settings

GUI_UTILS = Path(birdnet_analyzer.__file__).parent / "gui" / "utils.py"


def _webview_start_calls() -> list[ast.Call]:
    # Read, not run: pywebview is not in the test extra and CI has no display.
    tree = ast.parse(GUI_UTILS.read_text(encoding="utf-8"))

    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "start"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "webview"
    ]


def test_window_is_not_opened_in_debug_mode():
    calls = _webview_start_calls()

    assert calls, (
        f"No webview.start() call found in {GUI_UTILS}. This test guards the "
        "arguments of that call, so it would pass without checking anything."
    )

    def is_off(value: ast.expr) -> bool:
        return isinstance(value, ast.Constant) and value.value is False

    enabled = [
        keyword
        for call in calls
        for keyword in call.keywords
        if keyword.arg == "debug" and not is_off(keyword.value)
    ]

    assert not enabled, (
        "webview.start() enables debug mode at "
        + ", ".join(f"{GUI_UTILS.name}:{keyword.value.lineno}" for keyword in enabled)
        + ". That ships a devtools context menu to users."
    )


def test_every_font_file_of_the_theme_has_the_latin_letters():
    pytest.importorskip("gradio")
    from fontTools.ttLib import TTFont
    from gradio.themes.utils import fonts

    from birdnet_analyzer.gui import utils as gu

    theme = gu._theme()
    faces = [
        (font.name, weight)
        for font in (*theme._font, *theme._font_mono)
        if isinstance(font, fonts.LocalFont)
        for weight in font.weights
    ]
    assert faces

    letters = {ord(char) for char in string.ascii_letters + string.digits}
    incomplete = [
        face
        for face in faces
        if not letters <= TTFont(fonts.font_file_path(*face)).getBestCmap().keys()
    ]

    assert not incomplete


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS takes no storage path")
def test_the_window_keeps_its_browser_profile_with_the_app_data(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    from birdnet_analyzer.gui import utils as gu

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(settings, "APPDIR", tmp_path / settings.APP_NAME)

    path = gu._webview_storage_path()

    assert path == str(tmp_path / settings.APP_NAME / "webview")
    assert os.path.isdir(path)


def test_the_window_opens_with_the_default_profile_if_its_own_cannot_be_created(
    monkeypatch, tmp_path
):
    pytest.importorskip("gradio")
    from birdnet_analyzer.gui import utils as gu

    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setattr(settings, "webview_storage_directory", lambda: blocker / "sub")

    assert gu._webview_storage_path() is None
