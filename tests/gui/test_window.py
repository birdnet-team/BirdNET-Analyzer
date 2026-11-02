"""The desktop window the GUI opens."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import birdnet_analyzer
from birdnet_analyzer import settings

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.utils as gu  # noqa: E402

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


@pytest.fixture
def fake_webview(monkeypatch):
    """Stands in for pywebview and records the window it would open."""

    class Event:
        def __init__(self):
            self.handlers = []

        def __iadd__(self, handler):
            self.handlers.append(handler)
            return self

        def fire(self):
            for handler in self.handlers:
                handler()

    windows = []

    def create_window(title, url, **kwargs):
        window = SimpleNamespace(
            uid="window",
            url=url,
            kwargs=kwargs,
            events=SimpleNamespace(shown=Event(), loaded=Event()),
        )
        windows.append(window)

        return window

    webview = SimpleNamespace(
        settings={}, create_window=create_window, start=lambda **kwargs: None
    )
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setitem(sys.modules, "webview.platforms", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "webview.platforms.winforms",
        SimpleNamespace(BrowserView=SimpleNamespace(instances={})),
    )

    def launch(self, *args, **kwargs):
        return None, "http://127.0.0.1:7860/", None

    monkeypatch.setattr(gr.Blocks, "launch", launch)
    monkeypatch.setattr(gu, "_get_network_shortcuts", list)
    monkeypatch.setattr(gu, "shutdown_running_analyses", lambda: None)
    monkeypatch.setattr(gu, "_WINDOW", None)

    return windows


@pytest.mark.parametrize(
    ("theme", "background"),
    [("dark", gr.themes.Default().neutral_950), ("light", "#ffffff")],
)
def test_the_window_opens_in_the_colors_of_the_theme(
    appdir, fake_webview, theme, background
):
    settings.set_setting("theme", theme)

    gu.open_window([])

    (window,) = fake_webview
    assert window.url.endswith(f"?__theme={theme}")
    assert window.kwargs["background_color"] == background


def test_the_splash_screen_stays_until_the_window_is_shown(
    appdir, fake_webview, monkeypatch
):
    closed = []
    monkeypatch.setitem(
        sys.modules, "pyi_splash", SimpleNamespace(close=lambda: closed.append(True))
    )

    gu.open_window([])

    (window,) = fake_webview
    assert closed == []
    window.events.shown.fire()
    assert closed == [True]


def test_the_species_map_is_drawn_on_tab_selection_not_at_page_load(
    appdir, fake_webview, monkeypatch
):
    from birdnet_analyzer.gui import species

    built = []

    def launch_instead(demo, *args, **kwargs):
        built.append(demo)

        return None, "http://127.0.0.1:7860/", None

    monkeypatch.setattr(gr.Blocks, "launch", launch_instead)

    gu.open_window([species.build_species_tab])

    (demo,) = built
    maps = {
        block._id
        for block in demo.blocks.values()
        if isinstance(block, gr.Plot)
        and block.elem_id != "species-progress-placeholder"
    }

    def outputs_of(trigger):
        return {
            output._id
            for event in demo.fns.values()
            if any(event_trigger == trigger for _, event_trigger in event.targets)
            for output in event.outputs
        }

    assert maps
    assert maps <= outputs_of("select")
    assert not maps & outputs_of("load")
