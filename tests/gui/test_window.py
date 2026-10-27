"""The desktop window the GUI opens."""

import ast
from pathlib import Path

import birdnet_analyzer

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
