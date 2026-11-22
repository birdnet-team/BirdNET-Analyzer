"""The textboxes that show a selected file or folder."""

import ast
import unicodedata
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from birdnet_analyzer.gui import utils as gu  # noqa: E402

GUI_DIR = Path(gu.__file__).parent
PATHS = ["/data/Site A (north)/", "C:\\Users\\jo\\rec (1).wav", "relative/file.wav"]


def path_textbox():
    return gu.PathTextbox(rtl=True, interactive=False, elem_classes="path-textbox")


def as_a_session_copy(textbox):
    # What gradio does with a component once an event updated one of its properties.
    return type(textbox)(**textbox.constructor_args)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "variant", [path_textbox, lambda: as_a_session_copy(path_textbox())]
)
def test_a_path_is_shown_as_left_to_right_text_and_handed_back_unchanged(path, variant):
    textbox = variant()

    shown = textbox.postprocess(path)

    assert unicodedata.bidirectional(shown[0]) == "L"
    assert unicodedata.bidirectional(shown[-1]) == "L"
    assert path in shown
    assert textbox.preprocess(shown) == path


@pytest.mark.parametrize("nothing", [None, ""])
def test_no_selection_stays_empty(nothing):
    assert path_textbox().postprocess(nothing) == nothing
    assert path_textbox().preprocess(nothing) == nothing


def test_the_page_renders_it_as_an_ordinary_right_to_left_textbox():
    config = path_textbox().get_config()

    assert config["name"] == "textbox"
    assert config["rtl"] is True


def test_defining_it_leaves_no_stub_file_beside_the_module():
    assert not Path(gu.__file__).with_suffix(".pyi").exists()


def test_every_right_to_left_textbox_of_the_gui_is_one():
    plain = []

    for module in sorted(GUI_DIR.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        plain += [
            f"{module.name}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and any(keyword.arg == "rtl" for keyword in node.keywords)
            and ast.unparse(node.func) != "gu.PathTextbox"
        ]

    assert not plain
