import base64
import inspect
import io
import logging
import os
import sys
from types import SimpleNamespace

import pytest

gr = pytest.importorskip("gradio")

from PIL import Image  # noqa: E402

from birdnet_analyzer.gui import utils as gu  # noqa: E402


@pytest.fixture
def save_dialog(monkeypatch):
    def answer(path):
        webview = SimpleNamespace(FileDialog=SimpleNamespace(SAVE="save"))
        window = SimpleNamespace(create_file_dialog=lambda *a, **k: (str(path),))
        monkeypatch.setitem(sys.modules, "webview", webview)
        monkeypatch.setattr(gu, "_WINDOW", window)

    return answer


def _plot():
    """A semi-transparent image encoded the way gr.Plot hands a figure over."""
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), (255, 0, 0, 128)).save(buffer, "WEBP")
    data = base64.b64encode(buffer.getvalue()).decode()

    return SimpleNamespace(plot=f"data:image/webp;base64,{data}")


def test_a_plot_is_saved_under_the_chosen_name_despite_dots_in_the_path(
    save_dialog, tmp_path
):
    folder = tmp_path / "j.doe"
    folder.mkdir()
    save_dialog(folder / "plot.png")

    gu.download_plot(_plot(), "plot")

    assert os.listdir(folder) == ["plot.png"]
    assert Image.open(folder / "plot.png").format == "PNG"


@pytest.mark.parametrize(
    ("name", "saved", "image_format"),
    [
        ("x.jpg", "x.jpg", "JPEG"),
        ("x.JPEG", "x.JPEG", "JPEG"),
        ("x.webp", "x.webp", "WEBP"),
        ("x.bmp", "x.bmp", "BMP"),
        ("x.gif", "x.gif", "GIF"),
        ("x.tif", "x.tif", "TIFF"),
        ("x.xyz", "x.xyz.png", "PNG"),
    ],
)
def test_a_plot_is_saved_in_the_format_its_name_asks_for(
    save_dialog, tmp_path, name, saved, image_format
):
    save_dialog(tmp_path / name)

    gu.download_plot(_plot(), "plot")

    assert os.listdir(tmp_path) == [saved]
    assert Image.open(tmp_path / saved).format == image_format


def test_one_unreadable_network_shortcut_does_not_hide_the_others(
    monkeypatch, tmp_path, caplog
):
    for name in ("readable", "unreadable"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "target.lnk").touch()

    class FakeShellLink:
        def QueryInterface(self, iid):
            return self

        def Load(self, path):
            self.path = path

        def GetPath(self, flags):
            share = os.path.basename(os.path.dirname(self.path))

            if share == "unreadable":
                raise OSError("broken shortcut")

            return rf"\\server\{share}", None

    shell = SimpleNamespace(
        SHGetFolderPath=lambda *a: str(tmp_path),
        CLSID_ShellLink=None,
        IID_IShellLink=None,
        SLGP_RAWPATH=None,
    )
    pythoncom = SimpleNamespace(
        CoCreateInstance=lambda *a: FakeShellLink(),
        CLSCTX_INPROC_SERVER=None,
        IID_IPersistFile=None,
    )
    monkeypatch.setitem(sys.modules, "pythoncom", pythoncom)
    monkeypatch.setitem(
        sys.modules,
        "win32com.shell",
        SimpleNamespace(shell=shell, shellcon=SimpleNamespace(CSIDL_NETHOOD=None)),
    )
    unhandled = []
    monkeypatch.setattr(gu.utils, "write_error_log", unhandled.append)

    with caplog.at_level(logging.ERROR, logger=gu.logger.name):
        assert gu._get_network_shortcuts() == [r"\\server\readable"]

    assert unhandled == []
    assert [r.message for r in caplog.records if "unreadable" in r.message]


def test_the_error_handler_keeps_the_handlers_signature():
    def handler(path: str, progress=gr.Progress()):
        return path

    decorated = gu.gui_runtime_error_handler(handler)

    assert inspect.signature(decorated) == inspect.signature(handler)
    assert decorated.__wrapped__ is handler

    with gr.Blocks() as demo:
        gr.Button().click(decorated, gr.Textbox(), gr.Textbox())

    assert [fn.tracks_progress for fn in demo.fns.values()] == [True]


def test_a_gr_error_from_the_handler_keeps_its_message_and_stays_open(monkeypatch):
    error = gr.Error("x")
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    def handler():
        raise error

    with pytest.raises(gr.Error) as raised:
        gu.gui_runtime_error_handler(handler)()

    assert raised.value is error
    assert raised.value.message == "x"
    assert raised.value.duration is None
    assert logged == [error]


def test_any_other_error_reaches_the_user_as_a_gr_error(monkeypatch):
    error = ValueError("y")
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    def handler():
        raise error

    with pytest.raises(gr.Error) as raised:
        gu.gui_runtime_error_handler(handler)()

    assert raised.value.message == "y"
    assert raised.value.duration is None
    assert raised.value.__cause__ is error
    assert logged == [error]
