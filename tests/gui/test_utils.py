import base64
import errno
import inspect
import io
import logging
import os
import sys
from types import SimpleNamespace

import pytest

gr = pytest.importorskip("gradio")

from PIL import Image  # noqa: E402

from birdnet_analyzer.gui import state as gs  # noqa: E402
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
        SimpleNamespace(shell=shell, shellcon=SimpleNamespace(CSIDL_NETHOOD=0)),
    )
    unhandled = []
    monkeypatch.setattr(gu.utils, "write_error_log", unhandled.append)

    with caplog.at_level(logging.ERROR, logger=gu.logger.name):
        assert gu._get_network_shortcuts() == [r"\\server\readable"]

    assert unhandled == []
    assert [r.message for r in caplog.records if "unreadable" in r.message]


def test_a_profile_without_network_shortcuts_has_none(monkeypatch, tmp_path):
    def folder_path(hwnd, csidl, token, flags):
        # The shell refuses to name a folder that does not exist unless told not
        # to verify it.
        if not csidl & 0x4000:
            raise OSError("The system cannot find the path specified.")

        return str(tmp_path / "gone")

    monkeypatch.setitem(sys.modules, "pythoncom", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "win32com.shell",
        SimpleNamespace(
            shell=SimpleNamespace(SHGetFolderPath=folder_path),
            shellcon=SimpleNamespace(CSIDL_NETHOOD=0),
        ),
    )
    unhandled = []
    monkeypatch.setattr(gu.utils, "write_error_log", unhandled.append)

    assert gu._get_network_shortcuts() == []
    assert unhandled == []


def test_the_error_handler_keeps_the_handlers_signature():
    def handler(path: str, progress=gr.Progress()):
        return path

    decorated = gu.gui_runtime_error_handler(handler)

    assert inspect.signature(decorated) == inspect.signature(handler)
    assert decorated.__wrapped__ is handler

    with gr.Blocks() as demo:
        gr.Button().click(decorated, gr.Textbox(), gr.Textbox())

    assert [fn.tracks_progress for fn in demo.fns.values()] == [True]


def test_a_gr_error_from_the_handler_keeps_its_message_and_stays_open(
    monkeypatch, caplog
):
    error = gr.Error("x")
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    def handler():
        raise error

    with (
        caplog.at_level(logging.INFO, logger=gu.logger.name),
        pytest.raises(gr.Error) as raised,
    ):
        gu.gui_runtime_error_handler(handler)()

    assert raised.value is error
    assert raised.value.message == "x"
    assert raised.value.duration is None
    assert logged == []
    assert [(r.levelno, r.message) for r in caplog.records] == [(logging.INFO, "x")]


def test_a_gr_error_wrapping_a_failure_is_logged_with_it(monkeypatch):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    def handler():
        try:
            raise OSError("disk")
        except OSError as e:
            raise gr.Error("step") from e

    with pytest.raises(gr.Error) as raised:
        gu.gui_runtime_error_handler(handler)()

    assert raised.value.message == "step"
    assert logged == [raised.value]


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


@pytest.fixture
def download_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(gu.utils, "write_error_log", lambda e: None)
    monkeypatch.setenv("BIRDNET_APP_DATA", str(tmp_path))
    monkeypatch.delenv("BIRDNET_OFFLINE", raising=False)
    monkeypatch.setitem(
        gu.loc.LANGUAGE_LOOKUP, "model-download-failed-error", "{reason}|{directory}"
    )

    def message_for(error):
        def handler():
            raise error

        with pytest.raises(gr.Error) as raised:
            gu.gui_runtime_error_handler(handler)()

        assert raised.value.__cause__ is error

        return raised.value.message

    return message_for


def test_a_failed_connection_names_the_host_and_the_model_directory(
    download_failure, tmp_path
):
    import requests

    request = requests.Request("GET", "https://zenodo.org/records/1/model.onnx")
    error = requests.ConnectionError("Max retries exceeded", request=request.prepare())

    assert download_failure(error) == f"ConnectionError, zenodo.org|{tmp_path}"


def test_a_refused_download_is_shown_on_one_line(download_failure, tmp_path):
    from birdnet.utils.downloads import DownloadError

    error = DownloadError(
        "Failed to download the file. Status code: 200\n"
        "Expected size: 5 bytes, downloaded size: 3 bytes.",
        status_code=200,
    )

    assert download_failure(error) == (
        "Failed to download the file. Status code: 200 "
        f"Expected size: 5 bytes, downloaded size: 3 bytes|{tmp_path}"
    )


def test_downloads_switched_off_by_the_user_are_reported_as_such(
    download_failure, monkeypatch
):
    from birdnet.utils.downloads import DownloadError

    monkeypatch.setenv("BIRDNET_OFFLINE", "1")

    assert download_failure(DownloadError("disabled by BIRDNET_OFFLINE")) == (
        "disabled by BIRDNET_OFFLINE"
    )


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            FileNotFoundError(2, "No such file or directory", "birds.wav"),
            "gone|birds.wav",
        ),
        (PermissionError(13, "Permission denied", "R&D.csv"), "locked|R&amp;D.csv"),
        (OSError(errno.EROFS, "Read-only file system", "card.csv"), "locked|card.csv"),
        (OSError(errno.ENOSPC, "No space left on device", "out.csv"), "full"),
        (FileNotFoundError("No database found in db"), "No database found in db"),
        (ValueError("Set both lat and lon."), "Set both lat and lon."),
        (ValueError("a <b> c"), "a &lt;b&gt; c"),
        (
            RuntimeError("Resume state is incomplete. Delete 'x' and run again."),
            "Resume state is incomplete. Delete 'x' and run again.",
        ),
        (EOFError(), "EOFError()"),
        (
            IndexError("tuple index out of range"),
            "bug|IndexError: tuple index out of range",
        ),
        (TypeError("'<' not supported"), "bug|TypeError: '&lt;' not supported"),
        (TypeError("word " * 100), "bug|TypeError: " + "word " * 59 + "..."),
        (AssertionError(), "bug|AssertionError"),
    ],
    ids=lambda value: value if isinstance(value, str) else type(value).__name__,
)
def test_an_error_no_handler_explained_is_put_into_the_users_words(
    monkeypatch, error, message
):
    monkeypatch.setattr(gu.utils, "write_error_log", lambda e: None)
    texts = {
        "error-path-not-found": "gone|{path}",
        "error-path-no-access": "locked|{path}",
        "error-disk-full": "full",
        "error-unexpected": "bug|{error}",
    }

    for key, text in texts.items():
        assert key in gu.loc.LANGUAGE_LOOKUP
        monkeypatch.setitem(gu.loc.LANGUAGE_LOOKUP, key, text)

    def handler():
        raise error

    with pytest.raises(gr.Error) as raised:
        gu.gui_runtime_error_handler(handler)()

    assert raised.value.message == message


def test_a_missing_folder_is_named_instead_of_the_file_that_was_to_be_written(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(gu.utils, "write_error_log", lambda e: None)
    monkeypatch.setitem(gu.loc.LANGUAGE_LOOKUP, "error-path-not-found", "gone|{path}")
    species_list = tmp_path / "unplugged" / "lists" / "species.txt"

    def handler():
        species_list.write_text("Parus major_Great Tit")

    with pytest.raises(gr.Error) as raised:
        gu.gui_runtime_error_handler(handler)()

    assert raised.value.message == f"gone|{tmp_path / 'unplugged'}"


def test_a_language_is_saved_and_announced_once_per_choice(
    appdir, handler, monkeypatch
):
    warned = []
    monkeypatch.setattr(gr, "Warning", lambda message, **_: warned.append(message))
    on_language_change = handler(gu.build_settings, "on_language_change")

    on_language_change("de")
    on_language_change("de")

    assert gu.settings.get_setting("language-id") == "de"
    assert len(warned) == 1


@pytest.mark.parametrize(
    ("name", "value"), [("on_theme_change", "dark"), ("on_language_change", "de")]
)
def test_a_failed_settings_write_is_logged_and_shown(
    appdir, handler, monkeypatch, name, value
):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(gu.settings, "set_setting", fail)

    with pytest.raises(gr.Error, match="disk"):
        handler(gu.build_settings, name)(value)

    assert len(logged) == 1


def _species_controls():
    gu.sample_species_model_settings(gs.TabState("single"))


def test_a_failing_species_match_is_logged_at_debug_and_swallowed(
    appdir, handler, monkeypatch, caplog, tmp_path
):
    from birdnet_analyzer import model_utils

    species_file = tmp_path / "species.txt"
    species_file.write_text("Turdus merula_Common Blackbird\n")
    monkeypatch.setattr(model_utils, "acoustic_model_downloaded", lambda version: True)
    monkeypatch.setattr(
        model_utils, "acoustic_species_list", lambda version, locale: ["a"]
    )

    def fail(*args, **kwargs):
        raise RuntimeError("matcher bug")

    monkeypatch.setattr(model_utils, "match_species_to_model", fail)
    warn = handler(_species_controls, "warn_unmatched_species")

    with caplog.at_level(logging.DEBUG, logger=gu.logger.name):
        assert warn(str(species_file), gu.USE_BIRDNET_3_0, "en_us") is None

    debug = [r for r in caplog.records if r.levelno == logging.DEBUG and r.exc_info]
    assert [str(r.exc_info[1]) for r in debug] == ["matcher bug"]


def test_a_species_named_in_english_is_not_reported_in_another_locale(
    appdir, handler, monkeypatch, tmp_path
):
    from birdnet_analyzer import model_utils
    from birdnet_analyzer.gui import localization as loc

    species_file = tmp_path / "species.txt"
    species_file.write_text(
        "Troglodytes troglodytes_Eurasian Wren\nPekania pennanti_Fisher\n"
    )
    labels = {
        "de": ["Nannus troglodytes_Zaunkönig"],
        "en_us": ["Nannus troglodytes_Eurasian Wren"],
    }
    monkeypatch.setattr(model_utils, "acoustic_model_downloaded", lambda version: True)
    monkeypatch.setattr(
        model_utils, "acoustic_species_list", lambda version, locale: labels[locale]
    )
    warnings = []
    monkeypatch.setattr(gr, "Warning", lambda message, **_: warnings.append(message))
    warn = handler(_species_controls, "warn_unmatched_species")

    warn(str(species_file), gu.USE_BIRDNET_3_0, "de")

    assert warnings == [loc.localize("species-list-unmatched-warning").format(count=1)]


def test_an_unreadable_species_list_is_logged_and_shown(appdir, handler, monkeypatch):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(gu.utils, "read_lines", fail)

    with pytest.raises(gr.Error, match="disk"):
        handler(_species_controls, "on_species_file_change")("species.txt")

    assert len(logged) == 1


def test_the_splash_is_closed_only_when_the_bootloader_started_one(monkeypatch):
    closed = []
    monkeypatch.setitem(
        sys.modules, "pyi_splash", SimpleNamespace(close=lambda: closed.append(True))
    )

    monkeypatch.delenv("_PYI_SPLASH_IPC", raising=False)
    gu._close_splash()
    assert closed == []

    monkeypatch.setenv("_PYI_SPLASH_IPC", "1")
    gu._close_splash()
    assert closed == [True]


class _WindowEvent(list):
    def __iadd__(self, handler):
        self.append(handler)
        return self


def _open_window_with_a_fake_webview(monkeypatch, screen=(1920, 1080)):
    """Runs open_window up to the point where the real one blocks.

    Returns the URL and the other arguments of the window.
    """
    windows = []
    events = SimpleNamespace(shown=_WindowEvent(), loaded=_WindowEvent())
    webview = SimpleNamespace(
        settings={},
        screens=[SimpleNamespace(width=screen[0], height=screen[1])],
        create_window=lambda title, url, **kwargs: (
            windows.append((url, kwargs))
            or SimpleNamespace(events=events, uid="master")
        ),
        start=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setitem(sys.modules, "webview.platforms", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "webview.platforms.winforms", SimpleNamespace(BrowserView=None)
    )
    monkeypatch.setattr(gu, "_WINDOW", None)
    monkeypatch.setattr(gu, "_URL", "")
    monkeypatch.setattr(gu, "_get_network_shortcuts", list)
    monkeypatch.setattr(gu, "_exit_within", lambda seconds: None)
    monkeypatch.setattr(gu, "shutdown_running_analyses", lambda: None)

    try:
        gu.open_window(lambda: gr.Markdown("tab"))
    finally:
        gr.close_all()

    return windows[0]


@pytest.mark.parametrize("screen", [(1366, 768), (1280, 720), (1024, 600)])
def test_the_window_opens_maximized_on_a_screen_it_does_not_fit(
    appdir, monkeypatch, screen
):
    _, size = _open_window_with_a_fake_webview(monkeypatch, screen)

    assert size["maximized"]
    assert size["width"] <= screen[0]
    assert size["height"] <= screen[1]
    assert size["min_size"][0] < screen[0]
    assert size["min_size"][1] < screen[1]


def test_the_window_keeps_its_size_on_a_screen_it_fits(appdir, monkeypatch):
    _, size = _open_window_with_a_fake_webview(monkeypatch, (1920, 1080))

    assert (size["width"], size["height"]) == (1300, 900)
    assert size["min_size"] == (1300, 900)
    assert not size.get("maximized")


def _proxy_from_the_environment(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")


def _proxy_from_the_windows_settings(monkeypatch):
    import urllib.request

    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(
        urllib.request, "getproxies_registry", lambda: {"http": "127.0.0.1:9"}
    )


@pytest.mark.parametrize(
    "configure_proxy",
    [
        _proxy_from_the_environment,
        pytest.param(
            _proxy_from_the_windows_settings,
            marks=pytest.mark.skipif(
                sys.platform != "win32", reason="the registry holds them on Windows"
            ),
        ),
    ],
)
def test_the_window_opens_although_an_unreachable_proxy_is_configured(
    appdir, monkeypatch, configure_proxy
):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    configure_proxy(monkeypatch)

    url, _ = _open_window_with_a_fake_webview(monkeypatch)

    assert url.startswith("http://127.0.0.1:")
    assert "NO_PROXY" not in os.environ


def test_a_proxy_exclusion_of_the_user_is_kept(appdir, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "intranet.example")

    _open_window_with_a_fake_webview(monkeypatch)

    assert os.environ["NO_PROXY"] == "intranet.example"
