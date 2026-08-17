"""The GUI shows the birdnet library's model downloads."""

import sys
from unittest.mock import MagicMock

import pytest

gr = pytest.importorskip("gradio")

# gui.utils imports pywebview at module level, which the gui-tests extra lacks.
sys.modules.setdefault("webview", MagicMock(settings={}))


def test_download_progress_routes_library_updates_to_gradio(monkeypatch):
    # First-run model downloads happen inside birdnet.load; the frozen GUI diverts the
    # library's tqdm bar into the log file, so the updates must reach the gradio
    # progress bar (or toasts when there is none) through birdnet's callback hook.
    import birdnet
    import gradio as gr

    from birdnet_analyzer.gui import utils as gu

    calls = []
    infos = []
    warnings = []
    monkeypatch.setattr(gr, "Info", lambda msg, **kw: infos.append(msg))
    monkeypatch.setattr(gr, "Warning", lambda msg, **kw: warnings.append(msg))

    def fake_progress(value, desc=None, **kwargs):
        calls.append((value, desc))

    def registered():
        return birdnet.get_download_progress_callback()

    def report(**kwargs):
        registered()(
            birdnet.DownloadProgress(
                description="Downloading acoustic model v3.0",
                url="https://example.invalid/model",
                attempt=1,
                max_attempts=3,
                **kwargs,
            )
        )

    with gu.download_progress(fake_progress):
        report(status="started", bytes_done=0, bytes_total=1000)
        report(status="progress", bytes_done=250, bytes_total=1000)
        report(status="progress", bytes_done=5_000_000, bytes_total=None)
        report(status="finished", bytes_done=1000, bytes_total=1000)

    assert calls[0][0] == 0.25
    assert "acoustic model v3.0" in calls[0][1]
    assert "Downloading Downloading" not in calls[0][1]
    assert calls[1][0] == 0.0
    assert "5 MB" in calls[1][1]
    assert not infos, "no toasts while a progress bar is available"

    # Without a bar, the start of a download is announced as a toast instead.
    with gu.download_progress(None):
        report(status="started", bytes_done=0, bytes_total=1000)
        report(status="progress", bytes_done=250, bytes_total=1000)

    assert len(infos) == 1
    assert "acoustic model v3.0" in infos[0]

    # The hook is scoped: outside the block the library's default (no callback) is back.
    assert registered() is None


def test_download_progress_survives_ui_failures(monkeypatch):
    # In the library an exception escaping the callback aborts the download, so a
    # failing UI call (e.g. no request context for the toast) must be swallowed, and
    # a retry announcement must reach the user as a warning.
    import birdnet
    import gradio as gr

    from birdnet_analyzer.gui import utils as gu

    warnings = []
    monkeypatch.setattr(gr, "Warning", lambda msg, **kw: warnings.append(msg))

    def broken_progress(*args, **kwargs):
        raise RuntimeError("no request context")

    def registered():
        return birdnet.get_download_progress_callback()

    def update(status, **extra):
        return birdnet.DownloadProgress(
            description="Downloading acoustic model v3.0",
            url="https://example.invalid/model",
            bytes_done=0,
            bytes_total=1000,
            attempt=2,
            max_attempts=3,
            status=status,
            **extra,
        )

    with gu.download_progress(broken_progress):
        registered()(update("progress"))  # must not raise
        registered()(update("retrying", error="connection reset"))

    assert len(warnings) == 1
    assert "connection reset" in warnings[0]
    assert "(2/3)" in warnings[0]
