"""The GUI shows the birdnet library's model downloads."""

import sys
from unittest.mock import MagicMock

import pytest

gr = pytest.importorskip("gradio")

sys.modules.setdefault("webview", MagicMock(settings={}))


def test_download_progress_routes_library_updates_to_gradio(monkeypatch):
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
    assert calls[2][0] == 1.0, "finished shows the bar full"
    assert not infos, "no toasts while a progress bar is available"

    with gu.download_progress(None):
        report(status="started", bytes_done=0, bytes_total=1000)
        report(status="progress", bytes_done=250, bytes_total=1000)

    assert len(infos) == 1
    assert "acoustic model v3.0" in infos[0]

    assert registered() is None


def test_download_progress_survives_ui_failures(monkeypatch):
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


def test_overlapping_gui_events_keep_their_own_sink_and_unregister_cleanly():
    import threading

    import birdnet

    from birdnet_analyzer.gui import utils as gu

    def update(desc):
        return birdnet.DownloadProgress(
            description=desc,
            url="u",
            bytes_done=1,
            bytes_total=2,
            attempt=1,
            max_attempts=1,
            status="progress",
        )

    seen_a, seen_b = [], []
    a_entered, b_entered, a_left = (threading.Event() for _ in range(3))

    def event_a():
        with gu.download_progress(lambda v, desc=None, **k: seen_a.append(desc)):
            a_entered.set()
            b_entered.wait(5)
            birdnet.get_download_progress_callback()(update("Downloading alpha-model"))
        a_left.set()

    def event_b():
        a_entered.wait(5)
        with gu.download_progress(lambda v, desc=None, **k: seen_b.append(desc)):
            b_entered.set()
            a_left.wait(5)
            birdnet.get_download_progress_callback()(update("Downloading beta-model"))

    ta, tb = threading.Thread(target=event_a), threading.Thread(target=event_b)
    ta.start()
    tb.start()
    ta.join(10)
    tb.join(10)

    assert any("alpha-model" in d for d in seen_a)
    assert not any("beta-model" in d for d in seen_a)
    assert any("beta-model" in d for d in seen_b)
    assert not any("alpha-model" in d for d in seen_b)
    assert birdnet.get_download_progress_callback() is None
