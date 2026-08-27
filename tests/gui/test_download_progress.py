"""The GUI shows the birdnet library's model downloads."""

import pytest

gr = pytest.importorskip("gradio")


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


def _run_single_file_tab(monkeypatch, report):
    """Runs the single-file tab's handler over an ``analyze`` that calls ``report``.

    Returns the updates that reached the progress bar and the toasts that were
    raised instead. Gradio resolves the bar from a contextvar rather than from the
    argument, so a stand-in callback is the only way to observe it outside a live
    event.
    """
    import importlib

    import gradio as gr
    import pandas as pd

    from birdnet_analyzer.gui import single_file
    from birdnet_analyzer.gui import utils as gu

    records: list = []
    infos: list = []
    monkeypatch.setattr(gr, "Info", lambda msg, **kw: infos.append(msg))
    monkeypatch.setattr(
        gr.Progress, "_progress_callback", staticmethod(lambda: records.append)
    )

    class FakePredictions:
        def to_dataframe(self):
            return pd.DataFrame(
                columns=["species_name", "start_time", "end_time", "confidence"]
            )

    def fake_analyze(**kwargs):
        report(**kwargs)
        return FakePredictions()

    # birdnet_analyzer re-exports analyze, shadowing the submodule of the same name.
    analyze_module = importlib.import_module("birdnet_analyzer.analyze")
    monkeypatch.setattr(analyze_module, "analyze", fake_analyze)

    single_file.run_single_file_analysis(
        input_path="recording.wav",
        use_top_n=False,
        top_n=1,
        confidence=0.25,
        sensitivity=1.0,
        overlap=0.0,
        merge_consecutive=1,
        audio_speed=1.0,
        fmin=0,
        fmax=15000,
        species_list_choice=gu._ALL_SPECIES,
        species_list_file=None,
        lat=-1,
        lon=-1,
        week=-1,
        use_yearlong=True,
        sf_thresh=0.03,
        selected_model="BirdNET 2.4",
        custom_classifier_file=None,
        locale="en_us",
    )

    return [tracked for update in records for tracked in update], infos


def test_single_file_tab_shows_the_download_on_its_progress_bar(monkeypatch):
    """The single-file tab hands run_analysis a tracker, so no toast fallback.

    A first-use download is several hundred MB; without the bar the tab looks like
    it hangs for minutes on nothing but a toast.
    """
    import birdnet

    def report_download(**kwargs):
        # "started" is the update that falls back to a toast when there is no bar.
        for status, done in (("started", 0), ("progress", 250), ("finished", 1000)):
            birdnet.get_download_progress_callback()(
                birdnet.DownloadProgress(
                    description="Downloading acoustic model v3.0",
                    url="https://example.invalid/model",
                    bytes_done=done,
                    bytes_total=1000,
                    attempt=1,
                    max_attempts=3,
                    status=status,
                )
            )

    updates, infos = _run_single_file_tab(monkeypatch, report_download)

    # The label is localized; the model name in it is not.
    assert any("acoustic model v3.0" in (update.desc or "") for update in updates)
    assert not infos, "no toasts while a progress bar is available"


def test_single_file_tab_shows_analysis_progress_on_its_progress_bar(monkeypatch):
    """The same tracker carries the analysis itself, not just the download.

    A long recording is minutes of work after the download has finished.
    """
    from types import SimpleNamespace

    import birdnet_analyzer.gui.localization as loc

    def report_analysis(**kwargs):
        kwargs["on_update"](
            SimpleNamespace(processed_segments=3, total_segments=10, progress_pct=30.0)
        )

    updates, _ = _run_single_file_tab(monkeypatch, report_analysis)

    analyzing = [update for update in updates if update.index == 3]
    assert analyzing, "the per-segment callback never reached the bar"
    assert analyzing[0].length == 10

    # An absent key localizes to itself, which the GUI would then show as the unit.
    unit = analyzing[0].unit
    assert unit == loc.localize("progress-unit-segments")
    assert unit != "progress-unit-segments", "the unit label is an untranslated key"
