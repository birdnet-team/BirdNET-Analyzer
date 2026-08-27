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


MODEL_NAME = "acoustic model v3.0"


def _capture_progress(monkeypatch):
    """Captures what a handler's ``gr.Progress`` reports, and any toast it raises.

    Gradio resolves the bar from a contextvar rather than from the argument, so a
    stand-in callback is the only way to observe it outside a live event.
    """
    import gradio as gr

    records: list = []
    infos: list = []
    monkeypatch.setattr(gr, "Info", lambda msg, **kw: infos.append(msg))
    monkeypatch.setattr(
        gr.Progress, "_progress_callback", staticmethod(lambda: records.append)
    )

    return records, infos


def _emit_download():
    """Reports a model download the way the birdnet library does mid-load.

    Stays silent when nothing registered a sink, so a tab that never wraps its model
    call fails on the missing progress update rather than on a TypeError here.
    """
    import birdnet

    callback = birdnet.get_download_progress_callback()

    if callback is None:
        return

    # "started" is the update that falls back to a toast when there is no bar.
    for status, done in (("started", 0), ("progress", 250), ("finished", 1000)):
        callback(
            birdnet.DownloadProgress(
                description=f"Downloading {MODEL_NAME}",
                url="https://example.invalid/model",
                bytes_done=done,
                bytes_total=1000,
                attempt=1,
                max_attempts=3,
                status=status,
            )
        )


def _flatten(records):
    return [tracked for update in records for tracked in update]


def _assert_download_reached_the_bar(updates, infos):
    # The label is localized; the model name in it is not.
    assert any(MODEL_NAME in (update.desc or "") for update in updates), (
        "the model download never reached the progress bar"
    )
    assert not [msg for msg in infos if MODEL_NAME in msg], (
        "the download fell back to a toast although a bar was available"
    )


def _module(name):
    """Imports a submodule by name.

    ``birdnet_analyzer`` re-exports ``analyze``/``embeddings``/``search``/``train``
    as functions, shadowing the submodules of the same name, so attribute access -
    and monkeypatch's dotted string form, which uses it - reaches the function.
    """
    import importlib

    return importlib.import_module(name)


class _FakeDatabase:
    """The metadata and close surface the embeddings and search handlers touch."""

    class _Connection:
        def close(self):
            pass

    def __init__(self):
        self.db = self._Connection()

    def get_metadata(self, key):
        return {"AUDIO_SPEED": 1.0, "BANDPASS_FMIN": 0, "BANDPASS_FMAX": 15000}

    def insert_metadata(self, key, value):
        pass


def _run_single_file_tab(monkeypatch, report):
    """Runs the single-file tab's handler over an ``analyze`` that calls ``report``."""
    import pandas as pd

    from birdnet_analyzer.gui import single_file
    from birdnet_analyzer.gui import utils as gu

    records, infos = _capture_progress(monkeypatch)

    class FakePredictions:
        def to_dataframe(self):
            return pd.DataFrame(
                columns=["species_name", "start_time", "end_time", "confidence"]
            )

    def fake_analyze(**kwargs):
        report(**kwargs)
        return FakePredictions()

    monkeypatch.setattr(_module("birdnet_analyzer.analyze"), "analyze", fake_analyze)

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

    return _flatten(records), infos


def test_single_file_tab_shows_the_download_on_its_progress_bar(monkeypatch):
    """A first-use download is several hundred MB; a toast alone looks like a hang."""
    updates, infos = _run_single_file_tab(
        monkeypatch, lambda **kwargs: _emit_download()
    )

    _assert_download_reached_the_bar(updates, infos)


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


def test_embeddings_tab_shows_the_download_on_its_progress_bar(monkeypatch):
    """Building a database is the first thing a fresh install does, model and all."""
    from birdnet_analyzer.gui import embeddings as gui_embeddings

    records, infos = _capture_progress(monkeypatch)

    monkeypatch.setattr(
        gui_embeddings, "get_embeddings_database", lambda directory: _FakeDatabase()
    )
    monkeypatch.setattr(
        _module("birdnet_analyzer.embeddings.core"),
        "embeddings",
        lambda *args, **kwargs: _emit_download(),
    )

    gui_embeddings.run_embeddings_with_tqdm_tracking(
        input_path="recordings",
        db_directory="database",
        overlap=0.0,
        batch_size=1,
        producers_number=1,
        workers_number=1,
        audio_speed=1.0,
        fmin=0,
        fmax=15000,
        enable_file_output=False,
        file_output="",
    )

    _assert_download_reached_the_bar(_flatten(records), infos)


def test_search_tab_shows_the_download_on_its_progress_bar(monkeypatch):
    """Searching embeds the query sample, which loads the model on first use."""
    from birdnet_analyzer.gui import search as gui_search

    records, infos = _capture_progress(monkeypatch)

    def fake_get_search_results(*args, **kwargs):
        _emit_download()
        return []

    monkeypatch.setattr(gui_search, "get_search_database", lambda path: _FakeDatabase())
    monkeypatch.setattr(
        _module("birdnet_analyzer.search.utils"),
        "get_search_results",
        fake_get_search_results,
    )

    gui_search.run_search(
        "database", "recordings", "query.wav", 10, "cosine", "center", 0.0
    )

    _assert_download_reached_the_bar(_flatten(records), infos)


def test_train_tab_shows_the_download_on_its_progress_bar(monkeypatch):
    """Training embeds the training data first, which loads the model on first use."""
    from types import SimpleNamespace

    from birdnet_analyzer.gui import train as gui_train

    records, infos = _capture_progress(monkeypatch)

    def fake_train_model(**kwargs):
        _emit_download()
        history = SimpleNamespace(
            epoch=[0], history={"val_AUPRC": [0.5], "val_AUROC": [0.6]}
        )

        return history, {}

    monkeypatch.setattr(
        _module("birdnet_analyzer.train.utils"), "train_model", fake_train_model
    )

    gui_train.start_training(
        data_dir="training-data",
        test_data_dir=None,
        crop_mode="center",
        crop_overlap=0.0,
        fmin=0,
        fmax=15000,
        output_dir="classifiers",
        classifier_name="classifier",
        model_save_mode="replace",
        cache_mode="none",
        cache_file="",
        cache_file_name="",
        autotune=False,
        autotune_trials=1,
        autotune_folds=1,
        autotune_repeats=1,
        epochs=1,
        batch_size=1,
        learning_rate=0.001,
        focal_loss=False,
        focal_loss_gamma=2.0,
        focal_loss_alpha=0.25,
        hidden_units=0,
        dropout=0.0,
        label_smoothing=False,
        use_mixup=False,
        upsampling_ratio=0.0,
        upsampling_mode="repeat",
        model_formats=["tflite"],
        audio_speed=1.0,
        threads=1,
    )

    _assert_download_reached_the_bar(_flatten(records), infos)
