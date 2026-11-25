import inspect
import threading

import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.localization as loc  # noqa: E402
import birdnet_analyzer.gui.multi_file as mfa  # noqa: E402
from birdnet_analyzer import utils  # noqa: E402
from birdnet_analyzer.gui import analysis as ga  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


def test_a_failing_folder_pick_is_logged_and_shown(
    appdir, handler, monkeypatch, tmp_path
):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)
    monkeypatch.setattr(gu, "select_folder", lambda *a, **k: str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(gu, "get_audio_files_and_durations", fail)

    with pytest.raises(gr.Error, match="disk"):
        handler(mfa.build_multi_analysis_tab, "select_directory_on_empty")()

    assert len(logged) == 1


def test_the_batch_analysis_lets_birdnet_pick_the_worker_count_by_default(
    appdir, monkeypatch
):
    with gr.Blocks() as demo:
        mfa.build_multi_analysis_tab()

    run = next(
        e
        for e in demo.fns.values()
        if getattr(e.fn, "__name__", None) == "run_batch_analysis"
    )
    names = list(inspect.signature(run.fn).parameters)[: len(run.inputs)]
    inputs = dict(zip(names, run.inputs, strict=True))
    assert isinstance(inputs["workers_auto"], gr.Checkbox)
    assert isinstance(inputs["workers_number"], gr.Number)
    values = {name: component.value for name, component in inputs.items()}
    values["input_dir"] = "recordings"
    seen = {}
    monkeypatch.setattr(ga, "run_analysis", lambda **kwargs: seen.update(kwargs))

    def start(**overrides):
        run.fn(*[{**values, **overrides}[name] for name in names])
        return seen["n_workers"]

    assert start() is None
    assert start(workers_auto=False, workers_number=4) == 4


def test_analyze_is_usable_again_after_a_failed_run(appdir):
    with gr.Blocks() as demo:
        mfa.build_multi_analysis_tab()

    events = demo.fns
    run = next(
        e
        for e in events.values()
        if getattr(e.fn, "__name__", None) == "run_batch_analysis"
    )
    analyze_button = events[run.trigger_after].outputs[0]
    on_failure = next(
        e
        for e in events.values()
        if e.trigger_after == run._id and e.trigger_only_on_failure
    )
    updates = dict(zip(on_failure.outputs, on_failure.fn(), strict=True))

    assert updates[analyze_button]["interactive"] is True


def test_a_run_starts_without_the_result_of_the_run_before(appdir, handler):
    *_, result_list = handler(mfa.build_multi_analysis_tab, "prepare_run_ui")()

    assert (result_list["value"], result_list["headers"]) == ([], [""])


def _report_unreadable(files):
    utils.warn_about_unreadable_files(files, 3)


def _report_from_another_run(files):
    other = threading.Thread(target=_report_unreadable, args=(files,))
    other.start()
    other.join()


@pytest.mark.parametrize(
    ("report", "listed", "header"),
    [
        (
            _report_unreadable,
            ["recordings/broken.wav"],
            "multi-tab-result-dataframe-column-invalid-file-header",
        ),
        (
            _report_from_another_run,
            [],
            "multi-tab-result-dataframe-column-success-header",
        ),
    ],
    ids=["this run", "another run"],
)
def test_files_a_run_could_not_read_are_listed_although_it_analyzed_none_itself(
    appdir, monkeypatch, report, listed, header
):
    with gr.Blocks() as demo:
        mfa.build_multi_analysis_tab()

    run = next(
        e
        for e in demo.fns.values()
        if getattr(e.fn, "__name__", None) == "run_batch_analysis"
    )
    names = list(inspect.signature(run.fn).parameters)[: len(run.inputs)]
    values = {
        name: component.value for name, component in zip(names, run.inputs, strict=True)
    }
    values["input_dir"] = "recordings"
    monkeypatch.setattr(
        ga, "run_analysis", lambda **kwargs: report(["recordings/broken.wav"])
    )

    shown = run.fn(*[values[name] for name in names])

    assert (shown["value"], shown["headers"]) == (listed, [loc.LANGUAGE_LOOKUP[header]])
