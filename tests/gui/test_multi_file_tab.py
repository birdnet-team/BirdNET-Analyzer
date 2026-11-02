import inspect

import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.gui.multi_file as mfa  # noqa: E402
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


def test_the_start_button_comes_back_after_a_run_whether_or_not_it_failed(appdir):
    with gr.Blocks() as demo:
        mfa.build_multi_analysis_tab()

    events = demo.fns
    run = next(
        e
        for e in events.values()
        if getattr(e.fn, "__name__", None) == "run_batch_analysis"
    )
    prepare = events[run.trigger_after]
    followers = [e for e in events.values() if e.trigger_after == run._id]

    assert sorted(e.trigger_only_on_failure for e in followers) == [False, True]
    for follower in followers:
        start, pause = follower.fn()
        status = next(e for e in events.values() if e.trigger_after == follower._id)

        assert follower.outputs == prepare.outputs[:2]
        assert start["interactive"] is True
        assert pause["visible"] is False
        assert status.fn is mfa.refresh_resume_status
