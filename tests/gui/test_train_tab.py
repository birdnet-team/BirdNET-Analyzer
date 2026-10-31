import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.config as cfg  # noqa: E402
import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import train  # noqa: E402
from birdnet_analyzer.gui import utils as gu  # noqa: E402


def _radio(label_key):
    label = loc.localize(label_key)

    return next(
        c
        for c in gs.persisted_components()
        if isinstance(c, gr.Radio) and c.label == label
    )


def build():
    with gr.Blocks() as demo:
        train.build_train_tab()

    return demo, _radio("training-tab-base-model-radio-label")


def test_the_newest_version_is_the_default_base(appdir):
    _, base_model = build()

    assert base_model.value == cfg.DEFAULT_ACOUSTIC_MODEL_VERSION
    assert base_model.interactive
    assert {version for _, version in base_model.choices} >= {"2.4", "3.0"}


def test_loading_a_cache_disables_the_base_choice(appdir):
    demo, base_model = build()
    cache_mode = _radio("training-tab-cache-mode-radio-label")

    disabled = _fire(demo, cache_mode, {cache_mode: "load"})[base_model]
    enabled = _fire(demo, cache_mode, {cache_mode: "none"})[base_model]

    assert disabled["interactive"] is False
    assert enabled["interactive"] is True


def test_the_base_choice_is_built_disabled_when_a_cache_is_persisted(appdir):
    settings.set_tab_setting("train", "cache_mode_radio", "load")

    _, base_model = build()

    assert not base_model.interactive


def test_cache_model_version_reads_the_record_or_the_embeddings_width(tmp_path):
    recorded = tmp_path / "recorded.npz"
    np.savez(recorded, x_train=np.zeros((1, 1024)), model_version="3.0")
    unrecorded = tmp_path / "unrecorded.npz"
    np.savez(unrecorded, x_train=np.zeros((3, 1024), np.float32))
    unknown = tmp_path / "unknown.npz"
    np.savez(unknown, x_train=np.zeros((1, 512)))
    foreign = tmp_path / "foreign.npz"
    np.savez(foreign, labels=np.zeros(3))

    assert train.cache_model_version(str(recorded)) == "3.0"
    assert train.cache_model_version(str(unrecorded)) == "2.4"
    assert train.cache_model_version(str(unknown)) is None
    assert train.cache_model_version(str(foreign)) is None
    assert train.cache_model_version(str(tmp_path / "missing.npz")) is None


def _start_training(monkeypatch, tmp_path, fail_with=None, **overrides):
    from birdnet_analyzer.train import utils as train_utils

    seen = {}

    def fake_train_model(**kwargs):
        seen.update(kwargs)

        if fail_with:
            raise fail_with

        history = SimpleNamespace(
            epoch=[0], history={"val_AUPRC": [0.5], "val_AUROC": [0.5]}
        )
        return history, {}

    monkeypatch.setattr(train_utils, "train_model", fake_train_model)

    train.start_training(
        **{
            "data_dir": str(tmp_path),
            "test_data_dir": None,
            "crop_mode": "center",
            "crop_overlap": 0.0,
            "fmin": 0,
            "fmax": 15000,
            "output_dir": str(tmp_path),
            "classifier_name": "cc",
            "model_save_mode": "replace",
            "birdnet_version": "3.0",
            "cache_mode": "none",
            "cache_file": None,
            "cache_file_name": "cache.npz",
            "autotune": False,
            "autotune_trials": 1,
            "autotune_folds": 2,
            "autotune_repeats": 1,
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "focal_loss": False,
            "focal_loss_gamma": 2.0,
            "focal_loss_alpha": 0.25,
            "hidden_units": 0,
            "dropout": 0.0,
            "label_smoothing": False,
            "use_mixup": False,
            "upsampling_ratio": 0.0,
            "upsampling_mode": "repeat",
            "model_formats": ["tflite"],
            "audio_speed": 1.0,
            "threads": 1,
            "progress": None,
        }
        | overrides
    )

    return seen


def test_the_base_choice_reaches_training(monkeypatch, tmp_path):
    seen = _start_training(monkeypatch, tmp_path, birdnet_version="2.4")

    assert seen["birdnet"] == "2.4"


@pytest.mark.parametrize(("slider", "speed"), [(-2, 0.5), (0, 1.0), (3, 3.0)])
def test_the_speed_slider_reaches_training_as_a_factor(
    monkeypatch, tmp_path, slider, speed
):
    seen = _start_training(monkeypatch, tmp_path, audio_speed=slider)

    assert seen["audio_speed"] == speed


def test_an_out_of_range_bandpass_is_rejected(monkeypatch, tmp_path):
    message = loc.localize("validation-no-valid-frequency")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _start_training(monkeypatch, tmp_path, fmin=-1)


@pytest.mark.parametrize("picked", ["nothing", "a file"])
def test_saving_a_cache_needs_a_directory(monkeypatch, tmp_path, picked):
    cache_file = None

    if picked == "a file":
        cache_file = tmp_path / "train_cache.npz"
        cache_file.touch()

    message = loc.localize("validation-no-directory-selected")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _start_training(
            monkeypatch,
            tmp_path,
            cache_mode="save",
            cache_file=cache_file and str(cache_file),
        )


@pytest.mark.parametrize("name", ["", "  "])
def test_saving_a_cache_needs_a_file_name(monkeypatch, tmp_path, name):
    message = loc.localize("validation-no-valid-cache-file-name")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _start_training(
            monkeypatch,
            tmp_path,
            cache_mode="save",
            cache_file=str(tmp_path),
            cache_file_name=name,
        )


@pytest.mark.parametrize("picked", ["nothing", "a missing file"])
def test_training_from_a_cache_needs_an_existing_file(monkeypatch, tmp_path, picked):
    cache_file = None if picked == "nothing" else str(tmp_path / "missing.npz")
    message = loc.localize("validation-no-cache-file-selected")

    with pytest.raises(gr.Error, match=re.escape(message)):
        _start_training(monkeypatch, tmp_path, cache_mode="load", cache_file=cache_file)


def test_a_saved_cache_lands_in_the_picked_directory(monkeypatch, tmp_path):
    seen = _start_training(
        monkeypatch,
        tmp_path,
        cache_mode="save",
        cache_file=str(tmp_path),
        cache_file_name="cache.npz",
    )

    assert seen["save_cache_to"] == os.path.join(str(tmp_path), "cache.npz")


def test_a_cache_file_name_is_used_without_surrounding_spaces(monkeypatch, tmp_path):
    seen = _start_training(
        monkeypatch,
        tmp_path,
        cache_mode="save",
        cache_file=str(tmp_path),
        cache_file_name="  cache.npz  ",
    )

    assert seen["save_cache_to"] == os.path.join(str(tmp_path), "cache.npz")


def test_a_training_error_with_a_localization_key_is_shown_translated(
    monkeypatch, tmp_path
):
    key = "validation-no-audio-files-in-training-folders"
    error = ValueError("untranslated", key, "a, b")
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    with pytest.raises(gr.Error, match=re.escape(loc.localize(key).format("a, b"))):
        _start_training(monkeypatch, tmp_path, fail_with=error)

    assert logged == []


def test_any_other_training_error_is_shown_in_full(monkeypatch, tmp_path):
    error = PermissionError(13, "Permission denied", "cc.tflite")
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    with pytest.raises(gr.Error, match=r"Errno 13.*cc\.tflite"):
        _start_training(monkeypatch, tmp_path, fail_with=error)

    assert len(logged) == 1


def test_folder_labels_are_listed_without_surrounding_spaces(monkeypatch, tmp_path):
    (tmp_path / "a, b").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.setattr(train.gu, "select_folder", lambda **_: str(tmp_path))

    _, labels = train.select_subdirectories()

    assert labels == [["a"], ["b"]]


def test_the_autotune_folds_default_matches_the_cli(appdir):
    from birdnet_analyzer.cli import train_parser

    with gr.Blocks():
        train.build_train_tab()

    folds = _persisted(gr.Number, "training-tab-autotune-folds-number-label")

    assert folds.value == train_parser().get_default("autotune_n_splits")


def _persisted(cls, label_key):
    label = loc.localize(label_key)

    return next(
        c for c in gs.persisted_components() if isinstance(c, cls) and c.label == label
    )


def _fire(demo, trigger, values):
    """Runs every handler of ``trigger``'s change event, merging their updates."""
    updates = {}

    for event in demo.fns.values():
        if (trigger._id, "change") in event.targets:
            result = event.fn(*(values.get(c) for c in event.inputs))
            results = result if len(event.outputs) > 1 else (result,)

            for output, update in zip(event.outputs, results, strict=True):
                assert output not in updates, "two handlers write the same output"
                updates[output] = update

    return updates


@pytest.mark.parametrize("cache", ["none", "load"])
@pytest.mark.parametrize("crop", ["center", "segments"])
def test_the_crop_overlap_follows_cache_and_crop_mode_whichever_changed(
    appdir, cache, crop
):
    with gr.Blocks() as demo:
        train.build_train_tab()
    cache_mode = _radio("training-tab-cache-mode-radio-label")
    crop_mode = _radio("training-tab-crop-mode-radio-label")
    crop_overlap = _persisted(gr.Slider, "training-tab-crop-overlap-number-label")
    expected = {
        "visible": crop == "segments",
        "interactive": crop == "segments" and cache != "load",
    }

    for trigger in (cache_mode, crop_mode):
        update = _fire(demo, trigger, {cache_mode: cache, crop_mode: crop})[
            crop_overlap
        ]

        assert {key: update.get(key) for key in expected} == expected


@pytest.mark.parametrize(("folds", "editable"), [(1, False), (3, True)])
def test_the_autotune_repeats_are_editable_only_with_several_folds(
    appdir, folds, editable
):
    settings.set_tab_setting("train", "autotune_folds_number", folds)

    with gr.Blocks() as demo:
        train.build_train_tab()
    autotune_folds = _persisted(gr.Number, "training-tab-autotune-folds-number-label")
    repeats = _persisted(gr.Number, "training-tab-autotune-repeats-number-label")

    update = _fire(demo, autotune_folds, {autotune_folds: folds})[repeats]

    assert repeats.interactive is editable
    assert update["interactive"] is editable


def test_a_cache_mode_switch_is_a_single_update(appdir):
    with gr.Blocks() as demo:
        train.build_train_tab()
    cache_mode = _radio("training-tab-cache-mode-radio-label")

    listeners = [
        event
        for event in demo.fns.values()
        if (cache_mode._id, "change") in event.targets
    ]

    assert len(listeners) == 1


def test_changing_the_cache_mode_forgets_the_picked_cache_location(appdir):
    with gr.Blocks() as demo:
        train.build_train_tab()
    cache_mode = _radio("training-tab-cache-mode-radio-label")
    pickers = {
        getattr(event.fn, "__name__", ""): event.outputs for event in demo.fns.values()
    }
    picked = {
        *pickers["select_cache_directory_and_update"][:2],
        *pickers["on_cache_file_selection_click"][:3],
    }

    for mode in ("none", "load", "save"):
        updates = _fire(demo, cache_mode, {cache_mode: mode})

        for component in picked:
            update = updates[component]

            if isinstance(update, dict):
                assert "value" in update, component
                update = update["value"]

            assert update in (None, "")


def test_selecting_a_cache_shows_its_version(appdir, monkeypatch, tmp_path):
    cache = tmp_path / "cache.npz"
    np.savez(cache, x_train=np.zeros((1, 1024)), model_version="2.4")
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: str(cache))

    with gr.Blocks() as demo:
        train.build_train_tab()
    base_model = _radio("training-tab-base-model-radio-label")
    on_select = next(
        event.fn
        for event in demo.fns.values()
        if event.targets
        and event.targets[0][1] == "click"
        and base_model in event.outputs
    )

    *_, update = on_select()
    assert update == gr.update(value="2.4")


def test_a_failing_folder_listing_is_logged_and_shown(
    appdir, handler, monkeypatch, tmp_path
):
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)
    monkeypatch.setattr(gu, "select_folder", lambda **_: str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(train.utils, "list_subdirectories", fail)

    with pytest.raises(gr.Error, match="disk"):
        handler(train.build_train_tab, "selection_fn")()

    assert len(logged) == 1


def test_cache_model_version_lets_an_unexpected_error_through(tmp_path, monkeypatch):
    from birdnet_analyzer import base_models

    cache = tmp_path / "cache.npz"
    np.savez(cache, x_train=np.zeros((1, 1024)))

    def fail(width):
        raise TypeError("bug")

    monkeypatch.setattr(base_models, "version_for_embeddings_dim", fail)

    with pytest.raises(TypeError, match="bug"):
        train.cache_model_version(str(cache))
