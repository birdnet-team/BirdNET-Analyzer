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
from tests.params_files import (  # noqa: E402
    V2_4_0_TRAIN_FILE,
    V2_4_0_TRAIN_VALUES,
    wide_file,
)


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
    from birdnet_analyzer.train.errors import TrainingInputError

    key = "validation-no-audio-files-in-training-folders"
    error = TrainingInputError(key, "untranslated", "R&D, b")
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    with pytest.raises(gr.Error) as raised:
        _start_training(monkeypatch, tmp_path, fail_with=error)

    assert raised.value.message == loc.LANGUAGE_LOOKUP[key].format("R&amp;D, b")
    assert logged == []


def test_a_training_error_without_a_text_in_the_gui_keeps_its_own(
    monkeypatch, tmp_path
):
    from birdnet_analyzer.train.errors import TrainingInputError

    error = TrainingInputError("validation-not-translated-yet", "Only in English.")
    monkeypatch.setattr(gu.utils, "write_error_log", lambda e: None)

    with pytest.raises(gr.Error) as raised:
        _start_training(monkeypatch, tmp_path, fail_with=error)

    assert raised.value.message == "Only in English."


def test_any_other_training_error_is_logged_and_explained(monkeypatch, tmp_path):
    error = PermissionError(13, "Permission denied", "cc.tflite")
    logged = []
    monkeypatch.setattr(gu.utils, "write_error_log", logged.append)

    with pytest.raises(gr.Error) as raised:
        _start_training(monkeypatch, tmp_path, fail_with=error)

    assert raised.value.message == loc.LANGUAGE_LOOKUP["error-path-no-access"].format(
        path="cc.tflite"
    )
    assert len(logged) == 1


def test_a_failed_download_during_training_shows_the_download_message(
    monkeypatch, tmp_path
):
    import requests

    monkeypatch.setattr(gu.utils, "write_error_log", lambda e: None)
    monkeypatch.setenv("BIRDNET_APP_DATA", str(tmp_path))
    monkeypatch.delenv("BIRDNET_OFFLINE", raising=False)
    monkeypatch.setitem(
        loc.LANGUAGE_LOOKUP, "model-download-failed-error", "{reason}|{directory}"
    )
    request = requests.Request("GET", "https://zenodo.org/records/1/model.zip")
    error = requests.ConnectionError("Max retries exceeded", request=request.prepare())

    with pytest.raises(gr.Error) as raised:
        _start_training(monkeypatch, tmp_path, fail_with=error)

    assert raised.value.message == f"ConnectionError, zenodo.org|{tmp_path}"


def test_folder_labels_are_listed_without_surrounding_spaces(monkeypatch, tmp_path):
    (tmp_path / "a, b").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.setattr(train.gu, "select_folder", lambda **_: str(tmp_path))

    _, labels = train.select_subdirectories()

    assert labels == [["a"], ["b"]]


@pytest.mark.parametrize(
    ("version", "width", "learning_rate"), [("3.0", 1280, 0.1), ("2.4", 1024, 0.01)]
)
def test_a_classifier_trained_in_the_tab_has_the_default_settings_of_its_base(
    tmp_path, version, width, learning_rate
):
    import inspect

    from birdnet_analyzer import params
    from birdnet_analyzer.train.core import train as train_api
    from birdnet_analyzer.train.utils import _save_to_cache

    rng = np.random.default_rng(0)
    y = np.repeat(np.eye(2, dtype="float32"), 20, axis=0)
    x = (rng.normal(size=(40, width)) + 2 * y[:, :1]).astype("float32")
    cache = str(tmp_path / "cache.npz")
    _save_to_cache(
        cache, x, y, np.array([]), np.array([]), ["a", "b"], model_version=version
    )

    train.start_training(
        data_dir=None,
        test_data_dir=None,
        crop_mode="center",
        crop_overlap=0.0,
        fmin=0,
        fmax=15000,
        output_dir=str(tmp_path),
        classifier_name="cc",
        model_save_mode="replace",
        birdnet_version=cfg.DEFAULT_ACOUSTIC_MODEL_VERSION,
        cache_mode="load",
        cache_file=cache,
        cache_file_name="",
        model_formats=[],
        audio_speed=1,
        threads=1,
        progress=None,
    )

    recorded = params.load_train_params(str(tmp_path / "cc") + cfg.TRAIN_PARAMS_SUFFIX)
    defaults = inspect.signature(train_api).parameters
    hyperparameters = [
        "autotune",
        "epochs",
        "batch_size",
        "hidden_units",
        "dropout",
        "label_smoothing",
        "mixup",
        "use_focal_loss",
        "focal_loss_gamma",
        "focal_loss_alpha",
        "upsampling_mode",
        "upsampling_ratio",
    ]

    assert recorded["learning_rate"] == learning_rate
    assert {key: recorded[key] for key in hyperparameters} == {
        key: defaults[key].default for key in hyperparameters
    }


def test_a_preset_of_the_previous_version_loads_without_a_warning(
    appdir, handler, monkeypatch
):
    warned = []
    monkeypatch.setattr(gr, "Warning", lambda message, **_: warned.append(message))
    settings.save_preset(
        "train",
        "Before 3.0",
        {
            "classifier_name_textbox": "Gibbons",
            "output_format_checkboxgroup": ["tflite"],
            "birdnet_version_radio": "2.4",
            "cache_mode_radio": "none",
            "cache_file_name_textbox": "train_cache.npz",
            "fmin_number": 0,
            "fmax_number": 12000,
            "audio_speed_slider": 0,
            "crop_mode_radio": "first",
            "crop_overlap_slider": 0.0,
            "threads_number": 1,
            "model_save_mode_radio": "replace",
            "autotune_checkbox": False,
            "autotune_trials_number": 50,
            "autotune_folds_number": 1,
            "autotune_repeats_number": 1,
            "epochs_number": 50,
            "batch_size_number": 32,
            "learning_rate_number": 0.0001,
            "hidden_units_number": 0,
            "dropout_number": 0.0,
            "use_label_smoothing_checkbox": False,
            "upsampling_mode_radio": "repeat",
            "upsampling_ratio_slider": 0.0,
            "use_mixup_checkbox": False,
            "use_focal_loss_checkbox": False,
            "focal_loss_gamma_slider": 2.0,
            "focal_loss_alpha_slider": 0.25,
        },
    )

    handler(train.build_train_tab, "on_load")("Before 3.0")

    restored = settings.get_tab_settings("train")
    assert warned == []
    assert restored["crop_mode_radio"] == "center"
    assert restored["fmax_number"] == 12000


def test_the_parameters_file_of_v2_4_0_loads_with_its_base_and_no_warning(
    appdir, handler, monkeypatch, tmp_path
):
    warned = []
    monkeypatch.setattr(gr, "Warning", lambda message, **_: warned.append(message))
    file = wide_file(tmp_path, V2_4_0_TRAIN_VALUES, name=V2_4_0_TRAIN_FILE)
    monkeypatch.setattr(gu, "select_file", lambda *a, **k: file)

    handler(train.build_train_tab, "on_load_params_file")()

    restored = settings.get_tab_settings("train")
    assert warned == []
    assert restored.get("birdnet_version_radio") == "2.4"
    assert restored.get("crop_mode_radio") == "center"


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
