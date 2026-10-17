"""The train tab's base model choice.

The classifier is trained on one BirdNET version's embeddings, chosen with a radio
that defaults to the newest version. A cache file fixes the version it was created
with, so loading one shows that version and disables the choice.
"""

import numpy as np
import pytest

gr = pytest.importorskip("gradio")

import birdnet_analyzer.config as cfg  # noqa: E402
import birdnet_analyzer.gui.localization as loc  # noqa: E402
from birdnet_analyzer import settings  # noqa: E402
from birdnet_analyzer.gui import state as gs  # noqa: E402
from birdnet_analyzer.gui import train  # noqa: E402


@pytest.fixture
def appdir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "APPDIR", tmp_path)
    monkeypatch.setattr(settings, "STATE_SETTINGS_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(tmp_path / "error_log.txt"))
    monkeypatch.setattr(gs, "_PERSISTED", [])
    return tmp_path


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

    base_model = _radio("training-tab-base-model-radio-label")
    cache_mode = _radio("training-tab-cache-mode-radio-label")
    on_cache_mode_change = next(
        event.fn
        for event in demo.fns.values()
        if event.targets
        and event.targets[0] == (cache_mode._id, "change")
        and base_model in event.outputs
    )

    return base_model, on_cache_mode_change


def test_the_newest_version_is_the_default_base(appdir):
    base_model, _ = build()

    assert base_model.value == cfg.DEFAULT_ACOUSTIC_MODEL_VERSION
    assert base_model.interactive
    assert {version for _, version in base_model.choices} >= {"2.4", "3.0"}


def test_loading_a_cache_disables_the_base_choice(appdir):
    _, on_cache_mode_change = build()

    *_, update = on_cache_mode_change("load", None, None)
    assert update["interactive"] is False

    *_, update = on_cache_mode_change("none", None, None)
    assert update["interactive"] is True


def test_the_base_choice_is_built_disabled_when_a_cache_is_persisted(appdir):
    settings.set_tab_setting("train", "cache_mode_radio", "load")

    base_model, _ = build()

    assert not base_model.interactive


def test_cache_model_version_reads_the_record_or_the_embeddings_width(tmp_path):
    recorded = tmp_path / "recorded.npz"
    np.savez(recorded, x_train=np.zeros((1, 1024)), model_version="3.0")
    unrecorded = tmp_path / "unrecorded.npz"
    np.savez(unrecorded, x_train=np.zeros((3, 1024), np.float32))
    unknown = tmp_path / "unknown.npz"
    np.savez(unknown, x_train=np.zeros((1, 512)))

    assert train.cache_model_version(str(recorded)) == "3.0"
    assert train.cache_model_version(str(unrecorded)) == "2.4"
    assert train.cache_model_version(str(unknown)) is None
    assert train.cache_model_version(str(tmp_path / "missing.npz")) is None


def test_the_base_choice_reaches_training(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from birdnet_analyzer.train import utils as train_utils

    seen = {}

    def fake_train_model(**kwargs):
        seen.update(kwargs)
        history = SimpleNamespace(
            epoch=[0], history={"val_AUPRC": [0.5], "val_AUROC": [0.5]}
        )
        return history, {}

    monkeypatch.setattr(train_utils, "train_model", fake_train_model)

    train.start_training(
        data_dir=str(tmp_path),
        test_data_dir=None,
        crop_mode="center",
        crop_overlap=0.0,
        fmin=0,
        fmax=15000,
        output_dir=str(tmp_path),
        classifier_name="cc",
        model_save_mode="replace",
        birdnet_version="2.4",
        cache_mode="none",
        cache_file=None,
        cache_file_name="cache.npz",
        autotune=False,
        autotune_trials=1,
        autotune_folds=2,
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
        progress=None,
    )

    assert seen["birdnet"] == "2.4"


def test_selecting_a_cache_shows_its_version(appdir, monkeypatch, tmp_path):
    from birdnet_analyzer.gui import utils as gu

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
