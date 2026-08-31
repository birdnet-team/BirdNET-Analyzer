"""Tests for analysis session pause/cancel behavior in model_utils."""

import pytest

from birdnet_analyzer import model_utils


class FakeSession:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def test_pause_cancels_sessions_without_latching_shutdown():
    session = FakeSession()
    model_utils._register_session(session)

    try:
        assert model_utils.pause_active_analyses() == 1
        assert session.cancelled
        assert not model_utils._SHUTDOWN.is_set(), "pause must not latch shutdown"

        # After a pause, newly registered sessions must not be auto-cancelled
        # (unlike after cancel_active_analyses), so the run can be continued.
        new_session = FakeSession()
        model_utils._register_session(new_session)
        assert not new_session.cancelled
        model_utils._unregister_session(new_session)
    finally:
        model_utils._unregister_session(session)
        model_utils._SHUTDOWN.clear()


def test_pause_with_no_active_sessions_is_a_noop():
    assert model_utils.pause_active_analyses() == 0
    assert not model_utils._SHUTDOWN.is_set()


def test_language_for_version_keeps_supported_and_falls_back_otherwise():
    # The 2.4 and 3.0 models support overlapping but non-identical language sets, and
    # the library raises on an unsupported (version, locale) pair. _language_for_version
    # coerces such a locale to en_us so an analysis never crashes on the label language.
    from birdnet.globals import (
        MODEL_LANGUAGE_EN_US,
        VALID_MODEL_LANGUAGES_V2_4,
        VALID_MODEL_LANGUAGES_V3_0,
    )

    # en_us is the coercion target, so it must be valid for every version.
    assert MODEL_LANGUAGE_EN_US in VALID_MODEL_LANGUAGES_V2_4
    assert MODEL_LANGUAGE_EN_US in VALID_MODEL_LANGUAGES_V3_0

    only_v24 = sorted(set(VALID_MODEL_LANGUAGES_V2_4) - set(VALID_MODEL_LANGUAGES_V3_0))
    only_v30 = sorted(set(VALID_MODEL_LANGUAGES_V3_0) - set(VALID_MODEL_LANGUAGES_V2_4))
    # The whole point of the coercion is that the sets are not nested; if the library
    # ever makes one a superset of the other this test should be revisited.
    assert only_v24
    assert only_v30

    # A 2.4-only locale is kept for 2.4 but coerced to en_us for 3.0, and vice versa.
    assert model_utils._language_for_version(only_v24[0], "2.4") == only_v24[0]
    assert model_utils._language_for_version(only_v24[0], "3.0") == MODEL_LANGUAGE_EN_US
    assert model_utils._language_for_version(only_v30[0], "3.0") == only_v30[0]
    assert model_utils._language_for_version(only_v30[0], "2.4") == MODEL_LANGUAGE_EN_US


def test_supports_sensitivity_only_for_2_4_based_models():
    assert model_utils.supports_sensitivity("birdnet", "2.4")
    assert not model_utils.supports_sensitivity("birdnet", "3.1")
    assert model_utils.supports_sensitivity("birdnet", "3.0", classifier="cc.tflite")
    assert not model_utils.supports_sensitivity("birdnet", "3.0")
    assert not model_utils.supports_sensitivity("perch")


def test_run_inference_drops_sensitivity_for_3_0(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    seen = {}

    @contextmanager
    def fake_predict_session(**kwargs):
        seen.update(kwargs)
        session = MagicMock()
        session.run.return_value = "result"
        yield session

    fake_model = MagicMock()
    fake_model.predict_session = fake_predict_session
    monkeypatch.setattr(model_utils.birdnet, "load", lambda *a, **k: fake_model)

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    result = model_utils.run_inference(
        str(audio), model="birdnet", version="3.0", sigmoid_sensitivity=1.25
    )

    assert result == "result"
    assert seen["sigmoid_sensitivity"] == 1.0


def test_only_birdnet_3_0_can_use_the_gpu():
    assert model_utils.supports_gpu("birdnet", "3.0")
    assert not model_utils.supports_gpu("birdnet", "2.4")
    assert not model_utils.supports_gpu("perch", "3.0")
    assert not model_utils.supports_gpu("birdnet", "3.0", classifier="custom.tflite")


def test_effective_device_normalizes_and_rejects_unknown_devices():
    assert model_utils.effective_device("cpu") == "CPU"
    assert model_utils.effective_device(" CPU ") == "CPU"

    for unknown in ("TPU", "cuda", "GPU:x", ""):
        with pytest.raises(ValueError, match="Unknown device"):
            model_utils.effective_device(unknown)


def test_effective_device_keeps_the_gpu_when_it_can_be_delivered(monkeypatch):
    monkeypatch.setattr(model_utils, "gpu_available", lambda: True)

    assert model_utils.effective_device("GPU", "birdnet", "3.0") == "GPU"
    assert model_utils.effective_device("gpu:1", "birdnet", "3.0") == "GPU:1"


def test_effective_device_falls_back_to_cpu_instead_of_failing_in_a_worker(monkeypatch):
    monkeypatch.setattr(model_utils, "gpu_available", lambda: True)

    assert model_utils.effective_device("GPU", "birdnet", "2.4") == "CPU"
    assert model_utils.effective_device("GPU", "perch") == "CPU"
    assert model_utils.effective_device("GPU", classifier="custom.tflite") == "CPU"

    monkeypatch.setattr(model_utils, "gpu_available", lambda: False)
    assert model_utils.effective_device("GPU", "birdnet", "3.0") == "CPU"


def test_gpu_is_only_offered_for_a_cuda_capable_onnxruntime(monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    assert not model_utils.gpu_available()

    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    assert not model_utils.gpu_available()

    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert model_utils.gpu_available()
