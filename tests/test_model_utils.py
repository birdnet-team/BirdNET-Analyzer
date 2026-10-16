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


def test_supports_sensitivity_for_every_model_but_perch():
    assert model_utils.supports_sensitivity("birdnet")
    assert not model_utils.supports_sensitivity("perch")

    assert model_utils.effective_sensitivity(1.25, "birdnet") == 1.25
    assert model_utils.effective_sensitivity(1.25, "perch") == 1.0


def test_classifier_version_reads_each_file_once(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []

    def fake_inspect(path, *, library):
        calls.append((path, library))
        return SimpleNamespace(version="3.0")

    monkeypatch.setattr(model_utils.birdnet, "inspect_custom_tflite", fake_inspect)
    model_utils._inspect_classifier.cache_clear()
    classifier = tmp_path / "cc.tflite"
    classifier.write_bytes(b"v1")

    assert model_utils.classifier_version(str(classifier)) == "3.0"
    assert model_utils.classifier_version(str(classifier)) == "3.0"
    assert calls == [(str(classifier), "litert")]

    classifier.write_bytes(b"v2, longer")
    assert model_utils.classifier_version(str(classifier)) == "3.0"
    assert len(calls) == 2, "a changed file is inspected again"


def test_classifier_version_rejects_a_missing_file():
    with pytest.raises(FileNotFoundError):
        model_utils.classifier_version("nowhere/cc.tflite")


def test_classifier_batch_size_is_one_on_3_0():
    assert model_utils.classifier_batch_size("2.4", 8) == 8
    assert model_utils.classifier_batch_size("3.0", 8) == 1
    assert model_utils.classifier_batch_size("3.0", 1) == 1


def test_validate_min_conf_rejects_non_probabilities():
    model_utils.validate_min_conf(0.25)
    model_utils.validate_min_conf(0.0)  # the GUI's top-n path passes 0

    with pytest.raises(ValueError, match="probabilities"):
        model_utils.validate_min_conf(1.0)

    with pytest.raises(ValueError, match="probabilities"):
        model_utils.validate_min_conf(-0.1)


def test_run_inference_normalizes_perch_with_softmax(monkeypatch, tmp_path):
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
    monkeypatch.setattr(
        model_utils.birdnet, "load_perch_v2", lambda *a, **k: fake_model
    )

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    result = model_utils.run_inference(str(audio), model="perch")

    assert result == "result"
    assert seen["apply_softmax"] is True
    assert seen["apply_sigmoid"] is False


def test_run_inference_keeps_the_sensitivity_for_3_0(monkeypatch, tmp_path):
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
    assert seen["sigmoid_sensitivity"] == 1.25


@pytest.mark.parametrize(
    ("version", "batch_size", "sensitivity"), [("2.4", 4, 1.25), ("3.0", 1, 1.25)]
)
def test_run_inference_loads_a_classifier_on_its_own_version(
    monkeypatch, tmp_path, version, batch_size, sensitivity
):
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    seen = {}
    loaded = {}

    @contextmanager
    def fake_predict_session(**kwargs):
        seen.update(kwargs)
        session = MagicMock()
        session.run.return_value = "result"
        yield session

    fake_model = MagicMock()
    fake_model.predict_session = fake_predict_session

    def fake_load_custom(model_type, loaded_version, backend, classifier, labels):
        loaded.update(version=loaded_version, classifier=classifier, labels=labels)
        return fake_model

    monkeypatch.setattr(model_utils, "classifier_version", lambda path: version)
    monkeypatch.setattr(model_utils.birdnet, "load_custom", fake_load_custom)

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    result = model_utils.run_inference(
        str(audio), classifier="cc.tflite", batch_size=4, sigmoid_sensitivity=1.25
    )

    assert result == "result"
    assert loaded == {
        "version": version,
        "classifier": "cc.tflite",
        "labels": "cc_Labels.txt",
    }
    assert seen["batch_size"] == batch_size
    assert seen["sigmoid_sensitivity"] == sensitivity
