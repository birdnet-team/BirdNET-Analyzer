"""Real exports of a classifier head on each BirdNET base.

They need the base model's SavedModel, which CI does not pre-download (the 3.0 one
is a 500 MB zip), so each test skips unless that model is already cached.
"""

import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from birdnet_analyzer import model  # noqa: E402

LABELS = ["Sci a_Common a", "Sci b_Common b", "Sci c_Common c"]


def _audio(base):
    """A noise segment: digital silence takes the 3.0 SavedModel to NaN scores."""
    rng = np.random.default_rng(0)

    return rng.normal(0, 0.05, (1, base.segment_samples)).astype(np.float32)


def _pb_cached(version: str) -> bool:
    from birdnet.utils.helper import check_protobuf_model_files_exist
    from birdnet.utils.local_data import get_model_path

    return check_protobuf_model_files_exist(
        get_model_path("acoustic", version, "pb", "fp32")
    )


needs_3_0 = pytest.mark.skipif(not _pb_cached("3.0"), reason="3.0 pb model not cached")
needs_2_4 = pytest.mark.skipif(not _pb_cached("2.4"), reason="2.4 pb model not cached")


def _head(version: str):
    """A trained classifier as train_model hands it over: activation popped."""
    classifier = model.build_linear_classifier(
        len(LABELS), model.base_model(version).embeddings_dim
    )
    classifier.pop()

    return classifier


def _interpreter(path):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    return interpreter


def _read_labels(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


@needs_3_0
def test_tflite_export_on_3_0_mirrors_the_official_model(tmp_path):
    path = tmp_path / "cc.tflite"
    base = model.base_model("3.0")

    model.save_linear_classifier(_head("3.0"), str(path), LABELS, version="3.0")

    interpreter = _interpreter(path)
    signature = interpreter.get_signature_list()["serving_default"]
    assert signature["inputs"] == ["x"]
    assert sorted(signature["outputs"]) == ["embeddings", "predictions"]

    runner = interpreter.get_signature_runner("serving_default")
    out = runner(x=_audio(base))
    assert out["predictions"].shape == (1, len(LABELS))
    assert out["embeddings"].shape == (1, base.embeddings_dim)
    assert out["predictions"].min() >= 0
    assert out["predictions"].max() <= 1
    assert _read_labels(tmp_path / "cc_Labels.txt") == LABELS


@needs_3_0
def test_tflite_append_on_3_0_keeps_the_official_classes_first(tmp_path):
    path = tmp_path / "cc.tflite"
    base = model.base_model("3.0")
    _, official = base.load()

    model.save_linear_classifier(
        _head("3.0"), str(path), LABELS, mode="append", version="3.0"
    )

    runner = _interpreter(path).get_signature_runner("serving_default")
    out = runner(x=_audio(base))
    assert out["predictions"].shape == (1, len(official) + len(LABELS))
    assert out["predictions"].min() >= 0
    assert out["predictions"].max() <= 1
    assert _read_labels(tmp_path / "cc_Labels.txt") == [*official, *LABELS]


@needs_3_0
def test_raven_export_on_3_0_scores_probabilities_at_32_khz(tmp_path):
    path = tmp_path / "cc"
    base = model.base_model("3.0")

    model.save_raven_model(_head("3.0"), str(path), LABELS, version="3.0")

    loaded = tf.saved_model.load(str(path))
    basic = loaded.signatures["basic"]
    assert basic.structured_input_signature[1]["inputs"].shape.as_list() == [
        None,
        base.segment_samples,
    ]

    scores = basic(inputs=tf.constant(_audio(base)))["scores"].numpy()
    assert scores.shape == (1, len(LABELS))
    assert scores.min() >= 0
    assert scores.max() <= 1

    with open(path / "model_config.json", encoding="utf-8") as f:
        config = json.load(f)

    assert config["signatures"][0]["modelInputs"][0]["sampleRate"] == 32000.0
    assert "BirdNET 3.0" in config["modelDescription"]
    assert _read_labels(tmp_path / "cc_Labels.txt") == LABELS


@needs_3_0
def test_detached_export_on_3_0_takes_1280_embeddings(tmp_path):
    base = model.base_model("3.0")

    model.save_detached_classifier(
        _head("3.0"), str(tmp_path / "cc"), labels=LABELS, version="3.0"
    )

    interpreter = _interpreter(tmp_path / "cc_detached.tflite")
    (inp,) = interpreter.get_input_details()
    assert list(inp["shape"]) == [1, base.embeddings_dim]

    interpreter.set_tensor(inp["index"], np.zeros((1, base.embeddings_dim), np.float32))
    interpreter.invoke()
    out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    assert out.min() >= 0
    assert out.max() <= 1
    assert (tmp_path / "cc_detached" / "saved_model.pb").is_file()
    assert _read_labels(tmp_path / "cc_detached_Labels.txt") == LABELS


@needs_2_4
@pytest.mark.parametrize(("mode", "n_outputs"), [("replace", 3), ("append", 6522 + 3)])
def test_tflite_export_on_2_4_keeps_its_single_logit_output(tmp_path, mode, n_outputs):
    path = tmp_path / "cc.tflite"
    base = model.base_model("2.4")

    model.save_linear_classifier(
        _head("2.4"), str(path), LABELS, mode=mode, version="2.4"
    )

    interpreter = _interpreter(path)
    (inp,) = interpreter.get_input_details()
    (out,) = interpreter.get_output_details()
    assert list(inp["shape"]) == [1, base.segment_samples]
    assert list(out["shape"]) == [1, n_outputs]
    assert len(interpreter.get_signature_list()["serving_default"]["outputs"]) == 1
