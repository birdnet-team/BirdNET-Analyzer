import os

import numpy as np
import pytest

from birdnet_analyzer import utils

# The first bytes of a TFLite model. Not valid UTF-8, so reading it as text fails.
TFLITE_BYTES = bytes([0x1C, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4C, 0x33, 0xFF, 0xFE, 0x80])


def classifier(tmp_path, name="CustomClassifier.tflite"):
    path = tmp_path / name
    path.write_bytes(TFLITE_BYTES)

    return str(path)


def test_the_labels_of_a_trained_classifier_are_read(tmp_path):
    path = classifier(tmp_path)
    (tmp_path / "CustomClassifier_Labels.txt").write_text(
        "Species one\nSpecies two\n", encoding="utf-8"
    )

    assert utils.read_classifier_labels(path) == ["Species one", "Species two"]


def test_the_labels_of_a_birdnet_model_are_read(tmp_path):
    path = classifier(tmp_path, "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")
    (tmp_path / "BirdNET_GLOBAL_6K_V2.4_Labels.txt").write_text(
        "Cyanocitta cristata_Blue Jay\n", encoding="utf-8"
    )

    assert utils.read_classifier_labels(path) == ["Cyanocitta cristata_Blue Jay"]


def test_a_classifier_without_a_label_file_has_no_labels(tmp_path):
    # The classifier must not be mistaken for its own label file, which would read the
    # model as UTF-8 text.
    assert utils.read_classifier_labels(classifier(tmp_path)) is None


def test_a_birdnet_model_without_a_label_file_has_no_labels(tmp_path):
    path = classifier(tmp_path, "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")

    assert utils.read_classifier_labels(path) is None


def _mesh(figure):
    return figure.axes[0].collections[0].get_array()


def test_a_short_clip_is_drawn_frame_by_frame():
    sr = 48000
    figure = utils.spectrogram_from_audio(
        np.zeros(sr * 3, dtype="float32"), sr, fig_num="spec", fig_size=(20, 4)
    )

    assert _mesh(figure).shape[1] == 1 + sr * 3 // 512


def test_a_folder_scan_offers_only_formats_the_analysis_reads(tmp_path):
    for name in ("a.wav", "b.m4a", "c.wma", "d.flac"):
        (tmp_path / name).write_bytes(b"")

    found = utils.collect_audio_files(str(tmp_path))

    assert [os.path.basename(f) for f in found] == ["a.wav", "d.flac"]


def test_a_long_recording_is_drawn_at_pixel_resolution_keeping_peaks():
    sr = 48000
    signal = np.zeros(sr * 600, dtype="float32")
    signal[sr * 300 : sr * 300 + 480] = 1.0

    figure = utils.spectrogram_from_audio(signal, sr, fig_num="spec", fig_size=(20, 4))
    mesh = _mesh(figure)

    assert 20 * figure.dpi / 2 < mesh.shape[1] <= 20 * figure.dpi
    assert mesh.max() == pytest.approx(0)
    assert mesh.max(axis=0).argmax() == pytest.approx(mesh.shape[1] / 2, abs=2)
