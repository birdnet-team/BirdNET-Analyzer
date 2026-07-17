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
