"""Which of the birdnet library's downloads the model directory holds."""

import pytest

from birdnet_analyzer import model_store, settings
from tests.models import put_acoustic_2_4, put_acoustic_3_0, put_labels_2_4

# What the Windows installer verifies before it hands over the model files of an
# installation of 2.4.0 or older, and the note it writes beside the SavedModel.
TFLITE_SIZE_OF_2_4_0 = 51_726_412
SAVED_MODEL_SOURCE = (
    "https://zenodo.org/records/15050749/files/BirdNET_v2.4_protobuf.zip"
)


@pytest.fixture
def model_directory(monkeypatch, tmp_path):
    directory = tmp_path / "models"
    monkeypatch.setenv(settings.MODEL_DIR_ENV_VAR, str(directory))

    return directory


def test_the_installed_library_tells_the_state_of_every_model(model_directory):
    models = model_store.all_model_files()

    assert [files.key for files in models] == list(model_store._PROBES)
    assert not any(files.downloaded for files in models)
    assert all(files.download_size > 1_000_000 for files in models)


def test_a_model_counts_as_downloaded_once_its_files_are_there(model_directory):
    size = model_store.model_files(model_store.ACOUSTIC_2_4).download_size

    assert model_store.missing_download_size(model_store.ACOUSTIC_2_4) == size

    put_acoustic_2_4(model_directory)

    assert model_store.model_files(model_store.ACOUSTIC_2_4).downloaded
    assert model_store.missing_download_size(model_store.ACOUSTIC_2_4) == 0
    assert not model_store.model_files(model_store.TRAINING_2_4).downloaded


def test_the_default_model_counts_as_downloaded_once_its_files_are_there(
    model_directory,
):
    assert not model_store.model_files(model_store.ACOUSTIC_3_0).downloaded

    put_acoustic_3_0(model_directory)

    assert model_store.model_files(model_store.ACOUSTIC_3_0).downloaded
    assert not model_store.model_files(model_store.TRAINING_3_0).downloaded


def test_a_training_downloads_the_embeddings_model_and_the_export_model(
    model_directory,
):
    embeddings, export = model_store.training_keys("2.4")
    both = model_store.missing_download_size(embeddings, export)

    put_acoustic_2_4(model_directory)

    assert 0 < model_store.missing_download_size(embeddings, export) < both


def test_the_2_4_model_of_an_earlier_installation_is_used_without_a_download(
    model_directory, tmp_path
):
    bundle = tmp_path / "birdnet_models"
    v2_4 = bundle / "acoustic-models" / "v2.4"
    put_labels_2_4(v2_4 / "tf" / "labels")
    put_labels_2_4(v2_4 / "pb" / "labels")

    with open(v2_4 / "tf" / "model-fp32.tflite", "wb") as file:
        file.truncate(TFLITE_SIZE_OF_2_4_0)

    saved_model = v2_4 / "pb" / "model-fp32"
    (saved_model / "variables").mkdir(parents=True)

    for name in (
        "saved_model.pb",
        "variables/variables.data-00000-of-00001",
        "variables/variables.index",
    ):
        (saved_model / name).write_bytes(b"")

    (saved_model / ".birdnet_source").write_text(SAVED_MODEL_SOURCE)

    settings.seed_bundled_models(bundle)

    assert model_store.model_files(model_store.ACOUSTIC_2_4).downloaded
    assert model_store.model_files(model_store.TRAINING_2_4).downloaded


def test_the_labels_shipped_alone_do_not_pass_for_a_model(model_directory, tmp_path):
    bundle = tmp_path / "birdnet_models"
    put_labels_2_4(bundle / "acoustic-models" / "v2.4" / "tf" / "labels")
    put_labels_2_4(bundle / "acoustic-models" / "v2.4" / "pb" / "labels")

    settings.seed_bundled_models(bundle)

    assert not model_store.model_files(model_store.ACOUSTIC_2_4).downloaded
    assert not model_store.model_files(model_store.TRAINING_2_4).downloaded
