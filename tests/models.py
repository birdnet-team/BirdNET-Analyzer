def pb_model_cached(version: str) -> bool:
    from birdnet.acoustic.models.v2_4.pb import AcousticPBDownloaderV2_4
    from birdnet.acoustic.models.v3_0.pb import AcousticPBDownloaderV3_0
    from birdnet.utils.downloads import check_protobuf_model_files_exist

    downloader = (
        AcousticPBDownloaderV2_4 if version == "2.4" else AcousticPBDownloaderV3_0
    )
    model_path, _ = downloader._get_paths()

    return check_protobuf_model_files_exist(model_path)


def put_acoustic_2_4(directory) -> None:
    """Fills ``directory`` the way a download of the BirdNET 2.4 tflite model does."""
    from birdnet.acoustic.models.v2_4.tf import models

    tf = directory / "acoustic-models" / "v2.4" / "tf"
    put_labels_2_4(tf / "labels")

    with open(tf / "model-fp32.tflite", "wb") as file:
        file.truncate(models["fp32"].file_size)


def put_acoustic_3_0(directory) -> None:
    """Fills ``directory`` the way a download of the BirdNET 3.0 ONNX model does."""
    from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxDownloaderV3_0, models
    from birdnet.utils.local_data import get_birdnet_app_data_folder

    model, labels = (
        directory / path.relative_to(get_birdnet_app_data_folder())
        for path in AcousticOnnxDownloaderV3_0._get_paths("fp32")
    )
    labels.mkdir(parents=True)

    for language in AcousticOnnxDownloaderV3_0.AVAILABLE_LANGUAGES:
        (labels / f"{language}.txt").write_text("")

    with open(model, "wb") as file:
        file.truncate(models["fp32"].file_size)


def put_labels_2_4(directory) -> None:
    from birdnet.acoustic.models.v2_4.tf import AcousticTFDownloaderV2_4

    directory.mkdir(parents=True)

    for language in AcousticTFDownloaderV2_4.AVAILABLE_LANGUAGES:
        (directory / f"{language}.txt").write_text("")
