def pb_model_cached(version: str) -> bool:
    from birdnet.acoustic.models.v2_4.pb import AcousticPBDownloaderV2_4
    from birdnet.acoustic.models.v3_0.pb import AcousticPBDownloaderV3_0
    from birdnet.utils.downloads import check_protobuf_model_files_exist

    downloader = (
        AcousticPBDownloaderV2_4 if version == "2.4" else AcousticPBDownloaderV3_0
    )
    model_path, _ = downloader._get_paths()

    return check_protobuf_model_files_exist(model_path)
