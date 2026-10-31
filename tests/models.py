def pb_model_cached(version: str) -> bool:
    from birdnet.utils.downloads import check_protobuf_model_files_exist
    from birdnet.utils.local_data import get_model_path

    return check_protobuf_model_files_exist(
        get_model_path("acoustic", version, "pb", "fp32")
    )
