"""The models the app downloads through the birdnet library, and which are on disk.

Importable without TensorFlow or ONNX Runtime: the GUI asks before anything is loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

ACOUSTIC_3_0 = "acoustic-3.0"
ACOUSTIC_2_4 = "acoustic-2.4"
PERCH = "perch"
GEO_3_0 = "geo-3.0"
GEO_2_4 = "geo-2.4"
TRAINING_3_0 = "training-3.0"
TRAINING_2_4 = "training-2.4"


@dataclass(frozen=True)
class ModelFiles:
    key: str
    downloaded: bool
    download_size: int


def _acoustic_3_0() -> tuple[bool, int]:
    from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxDownloaderV3_0, models
    from birdnet.globals import MODEL_PRECISION_FP32 as FP32

    return (
        AcousticOnnxDownloaderV3_0._check_acoustic_model_available(FP32),
        models[FP32].dl_size,
    )


def _acoustic_2_4() -> tuple[bool, int]:
    from birdnet.acoustic.models.v2_4.tf import AcousticTFDownloaderV2_4, models
    from birdnet.globals import MODEL_PRECISION_FP32 as FP32

    return (
        AcousticTFDownloaderV2_4._check_acoustic_model_available(FP32),
        models[FP32].dl_size,
    )


def _perch() -> tuple[bool, int]:
    from birdnet.acoustic.models.perch_v2.pb import AcousticPBDownloaderPerchV2

    return (
        AcousticPBDownloaderPerchV2._check_acoustic_model_available("CPU"),
        AcousticPBDownloaderPerchV2.MODEL_DOWNLOAD_SIZE_CPU,
    )


def _geo_3_0() -> tuple[bool, int]:
    from birdnet.geo.models.v3_0.onnx import GeoOnnxDownloaderV3_0, models
    from birdnet.globals import MODEL_PRECISION_FP32 as FP32

    return (
        GeoOnnxDownloaderV3_0._check_geo_model_available(FP32),
        models[FP32].dl_size,
    )


def _geo_2_4() -> tuple[bool, int]:
    from birdnet.geo.models.v2_4.tf import GeoTFDownloaderV2_4

    return (
        GeoTFDownloaderV2_4._check_geo_model_available(),
        GeoTFDownloaderV2_4._model_info.dl_size,
    )


def _training_3_0() -> tuple[bool, int]:
    from birdnet.acoustic.models.v3_0 import pb

    return (
        pb.AcousticPBDownloaderV3_0._check_acoustic_model_available(),
        pb._PB_DL_SIZE,
    )


def _training_2_4() -> tuple[bool, int]:
    from birdnet.acoustic.models.v2_4 import pb

    return (
        pb.AcousticPBDownloaderV2_4._check_acoustic_model_available(),
        pb._PB_DL_SIZE,
    )


# In the order the settings tab lists them.
_PROBES: dict[str, Callable[[], tuple[bool, int]]] = {
    ACOUSTIC_3_0: _acoustic_3_0,
    ACOUSTIC_2_4: _acoustic_2_4,
    PERCH: _perch,
    GEO_3_0: _geo_3_0,
    GEO_2_4: _geo_2_4,
    TRAINING_3_0: _training_3_0,
    TRAINING_2_4: _training_2_4,
}


def model_files(key: str) -> ModelFiles | None:
    """What the model directory holds of one download.

    None when the installed birdnet library cannot tell: the probes read its
    downloader classes, which are not part of its public interface.
    """
    try:
        downloaded, download_size = _PROBES[key]()
    except Exception:
        logger.warning("The state of the %s model is unknown", key, exc_info=True)
        return None

    return ModelFiles(key, bool(downloaded), int(download_size))


def all_model_files() -> list[ModelFiles]:
    return [files for key in _PROBES if (files := model_files(key)) is not None]


def acoustic_key(version: str) -> str:
    return ACOUSTIC_3_0 if version == "3.0" else ACOUSTIC_2_4


def geo_key(version: str) -> str:
    """The geo model a location filter uses with the acoustic model ``version``."""
    return GEO_2_4 if version == "2.4" else GEO_3_0


def training_keys(version: str) -> tuple[str, str]:
    """The downloads a training on ``version`` uses: embeddings, then the export."""
    if version == "3.0":
        return ACOUSTIC_3_0, TRAINING_3_0

    return ACOUSTIC_2_4, TRAINING_2_4


def missing_download_size(*keys: str) -> int:
    """The bytes a first use of these models downloads; 0 when all are on disk."""
    return sum(
        files.download_size
        for key in keys
        if (files := model_files(key)) is not None and not files.downloaded
    )


_LIBRARY_FOLDERS = ("acoustic-models", "geo-models", "taxonomy-v3")


def stored_size(directory: str) -> int:
    """The bytes of the birdnet library's files in the model directory.

    Only its own folders are read: the directory is the user's choice and can
    be one with a lot else in it.
    """
    import os

    total = 0

    for folder in _LIBRARY_FOLDERS:
        for root, _dirs, names in os.walk(os.path.join(directory, folder)):
            for name in names:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue

    return total
