import os
from typing import Literal, get_args

from birdnet.globals import (
    ACOUSTIC_MODEL_VERSIONS,
    GEO_MODEL_VERSIONS,
    MODEL_LANGUAGE_EN_US,
    MODEL_LANGUAGES,
    VALID_MODEL_LANGUAGES_V2_4,
    VALID_MODEL_LANGUAGES_V3_0,
)


def _newest_version(versions: tuple[str, ...]) -> str:
    """Return the highest ``"<major>.<minor>"`` version from the given tuple."""
    return max(versions, key=lambda v: tuple(int(part) for part in v.split(".")))


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
RANDOM_SEED: int = 42

# The acoustic/geo model version used by default. Derived from the versions the
# installed birdnet library ships, so the analyzer follows the newest model without
# a code change. The geo model is not offered as a choice: its newest version always
# replaces the older ones (see birdnet_analyzer.model_utils.run_geomodel).
DEFAULT_ACOUSTIC_MODEL_VERSION: str = _newest_version(get_args(ACOUSTIC_MODEL_VERSIONS))
DEFAULT_GEO_MODEL_VERSION: str = _newest_version(get_args(GEO_MODEL_VERSIONS))

# The languages each acoustic model version can label its predictions in. v2.4 and
# v3.0 support different sets, so the CLI/GUI offer their union and the birdnet
# library validates the concrete (version, language) pair when a model is loaded.
ACOUSTIC_MODEL_LANGUAGES: dict[str, list[str]] = {
    "2.4": list(VALID_MODEL_LANGUAGES_V2_4),
    "3.0": list(VALID_MODEL_LANGUAGES_V3_0),
}
ALL_MODEL_LANGUAGES: list[str] = sorted(
    set(VALID_MODEL_LANGUAGES_V2_4) | set(VALID_MODEL_LANGUAGES_V3_0)
)

# Languages the (always newest) geo model can label its species list in. The v3.0
# geo and acoustic models share the same language set.
GEO_MODEL_LANGUAGES: list[str] = list(VALID_MODEL_LANGUAGES_V3_0)

MODEL_VERSION: str = f"V{DEFAULT_ACOUSTIC_MODEL_VERSION}"
SCORE_FUNCTIONS = Literal["cosine", "euclidean", "dot"]
CROP_MODES = Literal["center", "first", "segments"]
CODES_FILE: str = os.path.join(SCRIPT_DIR, "eBird_taxonomy_codes_2024E.json")
ALLOWED_FILETYPES: list[str] = [
    "wav",
    "flac",
    "mp3",
    "ogg",
    "m4a",
    "wma",
    "aiff",
    "aif",
]
RESULT_TYPES = Literal["table", "audacity", "kaleidoscope", "csv", "parquet"]
ADDITIONAL_COLUMNS = Literal[
    "lat", "lon", "week", "overlap", "sensitivity", "min_conf", "species_list", "model"
]
OUTPUT_RAVEN_FILENAME: str = "BirdNET_SelectionTable.txt"
OUTPUT_KALEIDOSCOPE_FILENAME: str = "BirdNET_Kaleidoscope.csv"
OUTPUT_CSV_FILENAME: str = "BirdNET_CombinedTable.csv"
OUTPUT_AUDACITY_FILENAME: str = "BirdNET_AudacityLabels.txt"
OUTPUT_PARQUET_FILENAME: str = "BirdNET_CombinedTable.parquet"
ANALYSIS_PARAMS_FILENAME: str = "birdnet.analyze-params.csv"
TRAIN_PARAMS_SUFFIX: str = ".birdnet.train-params.csv"
LABEL_LANGUAGE: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US
SAMPLE_CROP_MODES = Literal["center", "first", "segments", "smart"]
NON_EVENT_CLASSES: list[str] = ["noise", "other", "background", "silence"]
UPSAMPLING_MODES = Literal["repeat", "mean", "smote"]
TRAINED_MODEL_OUTPUT_FORMATS = Literal["tflite", "raven", "detached"]
TRAINED_MODEL_SAVE_MODES = Literal["replace", "append"]
AUTOTUNE_METRICS = Literal["val_loss", "val_AUPRC", "val_AUROC"]
