import os
from typing import Literal

from birdnet.globals import MODEL_LANGUAGE_EN_US, MODEL_LANGUAGES

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
RANDOM_SEED: int = 42
MODEL_VERSION: str = "V2.4"
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
