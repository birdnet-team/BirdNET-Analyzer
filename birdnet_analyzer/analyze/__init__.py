import os

import birdnet_analyzer.config as cfg
from birdnet_analyzer.analyze.core import analyze

POSSIBLE_ADDITIONAL_COLUMNS_MAP = {
    "lat": lambda: cfg.LATITUDE,
    "lon": lambda: cfg.LONGITUDE,
    "week": lambda: cfg.WEEK,
    "overlap": lambda: cfg.SIG_OVERLAP,
    "sensitivity": lambda: cfg.SIGMOID_SENSITIVITY,
    "min_conf": lambda: cfg.MIN_CONFIDENCE,
    "species_list": lambda: cfg.SPECIES_LIST_FILE or "",
    "model": lambda: os.path.basename(cfg.MODEL_PATH),
}

__all__ = [
    "analyze",
]
