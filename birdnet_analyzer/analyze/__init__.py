from typing import get_args

import birdnet_analyzer.config as cfg
from birdnet_analyzer.analyze.core import (
    analyze,
    save_as_audacity,
    save_as_csv,
    save_as_kaleidoscope,
    save_as_rtable,
)

POSSIBLE_ADDITIONAL_COLUMNS = list(get_args(cfg.ADDITIONAL_COLUMNS))

__all__ = [
    "analyze",
    "save_as_audacity",
    "save_as_csv",
    "save_as_kaleidoscope",
    "save_as_rtable",
]
