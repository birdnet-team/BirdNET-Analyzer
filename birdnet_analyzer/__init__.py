import logging
import os
import warnings

from birdnet_analyzer.settings import apply_model_directory

# Before anything can import birdnet: it resolves its model directory from
# BIRDNET_APP_DATA once, at import time.
apply_model_directory()

from birdnet_analyzer.analyze import analyze  # noqa: E402
from birdnet_analyzer.embeddings import embeddings  # noqa: E402
from birdnet_analyzer.search import search  # noqa: E402
from birdnet_analyzer.segments import segments  # noqa: E402
from birdnet_analyzer.species import species  # noqa: E402
from birdnet_analyzer.train import train  # noqa: E402

__version__ = "2.4.0"
__all__ = ["analyze", "embeddings", "search", "segments", "species", "train"]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"

warnings.filterwarnings("ignore")

# Library convention: emit records but leave the configuration to the host
# application. The shipped CLI and GUI configure handlers via logs.setup_logging().
logging.getLogger(__name__).addHandler(logging.NullHandler())
