import logging

from birdnet_analyzer.settings import apply_model_directory

# Library convention: emit records but leave the configuration to the host
# application. The shipped CLI and GUI configure handlers via logs.setup_logging().
logging.getLogger(__name__).addHandler(logging.NullHandler())

# The birdnet library reads BIRDNET_APP_DATA whenever it needs a model path.
apply_model_directory()

from birdnet_analyzer.analyze import analyze  # noqa: E402
from birdnet_analyzer.embeddings import embeddings  # noqa: E402
from birdnet_analyzer.search import search  # noqa: E402
from birdnet_analyzer.segments import segments  # noqa: E402
from birdnet_analyzer.species import species  # noqa: E402
from birdnet_analyzer.train import train  # noqa: E402

__version__ = "2.4.0"
__all__ = ["analyze", "embeddings", "search", "segments", "species", "train"]
