import os
import warnings

from absl import logging

from birdnet_analyzer.analyze import analyze
from birdnet_analyzer.embeddings import embeddings
from birdnet_analyzer.search import search
from birdnet_analyzer.segments import segments
from birdnet_analyzer.species import species
from birdnet_analyzer.train import train

__version__ = "2.2.0"
__all__ = ["analyze", "embeddings", "search", "segments", "species", "train"]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.set_verbosity(logging.ERROR)
warnings.filterwarnings("ignore")
