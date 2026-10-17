"""The BirdNET models a custom classifier can be trained on.

Importable without TensorFlow, so the GUI can consult it before training starts.
"""

from __future__ import annotations

from dataclasses import dataclass

from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.pb import AcousticPBDownloaderV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.pb import AcousticPBDownloaderV3_0


@dataclass(frozen=True)
class BaseModel:
    """The BirdNET model a custom classifier is trained on and exported with."""

    version: str
    segment_samples: int
    sample_rate: int
    embeddings_dim: int
    # 3.0 applies the sigmoid inside its graph and outputs probabilities, so every
    # export built on it does the same. 2.4 outputs logits; the pipeline's sigmoid
    # (and with it the sensitivity setting) applies to those.
    outputs_probabilities: bool

    def load(self):
        """The SavedModel and its English labels; downloads the model on first use."""
        import tensorflow as tf

        downloader = (
            AcousticPBDownloaderV2_4
            if self.version == "2.4"
            else AcousticPBDownloaderV3_0
        )
        path, labels = downloader.get_model_path_and_labels("en_us")

        return tf.saved_model.load(str(path)), list(labels)


_BASE_MODELS = {
    "2.4": BaseModel(
        "2.4",
        AcousticModelV2_4.get_segment_size_samples(),
        AcousticModelV2_4.get_sample_rate(),
        AcousticModelV2_4.get_embeddings_dim(),
        outputs_probabilities=False,
    ),
    "3.0": BaseModel(
        "3.0",
        AcousticModelV3_0.get_segment_size_samples(),
        AcousticModelV3_0.get_sample_rate(),
        AcousticModelV3_0.get_embeddings_dim(),
        outputs_probabilities=True,
    ),
}


def base_model(version: str) -> BaseModel:
    if version not in _BASE_MODELS:
        raise ValueError(
            f"Unsupported BirdNET version for training: {version}. "
            f"Supported versions are: {', '.join(_BASE_MODELS)}."
        )

    return _BASE_MODELS[version]


def version_for_embeddings_dim(dim: int) -> str:
    """The BirdNET version whose embeddings are ``dim`` wide.

    Caches written before training recorded the version identify their base model by
    nothing else.
    """
    for base in _BASE_MODELS.values():
        if base.embeddings_dim == dim:
            return base.version

    raise ValueError(f"No BirdNET model has {dim}-dimensional embeddings.")
