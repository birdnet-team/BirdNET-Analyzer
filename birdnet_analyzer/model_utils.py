from __future__ import annotations

import logging
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, cast

import birdnet

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    import numpy as np
    from birdnet.acoustic.inference.core.encoding.encoding_result import (
        AcousticFileEncodingResult,
    )
    from birdnet.acoustic.inference.core.perf_tracker import (
        AcousticProgressStats,
    )
    from birdnet.acoustic.inference.core.prediction.prediction_result import (
        AcousticFilePredictionResult,
    )
    from birdnet.acoustic.inference.session import (
        AcousticEncodingSession,
        AcousticSessionBase,
    )
    from birdnet.globals import (
        ACOUSTIC_MODEL_VERSIONS,
        MODEL_LANGUAGES,
        MODEL_LANGUAGES_V2_4,
        MODEL_LANGUAGES_V3_0,
    )

logger = logging.getLogger(__name__)

GLOBAL_PREFETCH_RATIO = 2


def _scientific_name(species_label: str) -> str:
    """Return the scientific-name key of a ``"Scientific name_Common name"`` label."""
    return species_label.split("_", 1)[0]


def match_species_to_model(
    requested_species: Collection[str], model_species: Collection[str]
) -> set[str]:
    """Map requested species onto a model's labels by scientific name.

    The geo model and the acoustic model can use different taxonomies and label
    languages, so their ``"Scientific name_Common name"`` strings rarely match
    exactly even when they mean the same bird (the common name differs). The
    scientific name is stable across both, so it is used as the join key.

    Returns the subset of ``model_species`` whose scientific name also occurs in
    ``requested_species`` - i.e. the labels the acoustic model actually knows, which
    is what its custom species list requires (an unknown species raises in the
    library). This lets the (global) geo model filter any acoustic model version.

    Args:
        requested_species: Species names to keep, e.g. a geo model prediction.
        model_species: The acoustic model's own species labels.

    Returns:
        The matching labels, taken verbatim from ``model_species``.
    """
    model_by_scientific_name: dict[str, str] = {}
    for label in model_species:
        # First label wins should a scientific name ever appear twice.
        model_by_scientific_name.setdefault(_scientific_name(label), label)

    requested_scientific_names = {_scientific_name(name) for name in requested_species}

    return {
        label
        for scientific_name, label in model_by_scientific_name.items()
        if scientific_name in requested_scientific_names
    }


# list of sessions so they can be cancelled from another
# thread. Access is guarded by a lock
# because sessions are registered from Gradio worker threads while
# cancel_active_analyses() may be called from the main thread.
_ACTIVE_SESSIONS: set[AcousticSessionBase] = set()
_ACTIVE_SESSIONS_LOCK = threading.Lock()
# Latched once shutdown begins. A session that registers after this point
# cancels itself immediately, so no analysis can slip past
# cancel_active_analyses() and keep running headless.
_SHUTDOWN = threading.Event()


def _register_session(session) -> None:
    """Track a live inference session so it can be cancelled on shutdown."""
    with _ACTIVE_SESSIONS_LOCK:
        _ACTIVE_SESSIONS.add(session)
        # Read the latch under the same lock cancel_active_analyses() holds, so a
        # session registered concurrently with shutdown is either seen by the
        # cancel loop or cancelled here - never missed by both.
        shutting_down = _SHUTDOWN.is_set()

    if shutting_down:
        with suppress(Exception):
            session.cancel()


def _unregister_session(session) -> None:
    with _ACTIVE_SESSIONS_LOCK:
        _ACTIVE_SESSIONS.discard(session)


def active_session_count() -> int:
    """Return the number of inference sessions currently running."""
    with _ACTIVE_SESSIONS_LOCK:
        return len(_ACTIVE_SESSIONS)


def cancel_active_analyses() -> int:
    """Signal every in-flight analysis to cancel and latch shutdown.

    Uses the birdnet session cancel event, which stops the inference pipeline
    from consuming new work and lets each session tear down its worker/producer
    subprocesses and shared memory cleanly via its context manager. Safe to call
    from any thread.

    Latches shutdown so any analysis started after this call (e.g. by a Gradio
    worker thread still finishing a queued request) is cancelled as soon as it
    registers, instead of running headless.

    Returns:
        The number of sessions that were asked to cancel.
    """
    with _ACTIVE_SESSIONS_LOCK:
        _SHUTDOWN.set()
        sessions = list(_ACTIVE_SESSIONS)

    for session in sessions:
        with suppress(Exception):
            session.cancel()

    return len(sessions)


def pause_active_analyses() -> int:
    """Cancel every in-flight analysis without latching shutdown.

    Unlike :func:`cancel_active_analyses`, new analyses may still be started
    afterwards. Used by the GUI to pause a run: the resume journal keeps the
    per-file progress, so re-running the same analysis continues where it
    stopped.

    Returns:
        The number of sessions that were asked to cancel.
    """
    with _ACTIVE_SESSIONS_LOCK:
        sessions = list(_ACTIVE_SESSIONS)

    for session in sessions:
        with suppress(Exception):
            session.cancel()

    return len(sessions)


def _language_for_version(language: MODEL_LANGUAGES, version: str) -> MODEL_LANGUAGES:
    """Return ``language`` if the model ``version`` supports it, else ``en_us``.

    The 2.4 and 3.0 models support overlapping but non-identical language sets (e.g.
    3.0 has no Italian, 2.4 no Croatian), and the birdnet library raises on an
    unsupported ``(version, locale)`` pair. A locale that was valid for the model the
    user last used - or that they typed on the CLI - would otherwise crash the whole
    analysis. Coercing to English keeps the run going; only the label language, not
    the detections, is affected. The caller should have already surfaced a warning to
    the user where possible (the GUI limits the locale choices to the model).
    """
    from birdnet.globals import (
        MODEL_LANGUAGE_EN_US,
        VALID_MODEL_LANGUAGES_V2_4,
        VALID_MODEL_LANGUAGES_V3_0,
    )

    supported = (
        VALID_MODEL_LANGUAGES_V3_0 if version == "3.0" else VALID_MODEL_LANGUAGES_V2_4
    )
    if language in supported:
        return language

    logger.warning(
        "Locale '%s' is not available for BirdNET %s; using '%s' for labels instead.",
        language,
        version,
        MODEL_LANGUAGE_EN_US,
    )
    return MODEL_LANGUAGE_EN_US


def run_inference(
    path,
    model="birdnet",
    version: ACOUSTIC_MODEL_VERSIONS = "3.0",
    top_k: int | None = 5,
    batch_size=1,
    n_workers: int | None = None,
    n_producers: int = 1,
    prefetch_ratio=GLOBAL_PREFETCH_RATIO,
    overlap_duration_s=0.0,
    bandpass_fmin=0,
    bandpass_fmax=15_000,
    sigmoid_sensitivity=1.0,
    speed=1.0,
    min_confidence=0.1,
    custom_species_list=None,
    label_language: MODEL_LANGUAGES = "en_us",
    classifier: str | None = None,
    cc_species_list: str | None = None,
    match_species_by_scientific_name: bool = False,
    callback: Callable[[AcousticProgressStats], None] | None = None,
    on_file_complete: Callable[[AcousticFilePredictionResult], None] | None = None,
) -> AcousticFilePredictionResult:
    if classifier:
        if not cc_species_list:
            cc_species_list = classifier.replace(".tflite", "_Labels.txt", 1)

        # Custom classifiers are trained on 2.4 embeddings (training does not support
        # 3.0 yet), so they are loaded on the 2.4 base regardless of ``version``.
        acoustic_model = birdnet.load_custom(
            "acoustic", "2.4", "tf", classifier, cc_species_list
        )
    elif model == "birdnet":
        # A locale valid for one model version can be unsupported by another, which
        # the library rejects; coerce it to a supported one so the run never fails on
        # the label language. The cast is then sound: the value is known to be in the
        # target version's set.
        lang = _language_for_version(label_language, version)
        if version == "3.0":
            # 3.0 ships an ONNX backend: numerically equivalent predictions
            # (confidence differs by ~1e-6), markedly faster CPU inference, and no
            # TensorFlow import in the workers.
            acoustic_model = birdnet.load(
                "acoustic", "3.0", "onnx", lang=cast("MODEL_LANGUAGES_V3_0", lang)
            )
        else:
            # 2.4 has no ONNX build, so it stays on the TensorFlow backend.
            acoustic_model = birdnet.load(
                "acoustic", version, "tf", lang=cast("MODEL_LANGUAGES_V2_4", lang)
            )
    elif model == "perch":
        acoustic_model = birdnet.load_perch_v2("CPU")
    else:
        raise ValueError(
            f"Unsupported model: {model}\nSupported models are: 'birdnet', 'perch' or "
            "use a custom classifier."
        )

    # A species list derived from the geo model can name species the acoustic model
    # does not know (the geo model is global and uses a different taxonomy), which the
    # library would reject. Reconcile it against the loaded model by scientific name.
    if custom_species_list is not None and match_species_by_scientific_name:
        custom_species_list = match_species_to_model(
            custom_species_list, acoustic_model.species_list
        )

    from birdnet.acoustic.inference.configs import InferenceConfig

    input_files = InferenceConfig.validate_input_files(path)

    # Only pass the kwarg when used: birdnet releases without the per-file
    # completion hook reject it (see supports_on_file_complete()).
    session_kwargs = (
        {"on_file_complete": on_file_complete} if on_file_complete is not None else {}
    )

    with acoustic_model.predict_session(
        top_k=top_k,
        batch_size=batch_size,
        prefetch_ratio=prefetch_ratio,
        overlap_duration_s=overlap_duration_s,
        bandpass_fmin=bandpass_fmin,
        bandpass_fmax=bandpass_fmax,
        sigmoid_sensitivity=sigmoid_sensitivity,
        speed=speed,
        default_confidence_threshold=min_confidence,
        custom_species_list=custom_species_list,
        progress_callback=callback,
        show_stats="progress",
        n_workers=n_workers,
        n_producers=n_producers,
        apply_sigmoid=model != "perch",
        max_n_files=len(input_files),
        **session_kwargs,
    ) as session:
        _register_session(session)
        try:
            return session.run(input_files)  # ty:ignore[invalid-return-type]
        finally:
            _unregister_session(session)


def supports_on_file_complete() -> bool:
    """Whether the installed birdnet provides the per-file completion hook.

    Resumable analysis needs ``on_file_complete`` (birdnet-team/birdnet#57);
    on older releases the feature is silently disabled.
    """
    from importlib.util import find_spec

    return find_spec("birdnet.acoustic.inference.core.file_completion") is not None


def run_geomodel(
    lat, lon, week=None, language: MODEL_LANGUAGES = "en_us", threshold: float = 0.03
) -> birdnet.GeoPredictionResult:
    from birdnet_analyzer.config import DEFAULT_GEO_MODEL_VERSION

    # The newest geo model replaces the older ones outright; it is never a choice.
    # ``language`` only affects the localized species names, so callers that match on
    # scientific name (e.g. acoustic species filtering) can leave it at the default.
    #
    # The ONNX backend is used rather than tf/pb: it imports no TensorFlow at all -
    # and the geo model runs here in the main process, so this keeps TF out of it
    # entirely - loads much faster, and returns the same species. (The v3.0 geo tf
    # backend also only supports TensorFlow 2.18/2.19 while we require >=2.20; ONNX
    # has no such constraint.)
    #
    # This targets the v3.0 geo model specifically: the ONNX backend and the v3.0
    # language set below only exist for it. Assert the (runtime-computed) newest
    # version is 3.0 so a future geo version fails loudly here - prompting an update -
    # instead of silently mismatching; it also lets the type checker resolve the
    # concrete overload, so the casts are then sound.
    assert DEFAULT_GEO_MODEL_VERSION == "3.0", (
        f"run_geomodel targets geo v3.0, but the newest geo model is "
        f"{DEFAULT_GEO_MODEL_VERSION}; update the backend/language handling for it."
    )
    language = _language_for_version(language, DEFAULT_GEO_MODEL_VERSION)
    model = birdnet.load(
        "geo",
        DEFAULT_GEO_MODEL_VERSION,
        "onnx",
        lang=cast("MODEL_LANGUAGES_V3_0", language),
    )
    return model.predict(lat, lon, week=week, min_confidence=threshold)


def _load_acoustic_for_embeddings(version: ACOUSTIC_MODEL_VERSIONS):
    """Load the acoustic model for embedding, avoiding TensorFlow where possible.

    3.0 has an ONNX backend (no TensorFlow, faster) and its embeddings are equivalent
    to the TensorFlow backend's; 2.4 only has TensorFlow. Embedding takes no label
    language, so unlike :func:`run_inference` this loader has nothing to coerce.
    """
    if version == "3.0":
        return birdnet.load("acoustic", "3.0", "onnx")

    return birdnet.load("acoustic", version, "tf")


def get_embeddings(
    path: str,
    version: ACOUSTIC_MODEL_VERSIONS = "2.4",
    batch_size=1,
    n_workers: int | None = None,
    n_producers: int = 1,
    prefetch_ratio=GLOBAL_PREFETCH_RATIO,
    overlap_duration_s=0.0,
    bandpass_fmin=0,
    bandpass_fmax=15_000,
    speed=1.0,
    callback: Callable[[AcousticProgressStats], None] | None = None,
) -> AcousticFileEncodingResult:
    model = _load_acoustic_for_embeddings(version)
    return model.encode(
        path,
        batch_size=batch_size,
        prefetch_ratio=prefetch_ratio,
        overlap_duration_s=overlap_duration_s,
        bandpass_fmin=bandpass_fmin,
        bandpass_fmax=bandpass_fmax,
        speed=speed,
        progress_callback=callback,
        n_workers=n_workers,
        n_producers=n_producers,
    )  # ty:ignore[invalid-return-type]


def get_embeddings_array_with_session(
    session: AcousticEncodingSession,
    signals: list[tuple[np.ndarray, int]],
) -> np.ndarray:
    result = session.run_arrays(signals)

    # result.embeddings has shape (n_inputs, n_segments, embed_dim).
    # Each input signal is a single segment, so squeeze the middle dim.
    # Return shape: (n_inputs, embed_dim)
    return result.embeddings[:, 0, :]


def encode_arrays_batched(
    session: AcousticEncodingSession,
    signals: list[tuple[np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a batch of single-segment signals through an open encoding session.

    Unlike :func:`get_embeddings_array_with_session`, this runs the whole batch through
    the library pipeline in a single call (enabling worker parallelism and real
    batching) and reports which inputs produced a valid embedding.

    Args:
        session: An open ``AcousticEncodingSession``.
        signals: A list of ``(signal, sample_rate)`` tuples. Each signal must be exactly
            one model segment long (e.g. 3 s) so that it yields exactly one segment.

    Returns:
        A tuple ``(embeddings, valid_mask)`` where ``embeddings`` has shape
        ``(n_inputs, embed_dim)`` and ``valid_mask`` is a boolean array of shape
        ``(n_inputs,)`` that is ``True`` where the corresponding input produced a valid
        embedding. Inputs that could not be processed (e.g. failed decoding) are marked
        ``False`` and their embedding row should be discarded by the caller.
    """
    import numpy as np

    result = session.run_arrays(signals)

    # embeddings/embeddings_masked have shape (n_inputs, n_segments, embed_dim).
    # This helper assumes each input is exactly one model segment. Guard against a
    # caller passing longer signals (or a mismatched session config), which would
    # otherwise silently drop the extra segments taken by the [:, 0, :] slice below.
    n_segments = result.embeddings.shape[1]
    if n_segments != 1:
        raise ValueError(
            "encode_arrays_batched expects one segment per input, but the session "
            f"produced {n_segments} segments per input. Pass signals that are exactly "
            "one model segment long (e.g. 3 s)."
        )

    # A segment is invalid when every value in its mask row is True (see
    # AcousticEncodingResultBase.to_structured_array).
    embeddings = result.embeddings[:, 0, :]
    valid_mask = ~result.embeddings_masked[:, 0, :].all(axis=1)

    return embeddings, np.asarray(valid_mask, dtype=bool)


def get_embeddings_array(
    signals: list[np.ndarray],
    version: ACOUSTIC_MODEL_VERSIONS = "2.4",
    batch_size=1,
    n_workers: int | None = None,
    n_producers: int = 1,
    prefetch_ratio=GLOBAL_PREFETCH_RATIO,
    bandpass_fmin=0,
    bandpass_fmax=15_000,
    speed=1.0,
    callback: Callable[[AcousticProgressStats], None] | None = None,
) -> np.ndarray:
    model = _load_acoustic_for_embeddings(version)
    sr = model.get_sample_rate()

    # encode_array was removed; use encode_session + run_arrays instead.
    # run_arrays expects (ndarray, sample_rate) tuples.
    inputs = [(sig, sr) for sig in signals]

    with model.encode_session(
        batch_size=batch_size,
        prefetch_ratio=prefetch_ratio,
        bandpass_fmin=bandpass_fmin,
        bandpass_fmax=bandpass_fmax,
        speed=speed,
        progress_callback=callback,
        n_workers=n_workers,
        n_producers=n_producers,
    ) as session:
        result = session.run_arrays(inputs)

    # result.embeddings has shape (n_inputs, n_segments, embed_dim).
    # Each input signal is a single segment, so squeeze the middle dim.
    # Return shape: (n_inputs, embed_dim)
    return result.embeddings[:, 0, :]
