from __future__ import annotations

import threading
from contextlib import suppress
from typing import TYPE_CHECKING

import birdnet

if TYPE_CHECKING:
    from collections.abc import Callable

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
    from birdnet.globals import ACOUSTIC_MODEL_VERSIONS, MODEL_LANGUAGES

GLOBAL_PREFETCH_RATIO = 2

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


def run_inference(
    path,
    model="birdnet",
    version: ACOUSTIC_MODEL_VERSIONS = "2.4",
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
    callback: Callable[[AcousticProgressStats], None] | None = None,
) -> AcousticFilePredictionResult:
    if classifier:
        if not cc_species_list:
            cc_species_list = classifier.replace(".tflite", "_Labels.txt", 1)

        acoustic_model = birdnet.load_custom(
            "acoustic", version, "tf", classifier, cc_species_list
        )
    elif model == "birdnet":
        acoustic_model = birdnet.load("acoustic", version, "tf", lang=label_language)
    elif model == "perch":
        acoustic_model = birdnet.load_perch_v2("CPU")
    else:
        raise ValueError(
            f"Unsupported model: {model}\nSupported models are: 'birdnet', 'perch' or "
            "use a custom classifier."
        )

    from birdnet.acoustic.inference.configs import InferenceConfig

    input_files = InferenceConfig.validate_input_files(path)

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
    ) as session:
        _register_session(session)
        try:
            return session.run(input_files)  # ty:ignore[invalid-return-type]
        finally:
            _unregister_session(session)


def run_geomodel(
    lat, lon, week=None, language: MODEL_LANGUAGES = "en_us", threshold: float = 0.03
) -> birdnet.GeoPredictionResult:
    model = birdnet.load("geo", "2.4", "tf", lang=language)
    return model.predict(lat, lon, week=week, min_confidence=threshold)


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
    model = birdnet.load("acoustic", version, "tf")
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
    model = birdnet.load("acoustic", version, "tf")
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
