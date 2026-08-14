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


def match_species_to_model(
    requested_species: Collection[str], model_species: Collection[str]
) -> tuple[set[str], list[str]]:
    """Reconcile requested species onto a model's own labels.

    A requested ``"Scientific name_Common name"`` label often does not match a model
    label exactly. Two things drift independently between model versions (and across
    label languages): common names get revised while the scientific name stays put
    (e.g. ``Columba livia`` "Rock Pigeon" -> "Rock Dove"), and taxa get reclassified so
    the scientific name changes while the common name is stable (e.g. ``Accipiter``
    -> ``Astur cooperii`` "Cooper's Hawk"). Each requested entry is therefore matched,
    in order, by: the exact label, the scientific name (language-independent), then the
    common name (only when it maps to a single model label). Matched labels are taken
    verbatim from ``model_species``, so they are always valid for the model.

    Args:
        requested_species: Species to keep - a user list or a geo-model prediction.
        model_species: The model's own species labels.

    Returns:
        ``(matched, unmatched)``: the set of model labels to use, and the requested
        entries that matched nothing (order preserved, for reporting to the user).
    """
    labels = set(model_species)
    by_scientific: dict[str, str] = {}
    by_common: dict[str, str] = {}
    ambiguous_common: set[str] = set()
    for label in model_species:
        # First label wins should a name ever map to more than one label.
        scientific, _, common = label.partition("_")
        by_scientific.setdefault(scientific, label)
        if common:
            key = common.casefold()
            if key in by_common and by_common[key] != label:
                # A common name shared by two species can't disambiguate; drop it.
                ambiguous_common.add(key)
            else:
                by_common.setdefault(key, label)

    matched: set[str] = set()
    unmatched: list[str] = []
    for requested in requested_species:
        if requested in labels:
            matched.add(requested)
            continue
        scientific, _, common = requested.partition("_")
        if scientific in by_scientific:
            matched.add(by_scientific[scientific])
            continue
        # With no underscore the whole entry may be a bare common name.
        key = (common or scientific).casefold()
        if key in by_common and key not in ambiguous_common:
            matched.add(by_common[key])
            continue
        unmatched.append(requested)

    return matched, unmatched


def _reconcile_species_list(
    species_list, model_species: Collection[str], *, strict: bool
) -> set[str]:
    """Reconcile a custom species list against a model's labels.

    ``species_list`` is either a path to a user ``--slist`` file/folder or an
    in-memory collection (e.g. a geo-model prediction). It is matched onto the model's
    own labels via :func:`match_species_to_model`.

    For a user file, species the model does not know are reported - skipped with a
    warning, or raised when ``strict`` - because the user expects every entry to
    count. A geo-derived collection is broader than the acoustic model on purpose (it
    can include non-birds), so its unmatched species are filtered out silently.
    """
    from pathlib import Path

    from birdnet_analyzer.utils import read_lines

    is_user_file = isinstance(species_list, (str, Path))
    if is_user_file:
        path = Path(species_list)
        if path.is_dir():
            # A folder is expected to contain a "species_list.txt" (see the CLI help).
            path = path / "species_list.txt"
        requested: Collection[str] = [s for s in read_lines(path, trim=True) if s]
    else:
        requested = species_list

    matched, unmatched = match_species_to_model(requested, model_species)

    if is_user_file and unmatched:
        listing = "\n  ".join(sorted(unmatched))
        if not matched:
            # An empty custom list means "no filter" to the library - it would then
            # analyze EVERY species, the opposite of what the user asked for. If
            # nothing matched, that is always an error, even without --strict.
            raise ValueError(
                f"None of the {len(requested)} species in the list are available in "
                f"the model:\n  {listing}"
            )
        if strict:
            raise ValueError(
                f"{len(unmatched)} of {len(requested)} species in the list are not "
                f"available in the model:\n  {listing}"
            )
        logger.warning(
            "%d of %d species in the list are not available in the model and were "
            "skipped:\n  %s",
            len(unmatched),
            len(requested),
            listing,
        )

    return matched


def acoustic_species_list(version: str, language: str = "en_us") -> list[str]:
    """The species labels of a BirdNET acoustic model version, in ``language``.

    Loads the model to read its labels: on a fresh install this downloads the model
    files (hundreds of MB) on first use, exactly as an analysis would - it is only
    cheap once the model is cached. Callers that must not trigger a download (e.g. a
    GUI preview) should gate on :func:`acoustic_model_downloaded` first. ``language`` is
    coerced to one the version supports; the backend matches ``run_inference``.
    """
    language = _language_for_version(cast("MODEL_LANGUAGES", language), version)
    if version == "3.0":
        model = birdnet.load(
            "acoustic", "3.0", "onnx", lang=cast("MODEL_LANGUAGES_V3_0", language)
        )
    else:
        model = birdnet.load(
            "acoustic", "2.4", "tf", lang=cast("MODEL_LANGUAGES_V2_4", language)
        )

    return list(model.species_list)


def acoustic_model_downloaded(version: str) -> bool:
    """Whether the acoustic model for ``version`` is already fully on disk.

    Lets a caller avoid triggering a large model download for a cheap best-effort
    check (see :func:`acoustic_species_list`). Uses the birdnet downloader's own file
    check; any failure (e.g. a library API change) reports False so the caller skips.
    """
    try:
        from birdnet.globals import MODEL_PRECISION_FP32

        if version == "3.0":
            from birdnet.acoustic.models.v3_0.onnx import (
                AcousticOnnxDownloaderV3_0 as downloader,
            )
        else:
            from birdnet.acoustic.models.v2_4.tf import (
                AcousticTFDownloaderV2_4 as downloader,
            )

        return bool(downloader._check_acoustic_model_available(MODEL_PRECISION_FP32))
    except Exception:
        return False


# Live sessions, so they can be cancelled from another thread. Lock-guarded: sessions
# register from Gradio worker threads while cancel_active_analyses() runs on the main.
_ACTIVE_SESSIONS: set[AcousticSessionBase] = set()
_ACTIVE_SESSIONS_LOCK = threading.Lock()
# Latched once shutdown begins: a session registering after this cancels itself, so
# none can slip past cancel_active_analyses() and keep running headless.
_SHUTDOWN = threading.Event()


def _register_session(session) -> None:
    """Track a live inference session so it can be cancelled on shutdown."""
    with _ACTIVE_SESSIONS_LOCK:
        _ACTIVE_SESSIONS.add(session)
        # Read the latch under cancel_active_analyses()'s lock, so a session racing
        # shutdown is caught by the cancel loop or cancels itself here, never missed.
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
    strict_species_list: bool = False,
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
        # Coerce the locale to one this version supports (the library rejects an
        # unsupported pair); the cast into the version's set is then sound.
        lang = _language_for_version(label_language, version)
        if version == "3.0":
            # 3.0 uses the ONNX backend: equivalent predictions (~1e-6 diff), faster
            # CPU inference, and no TensorFlow import in the workers.
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

    # A custom species list (user --slist or geo prediction) may name labels the model
    # lacks verbatim (names drift across versions/languages), which the library rejects;
    # reconcile it against the model's own labels first (see _reconcile_species_list).
    if custom_species_list is not None:
        custom_species_list = _reconcile_species_list(
            custom_species_list,
            acoustic_model.species_list,
            strict=strict_species_list,
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
        on_file_complete=on_file_complete,
    ) as session:
        _register_session(session)
        try:
            return session.run(input_files)  # ty:ignore[invalid-return-type]
        finally:
            _unregister_session(session)


def run_geomodel(
    lat, lon, week=None, language: MODEL_LANGUAGES = "en_us", threshold: float = 0.03
) -> birdnet.GeoPredictionResult:
    from birdnet_analyzer.config import DEFAULT_GEO_MODEL_VERSION

    # The newest geo model always replaces the older ones; never a choice. ``language``
    # only affects localized names, so scientific-name matchers can leave it default.
    #
    # ONNX backend, not tf/pb: imports no TensorFlow (kept out of this main-process
    # call), loads faster, same species. (v3.0 geo tf needs TF 2.18/2.19; we need
    # >=2.20.) The assert pins this to v3.0 so a future version fails loudly here
    # instead of silently mismatching, and lets the type checker prove the casts sound.
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

    # embeddings is (n_inputs, n_segments, embed_dim); each input is one segment, so
    # squeeze the middle dim to get (n_inputs, embed_dim).
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

    # Shapes are (n_inputs, n_segments, embed_dim). This helper assumes one segment per
    # input; guard so a longer signal isn't silently truncated by the [:, 0, :] slice.
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

    # embeddings is (n_inputs, n_segments, embed_dim); each input is one segment, so
    # squeeze the middle dim to get (n_inputs, embed_dim).
    return result.embeddings[:, 0, :]
