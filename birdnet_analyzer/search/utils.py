from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np
from perch_hoplite.db import brutalism
from perch_hoplite.db.search_results import SearchResult
from scipy.spatial.distance import euclidean

from birdnet_analyzer import audio, model_utils
from birdnet_analyzer.config import CROP_MODES, SCORE_FUNCTIONS

if TYPE_CHECKING:
    from perch_hoplite.db.sqlite_usearch_impl import SQLiteUSearchDB


def _get_usearch_metric_name(db: SQLiteUSearchDB) -> str | None:
    try:
        usearch_cfg = db.get_metadata("usearch_config")
    except KeyError:
        return None
    return str(usearch_cfg.get("metric_name", "")).upper() or None


def _search_ann_ip(
    db: SQLiteUSearchDB, query_embedding: np.ndarray, n_results: int
) -> list[SearchResult]:
    matches = db.ui.search(query_embedding, count=n_results)
    return [
        SearchResult(window_id=int(window_id), sort_score=1.0 - float(distance))
        for window_id, distance in zip(matches.keys, matches.distances, strict=False)
    ]


def cosine_sim(data: np.ndarray, query: np.ndarray) -> float | np.ndarray:
    if data.ndim == 2:
        norms = np.linalg.norm(data, axis=1) * np.linalg.norm(query)
        return data @ query / norms
    return np.dot(data, query) / (np.linalg.norm(data) * np.linalg.norm(query))


def euclidean_scoring(data: np.ndarray, query: np.ndarray) -> float | np.ndarray:
    if data.ndim == 2:
        return np.linalg.norm(data - query, axis=1)
    return euclidean(data, query)


def get_query_embedding(
    queryfile_path,
    crop_mode: CROP_MODES = "center",
    crop_overlap=0.0,
    bandpass_fmin=0,
    bandpass_fmax=15000,
    audio_speed=1.0,
    sig_length=3.0,
    sig_minlen=1.0,
):
    """
    Extracts the embedding for a query file. Reads only the first 3 seconds
    Args:
        queryfile_path: The path to the query file.
    Returns:
        The query embedding.
    """

    sig, rate = audio.open_audio_file(
        queryfile_path,
        duration=sig_length * audio_speed if crop_mode == "first" else None,
        fmin=bandpass_fmin,
        fmax=bandpass_fmax,
        speed=audio_speed,
    )

    if crop_mode == "center":
        sig_splits = [audio.crop_center(sig, rate, sig_length)]
    elif crop_mode == "first":
        sig_splits = [
            audio.split_signal(sig, rate, sig_length, crop_overlap, sig_minlen)[0]
        ]
    else:
        sig_splits = audio.split_signal(sig, rate, sig_length, crop_overlap, sig_minlen)

    return model_utils.get_embeddings_array(sig_splits, n_workers=1)


def get_search_results(
    queryfile_path: str,
    db: SQLiteUSearchDB,
    n_results=10,
    audio_speed=1.0,
    fmin=0,
    fmax=15000,
    score_function: SCORE_FUNCTIONS = "cosine",
    crop_mode: CROP_MODES = "center",
    crop_overlap=0.0,
    sig_length=3.0,
    sig_fmin=0,
    sig_fmax=15000,
):
    bandpass_fmin = max(0, min(sig_fmax, int(fmin)))
    bandpass_fmax = max(sig_fmin, min(sig_fmax, int(fmax)))
    audio_speed = max(0.01, audio_speed)
    sig_overlap = max(0.0, min(2.9, float(crop_overlap)))
    query_embeddings = get_query_embedding(
        queryfile_path,
        crop_mode=crop_mode,
        crop_overlap=sig_overlap,
        bandpass_fmin=bandpass_fmin,
        bandpass_fmax=bandpass_fmax,
        audio_speed=audio_speed,
        sig_length=sig_length,
    )

    if score_function == "cosine":
        score_fn = cosine_sim
    elif score_function == "dot":
        score_fn = np.dot
    elif score_function == "euclidean":
        score_fn = euclidean_scoring
    else:
        raise ValueError(
            f"Invalid score function. Choose {', '.join(get_args(SCORE_FUNCTIONS))}."
        )

    db_embeddings_count = db.count_embeddings()
    n_results = min(n_results, db_embeddings_count)
    if n_results <= 0:
        return []

    queries = np.asarray(query_embeddings, dtype=np.float32)
    usearch_metric_name = _get_usearch_metric_name(db)
    # ANN path is currently safe only for inner product scoring.
    use_ann = score_function == "dot" and usearch_metric_name == "IP"

    if use_ann:
        # The mean of the dot products is the dot product with the mean query.
        return _search_ann_ip(db, queries.mean(axis=0), n_results)

    # brute_search keeps the highest scores, so distances are negated for the scan.
    sign = -1.0 if score_function == "euclidean" else 1.0

    def mean_score(data: np.ndarray, query: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        return sign * np.mean([score_fn(data, q) for q in query], axis=0)

    return [
        SearchResult(window_id=int(r.window_id), sort_score=sign * float(r.sort_score))
        for r in brutalism.brute_search(db, queries, n_results, mean_score)
    ]
