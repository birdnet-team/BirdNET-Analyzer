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
        SearchResult(window_id=int(window_id), sort_score=float(score))
        for window_id, score in zip(matches.keys, matches.distances, strict=False)
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


def euclidean_scoring_inverse(
    data: np.ndarray, query: np.ndarray
) -> float | np.ndarray:
    return -euclidean_scoring(data, query)


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

    return model_utils.get_embeddings_array(sig_splits)


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
        # TODO: this is a bit hacky since the search function expects the score to be
        # high for similar embeddings
        score_fn = euclidean_scoring_inverse
    else:
        raise ValueError(
            f"Invalid score function. Choose {', '.join(get_args(SCORE_FUNCTIONS))}."
        )

    db_embeddings_count = db.count_embeddings()
    n_results = min(n_results, db_embeddings_count - 1)
    if n_results <= 0:
        return []

    usearch_metric_name = _get_usearch_metric_name(db)
    # ANN path is currently safe only for inner product scoring.
    use_ann = score_function == "dot" and usearch_metric_name == "IP"

    scores_by_embedding_id: dict[int, list[float]] = {}

    for embedding in query_embeddings:
        if use_ann:
            sorted_results = _search_ann_ip(db, embedding, n_results)
        else:
            results, scores = brutalism.threaded_brute_search(
                db,
                embedding,
                n_results,
                score_fn,  # ty:ignore[invalid-argument-type]
            )
            sorted_results = results.search_results

        if not use_ann and score_function == "euclidean":
            for result in sorted_results:
                result.sort_score *= -1

        for result in sorted_results:
            if result.window_id not in scores_by_embedding_id:
                scores_by_embedding_id[result.window_id] = []

            scores_by_embedding_id[result.window_id].append(result.sort_score)

    search_results: list[SearchResult] = []

    for window_id, scores in scores_by_embedding_id.items():
        search_results.append(
            SearchResult(
                window_id=window_id, sort_score=np.sum(scores) / len(query_embeddings)
            )
        )

    reverse = score_function != "euclidean"

    search_results.sort(key=lambda x: x.sort_score, reverse=reverse)

    return search_results[0:n_results]
