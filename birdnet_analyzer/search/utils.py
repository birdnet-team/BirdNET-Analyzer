import numpy as np
from perch_hoplite.db import brutalism
from perch_hoplite.db.search_results import SearchResult
from scipy.spatial.distance import euclidean

import birdnet_analyzer.config as cfg
from birdnet_analyzer import audio, model


def cosine_sim(a, b):
    if a.ndim == 2:
        return np.array([cosine_sim(a[i], b) for i in range(a.shape[0])])
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_scoring(a, b):
    if a.ndim == 2:
        return np.array([euclidean_scoring(a[i], b) for i in range(a.shape[0])])
    return euclidean(a, b)


def euclidean_scoring_inverse(a, b):
    return -euclidean_scoring(a, b)


def get_query_embedding(queryfile_path):
    """
    Extracts the embedding for a query file. Reads only the first 3 seconds
    Args:
        queryfile_path: The path to the query file.
    Returns:
        The query embedding.
    """

    # Load audio
    sig, rate = audio.open_audio_file(
        queryfile_path,
        duration=cfg.SIG_LENGTH * cfg.AUDIO_SPEED if cfg.SAMPLE_CROP_MODE == "first" else None,
        fmin=cfg.BANDPASS_FMIN,
        fmax=cfg.BANDPASS_FMAX,
        speed=cfg.AUDIO_SPEED,
    )

    # Crop query audio
    if cfg.SAMPLE_CROP_MODE == "center":
        sig_splits = [audio.crop_center(sig, rate, cfg.SIG_LENGTH)]
    elif cfg.SAMPLE_CROP_MODE == "first":
        sig_splits = [audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)[0]]
    else:
        sig_splits = audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

    return model.embeddings(sig_splits)


def get_search_results(
    queryfile_path, db, n_results, audio_speed, fmin, fmax, score_function: str, crop_mode, crop_overlap
):
    # Set bandpass frequency range
    cfg.BANDPASS_FMIN = max(0, min(cfg.SIG_FMAX, int(fmin)))
    cfg.BANDPASS_FMAX = max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(fmax)))
    cfg.AUDIO_SPEED = max(0.01, audio_speed)
    cfg.SAMPLE_CROP_MODE = crop_mode
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(crop_overlap)))

    # Get query embedding
    query_embeddings = get_query_embedding(queryfile_path)

    # Set score function
    if score_function == "cosine":
        score_fn = cosine_sim
    elif score_function == "dot":
        score_fn = np.dot
    elif score_function == "euclidean":
        score_fn = euclidean_scoring_inverse  # TODO: this is a bit hacky since the search function expects the score to be high for similar embeddings
    else:
        raise ValueError("Invalid score function. Choose 'cosine', 'euclidean' or 'dot'.")

    db_embeddings_count = db.count_embeddings()
    n_results = min(n_results, db_embeddings_count - 1)
    scores_by_embedding_id = {}

    for embedding in query_embeddings:
        results, scores = brutalism.threaded_brute_search(db, embedding, n_results, score_fn)
        sorted_results = results.search_results

        if score_function == "euclidean":
            for result in sorted_results:
                result.sort_score *= -1

        for result in sorted_results:
            if result.embedding_id not in scores_by_embedding_id:
                scores_by_embedding_id[result.embedding_id] = []
            scores_by_embedding_id[result.embedding_id].append(result.sort_score)

    results = []

    for embedding_id, scores in scores_by_embedding_id.items():
        results.append(SearchResult(embedding_id, np.sum(scores) / len(query_embeddings)))

    reverse = score_function != "euclidean"

    results.sort(key=lambda x: x.sort_score, reverse=reverse)

    return results[0:n_results]
