import numpy as np

from birdnet_analyzer.model import random_multilabel_split, random_split


def test_random_split_adds_negative_samples_only_once():
    x = np.arange(10).reshape(-1, 1)
    y = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
        ]
    )

    x_train, _, x_val, _ = random_split(
        x, y, np.random.default_rng(42), val_ratio=0.5
    )

    assert len(x_train) + len(x_val) == len(x)
    assert len(np.unique(x_train)) == len(x_train)
    assert not np.intersect1d(x_train, x_val).size


def test_random_multilabel_split_keeps_negative_combinations_in_training():
    x = np.arange(10).reshape(-1, 1)
    y = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    x_train, _, x_val, y_val = random_multilabel_split(
        x, y, np.random.default_rng(42), val_ratio=0.5
    )

    assert len(x_train) + len(x_val) == len(x)
    assert not np.intersect1d(x_train, x_val).size
    assert {6, 7}.issubset(set(x_train[:, 0]))
    assert not np.any(y_val == -1)
