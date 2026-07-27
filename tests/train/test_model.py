import numpy as np
import pytest

from birdnet_analyzer.model import (
    get_empty_class_exception,
    random_multilabel_split,
    random_split,
    upsampling,
)


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


@pytest.mark.parametrize("mode", ["repeat", "mean", "linear"])
def test_upsampling_balances_each_class_to_requested_ratio(mode):
    x = np.arange(12, dtype="float32").reshape(6, 2)
    y = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1]])

    x_upsampled, y_upsampled = upsampling(
        x, y, np.random.default_rng(42), is_binary=False, ratio=1.0, mode=mode
    )

    assert len(x_upsampled) == 10
    assert np.array_equal(y_upsampled.sum(axis=0), [5, 5])


@pytest.mark.parametrize("mode", ["repeat", "mean", "linear"])
def test_upsampling_rejects_empty_classes(mode):
    x = np.arange(8, dtype="float32").reshape(4, 2)
    y = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])

    with pytest.raises(get_empty_class_exception()) as error:
        upsampling(
            x, y, np.random.default_rng(42), is_binary=False, ratio=1.0, mode=mode
        )

    assert error.value.index == 1


def test_smote_only_interpolates_between_samples_of_the_same_class():
    x = np.array([[0.0], [1.0], [100.0], [101.0], [102.0]])
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])

    x_upsampled, y_upsampled = upsampling(
        x, y, np.random.default_rng(42), is_binary=False, ratio=1.0, mode="smote"
    )

    assert len(x_upsampled) == 6
    assert np.array_equal(y_upsampled.sum(axis=0), [3, 3])
    assert 0 <= x_upsampled[-1, 0] <= 1
