"""Tests for the training data-loading path (decode/crop + batched encoding).

These cover the correctness-critical pieces introduced when the loader was changed to
parallel decoding + a single batched encode call:

* ``_read_and_crop_file`` -- turns a file into fixed-length segments per crop mode and
  fails soft on unreadable files.
* ``encode_arrays_batched`` -- runs a batch through an encoding session and drops the
  rows the pipeline marked invalid (so labels stay aligned with valid embeddings).

They intentionally avoid loading the BirdNET model: crop tests use a synthetic wav and
the batched-encode test uses a fake session, so the suite stays fast.
"""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from birdnet_analyzer import model_utils
from birdnet_analyzer.train.utils import _read_and_crop_file

SR = 48000
SIG_LENGTH = 3.0


@pytest.fixture
def sine_wav():
    """A 5 s mono sine wav at 48 kHz -> long enough for several 3 s segments."""
    duration = 5.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False, dtype="float32")
    sig = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype("float32")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, sig, SR)
    yield path
    os.remove(path)


def _label():
    return np.array([1.0, 0.0], dtype="float32")


def test_read_and_crop_center_returns_single_segment(sine_wav):
    segs, labels = _read_and_crop_file(
        sine_wav, _label(), sample_rate=SR, crop_mode="center", sig_length=SIG_LENGTH
    )
    assert len(segs) == 1
    assert len(labels) == 1
    assert segs[0].dtype == np.float32
    assert segs[0].shape[0] == int(SIG_LENGTH * SR)
    # each segment gets its own copy of the label vector
    np.testing.assert_array_equal(labels[0], _label())


def test_read_and_crop_first_returns_single_segment(sine_wav):
    segs, labels = _read_and_crop_file(
        sine_wav, _label(), sample_rate=SR, crop_mode="first", sig_length=SIG_LENGTH
    )
    assert len(segs) == 1 == len(labels)
    assert segs[0].shape[0] == int(SIG_LENGTH * SR)


def test_read_and_crop_segments_returns_multiple_aligned(sine_wav):
    segs, labels = _read_and_crop_file(
        sine_wav, _label(), sample_rate=SR, crop_mode="segments", sig_length=SIG_LENGTH
    )
    # 5 s / 3 s -> at least two segments; labels stay 1:1 with segments
    assert len(segs) >= 2
    assert len(segs) == len(labels)
    assert all(s.shape[0] == int(SIG_LENGTH * SR) for s in segs)


def test_read_and_crop_bad_file_returns_empty():
    segs, labels = _read_and_crop_file(
        "does_not_exist.wav", _label(), sample_rate=SR, crop_mode="center"
    )
    assert segs == []
    assert labels == []


class _FakeResult:
    def __init__(self, embeddings, masked):
        # shapes: (n_inputs, n_segments=1, emb_dim)
        self.embeddings = embeddings
        self.embeddings_masked = masked


class _FakeSession:
    """Records the inputs it receives and returns a preconfigured result."""

    def __init__(self, result):
        self._result = result
        self.received = None

    def run_arrays(self, signals):
        self.received = list(signals)
        return self._result


def test_encode_arrays_batched_all_valid():
    emb = np.arange(3 * 4, dtype="float32").reshape(3, 1, 4)
    masked = np.zeros((3, 1, 4), dtype=bool)  # nothing masked -> all valid
    session = _FakeSession(_FakeResult(emb, masked))

    signals = [(np.zeros(4, dtype="float32"), SR) for _ in range(3)]
    embeddings, valid = model_utils.encode_arrays_batched(session, signals)

    assert embeddings.shape == (3, 4)
    assert valid.dtype == bool
    assert valid.tolist() == [True, True, True]
    # the whole batch is handed to the session in a single call
    assert len(session.received) == 3


def test_encode_arrays_batched_drops_masked_rows():
    emb = np.arange(3 * 4, dtype="float32").reshape(3, 1, 4)
    masked = np.zeros((3, 1, 4), dtype=bool)
    masked[1, 0, :] = True  # middle input fully masked -> invalid
    session = _FakeSession(_FakeResult(emb, masked))

    signals = [(np.zeros(4, dtype="float32"), SR) for _ in range(3)]
    embeddings, valid = model_utils.encode_arrays_batched(session, signals)

    assert valid.tolist() == [True, False, True]
    # caller filters with the mask; remaining rows are inputs 0 and 2, order preserved
    kept = embeddings[valid]
    np.testing.assert_array_equal(kept[0], emb[0, 0])
    np.testing.assert_array_equal(kept[1], emb[2, 0])
