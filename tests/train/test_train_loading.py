import contextlib
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from birdnet_analyzer import model_utils
from birdnet_analyzer.train.utils import (
    _check_cache_destination,
    _load_from_cache,
    _load_training_data,
    _read_and_crop_file,
    _save_to_cache,
)

SR = 48000
SIG_LENGTH = 3.0


def _sine_wav(path, duration, amplitude=0.3):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False, dtype="float32")
    sf.write(path, (amplitude * np.sin(2 * np.pi * 1000 * t)).astype("float32"), SR)

    return str(path)


def _label():
    return np.array([1.0, 0.0], dtype="float32")


@pytest.mark.parametrize(
    ("crop_mode", "duration", "n_segments"),
    [("center", 5.0, 1), ("first", 5.0, 1), ("segments", 5.0, 2), ("segments", 3.5, 1)],
)
def test_a_file_is_cropped_into_labelled_segments(
    tmp_path, crop_mode, duration, n_segments
):
    segs, labels = _read_and_crop_file(
        _sine_wav(tmp_path / "recording.wav", duration),
        _label(),
        sample_rate=SR,
        crop_mode=crop_mode,
        sig_length=SIG_LENGTH,
    )

    assert [s.shape for s in segs] == [(int(SIG_LENGTH * SR),)] * n_segments
    assert all(s.dtype == np.float32 for s in segs)
    np.testing.assert_array_equal(labels, [_label()] * n_segments)


def test_read_and_crop_bad_file_returns_empty():
    segs, labels = _read_and_crop_file(
        "does_not_exist.wav", _label(), sample_rate=SR, crop_mode="center"
    )
    assert segs == []
    assert labels == []


def test_load_training_data_lists_empty_folders_before_loading_model(tmp_path):
    (tmp_path / "blackbird").mkdir()
    (tmp_path / "bluebird").mkdir()
    robin_dir = tmp_path / "robin"
    robin_dir.mkdir()
    (robin_dir / "recording.wav").touch()

    with (
        patch.object(model_utils, "_load_acoustic_for_embeddings") as mock_load,
        pytest.raises(ValueError, match=r"blackbird.*bluebird") as error,
    ):
        _load_training_data(str(tmp_path))

    assert "robin" not in str(error.value)
    assert error.value.args[1] == "validation-no-audio-files-in-training-folders"
    mock_load.assert_not_called()


@pytest.mark.parametrize("overlap", [0.0, 2.0])
@pytest.mark.parametrize("crop_mode", ["first", "segments"])
def test_a_file_shorter_than_one_second_is_one_zero_padded_segment(
    tmp_path, crop_mode, overlap
):
    duration = 0.4
    segs, labels = _read_and_crop_file(
        _sine_wav(tmp_path / "call.wav", duration),
        _label(),
        sample_rate=SR,
        crop_mode=crop_mode,
        sig_length=SIG_LENGTH,
        overlap=overlap,
    )

    (segment,) = segs
    call_end = int(duration * SR)
    assert segment.shape == (int(SIG_LENGTH * SR),)
    assert np.abs(segment[:call_end]).max() > 0.2
    assert not segment[call_end:].any()
    np.testing.assert_array_equal(labels, [_label()])


@pytest.mark.parametrize("crop_mode", ["first", "segments"])
def test_an_empty_file_gives_no_segment(tmp_path, crop_mode):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype="float32"), SR)

    segs, labels = _read_and_crop_file(
        str(path), _label(), sample_rate=SR, crop_mode=crop_mode, sig_length=SIG_LENGTH
    )

    assert segs == []
    assert labels == []


@pytest.mark.parametrize("audio_speed", [0.5, 2.0])
def test_the_first_segment_is_audio_to_its_end_at_any_speed(tmp_path, audio_speed):
    (segment,), _ = _read_and_crop_file(
        _sine_wav(tmp_path / "recording.wav", 8.0),
        _label(),
        sample_rate=SR,
        crop_mode="first",
        sig_length=SIG_LENGTH,
        audio_speed=audio_speed,
    )

    assert segment.shape == (int(SIG_LENGTH * SR),)
    assert np.abs(segment[-SR // 10 :]).max() > 0.2


class _FakeResult:
    def __init__(self, embeddings, masked):
        # shapes: (n_inputs, n_segments=1, emb_dim)
        self.embeddings = embeddings
        self.embeddings_masked = masked


class _FakeSession:
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
    assert len(session.received) == 3


def test_encode_arrays_batched_drops_masked_rows():
    emb = np.arange(3 * 4, dtype="float32").reshape(3, 1, 4)
    masked = np.zeros((3, 1, 4), dtype=bool)
    masked[1, 0, :] = True  # middle input fully masked -> invalid
    session = _FakeSession(_FakeResult(emb, masked))

    signals = [(np.zeros(4, dtype="float32"), SR) for _ in range(3)]
    embeddings, valid = model_utils.encode_arrays_batched(session, signals)

    assert valid.tolist() == [True, False, True]
    kept = embeddings[valid]
    np.testing.assert_array_equal(kept[0], emb[0, 0])
    np.testing.assert_array_equal(kept[1], emb[2, 0])


def test_encode_arrays_batched_rejects_multi_segment():
    # If the session yields more than one segment per input, [:, 0, :] would silently
    # drop the rest -> the helper must raise instead.
    emb = np.zeros((2, 3, 4), dtype="float32")  # 3 segments per input
    masked = np.zeros((2, 3, 4), dtype=bool)
    session = _FakeSession(_FakeResult(emb, masked))

    signals = [(np.zeros(4, dtype="float32"), SR) for _ in range(2)]
    with pytest.raises(ValueError, match="one segment per input"):
        model_utils.encode_arrays_batched(session, signals)


def _cache_arrays(embeddings_dim):
    x = np.zeros((4, embeddings_dim), dtype="float32")
    y = np.zeros((4, 2), dtype="float32")
    return x, y


def test_cache_round_trip_keeps_the_model_version(tmp_path):
    # Recorded as 3.0 although the width says 2.4: the record is what comes back.
    x, y = _cache_arrays(1024)
    path = str(tmp_path / "cache.npz")

    _save_to_cache(
        path, x, y, np.array([]), np.array([]), ["a", "b"], model_version="3.0"
    )

    *_, version = _load_from_cache(path)
    assert version == "3.0"


@pytest.mark.parametrize(("embeddings_dim", "version"), [(1024, "2.4"), (1280, "3.0")])
def test_a_cache_without_a_version_is_identified_by_its_embeddings(
    tmp_path, embeddings_dim, version
):
    x, y = _cache_arrays(embeddings_dim)
    path = str(tmp_path / "cache.npz")
    np.savez(path, x_train=x, y_train=y, labels=np.array(["a", "b"], dtype=object))

    *_, loaded_version = _load_from_cache(path)
    assert loaded_version == version


def test_loading_a_cache_returns_its_version_not_the_requested_one(tmp_path):
    x, y = _cache_arrays(1024)
    path = str(tmp_path / "cache.npz")
    _save_to_cache(
        path, x, y, np.array([]), np.array([]), ["a", "b"], model_version="2.4"
    )

    *_, version = _load_training_data(path, model_version="3.0")
    assert version == "2.4"


@pytest.mark.parametrize(
    ("folders", "options", "key"),
    [
        (["a", "-a"], {}, "validation-no-negative-samples-in-binary-classification"),
        (["a"], {}, "validation-non-event-samples-required-in-binary-classification"),
        (
            ["a,b"],
            {"upsampling_ratio": 0.5, "upsampling_mode": "mean"},
            "validation-only-repeat-upsampling-for-multi-label",
        ),
    ],
)
def test_invalid_training_folders_raise_a_localizable_value_error(
    tmp_path, folders, options, key
):
    for folder in folders:
        (tmp_path / folder).mkdir()

    with pytest.raises(ValueError, match=key) as error:
        _load_training_data(str(tmp_path), **options)

    assert len(error.value.args) == 2
    assert error.value.args[1] == key


class _FakeAcousticModel:
    def get_sample_rate(self):
        return SR

    @contextlib.contextmanager
    def encode_session(self, **kwargs):
        yield None


@pytest.fixture
def fake_encoder(monkeypatch):
    monkeypatch.setattr(
        model_utils, "_load_acoustic_for_embeddings", lambda _: _FakeAcousticModel()
    )
    monkeypatch.setattr(
        model_utils,
        "encode_arrays_batched",
        lambda _, signals: (
            np.array([[np.abs(s).max()] * 4 for s, _ in signals], dtype="float32"),
            np.ones(len(signals), dtype=bool),
        ),
    )


def _write_wav(path):
    path.parent.mkdir(parents=True)
    sf.write(path, np.zeros(SR, dtype="float32"), SR)


@pytest.mark.usefixtures("fake_encoder")
def test_a_folder_of_spaced_labels_sets_each_label(tmp_path):
    _write_wav(tmp_path / "a, b" / "x.wav")

    _, y_train, _, _, _, labels, *_ = _load_training_data(str(tmp_path))

    assert labels == ["a", "b"]
    np.testing.assert_array_equal(y_train, [[1, 1]])


@pytest.mark.usefixtures("fake_encoder")
def test_no_file_trains_and_validates_from_its_audio_or_its_cache(
    tmp_path, monkeypatch
):
    import keras

    from birdnet_analyzer.train.utils import train_model

    for k in range(10):
        folder = tmp_path / "data" / "ab"[k % 2]
        folder.mkdir(parents=True, exist_ok=True)
        _sine_wav(folder / f"{k}.wav", 9.0, amplitude=0.1 + 0.05 * k)

    fits = []
    fit = keras.Model.fit

    def recording_fit(self, x, y, validation_data, **kwargs):
        fits.append((x[:, 0], validation_data[0][:, 0]))
        return fit(self, x, y, validation_data=validation_data, **kwargs)

    monkeypatch.setattr(keras.Model, "fit", recording_fit)
    cache = str(tmp_path / "cache.npz")
    options = {
        "crop_mode": "segments",
        "overlap": 1.5,
        "epochs": 1,
        "model_formats": [],
    }

    train_model(
        str(tmp_path / "data"), str(tmp_path / "cc"), save_cache_to=cache, **options
    )
    train_model(cache, str(tmp_path / "cc"), **options)

    assert len(fits) == 2

    for trained, validated in fits:
        assert len(validated)
        assert not set(np.round(trained, 2)) & set(np.round(validated, 2))


@pytest.mark.usefixtures("fake_encoder")
def test_a_failed_cache_save_aborts_loading(tmp_path, monkeypatch):
    _write_wav(tmp_path / "data" / "a" / "x.wav")
    _write_wav(tmp_path / "data" / "b" / "x.wav")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", fail)

    with pytest.raises(OSError, match="disk full"):
        _load_training_data(
            str(tmp_path / "data"), save_cache_to=str(tmp_path / "cache.npz")
        )


def test_an_unwritable_cache_destination_fails_before_the_model_loads(tmp_path):
    for folder in ("a", "b"):
        (tmp_path / "data" / folder).mkdir(parents=True)
        (tmp_path / "data" / folder / "x.wav").touch()

    blocker = tmp_path / "blocker"
    blocker.touch()

    with (
        patch.object(model_utils, "_load_acoustic_for_embeddings") as mock_load,
        pytest.raises(OSError, match="blocker"),
    ):
        _load_training_data(
            str(tmp_path / "data"), save_cache_to=str(blocker / "cache.npz")
        )

    mock_load.assert_not_called()


def test_checking_the_cache_destination_leaves_no_trace(tmp_path):
    existing = tmp_path / "cache.npz"
    existing.write_bytes(b"old cache")

    _check_cache_destination(str(existing))
    _check_cache_destination(str(tmp_path / "new" / "cache.npz"))

    assert [p.name for p in tmp_path.iterdir()] == ["cache.npz"]
    assert existing.read_bytes() == b"old cache"


def test_a_directory_where_the_cache_would_go_is_reported_as_one(tmp_path):
    (tmp_path / "cache.npz").mkdir()

    with pytest.raises(IsADirectoryError, match=r"cache\.npz"):
        _check_cache_destination(str(tmp_path / "cache"))
