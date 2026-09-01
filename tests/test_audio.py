import numpy as np
import pytest
import soundfile as sf

from birdnet_analyzer import audio


def _write_sine_wav(path, sample_rate, duration=1.0, freq=440.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sf.write(path, np.sin(2 * np.pi * freq * t).astype(np.float32), sample_rate)


def test_open_audio_file_resamples_non_native_rate(tmp_path):
    path = str(tmp_path / "44k.wav")
    _write_sine_wav(path, 44100)

    sig, rate = audio.open_audio_file(path)

    assert rate == 48000
    assert len(sig) == pytest.approx(48000, abs=2)


@pytest.mark.parametrize("speed", [0.5, 2.0])
def test_open_audio_file_speed_scales_duration(tmp_path, speed):
    path = str(tmp_path / "48k.wav")
    _write_sine_wav(path, 48000)

    sig, rate = audio.open_audio_file(path, speed=speed)

    assert rate == 48000
    assert len(sig) == pytest.approx(48000 / speed, abs=2)


def test_open_audio_file_speed_with_non_native_rate(tmp_path):
    path = str(tmp_path / "44k.wav")
    _write_sine_wav(path, 44100)

    sig, rate = audio.open_audio_file(path, speed=2.0)

    assert rate == 48000
    assert len(sig) == pytest.approx(24000, abs=2)
