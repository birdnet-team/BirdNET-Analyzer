"""Module containing audio helper functions."""

from math import isclose

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, lfilter

import birdnet_analyzer.config as cfg

RANDOM = np.random.RandomState(cfg.RANDOM_SEED)


def open_audio_file(
    path: str,
    sample_rate=48000,
    offset=0.0,
    duration=None,
    fmin=None,
    fmax=None,
    speed=1.0,
    sig_fmin=0,
    sig_fmax=15000,
):
    """Open an audio file.

    Opens an audio file with librosa and the given settings.

    Args:
        path: Path to the audio file.
        sample_rate: The sample rate at which the file should be processed.
        offset: The starting offset.
        duration: Maximum duration of the loaded content.
        fmin: Minimum frequency for bandpass filter.
        fmax: Maximum frequency for bandpass filter.
        sig_fmin: Minimum frequency of the original signal.
        sig_fmax: Maximum frequency of the original signal.
        speed: Speed factor for audio playback.

    Returns:
        Returns the audio time series and the sampling rate.
    """
    # Open file with librosa (uses ffmpeg or libav)
    if speed == 1.0:
        sig, rate = librosa.load(
            path,
            sr=sample_rate,
            offset=offset,
            duration=duration,
            mono=True,
            res_type="kaiser_fast",
        )

    else:
        # Load audio with original sample rate
        sig, rate = librosa.load(
            path, sr=None, offset=offset, duration=duration, mono=True
        )

        # Resample with "fake" sample rate
        sig = librosa.resample(
            sig,
            orig_sr=int(rate * speed),
            target_sr=sample_rate,
            res_type="kaiser_fast",
        )
        rate = sample_rate

    # Bandpass filter
    if fmin is not None and fmax is not None:
        sig = bandpass(sig, rate, fmin, fmax, sig_fmin=sig_fmin, sig_fmax=sig_fmax)
        # sig = bandpassKaiserFIR(sig, rate, fmin, fmax)

    return sig, rate


def get_audio_info(path):
    """
    Get basic information about an audio file.

    Args:
        path (str): The file path to the audio file.

    Returns:
        dict: A dictionary containing audio file information such as sample rate and
        duration.
    """
    info = sf.info(path)

    return {
        "samplerate": info.samplerate,
        "duration": info.duration,
    }


def get_audio_file_length(path):
    """
    Get the length of an audio file in seconds.

    Args:
        path (str): The file path to the audio file.

    Returns:
        float: The duration of the audio file in seconds.
    """
    # Open file with librosa (uses ffmpeg or libav)

    return librosa.get_duration(path=path, sr=None)  # ty:ignore[invalid-argument-type]


def get_sample_rate(path: str):
    """
    Get the sample rate of an audio file.

    Args:
        path (str): The file path to the audio file.

    Returns:
        int: The sample rate of the audio file.
    """
    return librosa.get_samplerate(path)


def save_signal(sig, fname: str, rate=48000):
    """Saves a signal to file.

    Args:
        sig: The signal to be saved.
        fname: The file path.

    Returns:
        None
    """

    sf.write(fname, sig, rate, "PCM_16")


def pad(sig, seconds, srate, amount=None, use_noise=False):
    """Creates a noise vector with the given shape.

    Args:
        sig: The original audio signal.
        shape: Shape of the noise.
        amount: The noise intensity.
        use_noise: Whether to use noise or zeros for padding.

    Returns:
        An numpy array of noise with the given shape.
    """

    target_len = int(srate * seconds)

    if len(sig) < target_len:
        noise_shape = target_len - len(sig)

        if use_noise:
            if amount is None:
                amount = RANDOM.uniform(0.1, 0.5)

            # Create Gaussian noise
            try:
                noise = RANDOM.normal(
                    min(sig) * amount, max(sig) * amount, noise_shape
                ).astype(sig.dtype)
            except:
                noise = np.zeros(noise_shape, dtype=sig.dtype)
        else:
            noise = np.zeros(noise_shape, dtype=sig.dtype)

        return np.concatenate((sig, noise))

    return sig


def split_signal(
    sig,
    rate,
    seconds=3.0,
    overlap=0.0,
    minlen=1.0,
    amount=None,
    use_noise_for_padding=False,
):
    """Split signal with overlap.

    Args:
        sig: The original signal to be split.
        rate: The sampling rate.
        seconds: The duration of a segment.
        overlap: The overlapping seconds of segments.
        minlen: Minimum length of a split.
        use_noise_for_padding: Whether to use noise for padding.
    Returns:
        A list of splits.
    """

    if rate is None or rate <= 0:
        raise ValueError("Invalid sample rate")
    if seconds is None or seconds <= 0:
        raise ValueError("Invalid segment duration")
    if overlap is None or overlap < 0:
        raise ValueError("Invalid overlap duration")
    if minlen is None or minlen <= 0 or minlen > seconds:
        raise ValueError("Invalid minimum segment length")
    if overlap >= seconds:
        raise ValueError("Overlap must be smaller than segment duration")

    # Make sure overlap is smaller then signal duration
    if isclose(overlap, seconds):
        overlap = seconds - 0.01

    # Number of frames per chunk, per step and per minimum signal
    chunksize = int(rate * seconds)
    stepsize = int(rate * (seconds - overlap))
    minsize = int(rate * minlen)

    # Start of last chunk
    lastchunkpos = int((sig.size - chunksize + stepsize - 1) / stepsize) * stepsize
    # Make sure at least one chunk is returned
    if lastchunkpos < 0:
        lastchunkpos = 0
    # Omit last chunk if minimum signal duration is underrun
    elif sig.size - lastchunkpos < minsize:
        lastchunkpos = lastchunkpos - stepsize

    # Append noise or empty signal of chunk duration, so all splits have desired length
    if use_noise_for_padding:
        # Random noise intensity
        if amount is None:
            amount = RANDOM.uniform(0.1, 0.5)
        # Create Gaussian noise
        try:
            noise = RANDOM.normal(
                loc=min(sig) * amount, scale=max(sig) * amount, size=chunksize
            ).astype(sig.dtype)
        except:
            noise = np.zeros(shape=chunksize, dtype=sig.dtype)
    else:
        noise = np.zeros(shape=chunksize, dtype=sig.dtype)

    data = np.concatenate((sig, noise))

    # Split signal with overlap
    sig_splits = []
    sig_splits.extend(
        data[i : i + chunksize] for i in range(0, lastchunkpos + 1, stepsize)
    )

    return sig_splits


def crop_center(sig, rate, seconds):
    """Crop signal to center.

    Args:
        sig: The original signal.
        rate: The sampling rate.
        seconds: The length of the signal.

    Returns:
        The cropped signal.
    """
    if len(sig) > int(seconds * rate):
        start = int((len(sig) - int(seconds * rate)) / 2)
        end = start + int(seconds * rate)
        sig = sig[start:end]
    else:
        sig = pad(sig, seconds, rate, 0.5)

    return sig


def smart_crop_signal(sig, rate, sig_length, sig_overlap, sig_minlen):
    """Smart crop audio signal based on peak detection.

    This function analyzes the audio signal to find peaks in energy/amplitude,
    which are more likely to contain relevant target signals (e.g., bird calls).
    Only the audio segments with the highest energy peaks are returned.

    Args:
        sig: The audio signal.
        rate: The sample rate of the audio signal.
        sig_length: The desired length of each snippet in seconds.
        sig_overlap: The overlap between snippets in seconds.
        sig_minlen: The minimum length of a snippet in seconds.

    Returns:
        A list of audio snippets with the highest energy/peaks.
    """

    # If signal is too short, just return it
    if len(sig) / rate <= sig_length:
        return [sig]

    # Split the signal into overlapping windows
    splits = split_signal(sig, rate, sig_length, sig_overlap, sig_minlen)

    if len(splits) <= 1:
        return splits

    # Calculate energy for each window
    energies = []
    for split in splits:
        # Calculate RMS energy
        energy = np.sqrt(np.mean(split**2))
        # Also consider peak values
        peak = np.max(np.abs(split))
        # Combine both metrics
        # TODO: Hardcoded weights, could be optimized or made configurable
        energies.append(energy * 0.7 + peak * 0.3)  # Weighted combination

    # Find peaks in the energy curve
    # Smooth energies first to avoid small fluctuations
    # TODO: kernel size is hardcoded, make it configurable?
    smoothed_energies = np.convolve(energies, np.ones(3) / 3, mode="same")
    peaks, _ = find_peaks(
        smoothed_energies, height=np.mean(smoothed_energies), distance=2
    )

    # If no clear peaks found, fall back to selecting top energy segments
    if len(peaks) < 2:
        # Sort segments by energy and take top segments (up to 3 or 1/3 of total,
        # whichever is more)
        num_segments = max(3, len(splits) // 3)
        indices = np.argsort(energies)[-num_segments:]
        return [splits[i] for i in sorted(indices)]

    # Return the audio segments corresponding to the peaks
    peak_splits = [splits[i] for i in peaks]

    # If we have too many peaks, select the strongest ones
    if len(peak_splits) > 5:
        peak_energies = [energies[i] for i in peaks]
        sorted_indices = np.argsort(peak_energies)[::-1]  # Sort in descending order
        peak_splits = [peak_splits[i] for i in sorted_indices[:5]]  # Take top 5

    return peak_splits


def bandpass(sig, rate, fmin, fmax, order=5, sig_fmin=0, sig_fmax=15000):
    """
    Apply a bandpass filter to the input signal.

    Args:
        sig (numpy.ndarray): The input signal to be filtered.
        rate (int): The sampling rate of the input signal.
        fmin (float): The minimum frequency for the bandpass filter.
        fmax (float): The maximum frequency for the bandpass filter.
        order (int, optional): The order of the filter. Default is 5.
        sig_fmin (float, optional): The minimum frequency of the original signal.
            Default is 0.
        sig_fmax (float, optional): The maximum frequency of the original signal.
            Default is 15000.

    Returns:
        numpy.ndarray: The filtered signal as a float32 array.
    """
    # Check if we have to bandpass at all
    if (fmin == sig_fmin and fmax == sig_fmax) or fmin > fmax:
        return sig

    from scipy.signal import butter

    nyquist = 0.5 * rate

    # Highpass?
    if fmin > sig_fmin and fmax == sig_fmax:
        low = fmin / nyquist
        b, a = butter(order, low, btype="high")
        sig = lfilter(b, a, sig)

    # Lowpass?
    elif fmin == sig_fmin and fmax < sig_fmax:
        high = fmax / nyquist
        b, a = butter(order, high, btype="low")
        sig = lfilter(b, a, sig)

    # Bandpass?
    elif fmin > sig_fmin and fmax < sig_fmax:
        low = fmin / nyquist
        high = fmax / nyquist
        b, a = butter(order, [low, high], btype="band")
        sig = lfilter(b, a, sig)

    return sig.astype("float32")
