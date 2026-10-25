from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class RateEstimate:
    bpm: float
    dominant_hz: float
    confidence: float
    peak_power: float
    total_band_power: float


def clean_respiratory_signal(
    raw: np.ndarray,
    fps: float,
    min_bpm: float = 6.0,
    max_bpm: float = 48.0,
) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("Signal must be a one-dimensional sequence with >= 8 samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains non-finite values.")

    x = signal.detrend(x, type="linear")
    low = (min_bpm / 60.0)
    high = (max_bpm / 60.0)
    nyquist = fps / 2.0

    if high >= nyquist:
        high = nyquist * 0.95
    if not (0 < low < high):
        raise ValueError("Invalid respiratory passband for the supplied frame rate.")

    sos = signal.butter(4, [low, high], btype="bandpass", fs=fps, output="sos")

    # sosfiltfilt needs enough samples. For short signals, fall back to sosfilt.
    pad_need = 3 * (2 * len(sos) + 1)
    if x.size > pad_need:
        filtered = signal.sosfiltfilt(sos, x)
    else:
        filtered = signal.sosfilt(sos, x)

    std = float(np.std(filtered))
    if std > 1e-12:
        filtered = (filtered - np.mean(filtered)) / std
    else:
        filtered = filtered - np.mean(filtered)
    return filtered


def estimate_rate_welch(
    filtered: np.ndarray,
    fps: float,
    min_bpm: float = 6.0,
    max_bpm: float = 48.0,
) -> RateEstimate:
    x = np.asarray(filtered, dtype=np.float64)
    nperseg = min(len(x), max(64, int(round(fps * 12))))
    nfft = max(2048, 2 ** int(np.ceil(np.log2(max(len(x), 64)))))
    freqs, power = signal.welch(x, fs=fps, nperseg=nperseg, nfft=nfft, detrend="constant")

    lo = min_bpm / 60.0
    hi = max_bpm / 60.0
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        raise ValueError("No frequency bins fall inside the respiratory band.")

    band_freqs = freqs[mask]
    band_power = power[mask]
    idx = int(np.argmax(band_power))
    dominant_hz = float(band_freqs[idx])
    peak_power = float(band_power[idx])
    total = float(np.sum(band_power)) + 1e-12

    # Peak share is an interpretable, bounded confidence proxy.
    confidence = max(0.0, min(1.0, peak_power / total))

    return RateEstimate(
        bpm=dominant_hz * 60.0,
        dominant_hz=dominant_hz,
        confidence=confidence,
        peak_power=peak_power,
        total_band_power=total,
    )
