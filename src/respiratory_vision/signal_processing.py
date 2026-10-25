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
    validate_sampling(fps, min_bpm, max_bpm)
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("Signal must be a one-dimensional sequence with >= 8 samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains non-finite values.")

    x = signal.detrend(x, type="linear")
    low = min_bpm / 60.0
    high = max_bpm / 60.0
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
    validate_sampling(fps, min_bpm, max_bpm)
    x = np.asarray(filtered, dtype=np.float64)
    if x.ndim != 1 or x.size < 8 or not np.all(np.isfinite(x)):
        raise ValueError("Expected a finite one-dimensional signal with >= 8 samples.")
    if np.std(x) < 1e-10:
        raise ValueError("No measurable periodic signal.")
    nperseg = min(len(x), max(64, int(round(fps * 12))))
    nfft = max(2048, 2 ** int(np.ceil(np.log2(max(len(x), 64)))))
    freqs, power = signal.welch(
        x, fs=fps, nperseg=nperseg, nfft=nfft, detrend="constant"
    )

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

    # Legacy field: single-bin spectral share, NOT probability or calibrated confidence.
    confidence = max(0.0, min(1.0, peak_power / total))

    return RateEstimate(
        bpm=dominant_hz * 60.0,
        dominant_hz=dominant_hz,
        confidence=confidence,
        peak_power=peak_power,
        total_band_power=total,
    )


def validate_sampling(fps: float, min_bpm: float, max_bpm: float) -> None:
    if (
        not np.all(np.isfinite([fps, min_bpm, max_bpm]))
        or not 0 < min_bpm < max_bpm < fps * 30
    ):
        raise ValueError(
            "Respiratory band must be positive and strictly below Nyquist."
        )


def _method(bpm=None, reasons=None, **diagnostics):
    reasons = reasons or []
    return {
        "bpm": None if bpm is None else float(bpm),
        "reliable": bpm is not None and not reasons,
        "reasons": reasons,
        "diagnostics": diagnostics,
    }


def estimate_rate_autocorrelation(
    x, fps, min_bpm=6.0, max_bpm=48.0, min_correlation=0.35
):
    validate_sampling(fps, min_bpm, max_bpm)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 8 or not np.all(np.isfinite(x)) or np.std(x) < 1e-10:
        return _method(reasons=["Missing or constant signal."])
    x = x - np.mean(x)
    corr = signal.correlate(x, x, mode="full", method="fft")[len(x) - 1 :]
    corr = corr / corr[0]
    lo, hi = (
        max(1, int(np.ceil(fps * 60 / max_bpm))),
        min(len(x) // 2, int(np.floor(fps * 60 / min_bpm))),
    )
    indices, _ = signal.find_peaks(corr)
    candidates = indices[(indices >= lo) & (indices <= hi)]
    diag = {
        "lags_seconds": (np.arange(hi + 1) / fps).tolist(),
        "correlation": corr[: hi + 1].tolist(),
    }
    if not len(candidates):
        return _method(
            reasons=["No periodic autocorrelation peak in valid lag range."], **diag
        )
    best = int(candidates[np.argmax(corr[candidates])])
    near = candidates[corr[candidates] >= max(min_correlation, 0.90 * corr[best])]
    if len(near):
        best = int(near[0])
    reasons = (
        [] if corr[best] >= min_correlation else ["Autocorrelation peak is too weak."]
    )
    return _method(
        fps * 60 / best,
        reasons,
        selected_lag=best,
        peak_correlation=float(corr[best]),
        **diag,
    )


def estimate_rate_peaks(x, fps, min_bpm=6.0, max_bpm=48.0, max_interval_cv=0.35):
    validate_sampling(fps, min_bpm, max_bpm)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 8 or not np.all(np.isfinite(x)) or np.std(x) < 1e-10:
        return _method(reasons=["Missing or constant signal."])
    peaks, props = signal.find_peaks(
        x, distance=max(1, int(np.ceil(fps * 60 / max_bpm))), prominence=0.5 * np.std(x)
    )
    intervals = np.diff(peaks) / fps
    valid = (intervals >= 60 / max_bpm) & (intervals <= 60 / min_bpm)
    kept = intervals[valid]
    diag = {
        "peak_indices": peaks.tolist(),
        "intervals_seconds": intervals.tolist(),
        "accepted_intervals": valid.tolist(),
        "prominences": props["prominences"].tolist(),
    }
    if len(kept) < 2:
        return _method(
            reasons=["At least two valid inter-peak intervals are required."], **diag
        )
    cv = float(np.std(kept) / np.mean(kept))
    reasons = []
    if cv > max_interval_cv:
        reasons.append("Respiratory intervals vary beyond the configured threshold.")
    if not np.all(valid):
        reasons.append(
            "One or more candidate cycle intervals are outside the respiratory band."
        )
    return _method(60 / np.median(kept), reasons, interval_cv=cv, **diag)


def analyze_methods(filtered, fps, config):
    x = np.asarray(filtered, dtype=float)
    args = (fps, config.min_bpm, config.max_bpm)
    if np.std(x) < 1e-10:
        return {
            name: _method(reasons=["Missing or constant signal."])
            for name in ["welch", "autocorrelation", "peaks"]
        }
    estimate = estimate_rate_welch(x, *args)
    nperseg = min(len(x), max(64, int(round(fps * 12))))
    nfft = max(2048, 2 ** int(np.ceil(np.log2(max(len(x), 64)))))
    f, p = signal.welch(x, fs=fps, nperseg=nperseg, nfft=nfft)
    mask = (f >= config.min_bpm / 60) & (f <= config.max_bpm / 60)
    concentration = float(
        p[mask & (abs(f - estimate.dominant_hz) <= 1 / 12)].sum()
        / max(p[mask].sum(), 1e-12)
    )
    halves = (
        [estimate_rate_welch(part, *args).bpm for part in np.array_split(x, 2)]
        if len(x) / fps >= 16 and all(np.std(a) > 1e-10 for a in np.array_split(x, 2))
        else []
    )
    reasons = []
    if concentration < config.min_spectral_concentration:
        reasons.append("Spectral energy is too dispersed.")
    inds, _ = signal.find_peaks(p)
    inds = inds[mask[inds]]
    harmonic = any(
        0.35 * p[mask].max() < p[i]
        and (
            abs(f[i] * 2 - estimate.dominant_hz) < 0.025
            or abs(f[i] / 2 - estimate.dominant_hz) < 0.025
        )
        for i in inds
    )
    if harmonic:
        reasons.append("Competing spectral peaks suggest harmonic ambiguity.")
    welch = _method(
        estimate.bpm,
        reasons,
        frequencies_hz=f.tolist(),
        psd=p.tolist(),
        dominant_hz=estimate.dominant_hz,
        concentration=concentration,
        spectral_peak_share=estimate.confidence,
        harmonic_ambiguity=bool(harmonic),
        half_rates_bpm=halves,
        rate_change_bpm=abs(halves[1] - halves[0]) if halves else 0.0,
    )
    return {
        "welch": welch,
        "autocorrelation": estimate_rate_autocorrelation(
            x, *args, config.min_autocorrelation
        ),
        "peaks": estimate_rate_peaks(x, *args, config.max_interval_cv),
    }
