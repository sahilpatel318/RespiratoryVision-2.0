import numpy as np

from respiratory_vision.signal_processing import (
    clean_respiratory_signal,
    estimate_rate_welch,
)


def test_known_sine_rate():
    fps = 15.0
    bpm = 18.0
    t = np.arange(int(fps * 30)) / fps
    x = np.sin(2 * np.pi * (bpm / 60.0) * t)
    filtered = clean_respiratory_signal(x, fps)
    estimate = estimate_rate_welch(filtered, fps)
    assert abs(estimate.bpm - bpm) <= 1.5
