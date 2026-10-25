import numpy as np
import pytest

from respiratory_vision import PipelineConfig
from respiratory_vision.fusion import fuse_estimates
from respiratory_vision.quality import assess_quality
from respiratory_vision.signal_processing import (
    analyze_methods,
    clean_respiratory_signal,
    estimate_rate_welch,
)


@pytest.mark.parametrize("bpm", [9, 12, 18, 24, 30, 42])
def test_three_estimators(bpm):
    fps = 15
    t = np.arange(60 * fps) / fps
    x = np.sin(2 * np.pi * bpm / 60 * t)
    methods = analyze_methods(clean_respiratory_signal(x, fps), fps, PipelineConfig())
    for method in methods.values():
        assert method["bpm"] is not None
        assert abs(method["bpm"] - bpm) < 1.6
    assert fuse_estimates(methods)["bpm"] is not None


@pytest.mark.parametrize(
    "fps,lo,hi", [(0, 6, 48), (1, 6, 48), (10, 30, 10), (float("nan"), 6, 48)]
)
def test_invalid_sampling(fps, lo, hi):
    with pytest.raises(ValueError):
        clean_respiratory_signal(np.ones(100), fps, lo, hi)


def test_zero_signal_rejects():
    with pytest.raises(ValueError):
        estimate_rate_welch(np.zeros(100), 10)
    c = PipelineConfig()
    methods = analyze_methods(np.zeros(300), 10, c)
    q = assess_quality(np.zeros(300), methods, fuse_estimates(methods), {}, 10, c)
    assert q["state"] == "LOW_QUALITY"


def test_harmonic_ambiguity():
    t = np.arange(900) / 15
    x = np.sin(2 * np.pi * 0.2 * t) + 0.9 * np.sin(2 * np.pi * 0.4 * t)
    c = PipelineConfig()
    methods = analyze_methods(clean_respiratory_signal(x, 15), 15, c)
    assert methods["welch"]["diagnostics"]["harmonic_ambiguity"]
    assert not assess_quality(x, methods, fuse_estimates(methods), {}, 15, c)[
        "accepted"
    ]


def test_disagreement_not_averaged():
    methods = {
        k: {"bpm": v, "reliable": True, "reasons": []}
        for k, v in [("welch", 12), ("peaks", 24), ("autocorrelation", 12)]
    }
    assert fuse_estimates(methods)["bpm"] is None


def test_irregular_intervals_and_motion_gate():
    t = np.arange(600) / 15
    phase = 2 * np.pi * (0.25 * t + 0.1 * np.sin(0.6 * t))
    x = np.sin(phase)
    c = PipelineConfig()
    methods = analyze_methods(clean_respiratory_signal(x, 15), 15, c)
    q = assess_quality(
        x, methods, fuse_estimates(methods), {"artifact_fraction": 0.5}, 15, c
    )
    assert q["state"] == "MOTION_ARTIFACT"
    assert len(methods["peaks"]["diagnostics"]["intervals_seconds"]) > 2
