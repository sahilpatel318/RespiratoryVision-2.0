import json

import numpy as np
import pytest

from respiratory_vision import BaselineEstimator, PipelineConfig, RespiratoryEstimator
from respiratory_vision.roi import ROI
from respiratory_vision.synthetic import (
    CONDITIONS,
    ChallengeConfig,
    challenge_config,
    generate_breathing_video,
    generate_challenge_video,
)


@pytest.mark.parametrize(
    "condition",
    [
        "clean",
        "camera_motion",
        "illumination_changes",
        "irregular_breathing",
        "low_amplitude",
    ],
)
def test_challenge_pipeline(tmp_path, condition):
    path = tmp_path / "test.mp4"
    c = challenge_config(condition, "development", 1)
    meta = generate_challenge_video(path, c)
    r = RespiratoryEstimator().analyze(path)
    json.dumps(r, allow_nan=False)
    assert r["schema_version"] == "2.0"
    assert r["clinical_use"] is False
    assert r["methods"]
    if condition in ["clean", "camera_motion", "illumination_changes"]:
        assert abs(r["methods"]["welch"]["bpm"] - meta["ground_truth_bpm"]) < 3
    if condition == "low_amplitude":
        assert r["estimate"]["bpm"] is None
        assert not r["quality"]["accepted"]
    if condition == "camera_motion":
        assert r["quality"]["features"]["background_tracking_fraction"] > 0.8


def test_original_baseline_operates(tmp_path):
    p = tmp_path / "baseline.mp4"
    generate_breathing_video(p, duration_seconds=16)
    assert abs(BaselineEstimator().analyze(p)["estimate"]["bpm"] - 18) < 2


def test_short_invalid_constant_and_roi(tmp_path):
    p = tmp_path / "short.mp4"
    generate_breathing_video(p, duration_seconds=2)
    assert RespiratoryEstimator().analyze(p)["quality"]["state"] == "INSUFFICIENT_DATA"
    assert (
        RespiratoryEstimator().analyze(tmp_path / "missing.mp4")["quality"]["state"]
        == "UNSUPPORTED_INPUT"
    )
    (tmp_path / "bad.mp4").write_bytes(b"not video")
    assert (
        RespiratoryEstimator().analyze(tmp_path / "bad.mp4")["estimate"]["bpm"] is None
    )
    generate_challenge_video(
        p, ChallengeConfig(amplitude_pixels=0, duration_seconds=12)
    )
    assert RespiratoryEstimator().analyze(p)["quality"]["state"] == "LOW_QUALITY"
    assert (
        RespiratoryEstimator().analyze(p, roi=ROI(0, 0, 0, 10))["quality"]["state"]
        == "UNSUPPORTED_INPUT"
    )


def test_seed_separation_and_config_variation():
    dev = [challenge_config(c, "development", i) for c in CONDITIONS for i in range(4)]
    test = [challenge_config(c, "test", i) for c in CONDITIONS for i in range(4)]
    assert {c.seed for c in dev}.isdisjoint(c.seed for c in test)
    assert {c.bpm for c in dev}.isdisjoint(c.bpm for c in test)
    assert len({c.seed for c in dev}) == len(dev)


def test_challenge_determinism(tmp_path):
    from respiratory_vision.video_io import iter_frames

    c = ChallengeConfig(duration_seconds=2, noise_std=2, seed=23)
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    assert generate_challenge_video(a, c) == generate_challenge_video(b, c)
    for x, y in zip(iter_frames(a), iter_frames(b)):
        assert np.array_equal(x, y)


def test_camera_only_compensation(tmp_path):
    p = tmp_path / "camera_only.mp4"
    generate_challenge_video(
        p, ChallengeConfig(amplitude_pixels=0, camera_amplitude=6, duration_seconds=16)
    )
    on = RespiratoryEstimator().analyze(p)
    off = RespiratoryEstimator(PipelineConfig(motion_compensation=False)).analyze(p)
    assert (
        on["quality"]["features"]["displacement_std_pixels"]
        < 0.25 * off["quality"]["features"]["displacement_std_pixels"]
    )
    assert on["quality"]["features"]["background_tracking_fraction"] > 0.8


def test_original_flat_background_is_not_torso_reference(tmp_path):
    p = tmp_path / "original.mp4"
    generate_breathing_video(p, duration_seconds=16)
    r = RespiratoryEstimator().analyze(p)
    assert r["quality"]["features"]["background_tracking_fraction"] == 0
    assert r["quality"]["features"]["camera_translation_rms_pixels"] == 0
    assert r["quality"]["accepted"]
