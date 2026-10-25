from pathlib import Path

from respiratory_vision import RespiratoryEstimator
from respiratory_vision.synthetic import generate_breathing_video


def test_end_to_end_synthetic(tmp_path: Path):
    video = tmp_path / "breathing.mp4"
    generate_breathing_video(video, bpm=18.0, duration_seconds=16.0, fps=15, seed=12)
    result = RespiratoryEstimator().analyze(video)
    assert abs(result["estimate"]["bpm"] - 18.0) <= 3.0
    assert result["clinical_use"] is False
