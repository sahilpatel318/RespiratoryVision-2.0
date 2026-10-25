import cv2
import numpy as np
import pytest

from respiratory_vision.motion import extract_motion_trace
from respiratory_vision.motion_compensation import estimate_background_translation
from respiratory_vision.roi import ROI, relative_roi
from respiratory_vision.video_io import timestamp_diagnostics


@pytest.mark.parametrize("shift", [(2.0, 3.0), (-3.0, 1.0), (0.5, -0.75)])
def test_camera_translation(shift):
    image = np.random.default_rng(7).integers(0, 255, (150, 200), dtype=np.uint8)
    moved = cv2.warpAffine(
        image,
        np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]),
        (200, 150),
        borderMode=cv2.BORDER_REFLECT,
    )
    estimate, ok, count = estimate_background_translation(
        image, moved, ROI(40, 45, 120, 75)
    )
    assert ok and count >= 6
    assert np.allclose(estimate, shift, atol=0.25)


def test_background_fallback_and_stream():
    frame = np.zeros((80, 100, 3), np.uint8)
    trace = extract_motion_trace(
        (frame for _ in range(10)), ROI(20, 20, 60, 40), compensate=True
    )
    assert trace.diagnostics["compensation_fallback"]
    assert np.allclose(trace.displacement, 0)


@pytest.mark.parametrize(
    "roi", [ROI(-1, 0, 10, 10), ROI(0, 0, 0, 10), ROI(90, 0, 20, 10)]
)
def test_invalid_roi(roi):
    with pytest.raises(ValueError):
        roi.crop(np.zeros((80, 100, 3), np.uint8))


def test_bad_relative_roi():
    with pytest.raises(ValueError):
        relative_roi(np.zeros((80, 100, 3)), 0, 0, 0, 1)


def test_timestamp_validation():
    assert timestamp_diagnostics([0, 0.1, 0.2], 10)["timing_valid"]
    assert not timestamp_diagnostics([0, 0.1, 0.05], 10)["timing_valid"]
    assert not timestamp_diagnostics([0, 0.1, 0.4], 10)["timing_valid"]
    assert timestamp_diagnostics([0, 0, 0], 10)["timestamp_source"] == "nominal_fps"
