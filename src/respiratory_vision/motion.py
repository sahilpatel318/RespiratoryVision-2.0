from collections.abc import Iterable
from dataclasses import dataclass, field

import cv2
import numpy as np

from .errors import InputError
from .motion_compensation import estimate_background_translation
from .roi import ROI


@dataclass
class MotionTrace:
    vertical_flow: np.ndarray
    displacement: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    global_translation: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    subregion_flow: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))


def _prep(gray, gaussian_kernel):
    return (
        cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)
        if gaussian_kernel > 1
        else gray
    )


def _aggregate(flow):
    magnitude = np.linalg.norm(flow, axis=2)
    moving = magnitude > max(1e-5, float(np.percentile(magnitude, 60)))
    samples = flow[..., 1][moving]
    if not len(samples):
        return 0.0
    lo, hi = np.percentile(samples, [10, 90])
    samples = samples[(samples >= lo) & (samples <= hi)]
    return float(samples.mean()) if len(samples) else 0.0


def extract_motion_trace(
    frames: Iterable[np.ndarray],
    roi: ROI,
    gaussian_kernel=5,
    compensate=False,
    fps=10.0,
    max_motion_pixels_second=80.0,
) -> MotionTrace:
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        raise InputError("No decoded frames.", "INSUFFICIENT_DATA")
    roi.validate(first)
    prev_full = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    prev = _prep(roi.crop(prev_full), gaussian_kernel)
    vertical, global_motion, regions, successful, speeds = (
        [0.0],
        [[0.0, 0.0]],
        [[0.0] * 4],
        [],
        [],
    )
    for frame in iterator:
        if frame.shape != first.shape:
            raise InputError("Frame dimensions change within the recording.")
        full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = _prep(roi.crop(full), gaussian_kernel)
        translation, ok, _ = (
            estimate_background_translation(prev_full, full, roi)
            if compensate
            else (np.zeros(2), False, 0)
        )
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        raw_speed = float(np.percentile(np.linalg.norm(flow, axis=2), 90)) * fps
        flow = flow - translation.reshape(1, 1, 2)
        parts = [
            _aggregate(part)
            for row in np.array_split(flow, 2, axis=0)
            for part in np.array_split(row, 2, axis=1)
        ]
        # Median of active quadrants limits influence of a single contaminated region.
        active = [p for p in parts if abs(p) > 1e-4]
        vertical.append(float(np.median(active)) if active else 0.0)
        regions.append(parts)
        global_motion.append(translation.tolist())
        successful.append(ok)
        speeds.append(max(raw_speed, float(np.linalg.norm(translation)) * fps))
        prev, prev_full = gray, full
    if len(vertical) < 3:
        raise InputError("At least three decoded frames required.", "INSUFFICIENT_DATA")
    region_traces = np.cumsum(regions, axis=0)
    valid = np.std(region_traces, axis=0) > 0.05
    if valid.sum() > 1:
        corr = np.corrcoef(region_traces[:, valid].T)
        consistency = float(np.median(corr[np.triu_indices(valid.sum(), 1)]))
    else:
        consistency = 1.0
    diagnostics = {
        "compensation_enabled": compensate,
        "background_tracking_fraction": float(np.mean(successful))
        if compensate
        else None,
        "compensation_fallback": bool(compensate and not all(successful)),
        "artifact_fraction": float(
            np.mean(np.asarray(speeds) > max_motion_pixels_second)
        ),
        "motion_speed_p95_pixels_second": float(np.percentile(speeds, 95)),
        "roi_consistency": consistency,
        "camera_translation_rms_pixels": float(
            np.sqrt(np.mean(np.asarray(global_motion) ** 2))
        ),
    }
    arr = np.asarray(vertical, dtype=float)
    return MotionTrace(
        arr, np.cumsum(arr), diagnostics, np.asarray(global_motion), np.asarray(regions)
    )
