from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .errors import InputError


@dataclass
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def probe_video(path: str | Path) -> VideoMetadata:
    if not Path(path).is_file():
        raise InputError("Video file is missing.")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InputError("Video cannot be decoded by this OpenCV installation.")
        fps, count, width, height = [
            float(cap.get(p))
            for p in [
                cv2.CAP_PROP_FPS,
                cv2.CAP_PROP_FRAME_COUNT,
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
            ]
        ]
        if (
            not np.all(np.isfinite([fps, count, width, height]))
            or not 2 <= fps <= 240
            or width < 4
            or height < 4
            or count < 1
        ):
            raise InputError(
                "Invalid video metadata or unsupported frame rate (supported: 2-240 FPS)."
            )
        return VideoMetadata(
            "local recording", fps, int(count), int(width), int(height), count / fps
        )
    finally:
        cap.release()


def timestamp_diagnostics(times, fps: float, tolerance=0.2) -> dict:
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return {
            "timing_valid": True,
            "timestamp_source": "nominal_fps",
            "warning": "Too few timestamps to verify.",
        }
    if np.all(times == 0):
        return {
            "timing_valid": True,
            "timestamp_source": "nominal_fps",
            "warning": "Decoder supplies no timestamps; constant frame rate assumed.",
        }
    delta = np.diff(times)
    valid = bool(
        np.all(np.isfinite(times))
        and np.all(delta > 0)
        and np.max(abs(delta * fps - 1)) <= tolerance
    )
    return {
        "timing_valid": valid,
        "timestamp_source": "decoder_checked_nominal_grid",
        "warning": None
        if valid
        else "Non-monotonic or variable timestamps; uniform sampling is unsupported.",
    }


def iter_frames(
    path: str | Path, timestamps: list | None = None
) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InputError("Video cannot be decoded.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if timestamps is not None:
                timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
            yield frame
    finally:
        cap.release()
