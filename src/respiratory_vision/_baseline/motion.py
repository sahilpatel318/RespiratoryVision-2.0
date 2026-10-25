from dataclasses import dataclass

import cv2
import numpy as np

from .roi import ROI


@dataclass
class MotionTrace:
    vertical_flow: np.ndarray
    displacement: np.ndarray


def _prep(gray: np.ndarray, gaussian_kernel: int) -> np.ndarray:
    if gaussian_kernel > 1:
        gray = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)
    return gray


def extract_motion_trace(
    frames: list[np.ndarray],
    roi: ROI,
    gaussian_kernel: int = 5,
) -> MotionTrace:
    """Extract robust vertical thoracic motion using dense Farneback optical flow."""
    if len(frames) < 3:
        raise ValueError("At least three video frames are required.")

    first = roi.crop(frames[0])
    prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    prev = _prep(prev, gaussian_kernel)

    vertical = [0.0]

    for frame in frames[1:]:
        current_roi = roi.crop(frame)
        gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        gray = _prep(gray, gaussian_kernel)

        flow = cv2.calcOpticalFlowFarneback(
            prev,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Robustly aggregate vertical motion. Suppress nearly-static pixels,
        # which otherwise dominate the median in large background areas.
        fy = flow[..., 1]
        fx = flow[..., 0]
        magnitude = np.sqrt(fx * fx + fy * fy)
        moving = magnitude > np.percentile(magnitude, 60)

        if np.any(moving):
            sample = fy[moving]
            # Trim extremes caused by local edges/outliers.
            lo, hi = np.percentile(sample, [10, 90])
            sample = sample[(sample >= lo) & (sample <= hi)]
            value = float(np.mean(sample)) if sample.size else 0.0
        else:
            value = float(np.mean(fy))

        vertical.append(value)
        prev = gray

    vertical_arr = np.asarray(vertical, dtype=np.float64)
    displacement = np.cumsum(vertical_arr)
    return MotionTrace(vertical_flow=vertical_arr, displacement=displacement)
