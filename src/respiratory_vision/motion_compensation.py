"""Robust background translation from forward/backward Lucas-Kanade tracks.

Translation is deliberately interpretable. Rotation, parallax and rolling shutter
remain unsupported; track failures are surfaced, never reported as corrections.
"""

import cv2
import numpy as np

from .roi import ROI


def estimate_background_translation(
    previous, current, roi: ROI
) -> tuple[np.ndarray, bool, int]:
    mask = np.full(previous.shape, 255, dtype=np.uint8)
    # Exclude torso and an 8-pixel guard from reference feature detection.
    mask[
        max(0, roi.y - 8) : roi.y + roi.h + 8, max(0, roi.x - 8) : roi.x + roi.w + 8
    ] = 0
    points = cv2.goodFeaturesToTrack(
        previous, maxCorners=150, qualityLevel=0.02, minDistance=6, mask=mask
    )
    if points is None or len(points) < 6:
        return np.zeros(2), False, 0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, winSize=(21, 21), maxLevel=3
    )
    if nxt is None:
        return np.zeros(2), False, 0
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, nxt, None, winSize=(21, 21), maxLevel=3
    )
    if back is None:
        return np.zeros(2), False, 0
    valid = (
        (status.ravel() == 1)
        & (back_status.ravel() == 1)
        & (np.linalg.norm(back - points, axis=2).ravel() < 1.0)
    )
    displacements = (nxt - points).reshape(-1, 2)[valid]
    if len(displacements) < 6:
        return np.zeros(2), False, len(displacements)
    # Require 2D reference coverage: a single torso edge outside the ROI is not a background model.
    reference = points.reshape(-1, 2)[valid]
    span = np.ptp(reference, axis=0)
    if span[0] < 0.35 * previous.shape[1] or span[1] < 0.35 * previous.shape[0]:
        return np.zeros(2), False, len(displacements)
    median = np.median(displacements, axis=0)
    residual = np.linalg.norm(displacements - median, axis=1)
    inliers = residual <= max(0.5, 3 * np.median(residual))
    if inliers.sum() < 6 or inliers.mean() < 0.6:
        return np.zeros(2), False, int(inliers.sum())
    return np.median(displacements[inliers], axis=0), True, int(inliers.sum())
