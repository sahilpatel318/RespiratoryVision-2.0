from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]


def relative_roi(frame: np.ndarray, x: float, y: float, w: float, h: float) -> ROI:
    height, width = frame.shape[:2]
    roi = ROI(
        x=int(round(width * x)),
        y=int(round(height * y)),
        w=max(2, int(round(width * w))),
        h=max(2, int(round(height * h))),
    )
    if roi.x + roi.w > width or roi.y + roi.h > height:
        raise ValueError("ROI is outside frame bounds.")
    return roi


def select_interactive_roi(frame: np.ndarray) -> ROI:
    x, y, w, h = cv2.selectROI(
        "Select upper-torso ROI and press ENTER",
        frame,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyAllWindows()
    if w <= 1 or h <= 1:
        raise ValueError("No valid ROI selected.")
    return ROI(int(x), int(y), int(w), int(h))
