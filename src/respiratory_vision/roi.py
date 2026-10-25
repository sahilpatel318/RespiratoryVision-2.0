from dataclasses import dataclass

import cv2
import numpy as np

from .errors import InputError


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int

    def validate(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if (
            any(
                not isinstance(v, (int, np.integer))
                for v in [self.x, self.y, self.w, self.h]
            )
            or self.x < 0
            or self.y < 0
            or self.w < 4
            or self.h < 4
            or self.x + self.w > width
            or self.y + self.h > height
        ):
            raise InputError(
                "ROI must be at least 4x4 pixels and fully inside the frame."
            )

    def crop(self, frame: np.ndarray) -> np.ndarray:
        self.validate(frame)
        return frame[self.y : self.y + self.h, self.x : self.x + self.w]


def relative_roi(frame, x, y, w, h) -> ROI:
    if (
        not np.all(np.isfinite([x, y, w, h]))
        or x < 0
        or y < 0
        or w <= 0
        or h <= 0
        or x + w > 1
        or y + h > 1
    ):
        raise InputError("Invalid relative ROI.")
    height, width = frame.shape[:2]
    roi = ROI(round(width * x), round(height * y), round(width * w), round(height * h))
    roi.validate(frame)
    return roi


def select_interactive_roi(frame) -> ROI:
    try:
        values = cv2.selectROI(
            "Select thoracic ROI and press ENTER",
            frame,
            showCrosshair=True,
            fromCenter=False,
        )
    except cv2.error as exc:
        raise InputError(
            "Interactive ROI requires an OpenCV desktop GUI; use a manual bounding box."
        ) from exc
    finally:
        cv2.destroyAllWindows()
    roi = ROI(*(int(v) for v in values))
    roi.validate(frame)
    return roi
