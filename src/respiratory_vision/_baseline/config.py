from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for respiratory-rate estimation."""

    min_bpm: float = 6.0
    max_bpm: float = 48.0
    roi_x: float = 0.20
    roi_y: float = 0.30
    roi_w: float = 0.60
    roi_h: float = 0.50
    gaussian_kernel: int = 5
    min_duration_seconds: float = 8.0

    def validate(self) -> None:
        if not (0 < self.min_bpm < self.max_bpm):
            raise ValueError("Expected 0 < min_bpm < max_bpm.")
        for value in (self.roi_x, self.roi_y, self.roi_w, self.roi_h):
            if not 0 <= value <= 1:
                raise ValueError("ROI fractions must be in [0, 1].")
        if self.roi_x + self.roi_w > 1 or self.roi_y + self.roi_h > 1:
            raise ValueError("ROI extends outside frame.")
        if self.gaussian_kernel < 1 or self.gaussian_kernel % 2 == 0:
            raise ValueError("gaussian_kernel must be a positive odd integer.")
