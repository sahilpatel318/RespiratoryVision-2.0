import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PipelineConfig:
    """Provisional engineering thresholds, not clinical calibration."""

    min_bpm: float = 6.0
    max_bpm: float = 48.0
    roi_x: float = 0.20
    roi_y: float = 0.30
    roi_w: float = 0.60
    roi_h: float = 0.50
    gaussian_kernel: int = 5
    min_duration_seconds: float = 8.0
    motion_compensation: bool = True
    min_cycles: float = 2.5
    agreement_bpm: float = 3.0
    min_displacement_std: float = 0.08
    min_spectral_concentration: float = 0.45
    min_autocorrelation: float = 0.35
    max_interval_cv: float = 0.35
    max_rate_change_bpm: float = 6.0
    max_motion_pixels_second: float = 80.0
    max_artifact_fraction: float = 0.10
    min_roi_consistency: float = 0.10
    timestamp_tolerance: float = 0.20

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite and numeric.")
        if not 0 < self.min_bpm < self.max_bpm:
            raise ValueError("Expected 0 < min_bpm < max_bpm.")
        if not (
            0 <= self.roi_x < 1
            and 0 <= self.roi_y < 1
            and 0 < self.roi_w <= 1
            and 0 < self.roi_h <= 1
        ):
            raise ValueError("ROI origin must be nonnegative and size positive.")
        if self.roi_x + self.roi_w > 1 or self.roi_y + self.roi_h > 1:
            raise ValueError("ROI extends outside frame.")
        if (
            not isinstance(self.gaussian_kernel, int)
            or self.gaussian_kernel < 1
            or self.gaussian_kernel % 2 == 0
        ):
            raise ValueError("gaussian_kernel must be a positive odd integer.")
        for name in [
            "min_duration_seconds",
            "min_cycles",
            "agreement_bpm",
            "min_displacement_std",
            "max_rate_change_bpm",
            "max_motion_pixels_second",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in [
            "min_spectral_concentration",
            "min_autocorrelation",
            "max_interval_cv",
            "max_artifact_fraction",
            "min_roi_consistency",
            "timestamp_tolerance",
        ]:
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
