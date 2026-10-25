from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .motion import extract_motion_trace
from .roi import relative_roi, select_interactive_roi
from .signal_processing import clean_respiratory_signal, estimate_rate_welch
from .video_io import iter_frames, probe_video


class RespiratoryEstimator:
    """End-to-end respiratory-rate estimator from RGB video."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.config.validate()

    def analyze(self, video_path: str | Path, interactive_roi: bool = False) -> dict[str, Any]:
        metadata = probe_video(video_path)
        if metadata.duration_seconds < self.config.min_duration_seconds:
            raise ValueError(
                f"Video is {metadata.duration_seconds:.1f}s; "
                f"minimum configured duration is {self.config.min_duration_seconds:.1f}s."
            )

        frames = list(iter_frames(video_path))
        if not frames:
            raise ValueError("Video contains no readable frames.")

        if interactive_roi:
            roi = select_interactive_roi(frames[0])
        else:
            roi = relative_roi(
                frames[0],
                self.config.roi_x,
                self.config.roi_y,
                self.config.roi_w,
                self.config.roi_h,
            )

        trace = extract_motion_trace(
            frames,
            roi,
            gaussian_kernel=self.config.gaussian_kernel,
        )
        filtered = clean_respiratory_signal(
            trace.displacement,
            metadata.fps,
            self.config.min_bpm,
            self.config.max_bpm,
        )
        estimate = estimate_rate_welch(
            filtered,
            metadata.fps,
            self.config.min_bpm,
            self.config.max_bpm,
        )

        timestamps = np.arange(len(filtered), dtype=float) / metadata.fps

        return {
            "video": asdict(metadata),
            "roi": asdict(roi),
            "estimate": asdict(estimate),
            "timestamps_seconds": timestamps.tolist(),
            "raw_displacement": trace.displacement.tolist(),
            "filtered_signal": filtered.tolist(),
            "method": "OpenCV Farneback optical flow + Butterworth band-pass + Welch PSD",
            "clinical_use": False,
        }
