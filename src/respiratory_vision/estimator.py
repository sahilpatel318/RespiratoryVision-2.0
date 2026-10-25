import time
from dataclasses import asdict
from itertools import chain
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .errors import InputError
from .fusion import fuse_estimates
from .motion import extract_motion_trace
from .quality import assess_quality
from .roi import ROI, relative_roi, select_interactive_roi
from .signal_processing import (
    analyze_methods,
    clean_respiratory_signal,
    validate_sampling,
)
from .video_io import iter_frames, probe_video, timestamp_diagnostics


class RespiratoryEstimator:
    """Streaming enhanced pipeline; expected input failures become explicit abstentions."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.config.validate()

    def analyze(
        self, video_path: str | Path, interactive_roi=False, roi: ROI | None = None
    ) -> dict:
        start = time.perf_counter()
        result = {
            "schema_version": "2.0",
            "clinical_use": False,
            "config": asdict(self.config),
            "video": None,
            "roi": None,
            "estimate": {"bpm": None},
            "methods": {},
            "fusion": {},
            "quality": {},
            "timestamps_seconds": [],
            "raw_displacement": [],
            "filtered_signal": [],
            "global_translation_pixels": [],
            "subregion_flow": [],
            "method": "Farneback + background translation + Butterworth + multi-method quality gate",
        }
        frames = None
        try:
            metadata = probe_video(video_path)
            result["video"] = asdict(metadata)
            validate_sampling(metadata.fps, self.config.min_bpm, self.config.max_bpm)
            if metadata.duration_seconds < self.config.min_duration_seconds:
                raise InputError(
                    "Video is shorter than the configured minimum duration.",
                    "INSUFFICIENT_DATA",
                )
            timestamps = []
            frames = iter_frames(video_path, timestamps)
            first = next(frames, None)
            if first is None:
                raise InputError("Video has no decoded frames.", "INSUFFICIENT_DATA")
            roi = (
                select_interactive_roi(first)
                if interactive_roi
                else roi
                or relative_roi(
                    first,
                    self.config.roi_x,
                    self.config.roi_y,
                    self.config.roi_w,
                    self.config.roi_h,
                )
            )
            roi.validate(first)
            result["roi"] = asdict(roi)
            trace = extract_motion_trace(
                chain([first], frames),
                roi,
                self.config.gaussian_kernel,
                self.config.motion_compensation,
                metadata.fps,
                self.config.max_motion_pixels_second,
            )
            timing = timestamp_diagnostics(
                timestamps, metadata.fps, self.config.timestamp_tolerance
            )
            result["timing"] = timing
            if not timing["timing_valid"]:
                raise InputError(timing["warning"])
            if abs(len(trace.displacement) - metadata.frame_count) > 1:
                raise InputError(
                    "Decoded frame count differs from metadata; possible truncated recording."
                )
            if (
                len(trace.displacement) / metadata.fps
                < self.config.min_duration_seconds
            ):
                raise InputError("Insufficient decoded duration.", "INSUFFICIENT_DATA")
            filtered = clean_respiratory_signal(
                trace.displacement,
                metadata.fps,
                self.config.min_bpm,
                self.config.max_bpm,
            )
            methods = analyze_methods(filtered, metadata.fps, self.config)
            fusion = fuse_estimates(methods, self.config.agreement_bpm)
            quality = assess_quality(
                trace.displacement,
                methods,
                fusion,
                trace.diagnostics,
                metadata.fps,
                self.config,
            )
            result.update(
                methods=methods,
                fusion=fusion,
                quality=quality,
                estimate={"bpm": fusion["bpm"] if quality["accepted"] else None},
                timestamps_seconds=(np.arange(len(filtered)) / metadata.fps).tolist(),
                raw_displacement=trace.displacement.tolist(),
                filtered_signal=filtered.tolist(),
                global_translation_pixels=trace.global_translation.tolist(),
                subregion_flow=trace.subregion_flow.tolist(),
            )
        except (InputError, ValueError, cv2.error) as exc:
            result["quality"] = {
                "state": getattr(exc, "state", "UNSUPPORTED_INPUT"),
                "accepted": False,
                "reasons": [
                    str(exc)
                    if not isinstance(exc, cv2.error)
                    else "OpenCV could not process the recording."
                ],
                "features": {},
                "thresholds_provisional": True,
                "probability_calibrated": False,
            }
        finally:
            if frames is not None:
                frames.close()
        result["processing_seconds"] = time.perf_counter() - start
        return result
