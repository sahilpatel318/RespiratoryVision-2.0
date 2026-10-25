from enum import Enum

import numpy as np
from scipy import signal

from .config import PipelineConfig


class QualityState(str, Enum):
    ACCEPTED = "ACCEPTED"
    LOW_QUALITY = "LOW_QUALITY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    MOTION_ARTIFACT = "MOTION_ARTIFACT"
    ESTIMATOR_DISAGREEMENT = "ESTIMATOR_DISAGREEMENT"
    UNSUPPORTED_INPUT = "UNSUPPORTED_INPUT"


def assess_quality(
    raw, methods: dict, fusion: dict, motion: dict, fps: float, config: PipelineConfig
) -> dict:
    amplitude = float(np.std(signal.detrend(raw))) if len(raw) > 1 else 0.0
    duration = len(raw) / fps
    features = {
        "displacement_std_pixels": amplitude,
        "duration_seconds": duration,
        "spectral_concentration": methods["welch"]["diagnostics"].get(
            "concentration", 0
        ),
        **motion,
    }
    reasons = []
    state = QualityState.ACCEPTED
    if duration < config.min_duration_seconds:
        state = QualityState.INSUFFICIENT_DATA
        reasons.append("Recording is shorter than the configured minimum duration.")
    elif motion.get("artifact_fraction", 0) > config.max_artifact_fraction:
        state = QualityState.MOTION_ARTIFACT
        reasons.append("Excessive frame-to-frame motion.")
    elif amplitude < config.min_displacement_std:
        state = QualityState.LOW_QUALITY
        reasons.append("Displacement amplitude is below the configured minimum.")
    elif motion.get("roi_consistency", 1) < config.min_roi_consistency:
        state = QualityState.MOTION_ARTIFACT
        reasons.append("Thoracic subregions have inconsistent motion.")
    elif (
        methods["welch"]["diagnostics"].get("rate_change_bpm", 0)
        > config.max_rate_change_bpm
    ):
        state = QualityState.LOW_QUALITY
        reasons.append(
            "Dominant frequency changes between recording halves; one rate is unsuitable."
        )
    elif methods["welch"]["diagnostics"].get("harmonic_ambiguity", False):
        state = QualityState.ESTIMATOR_DISAGREEMENT
        reasons.append("Competing spectral peaks suggest harmonic ambiguity.")
    elif (
        fusion["spread_bpm"] is not None and fusion["spread_bpm"] > config.agreement_bpm
    ):
        state = QualityState.ESTIMATOR_DISAGREEMENT
        reasons.extend(fusion["reasons"])
    elif fusion["bpm"] is None:
        state = QualityState.LOW_QUALITY
        reasons.extend(fusion["reasons"])
    elif duration * fusion["bpm"] / 60 < config.min_cycles:
        state = QualityState.INSUFFICIENT_DATA
        reasons.append("Too few respiratory cycles for a stable estimate.")
    return {
        "state": state.value,
        "accepted": state == QualityState.ACCEPTED,
        "reasons": reasons,
        "features": features,
        "thresholds_provisional": True,
        "probability_calibrated": False,
    }
