from itertools import combinations

import numpy as np


def fuse_estimates(methods: dict, tolerance_bpm: float = 3.0) -> dict:
    """Require all eligible methods to agree; never hide a dissenting method."""
    eligible = {
        k: v["bpm"]
        for k, v in methods.items()
        if v["reliable"] and v["bpm"] is not None
    }
    excluded = {k: v["reasons"] for k, v in methods.items() if k not in eligible}
    differences = {
        f"{a}/{b}": abs(eligible[a] - eligible[b]) for a, b in combinations(eligible, 2)
    }
    spread = max(eligible.values()) - min(eligible.values()) if eligible else None
    reasons = []
    if len(eligible) < 2:
        reasons.append("Fewer than two eligible estimators.")
    if spread is not None and spread > tolerance_bpm:
        reasons.append("Eligible estimators disagree beyond the configured tolerance.")
    return {
        "bpm": float(np.median(list(eligible.values()))) if not reasons else None,
        "eligible_methods": list(eligible),
        "excluded_methods": excluded,
        "spread_bpm": spread,
        "pairwise_difference_bpm": differences,
        "reasons": reasons,
    }
