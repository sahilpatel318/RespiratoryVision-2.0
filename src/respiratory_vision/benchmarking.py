"""Reproducible comparative evaluation. Metrics are conditional on available predictions."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import BaselineEstimator, PipelineConfig, RespiratoryEstimator
from .synthetic import CONDITIONS, challenge_config, generate_challenge_video

METHODS = (
    "original_baseline",
    "compensated_welch",
    "autocorrelation",
    "peaks",
    "quality_fusion",
)


def score_rows(rows: list[dict], seed=17) -> dict:
    valid = [r for r in rows if r["estimated_bpm"] is not None]
    errors = np.asarray(
        [r["estimated_bpm"] - r["reference_bpm"] for r in valid], dtype=float
    )
    n = len(rows)
    result = {
        "n_total": n,
        "n_predictions": len(valid),
        "acceptance_rate": len(valid) / n if n else 0.0,
        "rejection_rate": 1 - len(valid) / n if n else 0.0,
        "mae": None,
        "rmse": None,
        "mean_signed_error": None,
        "median_absolute_error": None,
        "within_2_bpm_percent": None,
        "within_3_bpm_percent": None,
        "mae_95ci": None,
        "mean_processing_seconds": float(
            np.mean(
                [
                    r["processing_seconds"]
                    for r in rows
                    if r["processing_seconds"] is not None
                ]
            )
        )
        if any(r["processing_seconds"] is not None for r in rows)
        else None,
    }
    if len(errors):
        result.update(
            mae=float(abs(errors).mean()),
            rmse=float(np.sqrt(np.mean(errors**2))),
            mean_signed_error=float(errors.mean()),
            median_absolute_error=float(np.median(abs(errors))),
            within_2_bpm_percent=float(100 * np.mean(abs(errors) <= 2)),
            within_3_bpm_percent=float(100 * np.mean(abs(errors) <= 3)),
            absolute_error_quantiles_bpm={
                str(q): float(np.quantile(abs(errors), q))
                for q in [0, 0.25, 0.5, 0.75, 0.9, 1]
            },
        )
        # Cluster by independent source/participant when a private group key is supplied.
        groups = {r.get("_group", r["case_id"]) for r in valid}
        if len(groups) >= 8:
            group_errors = {
                g: [
                    abs(r["estimated_bpm"] - r["reference_bpm"])
                    for r in valid
                    if r.get("_group", r["case_id"]) == g
                ]
                for g in sorted(groups)
            }
            group_arrays = list(group_errors.values())
            rng = np.random.default_rng(seed)
            draws = [
                np.mean(
                    np.concatenate(
                        [
                            group_arrays[i]
                            for i in rng.integers(0, len(groups), len(groups))
                        ]
                    )
                )
                for _ in range(1000)
            ]
            result["mae_95ci"] = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
    return result


def evaluate_video(
    path, reference_bpm, case_id, condition, config=None, checkpoint=None
) -> tuple[list[dict], dict]:
    rows = []
    start = time.perf_counter()
    try:
        baseline = BaselineEstimator().analyze(path)
        pred = baseline["estimate"]["bpm"]
        baseline_state = "AVAILABLE"
    except (ValueError, FileNotFoundError, RuntimeError):
        pred = None
        baseline_state = "INPUT_FAILURE"
    baseline_time = time.perf_counter() - start
    result = RespiratoryEstimator(config).analyze(path)
    for name in METHODS:
        if name == "original_baseline":
            estimate, state, elapsed = pred, baseline_state, baseline_time
        elif name == "quality_fusion":
            estimate, state, elapsed = (
                result["estimate"]["bpm"],
                result["quality"]["state"],
                result["processing_seconds"],
            )
        else:
            key = "welch" if name == "compensated_welch" else name
            method = result["methods"].get(key, {})
            # Individual estimators are compared ungated; eligibility is recorded separately.
            estimate = method.get("bpm")
            state = (
                "AVAILABLE"
                if method.get("reliable")
                else "METHOD_FLAGGED"
                if estimate is not None
                else "UNAVAILABLE"
            )
            elapsed = result["processing_seconds"]
        rows.append(
            {
                "case_id": case_id,
                "condition": condition,
                "method": name,
                "reference_bpm": float(reference_bpm),
                "estimated_bpm": estimate,
                "signed_error_bpm": None
                if estimate is None
                else estimate - reference_bpm,
                "absolute_error_bpm": None
                if estimate is None
                else abs(estimate - reference_bpm),
                "status": state,
                "processing_seconds": elapsed,
            }
        )
    if checkpoint:
        from .training import predict_checkpoint

        start = time.perf_counter()
        estimate = (
            predict_checkpoint(
                checkpoint, result["filtered_signal"], result["video"]["fps"]
            )
            if result["filtered_signal"]
            else None
        )
        rows.append(
            {
                "case_id": case_id,
                "condition": condition,
                "method": "temporal_cnn",
                "reference_bpm": float(reference_bpm),
                "estimated_bpm": estimate,
                "signed_error_bpm": None
                if estimate is None
                else estimate - reference_bpm,
                "absolute_error_bpm": None
                if estimate is None
                else abs(estimate - reference_bpm),
                "status": "AVAILABLE" if estimate is not None else "UNAVAILABLE",
                "processing_seconds": result["processing_seconds"]
                + time.perf_counter()
                - start,
            }
        )
    return rows, result


def write_benchmark(rows, output, provenance: dict) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    summary = {
        "provenance": provenance,
        "metrics_definition": "Errors among available predictions. Always inspect coverage; individual methods are ungated. CI: seeded percentile bootstrap of independent recording/subject clusters, >=8 clusters, conditional on acceptance.",
        "methods": {
            m: score_rows([r for r in rows if r["method"] == m]) for m in methods
        },
        "by_condition": {
            c: {
                m: score_rows(
                    [r for r in rows if r["method"] == m and r["condition"] == c]
                )
                for m in methods
            }
            for c in sorted({r["condition"] for r in rows})
        },
    }
    accepted = {
        r["case_id"]
        for r in rows
        if r["method"] == "quality_fusion" and r["estimated_bpm"] is not None
    }
    summary["matched_fusion_subset"] = {
        m: score_rows(
            [r for r in rows if r["method"] == m and r["case_id"] in accepted]
        )
        for m in methods
    }
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    for name, subset in [
        ("results.csv", public),
        (
            "failure_cases.csv",
            [
                r
                for r in public
                if r["estimated_bpm"] is None
                or r["absolute_error_bpm"] > 3
                or r["status"] == "METHOD_FLAGGED"
            ],
        ),
    ]:
        with (output / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(public[0]))
            writer.writeheader()
            writer.writerows(subset)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [summary["methods"][m]["mae"] for m in methods]
    ax.bar(methods, [v or 0 for v in vals], color="#087e8b")
    ax.set_ylabel("MAE on available predictions (BPM)")
    ax.set_ylim(0, max(0.5, max([v or 0 for v in vals]) * 1.35 + 0.2))
    ax.tick_params(axis="x", rotation=20)
    for i, m in enumerate(methods):
        s = summary["methods"][m]
        ax.text(
            i,
            (vals[i] or 0) + 0.05,
            f"{s['n_predictions']}/{s['n_total']} available",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output / "method_comparison.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in methods:
        errors = [
            r["signed_error_bpm"]
            for r in rows
            if r["method"] == m and r["signed_error_bpm"] is not None
        ]
        if errors:
            ax.hist(errors, bins=12, alpha=0.4, label=m)
    ax.set(xlabel="Signed error (BPM)", ylabel="Recordings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "error_distribution.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    conditions = list(summary["by_condition"])
    ax.bar(
        conditions,
        [
            summary["by_condition"][c]
            .get("quality_fusion", {})
            .get("acceptance_rate", 0)
            * 100
            for c in conditions
        ],
        color="#087e8b",
    )
    ax.set(ylabel="Quality-gated acceptance (%)", ylim=(0, 105))
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output / "quality_by_condition.png", dpi=140)
    plt.close(fig)
    return summary


def run_benchmark(
    output="outputs/benchmark",
    split="test",
    per_condition=4,
    conditions=CONDITIONS,
    checkpoint=None,
) -> dict:
    if per_condition < 1 or per_condition > 50 or not conditions:
        raise ValueError("Use 1-50 cases per condition and at least one condition.")
    # Index must stay below seed block width; development/test seeds and configurations differ.
    config = PipelineConfig()
    rows, cases = [], []
    with tempfile.TemporaryDirectory(prefix="respiratory-synthetic-") as directory:
        for condition in conditions:
            for index in range(per_condition):
                c = challenge_config(condition, split, index)
                case_id = f"{split}-{condition}-{index:02d}"
                path = Path(directory) / f"{case_id}.mp4"
                metadata = generate_challenge_video(path, c)
                case_rows, result = evaluate_video(
                    path,
                    metadata["ground_truth_bpm"],
                    case_id,
                    condition,
                    config,
                    checkpoint,
                )
                rows.extend(case_rows)
                cases.append(
                    {
                        "case_id": case_id,
                        "generation": asdict(c),
                        "reference_bpm": metadata["ground_truth_bpm"],
                        "quality": result["quality"],
                        "method_reasons": {
                            k: v["reasons"] for k, v in result["methods"].items()
                        },
                    }
                )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.json").write_text(
        json.dumps(cases, indent=2, allow_nan=False), encoding="utf-8"
    )
    provenance = {
        "kind": "synthetic",
        "split": split,
        "generator": "challenge-v1",
        "pipeline_config": asdict(config),
        "case_count": len(cases),
        "per_condition": per_condition,
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "scipy", "opencv-python", "matplotlib"]
        },
        "clinical_validation": False,
        "cnn_checkpoint_used": bool(checkpoint),
        "cnn_checkpoint_sha256": hashlib.sha256(
            Path(checkpoint).read_bytes()
        ).hexdigest()
        if checkpoint
        else None,
    }
    return write_benchmark(rows, output, provenance)
