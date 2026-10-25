import argparse
import csv
import tempfile
from pathlib import Path

from respiratory_vision import BaselineEstimator as RespiratoryEstimator
from respiratory_vision.metrics import regression_metrics
from respiratory_vision.synthetic import generate_breathing_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/synthetic_benchmark.csv")
    args = parser.parse_args()

    rates = [12.0, 18.0, 24.0, 30.0]
    rows = []
    estimator = RespiratoryEstimator()

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        for i, bpm in enumerate(rates):
            path = temp / f"synthetic_{int(bpm)}bpm.mp4"
            generate_breathing_video(
                path,
                bpm=bpm,
                duration_seconds=18.0,
                fps=10,
                seed=100 + i,
            )
            result = estimator.analyze(path)
            pred = result["estimate"]["bpm"]
            rows.append(
                {
                    "ground_truth_bpm": bpm,
                    "estimated_bpm": pred,
                    "absolute_error_bpm": abs(pred - bpm),
                    "confidence": result["estimate"]["confidence"],
                }
            )

    metrics = regression_metrics(
        [r["ground_truth_bpm"] for r in rows],
        [r["estimated_bpm"] for r in rows],
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"MAE: {metrics.mae:.3f} BPM")
    print(f"RMSE: {metrics.rmse:.3f} BPM")
    print(f"Mean bias: {metrics.mean_bias:.3f} BPM")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
