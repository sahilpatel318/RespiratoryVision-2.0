import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from respiratory_vision import BaselineEstimator, PipelineConfig, RespiratoryEstimator
from respiratory_vision.reporting import write_analysis_report
from respiratory_vision.roi import ROI
from respiratory_vision.visualization import diagnostic_figure


def main():
    p = argparse.ArgumentParser(
        description="Analyze authorized local video; research use only."
    )
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="outputs/result.json")
    p.add_argument("--plot", default="outputs/signal.png")
    p.add_argument("--report", default="outputs/report.html")
    p.add_argument("--interactive-roi", action="store_true")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    p.add_argument("--min-bpm", type=float)
    p.add_argument("--max-bpm", type=float)
    p.add_argument("--no-compensation", action="store_true")
    p.add_argument("--baseline", action="store_true")
    p.add_argument(
        "--config",
        help="JSON containing a PipelineConfig dictionary or a full exported result",
    )
    p.add_argument("--reference-bpm", type=float)
    a = p.parse_args()
    values = {}
    if a.config:
        values = json.loads(Path(a.config).read_text())
        values = values.get("config", values)
    if a.min_bpm is not None:
        values["min_bpm"] = a.min_bpm
    if a.max_bpm is not None:
        values["max_bpm"] = a.max_bpm
    if a.no_compensation:
        values["motion_compensation"] = False
    config = PipelineConfig(**values)
    if a.baseline:
        if a.roi or a.config or a.no_compensation:
            p.error(
                "--baseline supports original relative/default or interactive ROI only."
            )
        from respiratory_vision._baseline.config import PipelineConfig as OriginalConfig

        result = BaselineEstimator(
            OriginalConfig(min_bpm=config.min_bpm, max_bpm=config.max_bpm)
        ).analyze(a.video, a.interactive_roi)
    else:
        result = RespiratoryEstimator(config).analyze(
            a.video, a.interactive_roi, ROI(*a.roi) if a.roi else None
        )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if not a.baseline:
        fig = diagnostic_figure(result)
        plot = Path(a.plot)
        plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot, dpi=140)
        plt.close(fig)
        write_analysis_report(
            result,
            a.report,
            a.reference_bpm,
            "Local user-supplied video and optional scalar reference",
        )
    print(
        json.dumps(
            {
                "estimated_bpm": result["estimate"]["bpm"],
                "quality": result.get(
                    "quality", {"state": "ORIGINAL_BASELINE_UNGATED"}
                ),
                "result_json": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
