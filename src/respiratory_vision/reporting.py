import base64
import csv
import io
import json
import math
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt

from .visualization import diagnostic_figure

NOTICE = "Research prototype. Not for diagnosis, medical decisions, or pediatric clinical monitoring."
STYLE = "body{font:16px/1.6 system-ui,sans-serif;color:#173242;max-width:1120px;margin:40px auto;padding:0 24px;background:#f6f8fa}h1,h2{color:#086d76}table{border-collapse:collapse;background:white}td,th{border:1px solid #cbd5df;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e8eef2;padding:16px}img{max-width:100%}.notice{border-left:5px solid #d57a29;padding:12px;background:#fff3dc}"


def signal_csv(result: dict) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["seconds", "raw_displacement_pixels", "filtered_normalized"])
    writer.writerows(
        zip(
            result["timestamps_seconds"],
            result["raw_displacement"],
            result["filtered_signal"],
        )
    )
    return out.getvalue()


def analysis_html(
    result: dict,
    reference_bpm=None,
    provenance="Local recording; no reference supplied",
) -> str:
    if reference_bpm is not None and (
        not math.isfinite(reference_bpm) or reference_bpm <= 0
    ):
        raise ValueError("Reference BPM must be finite and positive.")
    prediction = result["estimate"]["bpm"]
    accuracy = "No reference measurement supplied: accuracy cannot be calculated for this recording."
    if reference_bpm is not None:
        accuracy = f"Reference: {reference_bpm:.3f} BPM. " + (
            f"Absolute error: {abs(prediction - reference_bpm):.3f} BPM (one recording, not an accuracy study)."
            if prediction is not None
            else "Pipeline abstained; no accepted prediction error is available."
        )
    fig = diagnostic_figure(result)
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=120)
    plt.close(fig)
    uri = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    rows = "".join(
        f"<tr><td>{escape(k)}</td><td>{v['bpm']}</td><td>{v['reliable']}</td><td>{escape('; '.join(v['reasons']))}</td></tr>"
        for k, v in result["methods"].items()
    )
    machine = dict(
        result, report_reference_bpm=reference_bpm, report_provenance=provenance
    )
    payload = json.dumps(machine, indent=2, allow_nan=False)
    download = (
        "data:application/json;base64," + base64.b64encode(payload.encode()).decode()
    )
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>RespiratoryVision research report</title><style>{STYLE}</style>
+<h1>RespiratoryVision 2.0</h1><p>Video-based respiratory motion research report</p><p class="notice">{NOTICE}</p>
+<h2>Objective and provenance</h2><p>Investigate recovery of periodic thoracic motion from ordinary video using interpretable signal processing.</p><p>{escape(provenance)}</p>
+<h2>Measured result</h2><p>Status: <strong>{escape(result["quality"]["state"])}</strong>. Accepted estimate: {prediction if prediction is not None else "Abstained"} BPM.</p><p>{escape(accuracy)}</p>
+<h2>Methodology</h2><p>Streaming grayscale ROI, Farneback flow, optional robust background translation subtraction, active-quadrant aggregation, displacement integration, detrending and Butterworth band-pass filtering. Welch, autocorrelation and cycle-interval estimates are compared before provisional quality gating. Spectral share is not probability.</p>
+<h2>Signal diagnostics and method comparison</h2><img alt="Measured pipeline diagnostic plots" src="{uri}"><table><tr><th>Method</th><th>BPM</th><th>Eligible</th><th>Reasons</th></tr>{rows}</table>
+<h2>Quality decisions</h2><pre>{escape(json.dumps(result["quality"], indent=2))}</pre>
+<h2>Experimental configuration</h2><pre>{escape(json.dumps(result["config"], indent=2))}</pre>
+<h2>Failures and limitations</h2><p>Synthetic validation does not establish performance on people. Translation compensation cannot resolve rotation, parallax or independent body movement. Geometric ROI requires appropriate framing; automatic anatomy detection is unavailable. A single rate cannot characterize pauses or changing respiration. Thresholds are provisional and uncalibrated. Inspect exclusions, rejected cycles and background tracking coverage.</p>
+<h2>Reproduce</h2><pre>python -m pip install -e ".[dev]"
+python scripts/run_pipeline.py --video PATH_TO_AUTHORIZED_VIDEO --report outputs/report.html
+python scripts/run_full_benchmark.py --split test --output outputs/benchmark</pre><p>Use the exported configuration with --config; include --roi x y w h when a manual ROI was used.</p>
+<h2>Machine-readable result</h2><a download="result.json" href="{download}">Download JSON</a>
+<h2>Responsible use</h2><p>{NOTICE} Process only authorized recordings. Derived signals and reports can still be sensitive; retain and share them under the applicable data-use agreement.</p></html>'''.replace(
        "\n+", "\n"
    )


def write_analysis_report(
    result,
    output,
    reference_bpm=None,
    provenance="Local recording; no reference supplied",
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        analysis_html(result, reference_bpm, provenance), encoding="utf-8"
    )
    payload = dict(
        result, report_reference_bpm=reference_bpm, report_provenance=provenance
    )
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    output.with_suffix(".csv").write_text(signal_csv(result), encoding="utf-8")


def write_benchmark_report(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    rows = "".join(
        f"<tr><td>{escape(m)}</td><td>{s['mae']}</td><td>{s['rmse']}</td><td>{s['n_predictions']}/{s['n_total']}</td><td>{s['mae_95ci']}</td></tr>"
        for m, s in summary["methods"].items()
    )
    images = "".join(
        '<img alt="'
        + name
        + '" src="data:image/png;base64,'
        + base64.b64encode((directory / (name + ".png")).read_bytes()).decode()
        + '">'
        for name in ["method_comparison", "error_distribution", "quality_by_condition"]
    )
    failures = escape((directory / "failure_cases.csv").read_text())
    provenance = summary["provenance"]
    command = f"python scripts/run_full_benchmark.py --split {provenance.get('split', 'test')} --per-condition {provenance.get('per_condition', 4)} --output outputs/{directory.name}"
    if provenance.get("cnn_checkpoint_used"):
        command += " --checkpoint outputs/model/temporal_cnn.pt"
    html = f'<html lang="en"><meta charset="utf-8"><title>RespiratoryVision benchmark</title><style>{STYLE}</style><h1>RespiratoryVision benchmark</h1><p class="notice">{NOTICE}</p><h2>Executed experiments</h2><pre>{escape(json.dumps(summary["provenance"], indent=2))}</pre><p>{escape(summary["metrics_definition"])}</p><table><tr><th>Method</th><th>MAE</th><th>RMSE</th><th>Available</th><th>MAE 95% CI</th></tr>{rows}</table>{images}<h2>Failure cases</h2><pre>{failures}</pre><h2>Limitations</h2><p>Small synthetic stress grid, no human recordings. Error is conditional on prediction availability. Bootstrap intervals describe this generated mixture, not clinical uncertainty. Inspect matched_fusion_subset in summary.json for a like-for-like comparison. Timing includes shared enhanced extraction for each method and is not additive.</p><h2>Reproduce</h2><pre>{escape(command)}</pre></html>'
    (directory / "report.html").write_text(html, encoding="utf-8")
