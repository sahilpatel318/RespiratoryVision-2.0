"""Local Streamlit dashboard; all estimates come from the installed package."""

import json
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from respiratory_vision import PipelineConfig, RespiratoryEstimator
from respiratory_vision.benchmarking import run_benchmark
from respiratory_vision.reporting import NOTICE, analysis_html, signal_csv
from respiratory_vision.roi import ROI
from respiratory_vision.video_io import iter_frames
from respiratory_vision.visualization import diagnostic_figure

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data/sample/synthetic_breathing_18bpm.mp4"


def method_table(result):
    return pd.DataFrame(
        [
            {
                "Method": k,
                "BPM": v["bpm"],
                "Eligible": v["reliable"],
                "Exclusion reasons": "; ".join(v["reasons"]),
            }
            for k, v in result["methods"].items()
        ]
    )


def show_summary(summary):
    table = pd.DataFrame(summary["methods"]).T
    st.dataframe(
        table[
            [
                "n_total",
                "n_predictions",
                "mae",
                "rmse",
                "acceptance_rate",
                "within_3_bpm_percent",
            ]
        ],
        width="stretch",
    )
    st.caption(
        "MAE/RMSE use available estimates only. Coverage is part of the result. Individual methods are ungated."
    )
    chart = pd.DataFrame(
        {
            c: {
                "Acceptance (%)": v.get("quality_fusion", {}).get("acceptance_rate", 0)
                * 100
            }
            for c, v in summary["by_condition"].items()
        }
    ).T
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(chart.index, chart["Acceptance (%)"], color="#087e8b")
    ax.set(ylabel="Quality-gated acceptance (%)", ylim=(0, 105))
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.json(summary["provenance"], expanded=False)


def main():
    st.set_page_config(
        page_title="RespiratoryVision | Research Laboratory",
        page_icon="🫁",
        layout="wide",
    )
    st.title("RespiratoryVision 2.0")
    st.markdown(
        "**Video-based respiratory motion analysis** · Local research laboratory"
    )
    st.warning(NOTICE)
    st.sidebar.header("Analysis settings")
    st.sidebar.caption(
        "Thresholds are provisional engineering choices. No clinical confidence probability is reported."
    )
    low = st.sidebar.number_input(
        "Minimum BPM", min_value=1.0, max_value=100.0, value=6.0
    )
    high = st.sidebar.number_input(
        "Maximum BPM", min_value=2.0, max_value=200.0, value=48.0
    )
    compensate = st.sidebar.checkbox("Compensate camera translation", value=True)
    roi_mode = st.sidebar.selectbox(
        "ROI selection", ["Default geometric ROI", "Manual bounding box"]
    )
    manual = None
    if roi_mode == "Manual bounding box":
        values = [
            st.sidebar.number_input(
                label, min_value=0 if i < 2 else 4, value=value, step=1
            )
            for i, (label, value) in enumerate(
                zip(
                    ["X (pixels)", "Y (pixels)", "Width (pixels)", "Height (pixels)"],
                    [40, 45, 120, 75],
                )
            )
        ]
        manual = ROI(*(int(v) for v in values))
    st.sidebar.caption(
        "Automatic anatomy localization is unavailable. Use the preview to set a manual ROI. Desktop drag selection is available through the CLI."
    )
    with st.sidebar.expander("Quality thresholds"):
        agreement = st.number_input(
            "Maximum method spread (BPM)", min_value=0.1, value=3.0
        )
        amplitude = st.number_input(
            "Minimum motion SD (pixels)", min_value=0.001, value=0.08, format="%.3f"
        )
        min_duration = st.number_input(
            "Minimum duration (seconds)", min_value=4.0, value=8.0
        )
    tabs = st.tabs(
        [
            "Analyze Video",
            "Signal Investigation",
            "Benchmark Laboratory",
            "Research Findings",
        ]
    )
    with tabs[0]:
        st.subheader("Analyze an authorized recording")
        source = st.radio(
            "Recording source",
            ["Included synthetic 18 BPM demo", "Upload local video"],
            horizontal=True,
        )
        upload = (
            st.file_uploader(
                "Video (processed locally, temporary copy removed after analysis)",
                type=["mp4", "avi", "mov", "mkv"],
            )
            if source == "Upload local video"
            else None
        )
        available = source == "Included synthetic 18 BPM demo" or upload is not None
        if source == "Included synthetic 18 BPM demo":
            frames = iter_frames(SAMPLE)
            try:
                first = next(frames)
                preview = first.copy()
                selected = manual or ROI(40, 45, 120, 75)
                cv2.rectangle(
                    preview,
                    (selected.x, selected.y),
                    (selected.x + selected.w, selected.y + selected.h),
                    (0, 200, 180),
                    2,
                )
                st.image(
                    preview,
                    channels="BGR",
                    caption="Synthetic demo · proposed ROI",
                    width=480,
                )
            finally:
                frames.close()
            st.caption(
                "Known reference: 18 BPM. Synthetic engineering demonstration only."
            )
        elif upload is not None:
            st.video(upload.getvalue())
        if st.button("Run analysis", type="primary", disabled=not available):
            try:
                config = PipelineConfig(
                    min_bpm=low,
                    max_bpm=high,
                    motion_compensation=compensate,
                    agreement_bpm=agreement,
                    min_displacement_std=amplitude,
                    min_duration_seconds=min_duration,
                )
                with st.spinner("Extracting motion and checking estimator agreement…"):
                    if source == "Included synthetic 18 BPM demo":
                        result = RespiratoryEstimator(config).analyze(
                            SAMPLE, roi=manual
                        )
                        reference = 18.0
                        provenance = "Included non-clinical synthetic video; original generator, seed 7; known 18 BPM reference."
                    else:
                        with tempfile.TemporaryDirectory(
                            prefix="respiratory-upload-"
                        ) as directory:
                            path = Path(directory) / (
                                "recording" + Path(upload.name).suffix.lower()
                            )
                            path.write_bytes(upload.getvalue())
                            result = RespiratoryEstimator(config).analyze(
                                path, roi=manual
                            )
                        reference = None
                        provenance = "Locally uploaded recording. No independent reference supplied; authorization is the user responsibility."
                st.session_state["analysis"] = result
                st.session_state["reference"] = reference
                st.session_state["provenance"] = provenance
            except ValueError as exc:
                st.error(str(exc))
        if "analysis" in st.session_state:
            result = st.session_state["analysis"]
            st.caption(
                "Last completed analysis. Change settings and run again to refresh this result."
            )
            cols = st.columns(3)
            bpm = result["estimate"]["bpm"]
            cols[0].metric(
                "Accepted respiratory rate",
                f"{bpm:.2f} BPM" if bpm is not None else "Abstained",
            )
            cols[1].metric("Quality state", result["quality"]["state"])
            cols[2].metric("Processing time", f"{result['processing_seconds']:.2f} s")
            if result["quality"]["reasons"]:
                st.info(" ".join(result["quality"]["reasons"]))
            st.dataframe(method_table(result), width="stretch")
            with st.expander("Quality features and exact configuration"):
                st.json(result["quality"])
                st.json(result["config"])
                if result.get("timing"):
                    st.json(result["timing"])
            cols = st.columns(3)
            cols[0].download_button(
                "Download JSON",
                json.dumps(result, indent=2, allow_nan=False),
                "respiratory_result.json",
                "application/json",
            )
            cols[1].download_button(
                "Download signal CSV",
                signal_csv(result),
                "respiratory_signal.csv",
                "text/csv",
            )
            cols[2].download_button(
                "Download HTML report",
                analysis_html(
                    result,
                    st.session_state["reference"],
                    st.session_state["provenance"],
                ),
                "respiratory_report.html",
                "text/html",
            )
        st.caption(
            "Uploads remain in this local process. Temporary video copies are removed after each analysis. Clear the session to discard derived results. Reports may still contain sensitive derived data."
        )
        if st.button("Clear derived analysis"):
            for key in ["analysis", "reference", "provenance"]:
                st.session_state.pop(key, None)
            st.rerun()
    with tabs[1]:
        st.subheader("Inspect the evidence behind the estimate")
        if "analysis" not in st.session_state:
            st.info("Run an analysis in the first tab to inspect its measured signals.")
        else:
            result = st.session_state["analysis"]
            fig = diagnostic_figure(result)
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(method_table(result), width="stretch")
            st.json(result["fusion"], expanded=False)
            st.caption(
                "Cycle markers are candidate peaks. Invalid intervals and method exclusions are preserved in JSON. The three methods share one waveform and are not statistically independent."
            )
    with tabs[2]:
        st.subheader("Reproducible benchmark laboratory")
        st.write(
            "Run four development cases: clean motion, camera movement, near-zero amplitude and combined stress. The larger held-out test uses separate seeds and configurations."
        )
        if st.button("Run demo benchmark"):
            with st.spinner("Generating and evaluating four synthetic recordings…"):
                with tempfile.TemporaryDirectory(
                    prefix="respiratory-demo-"
                ) as directory:
                    summary = run_benchmark(
                        directory,
                        "development",
                        1,
                        ("clean", "camera_motion", "low_amplitude", "combined_stress"),
                    )
                    st.session_state["benchmark"] = summary
                    st.session_state["benchmark_csv"] = (
                        Path(directory) / "results.csv"
                    ).read_text()
        if "benchmark" in st.session_state:
            show_summary(st.session_state["benchmark"])
            st.download_button(
                "Download benchmark CSV",
                st.session_state["benchmark_csv"],
                "benchmark.csv",
                "text/csv",
            )
        else:
            st.info("Click Run demo benchmark to generate measured results locally.")
    with tabs[3]:
        st.subheader("Research findings and next experiments")
        st.write(
            "The pipeline measures apparent vertical motion inside a chosen thoracic ROI. Robust background translation subtraction, spectral analysis, autocorrelation and cycle intervals provide inspectable evidence. The quality gate can withhold a final rate."
        )
        summary_path = ROOT / "outputs/benchmark/summary.json"
        if summary_path.exists():
            st.markdown("**Executed held-out synthetic benchmark**")
            show_summary(json.loads(summary_path.read_text()))
        else:
            st.info(
                "No saved held-out benchmark found. Run scripts/run_full_benchmark.py to populate this section."
            )
        st.markdown(
            "**Limitations:** geometric ROI, translation-only camera model, shared-signal estimator errors, provisional thresholds, short synthetic clips, no authorized real dataset or clinical validation. A periodic non-respiratory movement may still be accepted."
        )
        st.markdown(
            "**Next experiments:** evaluate synchronized authorized reference recordings; split by subject; analyze longer changing-rate sequences; assess rotation and parallax; calibrate rejection thresholds on a separate development cohort."
        )
        st.caption(
            "No asthma prediction or pediatric monitoring claims. Quality scores are engineering diagnostics, not probabilities of correctness."
        )


if __name__ == "__main__":
    main()
