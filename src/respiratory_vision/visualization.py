"""Figures use pipeline diagnostics directly; no invented or illustrative traces."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def diagnostic_figure(result: dict):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    t = np.asarray(result["timestamps_seconds"])
    axes[0, 0].plot(t, result["raw_displacement"], color="#087e8b")
    axes[0, 0].set(
        title="Integrated local motion", xlabel="Seconds", ylabel="Pixels (relative)"
    )
    axes[0, 1].plot(t, result["filtered_signal"], color="#087e8b")
    indices = (
        result["methods"]
        .get("peaks", {})
        .get("diagnostics", {})
        .get("peak_indices", [])
    )
    if len(indices):
        axes[0, 1].scatter(
            t[indices],
            np.asarray(result["filtered_signal"])[indices],
            color="#d77628",
            s=25,
            label="Candidate cycles",
        )
        axes[0, 1].legend()
    axes[0, 1].set(
        title="Band-pass waveform and candidate peaks",
        xlabel="Seconds",
        ylabel="Standard deviations",
    )
    w = result["methods"].get("welch", {}).get("diagnostics", {})
    if "frequencies_hz" in w:
        axes[1, 0].plot(np.asarray(w["frequencies_hz"]) * 60, w["psd"])
        axes[1, 0].axvline(w["dominant_hz"] * 60, color="#d77628", linestyle="--")
    axes[1, 0].set(
        title="Welch power spectral density",
        xlabel="BPM",
        ylabel="PSD",
        xlim=(0, result["config"]["max_bpm"] * 1.2),
    )
    a = result["methods"].get("autocorrelation", {}).get("diagnostics", {})
    if "correlation" in a:
        axes[1, 1].plot(a["lags_seconds"], a["correlation"])
        if "selected_lag" in a:
            axes[1, 1].axvline(
                a["selected_lag"] / result["video"]["fps"],
                color="#d77628",
                linestyle="--",
            )
    axes[1, 1].set(
        title="Autocorrelation", xlabel="Lag (seconds)", ylabel="Normalized correlation"
    )
    methods = {k: v for k, v in result["methods"].items() if v["bpm"] is not None}
    axes[2, 0].bar(
        list(methods),
        [v["bpm"] for v in methods.values()],
        color=["#087e8b" if v["reliable"] else "#d77628" for v in methods.values()],
    )
    axes[2, 0].set(title="Method estimates (orange = flagged)", ylabel="BPM")
    camera = np.asarray(result.get("global_translation_pixels", []))
    if camera.size:
        axes[2, 1].plot(t, camera[:, 1], label="Estimated vertical camera translation")
        axes[2, 1].legend(fontsize=8)
    axes[2, 1].set(title="Reference motion", xlabel="Seconds", ylabel="Pixels / frame")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("RespiratoryVision | " + result["quality"]["state"], fontweight="bold")
    fig.tight_layout()
    return fig
