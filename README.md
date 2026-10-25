# RespiratoryVision 2.0 — Level 3 research platform

An in-place extension of the supplied RespiratoryVision Python prototype. The original package, samples, deterministic generator, CNN architecture and commands remain recognizable. The original estimator and its dependencies are frozen under `src/respiratory_vision/_baseline/` for reproducible comparison.

**Research/education only. Not for diagnosis, medical decisions, asthma prediction, or pediatric clinical monitoring. Synthetic recovery is not clinical validation.**

## Install (Python 3.11+, CPU, Windows PowerShell)

Open a terminal in this repository folder. With Python on PATH:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

If `python` is unavailable, use `py -3.12 -m venv .venv`, or the full path to an installed Python executable. Activation is optional; the explicit executable paths avoid PowerShell execution-policy changes. On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. Only Windows/Python 3.12 was exercised here. OpenCV codec availability varies by platform.

For the existing Codex workspace, the already installed interpreter is `..\..\work\.venv\Scripts\python.exe` relative to this repository. It uses existing system NumPy/SciPy; a fresh environment is recommended for independent reproduction. `requirements-tested.txt` records the exact dependency versions used. Optional CNN dependencies are separate:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
```

## Run the included demonstration

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --video data/sample/synthetic_breathing_18bpm.mp4 --reference-bpm 18
.\.venv\Scripts\python.exe -m streamlit run app.py
```

With an activated environment, the equivalent dashboard command is `streamlit run app.py`. It binds to `127.0.0.1:8501`, uses no accounts/API keys and disables usage telemetry. Choose **Included synthetic 18 BPM demo**, then **Run analysis**. Four tabs provide analysis/export, signal investigation, a four-case development benchmark, and actual saved research findings. The upload limit is 200 MB; use the CLI for larger recordings. Browser playback depends on codec support; the packaged MPEG-4 demo has an OpenCV frame preview.

Default CLI outputs are `outputs/result.json`, `outputs/signal.png`, `outputs/report.html`, plus report JSON/CSV. `estimate.bpm` is JSON null on abstention. Method estimates and exclusions remain available. Quality states are `ACCEPTED`, `LOW_QUALITY`, `INSUFFICIENT_DATA`, `MOTION_ARTIFACT`, `ESTIMATOR_DISAGREEMENT`, and `UNSUPPORTED_INPUT`. No probability of correctness is claimed.

## ROI, compensation and compatibility

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --video data/sample/synthetic_breathing_18bpm.mp4 --roi 40 45 120 75 --min-bpm 6 --max-bpm 48
.\.venv\Scripts\python.exe scripts/run_pipeline.py --video data/sample/synthetic_breathing_18bpm.mp4 --interactive-roi
.\.venv\Scripts\python.exe scripts/run_pipeline.py --video data/sample/synthetic_breathing_18bpm.mp4 --no-compensation
.\.venv\Scripts\python.exe scripts/run_pipeline.py --video data/sample/synthetic_breathing_18bpm.mp4 --baseline --output outputs/original_result.json
.\.venv\Scripts\python.exe scripts/evaluate_synthetic.py --output outputs/original_four_rate_benchmark.csv
.\.venv\Scripts\python.exe scripts/generate_synthetic_video.py --output outputs/demo_24.mp4 --bpm 24 --duration 24 --fps 15
```

Interactive selection requires a graphical desktop and GUI-enabled OpenCV; it cannot be automated in a headless terminal. No automatic anatomy detector is claimed: the explicit fallback is the existing geometric ROI or a manual box. The enhanced reader streams frames and retains O(N) traces, not O(N × frame size) images. The frozen original baseline retains its original memory behavior and should be used on bounded clips.

All `PipelineConfig` quality thresholds can be supplied through `--config settings.json`. A full exported result is also accepted as a config source; supply its absolute manual ROI separately with `--roi` if used. Config defaults and rationale are in [methodology](docs/methodology.md). The original `RateEstimate.confidence` field remains in the legacy API as spectral peak share; the enhanced result omits that misleading field.

## Reproduce executed experiments

```powershell
.\.venv\Scripts\python.exe scripts/run_full_benchmark.py --split development --per-condition 2 --output outputs/development
.\.venv\Scripts\python.exe scripts/run_full_benchmark.py --split test --output outputs/benchmark
```

The default test grid has **32 clips: 8 stress conditions × 4 rates**. It compares the exact original baseline, enhanced compensated Welch, autocorrelation, cycle peaks and quality-gated fusion. Expect a few minutes on CPU. Conditions: clean, mild noise, camera motion, low amplitude, illumination changes, occlusion, irregular breathing and combined stress. Generated videos live in temporary directories and are removed after evaluation.

Outputs: `results.csv`, `summary.json`, `cases.json`, `failure_cases.csv`, `method_comparison.png`, `error_distribution.png`, `quality_by_condition.png`, and a standalone `report.html`. Per-case generation settings and seeds are retained. Metrics include MAE, RMSE, signed bias, median absolute error, ±2/±3 BPM, rejection rates, condition breakdown, error quantiles and timing. Errors are conditional on prediction availability; compare `matched_fusion_subset` as well as overall scores. A rejected clip never disappears from the denominator. Timing is hardware dependent and excluded from numerical reproducibility assertions.

See [actual results](docs/results.md), [test evidence](VALIDATION.md), and [experimental protocol](docs/experimental_protocol.md). No authorized real dataset was supplied. All shipped data are synthetic.

## Authorized real-recording interface

CSV columns (paths relative to the manifest location are supported):

```csv
recording_id,subject_id,video_path,reference_bpm,condition,source
local001,subject001,recordings/example.mp4,18,rest,authorized-study
```

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_dataset.py --manifest C:/authorized/manifest.csv --partition test --output outputs/real
.\.venv\Scripts\python.exe scripts/evaluate_dataset.py --manifest data/sample/manifest.csv --kind synthetic_fixture --output outputs/fixture
```

`--partition test` / `development` uses deterministic subject-disjoint splitting. Every subject ID must be present and at least two subjects are required. `--partition all` evaluates supplied labels without creating a split. Duplicate IDs, duplicate resolved video paths, missing files, invalid labels and schema errors are rejected. Scalar references must describe the same recording interval; the adapter cannot verify synchronization or consent. Public reports replace recording and condition labels with sequential codes and omit subject IDs, source free text and paths. Unknown subjects share one bootstrap cluster, suppressing unjustified independent-recording intervals. Derived outputs are not guaranteed anonymous. Keep real recordings and reports outside version control.

## Optional temporal CNN

The original `TemporalCNNRegressor` architecture is retained. The training workflow adds seeds, disjoint waveform train/validation/test sets, validation-selected checkpoints, early stopping, history, metadata and classical comparisons.

```powershell
.\.venv\Scripts\python.exe scripts/train_signal_model.py --epochs 12 --patience 4 --output outputs/model/temporal_cnn.pt
.\.venv\Scripts\python.exe scripts/run_full_benchmark.py --split test --checkpoint outputs/model/temporal_cnn.pt --output outputs/benchmark_cnn
```

Only pass a trusted checkpoint generated by this project. Loading uses CPU and `weights_only=True`; schema and architecture are checked. Inference needs at least the trained 20-second duration, resamples to 15 Hz, and never stretches a short clip. The CNN is an ungated research comparator and does not control accepted video estimates. Synthetic waveform training does not establish superiority or clinical generalization. See actual training results in `outputs/model/temporal_cnn.json` when training has executed.

## Architecture and documentation

`estimator.py` orchestrates validated video → ROI → streaming Farneback and background tracking → displacement → band-pass → three estimators → fusion → quality gate. `benchmarking.py`, `dataset.py`, `reporting.py` and `app.py` all use the same estimator. No disconnected UI mock data are used.

- [Repository audit](docs/repository_audit.md) and [file changes](docs/changes.md)
- [Architecture](docs/architecture.md) and [methodology](docs/methodology.md)
- [Protocol](docs/experimental_protocol.md), [results](docs/results.md), [limitations](docs/limitations.md)
- [Ethics/privacy](docs/ethics_privacy.md), [verified references](docs/literature_review.md), [research summary](docs/research_summary.md)

MIT license; original copyright preserved. See LICENSE.
