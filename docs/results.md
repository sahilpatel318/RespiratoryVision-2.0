# Executed results — 2026-10-25

## Original baseline before modification

All five original tests passed. The original 12/18/24/30 BPM benchmark produced MAE **0.138 BPM**, RMSE **0.166 BPM**, signed bias **−0.126 BPM**. Raw logs and rows: `outputs/baseline/`. The retained original benchmark was rerun after the upgrade.

## Held-out challenge videos

32 generated recordings, eight conditions × four rates (13/19/25/31 BPM), 24 seconds each, 12 FPS, test seed family 51000. These are engineering stress cases, not human participants. Exact configs and thresholds are in `outputs/benchmark/cases.json` and `summary.json`.

| Method | MAE (BPM) | RMSE (BPM) | Predictions | Within ±3 BPM |
|---|---:|---:|---:|---:|
| original_baseline | 2.619 | 5.456 | 32/32 | 75.0% |
| compensated_welch | 1.608 | 3.923 | 32/32 | 84.4% |
| autocorrelation | 2.298 | 4.753 | 32/32 | 75.0% |
| peaks | 0.925 | 1.646 | 32/32 | 90.6% |
| quality_fusion | 0.145 | 0.180 | 20/32 | 100.0% |
| temporal_cnn | 1.169 | 1.954 | 32/32 | 93.8% |

Individual estimators and the CNN are ungated comparators. Their 100% availability does not indicate quality or safety. Fusion's error is conditional on its **62.5% acceptance** (20/32); all 12 rejections are retained in the CSV. All four low-amplitude, four irregular-breathing and four combined-stress clips were rejected; the other five conditions were accepted. These deliberately limited conditions must not be used as evidence of broad clinical performance.

The original-baseline and enhanced algorithms differ in aggregation as well as compensation, so their difference is not a pure compensation-only ablation. A camera-only regression test separately checks that enabling compensation suppresses global motion.

## Matched accepted subset (same 20 clips)

- original_baseline: 1.804 BPM MAE.
- compensated_welch: 0.115 BPM MAE.
- autocorrelation: 0.158 BPM MAE.
- peaks: 0.190 BPM MAE.
- quality_fusion: 0.145 BPM MAE.

Compensated Welch (0.115 BPM) is slightly better than fusion (0.145 BPM) on the same accepted clips. Fusion adds a rejection decision; this experiment does not establish that combining estimators is always more accurate. Its bootstrap 95% MAE interval is approximately 0.098–0.191 BPM, conditional on the accepted generated mixture. It is not a per-recording confidence interval or clinical uncertainty estimate.

Development results (16 clips): original baseline MAE 1.309 BPM; compensated Welch .961; ACF 1.406; peaks .733; fusion .533 at 75% coverage. Final thresholds were not selected against held-out scores. The sample-driven reference-coverage correction is recorded in the audit.

## CNN experiment

The existing CNN trained on 600 generated waveforms, validated on 120, tested on 120 with seeds 7/8/9. Validation selected epoch 12 of 12. On held-out waveforms, CNN MAE was **0.492 BPM** versus Welch **0.146 BPM**. The model was worse than Welch in this experiment. On the independent 32-video challenge grid, the CNN's MAE was **1.169 BPM** (32/32 predictions), worse than the peak comparator's .925 BPM but better than compensated Welch across that full stress mixture. Domains, coverage and stationary-reference definitions differ; do not generalize these rankings.

`outputs/model/temporal_cnn.pt` is an actually trained CPU checkpoint; its JSON contains history, hyperparameters, seeds and all 120 test predictions. Waveform-comparison runtime was not measured and is null. Video inference timing includes extraction. No checkpoint is used by the accepted-rate gate.

## Reproducibility and artifacts

The full 32-case test grid was generated and evaluated twice, once with the optional CNN. All **160 classical method rows matched exactly after excluding wall-clock processing time**; see `outputs/reproducibility.json`. These repeated runs are not independent extra test observations.

- `outputs/benchmark/report.html`: standalone report, figures and failure rows.
- `outputs/benchmark/summary.json`: full metrics, CIs, per-condition and matched-subset scores.
- `outputs/benchmark/results.csv`, `failure_cases.csv`: every available/rejected case.
- `outputs/benchmark_cnn/`: optional neural comparison on the same generated test grid.
- `outputs/sample_report.html`: original 18 BPM sample analysis (accepted 18.182 BPM).
- `outputs/fixture/`: four supplied synthetic samples processed through the real-recording manifest adapter, explicitly labeled synthetic_fixture.

No real or clinical performance was measured. The manifest adapter's functionality and grouping were tested with safe fixtures only.

Historical outputs such as sample_result.json, test_12_result.json and their signal plots were supplied in the original ZIP and are retained unchanged. They use the original schema (including the uncalibrated confidence field). Use sample_v2.json/sample_report.html and the benchmark folders for the upgraded results.
