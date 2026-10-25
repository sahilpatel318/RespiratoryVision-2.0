# Experimental protocol

## Baseline and independent challenges

First execute the five unmodified tests and four-rate original benchmark; logs are retained under outputs/baseline. The original generator remains deterministic and unchanged. Expanded challenge videos use textured stationary backgrounds and a subpixel translated torso texture, separate from that legacy generator.

Development seeds start at 11000, test seeds at 51000; add 100×condition index + within-condition index. Up to 50 examples per condition avoids overlapping blocks. Development rates cycle 12/18/24/30 BPM, test rates 13/19/25/31 BPM. Development uses 10 FPS, phase .13; test uses 12 FPS, phase .71 and 1.2×severity. Default duration is 24 seconds. Each case stores full generating settings and the time-average ground truth. Videos also have JSON sidecars with per-frame true phase rate, respiratory displacement and camera translation before temporary cleanup.

| Condition | Simulation |
|---|---|
| clean | 4-pixel sinusoidal vertical torso displacement |
| mild_noise | Independent Gaussian pixel noise SD 4×severity |
| camera_motion | Whole-frame sinusoidal x/y translation, vertical amplitude 7×severity pixels, .23 Hz |
| low_amplitude | .025×severity pixel respiratory amplitude |
| illumination_changes | Global additive sinusoidal brightness, amplitude 35×severity, .17 Hz |
| occlusion | 45% torso width covered during 35–65% of clip |
| irregular_breathing | 30% sinusoidal phase-rate modulation, +6×severity BPM linear change, 8% duration pause |
| combined_stress | Noise SD 7×severity, camera amplitude 9×severity, illumination 30, 35% occlusion, 12-pixel 1.4-Hz local artifact during 32–45% of clip, 20% rate modulation |

For changing/paused breathing, reference is the duration-average phase rate including pauses. This summary is not identical to a dominant periodic frequency or median cycle rate. Large errors there can reflect an unsuitable scalar target as well as extraction failure. No claim is made that these perturbations model clinical physiology.

Use development examples to investigate methods; freeze config before the final test run. A false background-track bug was identified from the supplied original sample plot and fixed with a reference-coverage guard before final reporting; the superseded held-out output was not inspected for tuning. No final test thresholds were fitted. Numerical results repeat with identical versions/seeds; codec implementations and floating-point versions can alter results. Runtime will vary. Smoke fixtures and repeated reproducibility runs are tests, not additional independent accuracy samples.

## Metrics and interpretation

Every clip produces one row per comparator, including failures/abstentions. Original baseline and individual methods are ungated comparators: their prediction availability is not a quality guarantee. METHOD_FLAGGED records a method estimate that fails its eligibility checks. Fusion acceptance is a distinct outcome. Compare conditional error together with availability and on the matched fusion-accepted subset.

MAE, RMSE, signed error, median absolute error, error quantiles and ±2/±3 BPM proportions use available estimates. Rejections remain in n_total and rejection rate. Failure CSV includes null estimates, >3-BPM errors, and flagged individual estimates. Reports include full per-condition summaries and method timing. Enhanced methods share one extraction and processing-time measurement, so those times must not be added.

95% MAE intervals use 1,000 seeded percentile bootstrap draws of independent recording clusters (synthetic) or subject clusters (real adapter), only with >=8 available clusters. These are descriptive intervals for this synthetic mixture, conditional on acceptance. They do not calibrate a prediction or quantify clinical uncertainty. Per-condition test samples are too few for reported intervals. Unknown real subject IDs share a cluster to avoid assuming independence.

## Real data and leakage

No authorized human dataset was provided. Adapter tests use generated videos only. Manifest subject IDs are used internally for group splits and bootstrap clusters, never copied into public CSV/JSON. A subject cannot cross development/test boundaries. An all-recordings evaluation is not a held-out study. Independent real validation requires matched timing, valid references, permission and participant-level separation. Duplicate resolved video paths are rejected; this does not detect copies renamed to different files or incorrect subject metadata.

## Neural experiment

The unchanged CNN is trained only on generated stationary waveforms with harmonic, drift and noise. Train/validation/test seeds are 7/8/9 by default. Validation chooses the checkpoint; test data are generated after training selection. Training is CPU deterministic, with explicit hyperparameters, early stopping and history. Inputs are 300 samples at 15 Hz after the same 6–48 BPM detrend/filter/zscore procedure. Video comparison uses the first 20 seconds of the filtered trace, resampled to 15 Hz and normalized; boundary and generator domain shifts remain. No clinical model exists. A trained checkpoint is required to include CNN video predictions.
