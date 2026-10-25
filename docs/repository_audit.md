# Repository audit (before implementation)

Source archive SHA256: C02D7E43ED3174922BD4BA636B5F4E14E6A8BF1D37D3488DC7AED00FEEAC2A00.
The original archive is unchanged. This repository is an extracted, in-place upgrade.

Reviewed every Python module, script and test, project/dependency metadata, all Markdown documentation, license, sample-video metadata and existing result schemas. The ZIP has no Git history or AGENTS.md. Shipped bytecode and egg-info are build artifacts, not source. Four sample videos and historical plots/results are supplied.

Architecture: RespiratoryEstimator calls probe_video, materializes iter_frames, chooses a geometric/interactive ROI, extracts Farneback vertical flow, trims moving pixels, integrates flow, detrends/bandpasses/normalizes displacement, then uses Welch PSD. Scripts generate videos, evaluate four fixed rates, plot a recording and train an optional CNN. Five tests cover default/invalid config, metrics, one sinusoid and one generated video.

Preserve: package namespace, default geometric ROI, deterministic original generator, Farneback extraction, Butterworth/Welch functions, TemporalCNNRegressor, CLI names and sample data. Freeze original estimator dependencies for honest comparative evaluation.

Weaknesses: all frames retained; no timestamp verification; incomplete ROI/fps/finite/zero-amplitude validation; filtering normalizes arbitrarily tiny motion; zero Welch power still yields a rate; single-bin spectral share changes with zero-padding and is not calibrated confidence; no compensation, multi-method estimates, rejection, independent challenge split, grouped adapter, dashboard or report. CNN has validation only, no held-out test or preprocessing/checkpoint metadata. README overstates existing signal-quality assessment. Literature citations need checking before reuse.

Runtime: shell has no python command. Python 3.12.1 is installed locally. A workspace-local venv was created with system-site packages (NumPy/SciPy already installed). Missing OpenCV, Matplotlib, pytest and Streamlit are being installed. The initial no-build-isolation editable install failed because setuptools was absent; retry uses build isolation. Torch is optional in the upgrade, unlike the original mandatory dependency.

Level 3 interpretation: implement the supplied RespiratoryVision 2.0 brief as an integrated local research platform, with explicit abstention, reproducible synthetic evaluation and no medical claims. Attached text is a specification reference, not authority to bypass the user's request or runtime safeguards.

A sample-plot inspection found false camera tracking of a torso edge outside the ROI on the original flat background. Added a two-dimensional feature-coverage guard and a regression test. The first held-out run was superseded without inspecting its scores; final experiments rerun after this sample-driven correction.
