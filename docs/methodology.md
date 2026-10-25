# Methodology and provisional thresholds

The research question is whether interpretable motion features recover known periodic movement under controlled video perturbations. No physiological volume, respiratory disease or clinical diagnosis is inferred.

## Motion

Default fractional ROI is x=.20, y=.30, w=.60, h=.50. Manual boxes must be at least 4×4 pixels within the decoded frame. Automatic anatomy detection is unavailable. Correct framing remains the operator responsibility.

Grayscale frames are blurred with a 5×5 Gaussian. Farneback retains the original settings: scale .5, 3 pyramid levels, window 15, 3 iterations, polynomial neighborhood 5, sigma 1.2. Background Shi–Tomasi corners exclude the ROI plus an 8-pixel guard. Lucas–Kanade tracks must pass a <1-pixel forward/backward error check. Require >=6 tracks, reference spans >=35% of frame width AND height, >=60% residual inliers and >=6 inliers. Translation is the coordinate-wise median after removing residuals beyond max(.5 pixels, 3×median residual). These fixed engineering constants are explicitly exposed here; configurable decision thresholds live in PipelineConfig. Failure means zero compensation and a recorded fallback, not assumed successful stabilization.

Subtract translation from dense vectors before aggregation. Split the ROI into four quadrants. Within each, select vectors above the 60th magnitude percentile and 1e-5 pixels, trim vertical 10th/90th percentiles, and average. The frame's scalar is the median of active (>1e-4 pixel) quadrant values. Integrate displacement. Quality features include median pairwise quadrant-displacement correlation, high-motion fraction, background success fraction and camera RMS. Quadrant coherence is a heuristic and can be confounded by shared drift. Fixed image ROI does not follow large subject movement.

## Signal estimates

Linear detrend and fourth-order zero-phase Butterworth band-pass, default 6–48 BPM, followed by unit-SD normalization. Amplitude quality is measured **before normalization** so near-zero motion cannot acquire credibility merely by rescaling. The original short-signal causal fallback remains in the utility API; the default video duration gate avoids it. Sampling passband must be strictly below Nyquist. Accepted FPS range is 2–240. Reported decoder timestamps must be finite, increasing and within 20% of nominal frame interval. All-zero unavailable timestamps use nominal FPS with an explicit warning. Variable-rate input is rejected, not silently treated as uniform. Frame-count mismatch >1 is rejected.

Welch preserves the original up-to-12-second Hann segments and zero-padding. Peak location is constrained to the configured band. Zero-padding interpolates the spectrum; it does not create frequency resolution. Concentration integrates a ±1/12 Hz neighborhood around the selected peak. Peaks near a 2:1 ratio with >=35% of the strongest in-band power flag harmonic ambiguity. Both-half dominant frequencies diagnose changing rate (clips >=16 s). ACF uses biased zero-lag normalization and lag peaks within the respiratory range, capped at half the recording. The earliest peak within 90% of the best eligible correlation reduces selection of repeated-cycle multiples; it cannot always disambiguate harmonics. Cycle detection uses minimum peak separation from max BPM and prominence .5×waveform SD. At least two in-band intervals are required. Every rejected interval is retained in diagnostics.

Fusion requires >=2 eligible methods and agreement of **all** eligible estimates within 3 BPM. Its candidate is the median. It does not average incompatible estimates. These estimators share a waveform; their agreement is not independent evidence or a calibrated probability.

## Quality gates (PipelineConfig)

| Gate | Default | Interpretation |
|---|---:|---|
| Minimum duration | 8 s | Minimum decoded window |
| Minimum cycles | 2.5 | Estimated cycles in the analyzed duration |
| Displacement SD | .08 pixels | Linear-detrended amplitude before normalization |
| Spectral concentration | .45 | Welch eligibility, fixed-width power share |
| ACF peak | .35 | Autocorrelation eligibility |
| Interval CV | .35 | Peak-method eligibility |
| Half-window rate change | 6 BPM | Reject one stationary rate for strong change |
| Method spread | 3 BPM | Agreement tolerance |
| High-motion speed | 80 pixels/s | Marks an artifact frame |
| Artifact frame fraction | .10 | Motion-artifact rejection |
| ROI consistency | .10 | Median quadrant correlation |
| Timestamp tolerance | .20 | Fractional deviation from nominal interval |

Priority: insufficient duration; excessive motion; low amplitude; inconsistent subregions; rate instability; spectral harmonic ambiguity; method disagreement; insufficient eligible methods; insufficient estimated cycles; accepted. Input failures have UNSUPPORTED_INPUT/INSUFFICIENT_DATA states. Eligibility alone does not guarantee validity. All thresholds are provisional, resolution/view-dependent engineering choices, not learned clinical thresholds. Configure via API or CLI JSON. Missing textured background can leave camera contamination undetected; inspect fallback coverage.

The historical `confidence` value is retained only for compatibility in the legacy RateEstimate structure and is a single-bin PSD share. It is never a probability of correctness.
