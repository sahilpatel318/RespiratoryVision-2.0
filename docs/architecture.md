# Architecture

This upgrades the provided repository rather than replacing it. `_baseline/` contains byte-for-byte copies of the original estimator, configuration, motion, signal, ROI and video modules. The original synthetic generator and TemporalCNNRegressor are retained. Legacy tests remain in place.

```text
video_io: probe + streaming decoder + timestamp diagnostics
  -> roi: geometric / manual pixels / interactive desktop
  -> motion: Farneback, active-quadrant robust aggregation, O(N) traces
     -> motion_compensation: background LK tracks, forward/backward checks,
        2D coverage, median/MAD-like residual rejection, translation subtraction
  -> signal_processing: detrend, Butterworth, Welch / ACF / cycles
  -> fusion: median only when all eligible estimators agree
  -> quality: explicit states and accepted rate or null
  -> reporting + visualization + CLI + Streamlit

synthetic -> benchmarking -> results/summary/failures/figures/HTML
authorized manifest -> dataset -> same benchmarking evaluator
optional training -> existing model -> metadata checkpoint -> comparator
```

Configuration is immutable and validated. Expected input/decoder failures yield stable structured abstentions. Invalid configuration raises ValueError. Missing dependencies or programmer errors are not swallowed as successful analyses. JSON output contains no input pathname; individual batch exports omit identifiers and source text. Baseline-only output preserves the old schema and path behavior, so do not publish it for identifiable recordings.

Memory: enhanced extraction holds a constant number of frames plus O(N) scalar traces and timing data. Filters/FFT/reporting still operate on whole traces; the system is offline, not a bounded-memory real-time monitor. The frozen baseline intentionally materializes frames for fair reproduction.

The dashboard stores derived results in session state and removes temporary uploaded video copies in a context manager, including on failures. Browser upload bytes remain in Streamlit session memory while the upload widget holds them. Clearing derived analysis removes results; remove the upload or close the session to release the video bytes. No video or participant data are sent to a cloud service.
