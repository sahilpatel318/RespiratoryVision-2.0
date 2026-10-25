# Research summary

RespiratoryVision investigates whether ordinary RGB video can support interpretable estimation of periodic thoracic motion. The supplied prototype was extended in place into a local research platform with streaming extraction, robust reference-motion subtraction, multiple estimators, explicit abstention, reproducible experiments and an inspectable dashboard.

Optical flow estimates apparent local image displacement; integrating its vertical component produces a displacement-like signal. This is useful for a controlled torso-motion model, but camera movement produces the same observable displacement and motivates separate background references. Sparse or clustered reference features can be misleading, so the implementation checks coverage and reports fallback.

Welch PSD, autocorrelation and cycle intervals examine different properties of the same waveform. Their agreement can support an engineering gate, but they are not independent tests of physiology. Harmonics, periodic artifacts and short recordings can confound all three. Thresholds remain provisional.

The original five tests and four-rate benchmark were run before changes. The final 32-clip synthetic challenge grid produced 2.619 BPM baseline MAE. Fusion accepted 20 clips with .145 BPM conditional MAE and rejected 12. On those same 20 clips, compensated Welch achieved .115 BPM, slightly better than fusion. Thus abstention is a substantive behavior, not proof that fusion universally improves numerical estimates. Full classical outputs repeated exactly across 160 matched rows in a second run, excluding timing.

The original CNN architecture was trained with separated train/validation/test seeds and validation-selected checkpointing. On 120 held-out synthetic waveforms its .492 BPM MAE was worse than Welch's .146 BPM. On generated video stress cases it achieved 1.169 BPM MAE, demonstrating that comparisons depend on the evaluation domain.

Patient-level leakage matters because recordings/windows from one person can share appearance, movement, acquisition context and physiology. The manifest adapter keeps subjects disjoint in development/test partitions and uses subject clusters for interval calculations, while omitting their IDs from public reports. Correct source metadata and genuinely independent authorized recordings remain necessary.

The project demonstrates software integration, controlled signal recovery, artifact stress testing, explicit failure handling and reproducibility. It does not establish clinical performance, pediatric applicability, disease detection or safe monitoring. Next work should use authorized synchronized reference recordings, participant-level splits, separate calibration data, longer time-varying analysis and evaluation of camera rotation/parallax and framing variation.

## Evidence-based resume bullets

- Extended an existing Python/OpenCV respiratory-motion prototype with streaming extraction, background translation compensation, Welch/autocorrelation/cycle estimators and explicit quality-based abstention.
- Built a reproducible 32-video, eight-condition synthetic evaluation framework with coverage-aware metrics, failure exports, subject-disjoint dataset splits and exact reproduction of 160 classical result rows across two runs.
- Integrated a four-tab local Streamlit research dashboard, standalone HTML/JSON/CSV reporting and a deterministic CPU PyTorch comparison using independent training, validation and test signals.
