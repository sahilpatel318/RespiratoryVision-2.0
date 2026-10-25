# Known limitations

- No human, pediatric, asthma or clinical validation. No diagnosis or exacerbation prediction.
- Default/manual image ROI is fixed, not anatomical segmentation or robust subject tracking. Interactive selection requires a desktop.
- Compensation models translation only. Rotation, parallax, rolling shutter, moving background objects and body movements can corrupt it. Insufficient or spatially clustered features cause explicit fallback. Fallback is diagnostic, not proof that camera movement is absent.
- Periodic non-respiratory motion can fool all three estimators. They share an input signal and their agreement is not independent statistical evidence.
- Engineering quality thresholds are provisional, spatial-resolution dependent and not optimized or calibrated against clinically annotated data. High-quality classification is not a safety guarantee.
- Low rates require longer clips; default 8 seconds is only a lower bound, with a separate estimated-cycle gate. Subtle or paused breathing can cause abstention without indicating disease.
- Whole-clip stationarity is assumed. Changing rates, pauses and irregular timing need future sliding-window characterization. Their mean phase-rate reference can differ from a spectral or median-cycle estimate.
- Some decoders do not supply timestamps. The system explicitly records a nominal-FPS assumption in that case. Variable-rate streams are rejected; resampling such video is not implemented.
- Enhanced video extraction streams frames, but signals/PSD/reporting are offline and use O(N) trace memory. Original baseline still stores decoded frames.
- Synthetic perturbations are simple engineering stressors. Small fixed grids and bootstrap intervals are not population-wide performance estimates.
- CNN training uses synthetic waveforms only. It may be worse than classical methods and is never substituted into the accepted-rate decision.
- Local codec availability varies. OpenCV-written MP4V may not play in every browser; the demo also provides an image preview. Browser uploads have a 200 MB limit.
- Local Streamlit startup and AppTest analysis were verified. A live browser connection in the managed execution environment timed out; visual browser verification there remains limited. No network/security settings were changed.
