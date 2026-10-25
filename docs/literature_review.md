# Verified methodological references

This is a focused reference list, not a systematic review. References were checked during implementation; they do not validate this repository.

1. [OpenCV optical flow documentation](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html) describes sparse Lucas–Kanade tracking and dense Farneback optical flow, including brightness-constancy and local-motion assumptions. These are the implemented motion primitives. Illumination and independent movement can violate their assumptions.
2. [SciPy Welch documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html) specifies segment averaging, overlap and zero-padding. The implemented power spectrum follows this API; zero-padding is not additional temporal evidence.
3. Bae, S., Borac, S., Emre, Y. et al. (2022). [Prospective validation of smartphone-based heart rate and respiratory rate measurement algorithms](https://www.nature.com/articles/s43856-022-00102-x). Communications Medicine 2, 40. DOI: 10.1038/s43856-022-00102-x. This is an example of independent prospective reference measurement and governed human video research. Its algorithm, participants and performance are not those of RespiratoryVision and its results cannot be transferred to this prototype.

Future research should compare synchronized reference measurements, hold out participants rather than windows, report movement-dependent failures, and justify any thresholds on separate development data. The original archive's broader literature list has been replaced with these directly verified sources rather than retaining unchecked claims.
