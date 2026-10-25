# File changes relative to the supplied ZIP

This is an extracted in-place upgrade. The original ZIP remains unchanged. Generated results and the synthetic manifest are described separately below.

## Modified source/documentation

- `docs/ethics_privacy.md`
- `docs/literature_review.md`
- `docs/methodology.md`
- `docs/results.md`
- `pyproject.toml`
- `README.md`
- `requirements.txt`
- `scripts/evaluate_synthetic.py`
- `scripts/generate_synthetic_video.py`
- `scripts/run_pipeline.py`
- `scripts/train_signal_model.py`
- `src/respiratory_vision/__init__.py`
- `src/respiratory_vision/config.py`
- `src/respiratory_vision/estimator.py`
- `src/respiratory_vision/metrics.py`
- `src/respiratory_vision/motion.py`
- `src/respiratory_vision/roi.py`
- `src/respiratory_vision/signal_processing.py`
- `src/respiratory_vision/synthetic.py`
- `src/respiratory_vision/video_io.py`
- `tests/test_config.py`
- `tests/test_signal_processing.py`
- `VALIDATION.md`

## Added source/documentation

- `.gitignore`
- `.streamlit/config.toml`
- `app.py`
- `docs/architecture.md`
- `docs/experimental_protocol.md`
- `docs/limitations.md`
- `docs/repository_audit.md`
- `docs/research_summary.md`
- `requirements-ml-tested.txt`
- `requirements-tested.txt`
- `scripts/evaluate_dataset.py`
- `scripts/run_full_benchmark.py`
- `src/respiratory_vision/_baseline/__init__.py`
- `src/respiratory_vision/_baseline/config.py`
- `src/respiratory_vision/_baseline/estimator.py`
- `src/respiratory_vision/_baseline/motion.py`
- `src/respiratory_vision/_baseline/roi.py`
- `src/respiratory_vision/_baseline/signal_processing.py`
- `src/respiratory_vision/_baseline/video_io.py`
- `src/respiratory_vision/benchmarking.py`
- `src/respiratory_vision/dataset.py`
- `src/respiratory_vision/errors.py`
- `src/respiratory_vision/fusion.py`
- `src/respiratory_vision/motion_compensation.py`
- `src/respiratory_vision/quality.py`
- `src/respiratory_vision/reporting.py`
- `src/respiratory_vision/training.py`
- `src/respiratory_vision/visualization.py`
- `tests/test_advanced_signal.py`
- `tests/test_app_and_reporting.py`
- `tests/test_benchmark_integration.py`
- `tests/test_dataset.py`
- `tests/test_motion.py`
- `tests/test_quality_pipeline.py`
- `tests/test_training.py`

## Additional deliverables

- `data/sample/manifest.csv`: safe adapter fixture using the four supplied synthetic videos.
- `outputs/baseline/`: pre-change evidence.
- `outputs/benchmark/`, `outputs/development/`, `outputs/benchmark_cnn/`: actually executed comparison data, plots and reports.
- `outputs/model/`: trained checkpoint and waveform evaluation/history.
- `outputs/fixture/`: manifest evaluation outputs.
- `outputs/sample_v2.*`, `sample_report.*`: enhanced sample diagnostics/export.
- `outputs/test_results.txt`, `reproducibility.json`: test and repeatability evidence.

Original sample media, license and historical result files are retained. The original CNN class is unchanged. `_baseline/` files match their original counterparts byte-for-byte. Cache files and egg-info are excluded from the delivery archive.
