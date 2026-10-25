"""Optional CPU PyTorch experiment with independent synthetic waveform splits."""

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .benchmarking import score_rows
from .config import PipelineConfig
from .fusion import fuse_estimates
from .model import TemporalCNNRegressor
from .signal_processing import analyze_methods, clean_respiratory_signal


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 12
    patience: int = 4
    batch_size: int = 64
    learning_rate: float = 0.001
    train_size: int = 600
    validation_size: int = 120
    test_size: int = 120
    samples: int = 300
    fps: float = 15.0
    seed: int = 7

    def validate(self):
        if (
            min(
                self.epochs,
                self.patience,
                self.batch_size,
                self.train_size,
                self.validation_size,
                self.test_size,
            )
            < 1
            or self.samples < 120
            or self.fps <= 1.6
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("Invalid training configuration.")


def make_dataset(n: int, samples: int, fps: float, seed: int):
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / fps
    x = np.zeros((n, 1, samples), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        bpm = rng.uniform(8, 36)
        phase = rng.uniform(0, 2 * np.pi)
        raw = (
            np.sin(2 * np.pi * bpm / 60 * t + phase)
            + 0.2 * np.sin(4 * np.pi * bpm / 60 * t + phase / 2)
            + 0.15 * np.sin(2 * np.pi * rng.uniform(0.01, 0.05) * t)
            + rng.normal(0, 0.18, samples)
        )
        x[i, 0] = clean_respiratory_signal(raw, fps)
        y[i] = bpm
    return torch.from_numpy(x), torch.from_numpy(y)


def train_model(config: TrainingConfig, output: str | Path) -> dict:
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    x_train, y_train = make_dataset(
        config.train_size, config.samples, config.fps, config.seed
    )
    x_val, y_val = make_dataset(
        config.validation_size, config.samples, config.fps, config.seed + 1
    )
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = TemporalCNNRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.L1Loss()
    best = float("inf")
    stale = 0
    history = []
    best_state = None
    best_epoch = 0
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append((loss.item(), len(yb)))
        model.eval()
        with torch.no_grad():
            val = float(torch.mean(abs(model(x_val) - y_val)))
        history.append(
            {
                "epoch": epoch + 1,
                "training_mae": sum(v * n for v, n in losses) / len(y_train),
                "validation_mae": val,
            }
        )
        if val < best:
            best = val
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "architecture": "TemporalCNNRegressor",
        "state_dict": best_state,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "preprocessing": {
            "method": "Butterworth 6-48 BPM, linear detrend, per-window zscore",
            "fps": config.fps,
            "samples": config.samples,
        },
        "split_seeds": {
            "training": config.seed,
            "validation": config.seed + 1,
            "test": config.seed + 2,
        },
        "provenance": "synthetic waveforms only; no human or video training data",
        "clinical_use": False,
    }
    torch.save(checkpoint, output)
    # Test examples are generated only after model/epoch selection is finished.
    x_test, y_test = make_dataset(
        config.test_size, config.samples, config.fps, config.seed + 2
    )
    with torch.no_grad():
        predictions = model(x_test).numpy()
    rows = []
    for index, (x, y, pred) in enumerate(
        zip(x_test.numpy(), y_test.numpy(), predictions)
    ):
        methods = analyze_methods(x[0], config.fps, PipelineConfig())
        estimates = {k: v["bpm"] for k, v in methods.items()}
        estimates["agreement_fusion"] = fuse_estimates(methods)["bpm"]
        estimates["temporal_cnn"] = float(pred)
        for name, estimate in estimates.items():
            rows.append(
                {
                    "case_id": f"test-{index:04d}",
                    "method": name,
                    "reference_bpm": float(y),
                    "estimated_bpm": estimate,
                    "processing_seconds": None,
                }
            )
    comparison = {
        name: score_rows([r for r in rows if r["method"] == name]) for name in estimates
    }
    report = {
        "config": asdict(config),
        "history": history,
        "best_epoch": best_epoch,
        "split_seeds": checkpoint["split_seeds"],
        "test_comparison": comparison,
        "test_predictions": rows,
        "limitations": "Independent synthetic stationary waveforms, not clinical or independent real-video validation. Runtime is not measured in waveform comparison. agreement_fusion is signal-only, not the video quality gate.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def load_checkpoint(path):
    torch.set_num_threads(2)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("architecture") != "TemporalCNNRegressor"
    ):
        raise ValueError("Unsupported checkpoint schema or architecture.")
    model = TemporalCNNRegressor()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_checkpoint(path, filtered_signal, fps):
    model, checkpoint = load_checkpoint(path)
    meta = checkpoint["preprocessing"]
    samples = int(meta["samples"])
    target_fps = float(meta["fps"])
    x = np.asarray(filtered_signal, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid model input.")
    if len(x) / fps < samples / target_fps or np.std(x) < 1e-10:
        return None
    # Fixed physical duration and rate, never stretch a short clip to the training length.
    resampled = np.interp(np.arange(samples) / target_fps, np.arange(len(x)) / fps, x)
    resampled = (resampled - resampled.mean()) / (resampled.std() + 1e-12)
    with torch.no_grad():
        return float(
            model(torch.from_numpy(resampled.astype(np.float32))[None, None, :]).item()
        )
