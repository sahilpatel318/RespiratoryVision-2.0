"""Run optional native ML checks in child processes, isolated from Streamlit threads."""

import importlib.util
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="Optional torch dependency is not installed",
)


def run_python(source, *args):
    completed = subprocess.run(
        [sys.executable, "-c", source, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_deterministic_datasets_and_distinct_splits():
    run_python("""
import torch
from respiratory_vision.training import make_dataset
first, truth = make_dataset(4, 300, 15, 7)
repeat, repeated_truth = make_dataset(4, 300, 15, 7)
other, _ = make_dataset(4, 300, 15, 8)
assert torch.equal(first, repeat) and torch.equal(truth, repeated_truth)
assert not torch.equal(first, other)
""")


def test_checkpoint_and_holdout(tmp_path):
    run_python(
        """
import sys
import numpy as np
from respiratory_vision.training import TrainingConfig, train_model, load_checkpoint, predict_checkpoint, make_dataset
path = sys.argv[1]
report = train_model(TrainingConfig(epochs=2, train_size=12, validation_size=4, test_size=4, batch_size=4), path)
_, checkpoint = load_checkpoint(path)
assert len(set(checkpoint['split_seeds'].values())) == 3
assert report['test_comparison']['temporal_cnn']['n_total'] == 4
assert report['test_comparison']['temporal_cnn']['mean_processing_seconds'] is None
x, _ = make_dataset(1, 300, 15, 99)
assert np.isfinite(predict_checkpoint(path, x.numpy()[0,0], 15))
assert predict_checkpoint(path, np.ones(100), 15) is None
""",
        tmp_path / "model.pt",
    )
