import csv
import json

import pytest

from respiratory_vision.benchmarking import run_benchmark
from respiratory_vision.metrics import regression_metrics


def test_reproducible_benchmark_and_failure_exports(tmp_path):
    first = run_benchmark(tmp_path / "a", "development", 1, ("clean",))
    second = run_benchmark(tmp_path / "b", "development", 1, ("clean",))
    for name in first["methods"]:
        for key in ["mae", "rmse", "acceptance_rate", "n_total"]:
            assert first["methods"][name][key] == second["methods"][name][key]
    assert (
        json.loads((tmp_path / "a/cases.json").read_text())[0]["generation"]["seed"]
        == 11000
    )
    with (tmp_path / "a/results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert all(r["case_id"] == "development-clean-00" for r in rows)
    for name in [
        "summary.json",
        "failure_cases.csv",
        "method_comparison.png",
        "error_distribution.png",
        "quality_by_condition.png",
    ]:
        assert (tmp_path / "a" / name).is_file()


@pytest.mark.parametrize(
    "truth,pred", [([], []), ([1], [float("nan")]), ([[1]], [[2]])]
)
def test_invalid_metrics(truth, pred):
    with pytest.raises(ValueError):
        regression_metrics(truth, pred)
