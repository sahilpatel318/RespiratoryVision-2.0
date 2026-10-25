import csv

import pytest

from respiratory_vision.benchmarking import score_rows
from respiratory_vision.dataset import (
    Recording,
    evaluate_manifest,
    group_split,
    load_manifest,
)
from respiratory_vision.synthetic import generate_breathing_video

FIELDS = [
    "recording_id",
    "subject_id",
    "video_path",
    "reference_bpm",
    "condition",
    "source",
]


def manifest(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def test_schema_and_missing(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("hello,world\na,b\n")
    with pytest.raises(ValueError):
        load_manifest(path)
    manifest(path, [["a", "s", "missing.mp4", 18, "rest", "authorized fixture"]])
    with pytest.raises(ValueError):
        load_manifest(path)


def test_group_leakage():
    rows = [
        Recording(str(i), str(i // 2), "unused", 18, "rest", "fixture")
        for i in range(10)
    ]
    dev, test = group_split(rows)
    assert {r.subject_id for r in dev}.isdisjoint(r.subject_id for r in test)
    assert group_split(rows) == (dev, test)
    with pytest.raises(ValueError):
        group_split(rows[:2])
    with pytest.raises(ValueError):
        group_split([Recording("a", "", "unused", 18, "rest", "fixture")])


def test_fixture_evaluation_and_privacy(tmp_path):
    p = tmp_path / "private_name.mp4"
    generate_breathing_video(p, duration_seconds=12)
    m = tmp_path / "manifest.csv"
    manifest(
        m,
        [
            [
                "private-record",
                "private-subject",
                p.name,
                18,
                "private-condition",
                "private-source",
            ]
        ],
    )
    result = evaluate_manifest(m, tmp_path / "report", kind="synthetic_fixture")
    assert result["provenance"]["kind"] == "synthetic_fixture"
    assert result["methods"]["original_baseline"]["n_total"] == 1
    for path in (tmp_path / "report").glob("*.json"):
        assert "private-" not in path.read_text()
        assert "private_name" not in path.read_text()
    manifest(m, [["a", "s", p.name, 18, "c", "s"], ["b", "s", p.name, 18, "c", "s"]])
    with pytest.raises(ValueError):
        load_manifest(m)


def test_metrics_keep_rejections():
    rows = [
        dict(case_id="a", reference_bpm=18, estimated_bpm=20, processing_seconds=1),
        dict(case_id="b", reference_bpm=18, estimated_bpm=None, processing_seconds=2),
    ]
    s = score_rows(rows)
    assert s["mae"] == 2 and s["acceptance_rate"] == 0.5 and s["n_total"] == 2
    assert s["mae_95ci"] is None
