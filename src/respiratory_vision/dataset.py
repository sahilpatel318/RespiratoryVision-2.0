"""Authorized local manifest adapter. Public exports omit IDs, paths and free text."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .benchmarking import evaluate_video, write_benchmark

REQUIRED = {
    "recording_id",
    "subject_id",
    "video_path",
    "reference_bpm",
    "condition",
    "source",
}


@dataclass(frozen=True)
class Recording:
    recording_id: str
    subject_id: str
    video_path: Path
    reference_bpm: float
    condition: str
    source: str


def load_manifest(path: str | Path) -> list[Recording]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(
                "Manifest must contain recording_id, subject_id, video_path, reference_bpm, condition, source."
            )
        rows = []
        ids = set()
        paths = set()
        for index, row in enumerate(reader, 2):
            try:
                rid = row["recording_id"].strip()
                video = Path(row["video_path"])
                video = (
                    (path.parent / video).resolve()
                    if not video.is_absolute()
                    else video.resolve()
                )
                bpm = float(row["reference_bpm"])
                if (
                    not rid
                    or rid in ids
                    or video in paths
                    or not video.is_file()
                    or not math.isfinite(bpm)
                    or bpm <= 0
                ):
                    raise ValueError()
                if not row["source"].strip() or not row["condition"].strip():
                    raise ValueError()
                rows.append(
                    Recording(
                        rid,
                        row["subject_id"].strip(),
                        video,
                        bpm,
                        row["condition"].strip(),
                        row["source"].strip(),
                    )
                )
                ids.add(rid)
                paths.add(video)
            except (ValueError, TypeError, KeyError):
                raise ValueError(
                    f"Invalid, duplicate or unavailable recording on manifest row {index}."
                ) from None
    if not rows:
        raise ValueError("Manifest contains no recordings.")
    return rows


def group_split(
    records: list[Recording], test_fraction=0.25, seed=7
) -> tuple[list[Recording], list[Recording]]:
    if not 0 < test_fraction < 1 or any(not r.subject_id for r in records):
        raise ValueError(
            "Group splits require subject IDs for every recording and a fraction in (0,1)."
        )
    groups = sorted({r.subject_id for r in records})
    if len(groups) < 2:
        raise ValueError("At least two independent subjects are required.")
    np.random.default_rng(seed).shuffle(groups)
    n = max(1, min(len(groups) - 1, round(len(groups) * test_fraction)))
    test = set(groups[:n])
    return [r for r in records if r.subject_id not in test], [
        r for r in records if r.subject_id in test
    ]


def evaluate_manifest(manifest, output, kind="real", partition="all", seed=7):
    if kind not in ("real", "synthetic_fixture") or partition not in (
        "all",
        "development",
        "test",
    ):
        raise ValueError("Invalid dataset kind or partition.")
    records = load_manifest(manifest)
    if partition != "all":
        development, test = group_split(records, seed=seed)
        records = test if partition == "test" else development
    conditions = {
        c: f"condition-{i + 1:02d}"
        for i, c in enumerate(sorted({r.condition for r in records}))
    }
    rows = []
    diagnostics = []
    for index, record in enumerate(records):
        case_id = f"recording-{index + 1:04d}"
        case_rows, result = evaluate_video(
            record.video_path,
            record.reference_bpm,
            case_id,
            conditions[record.condition],
        )
        for row in case_rows:
            row["_group"] = record.subject_id or "unknown-subject-cluster"
        rows.extend(case_rows)
        diagnostics.append(
            {
                "case_id": case_id,
                "quality": result["quality"],
                "method_reasons": {
                    k: v["reasons"] for k, v in result["methods"].items()
                },
            }
        )
    provenance = {
        "kind": kind,
        "partition": partition,
        "split_seed": seed,
        "reference": "user-supplied scalar BPM; synchronization/authorization not verified by software",
        "subject_count": len({r.subject_id for r in records if r.subject_id}),
        "privacy": "IDs, file paths, condition labels and source free text omitted. Derived signals/metrics are not guaranteed anonymous.",
        "clinical_validation": False,
    }
    summary = write_benchmark(rows, output, provenance)
    import json

    (Path(output) / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary
