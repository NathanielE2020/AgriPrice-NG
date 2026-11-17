"""Preparation orchestration and chronological split helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .data_loader import load_approved_rows
from .feature_engineering import build_feature_rows, build_monthly_series


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


FEATURE_COLUMNS = [
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "rolling_mean_3",
    "rolling_std_3",
    "month_sin",
    "month_cos",
    "trend_index",
]
TARGET_COLUMN = "target_price"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_approved_data() -> dict:
    approved_rows, source_counts = load_approved_rows()
    monthly_rows = build_monthly_series(approved_rows)
    feature_rows, diagnostics = build_feature_rows(monthly_rows)
    _write_csv(PROCESSED_DIR / "approved_monthly_reindexed.csv", monthly_rows)
    _write_csv(PROCESSED_DIR / "approved_training_features.csv", feature_rows)
    _write_csv(PROCESSED_DIR / "training_sample_counts.csv", diagnostics)
    return {
        "approved_rows": approved_rows,
        "monthly_rows": monthly_rows,
        "feature_rows": feature_rows,
        "diagnostics": diagnostics,
        "source_counts": source_counts,
    }


def chronological_split(rows: list[dict], train_fraction: float = 0.70, validation_fraction: float = 0.15):
    """Split one series by sorted calendar rows; never shuffle."""

    ordered = sorted(rows, key=lambda row: row["month"])
    n = len(ordered)
    if n < 10:
        return ordered, [], []
    train_end = max(1, int(n * train_fraction))
    validation_end = max(train_end + 1, int(n * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, n - 1)
    return ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]


def split_feature_rows(feature_rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in feature_rows:
        grouped[row["series_id"]].append(row)
    output: dict[str, dict[str, list[dict]]] = {}
    for series_id, rows in grouped.items():
        train, validation, test = chronological_split(rows)
        output[series_id] = {"train": train, "validation": validation, "test": test}
    return output
