"""Preparation helpers for the isolated aggregate-series experiment."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from .parquet_compat import iter_months, month_index


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
AGGREGATE_PATH = PROCESSED_DIR / "commodity_monthly_aggregates.csv"

APPROVED_AGGREGATE_SERIES = (
    ("Yam", "2.5 KG", "Retail"),
    ("Rice (imported)", "50 KG", "Wholesale"),
    ("Oil (palm)", "L", "Retail"),
    ("Rice (imported)", "2.8 KG", "Retail"),
    ("Rice (local)", "2.7 KG", "Retail"),
    ("Millet", "2.6 KG", "Retail"),
)

AGGREGATE_FEATURE_COLUMNS = [
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "rolling_mean_3",
    "rolling_std_3",
    "month_sin",
    "month_cos",
    "year_index",
]
AGGREGATE_TARGET_COLUMN = "target_price"


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def series_id(key: tuple[str, str, str]) -> str:
    return " | ".join(key)


def load_approved_aggregate_rows() -> list[dict]:
    """Load only the approved aggregate keys and validate their row shape."""

    if not AGGREGATE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {AGGREGATE_PATH}; run the existing market-level pipeline once to create the aggregate audit file."
        )
    approved = set(APPROVED_AGGREGATE_SERIES)
    rows: list[dict] = []
    for source in _read_csv(AGGREGATE_PATH):
        key = (source["commodity"], source["unit"], source["price_type"])
        if key not in approved:
            continue
        row = dict(source)
        row["mean_price"] = float(source["mean_price"])
        row["median_price"] = float(source["median_price"])
        row["n_markets"] = int(source["n_markets"])
        row["n_records"] = int(source["n_records"])
        row["observed"] = True
        row["series_id"] = series_id(key)
        rows.append(row)

    found = {(row["commodity"], row["unit"], row["price_type"]) for row in rows}
    missing = [key for key in APPROVED_AGGREGATE_SERIES if key not in found]
    if missing:
        raise ValueError(f"Approved aggregate series missing from {AGGREGATE_PATH}: {missing}")

    duplicate_keys = [
        key
        for key, group in _group_rows(rows).items()
        if len(group) != len({row["month"] for row in group})
    ]
    if duplicate_keys:
        raise ValueError(f"Duplicate aggregate group-month rows found: {duplicate_keys}")
    return rows


def _group_rows(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["commodity"], row["unit"], row["price_type"])].append(row)
    return grouped


def build_monthly_aggregate_series(rows: list[dict]) -> list[dict]:
    """Reindex each approved aggregate to calendar months without imputing prices."""

    output: list[dict] = []
    for key, group in sorted(_group_rows(rows).items()):
        by_month = {row["month"]: row for row in group}
        months = sorted(by_month, key=month_index)
        for month in iter_months(months[0], months[-1]):
            source = by_month.get(month)
            output.append(
                {
                    "commodity": key[0],
                    "unit": key[1],
                    "price_type": key[2],
                    "series_id": series_id(key),
                    "month": month,
                    "date": f"{month}-15",
                    "mean_price": source["mean_price"] if source else None,
                    "median_price": source["median_price"] if source else None,
                    "n_markets": source["n_markets"] if source else 0,
                    "n_records": source["n_records"] if source else 0,
                    "currency_values": source.get("currency_values", "") if source else "",
                    "observed": source is not None,
                }
            )
    return output


def _calendar_features(month: str) -> tuple[float, float, float]:
    month_number = int(month.split("-")[1])
    year = int(month.split("-")[0])
    angle = 2.0 * math.pi * (month_number - 1) / 12.0
    return math.sin(angle), math.cos(angle), float(year)


def build_aggregate_feature_rows(monthly_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Create leakage-safe lags/rolling features from exact calendar months."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in monthly_rows:
        grouped[row["series_id"]].append(row)

    features: list[dict] = []
    diagnostics: list[dict] = []
    for current_series_id, rows in sorted(grouped.items()):
        by_month = {row["month"]: row for row in rows}
        months = sorted(by_month, key=lambda item: month_index(item))
        observed_count = sum(bool(row["observed"]) for row in rows)
        usable_count = 0
        for month in months:
            current = by_month[month]
            if not current["observed"]:
                continue
            current_index = month_index(month)
            values: dict[int, float | None] = {
                month_index(item_month): by_month[item_month]["mean_price"]
                for item_month in months
            }
            history = [values.get(current_index - lag) for lag in (1, 2, 3)]
            rolling_values = [values.get(current_index - offset) for offset in (1, 2, 3)]
            if any(value is None for value in history + rolling_values):
                continue
            numeric_history = [float(value) for value in rolling_values]
            rolling_mean = sum(numeric_history) / len(numeric_history)
            rolling_std = math.sqrt(
                sum((value - rolling_mean) ** 2 for value in numeric_history) / len(numeric_history)
            )
            month_sin, month_cos, year_index = _calendar_features(month)
            features.append(
                {
                    **current,
                    "price_lag_1": float(history[0]),
                    "price_lag_2": float(history[1]),
                    "price_lag_3": float(history[2]),
                    "rolling_mean_3": rolling_mean,
                    "rolling_std_3": rolling_std,
                    "month_sin": month_sin,
                    "month_cos": month_cos,
                    "year_index": year_index,
                    AGGREGATE_TARGET_COLUMN: float(current["mean_price"]),
                }
            )
            usable_count += 1

        diagnostics.append(
            {
                "series_id": current_series_id,
                "commodity": rows[0]["commodity"],
                "unit": rows[0]["unit"],
                "price_type": rows[0]["price_type"],
                "calendar_rows": len(rows),
                "observed_rows": observed_count,
                "usable_feature_rows": usable_count,
                "excluded_missing_history_rows": observed_count - usable_count,
                "lags": "1,2,3",
                "rolling_windows": "3",
                "target_column": AGGREGATE_TARGET_COLUMN,
            }
        )
    return features, diagnostics


def chronological_split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: row["month"])
    n = len(ordered)
    if n < 10:
        return ordered, [], []
    train_end = max(1, int(n * 0.70))
    validation_end = max(train_end + 1, int(n * 0.85))
    validation_end = min(validation_end, n - 1)
    return ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]


def prepare_aggregate_data() -> dict:
    rows = load_approved_aggregate_rows()
    monthly_rows = build_monthly_aggregate_series(rows)
    feature_rows, diagnostics = build_aggregate_feature_rows(monthly_rows)
    _write_csv(PROCESSED_DIR / "aggregate_monthly_reindexed.csv", monthly_rows)
    _write_csv(PROCESSED_DIR / "aggregate_training_features.csv", feature_rows)
    return {
        "source_rows": rows,
        "monthly_rows": monthly_rows,
        "feature_rows": feature_rows,
        "diagnostics": diagnostics,
    }
