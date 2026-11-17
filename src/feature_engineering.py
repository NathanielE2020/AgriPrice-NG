"""Leakage-safe monthly reindexing, calendar lags, and rolling features."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .data_loader import series_key
from .parquet_compat import iter_months, month_index, month_key


# The approved series have long calendar gaps. Requiring lag 6/rolling 6
# would remove almost every row. Lags 1--3 and a 3-month rolling statistic
# still capture recent calendar history while preserving honest missingness.
LAGS = (1, 2, 3)
ROLLING_WINDOWS = (3,)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _calendar_feature(month: str) -> tuple[float, float, int]:
    month_number = int(month.split("-")[1])
    angle = 2.0 * math.pi * (month_number - 1) / 12.0
    return math.sin(angle), math.cos(angle), month_index(month)


def build_monthly_series(rows: Iterable[dict]) -> list[dict]:
    """Reindex each approved series to every calendar month in its span.

    Missing target months remain ``None``.  They are never forward-filled or
    interpolated.  The returned rows are useful for auditing and for ensuring
    that lag 1 means the immediately preceding calendar month.
    """

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[series_key(row)].append(row)

    output: list[dict] = []
    for key, group_rows in sorted(grouped.items()):
        observed_by_month: dict[str, dict] = {}
        for row in group_rows:
            month = month_key(row["date"])
            if month in observed_by_month:
                raise ValueError(f"Duplicate monthly row for {key} at {month}")
            observed_by_month[month] = row

        months = sorted(observed_by_month)
        for month in iter_months(months[0], months[-1]):
            source = observed_by_month.get(month)
            base = {
                "commodity": key[0],
                "market": key[1],
                "unit": key[2],
                "price_type": key[3],
                "market_id": source.get("market_id") if source else group_rows[0].get("market_id"),
                "state": source.get("admin1") if source else group_rows[0].get("admin1"),
                "month": month,
                "date": f"{month}-15",
                "price": float(source["price"]) if source is not None else None,
                "priceflag": source.get("priceflag") if source else None,
                "observed": source is not None,
            }
            output.append(base)
    return output


def build_feature_rows(monthly_rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Build complete-feature rows and per-series preparation diagnostics."""

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in monthly_rows:
        grouped[(row["commodity"], row["market"], row["unit"], row["price_type"])].append(row)

    feature_rows: list[dict] = []
    diagnostics: list[dict] = []
    required_feature_names = [
        *(f"price_lag_{lag}" for lag in LAGS),
        *(f"rolling_mean_{window}" for window in ROLLING_WINDOWS),
        *(f"rolling_std_{window}" for window in ROLLING_WINDOWS),
    ]

    for key, rows in sorted(grouped.items()):
        by_month = {row["month"]: row for row in rows}
        months = sorted(by_month, key=month_index)
        usable = 0
        observed = sum(row["observed"] for row in rows)
        for month in months:
            current = by_month[month]
            if current["price"] is None:
                continue
            current_index = month_index(month)
            values_by_index = {
                month_index(item_month): by_month[item_month]["price"]
                for item_month in months
            }
            features: dict[str, float] = {}
            complete = True
            for lag in LAGS:
                value = values_by_index.get(current_index - lag)
                features[f"price_lag_{lag}"] = value
                if value is None:
                    complete = False
            for window in ROLLING_WINDOWS:
                history = [
                    values_by_index.get(current_index - offset)
                    for offset in range(1, window + 1)
                ]
                if any(value is None for value in history):
                    complete = False
                    features[f"rolling_mean_{window}"] = None
                    features[f"rolling_std_{window}"] = None
                else:
                    numeric_history = [float(value) for value in history]
                    features[f"rolling_mean_{window}"] = _mean(numeric_history)
                    features[f"rolling_std_{window}"] = _std(numeric_history)

            month_sin, month_cos, trend_index = _calendar_feature(month)
            features["month_sin"] = month_sin
            features["month_cos"] = month_cos
            features["trend_index"] = float(trend_index)
            if complete and all(features[name] is not None for name in required_feature_names):
                usable += 1
                feature_rows.append(
                    {
                        **current,
                        "series_id": " | ".join(key),
                        **features,
                        "target_price": float(current["price"]),
                    }
                )

        diagnostics.append(
            {
                "series_id": " | ".join(key),
                "commodity": key[0],
                "market": key[1],
                "unit": key[2],
                "price_type": key[3],
                "calendar_rows": len(rows),
                "observed_rows": observed,
                "usable_feature_rows": usable,
                "excluded_missing_history_rows": observed - usable,
                "lags": ",".join(str(lag) for lag in LAGS),
                "rolling_windows": ",".join(str(window) for window in ROLLING_WINDOWS),
            }
        )

    return feature_rows, diagnostics
