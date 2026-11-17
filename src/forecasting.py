"""Recursive calendar-month forecasts for trained tabular models."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .feature_engineering import LAGS, ROLLING_WINDOWS
from .parquet_compat import month_from_index, month_index
from .preprocessing import FEATURE_COLUMNS


def _feature_vector(history: dict[str, float], month: str) -> list[float] | None:
    current_index = month_index(month)
    values: list[float] = []
    for lag in LAGS:
        value = history.get(month_from_index(current_index - lag))
        if value is None:
            return None
        values.append(float(value))
    for window in ROLLING_WINDOWS:
        history_values = [
            history.get(month_from_index(current_index - offset))
            for offset in range(1, window + 1)
        ]
        if any(value is None for value in history_values):
            return None
        numeric = [float(value) for value in history_values]
        mean = sum(numeric) / len(numeric)
        std = math.sqrt(sum((value - mean) ** 2 for value in numeric) / len(numeric))
        values.extend([mean, std])
    month_number = int(month.split("-")[1])
    angle = 2.0 * math.pi * (month_number - 1) / 12.0
    values.extend([math.sin(angle), math.cos(angle), float(current_index)])
    return values


def recursive_forecast(
    model,
    monthly_rows: list[dict],
    series_id: str,
    model_name: str,
    horizons: tuple[int, ...] = (1, 3, 6),
) -> list[dict]:
    observed = {
        row["month"]: float(row["price"])
        for row in monthly_rows
        if row["price"] is not None
    }
    if not observed:
        return [
            {
                "series_id": series_id,
                "model": model_name,
                "horizon_months": horizon,
                "forecast_month": None,
                "forecast_price": None,
                "status": "unavailable_no_observed_history",
            }
            for horizon in horizons
        ]
    last_month = max(observed, key=month_index)
    predicted_by_step: dict[int, float] = {}
    failed_step: int | None = None
    working_history = dict(observed)
    for step in range(1, max(horizons) + 1):
        target_month = month_from_index(month_index(last_month) + step)
        vector = _feature_vector(working_history, target_month)
        if vector is None:
            failed_step = step
            break
        prediction = float(model.predict(np.asarray([vector], dtype=float))[0])
        working_history[target_month] = prediction
        predicted_by_step[step] = prediction

    # Emit exactly one row for each requested horizon.  If a calendar gap
    # prevents a recursive feature vector, retain any earlier successful
    # horizons and mark only the affected later horizons unavailable.  The
    # unavailable row still names its true requested calendar month.
    predictions: list[dict] = []
    for horizon in horizons:
        forecast_month = month_from_index(month_index(last_month) + horizon)
        if horizon in predicted_by_step:
            predictions.append(
                {
                    "series_id": series_id,
                    "model": model_name,
                    "horizon_months": horizon,
                    "forecast_month": forecast_month,
                    "forecast_price": predicted_by_step[horizon],
                    "status": "recursive_forecast",
                }
            )
        else:
            predictions.append(
                {
                    "series_id": series_id,
                    "model": model_name,
                    "horizon_months": horizon,
                    "forecast_month": forecast_month,
                    "forecast_price": None,
                    "status": "unavailable_missing_calendar_history"
                    if failed_step is not None
                    else "unavailable_forecast_horizon",
                }
            )
    return predictions


def write_forecasts(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "series_id",
        "model",
        "horizon_months",
        "forecast_month",
        "forecast_price",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
