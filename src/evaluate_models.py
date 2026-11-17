"""Metrics, chronological splits, and leakage-safe stacking utilities."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .preprocessing import FEATURE_COLUMNS, TARGET_COLUMN


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[name]) for name in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.asarray([float(row[TARGET_COLUMN]) for row in rows], dtype=float)
    return x, y


def safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    mask = actual != 0
    if not np.any(mask):
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    if len(actual) == 0:
        return {"rmse": None, "mae": None, "mape": None, "r2": None}
    errors = predicted - actual
    rmse = float(math.sqrt(float(np.mean(errors**2))))
    mae = float(np.mean(np.abs(errors)))
    mape = safe_mape(actual, predicted)
    if len(actual) < 2 or float(np.var(actual)) == 0.0:
        r2 = None
    else:
        r2 = float(1.0 - (np.sum(errors**2) / np.sum((actual - np.mean(actual)) ** 2)))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def create_metric_row(
    *,
    series_id: str,
    model: str,
    split: str,
    train_count: int,
    validation_count: int,
    test_count: int,
    metrics: dict,
    status: str = "trained",
    notes: str = "",
) -> dict:
    return {
        "series_id": series_id,
        "model": model,
        "split": split,
        "n_train": train_count,
        "n_validation": validation_count,
        "n_test": test_count,
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "mape": metrics.get("mape"),
        "r2": metrics.get("r2"),
        "status": status,
        "notes": notes,
    }
