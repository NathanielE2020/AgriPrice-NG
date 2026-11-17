"""Isolated TensorFlow LSTM experiment for the approved aggregate series.

This module is intentionally separate from the original market-level runner
and from ``src.final_validation``.  It reads the already prepared aggregate
monthly file and writes only below ``models/lstm_experiment``,
``reports/lstm_experiment`` and ``outputs/lstm_experiment``.

The experiment uses complete calendar-contiguous 12-month windows.  Missing
months are never filled, and transformations are selected on the validation
partition before the untouched test partition is opened.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

from .aggregate_preprocessing import APPROVED_AGGREGATE_SERIES, series_id
from .parquet_compat import month_from_index, month_index
from .plotting import write_actual_predicted_svg


ROOT = Path(__file__).resolve().parents[1]
MONTHLY_PATH = ROOT / "data" / "processed" / "aggregate_monthly_reindexed.csv"
FINAL_COMPARISON_PATH = ROOT / "reports" / "final_validation" / "final_model_comparison.csv"
MODELS_DIR = ROOT / "models" / "lstm_experiment"
REPORT_DIR = ROOT / "reports" / "lstm_experiment"
OUTPUT_DIR = ROOT / "outputs" / "lstm_experiment"
PLOTS_DIR = OUTPUT_DIR / "plots"
FORECAST_DIR = OUTPUT_DIR / "forecasts"

RANDOM_STATE = 42
WINDOW = 12
TRANSFORMS = ("original", "log1p", "price_change", "percentage_change")
HORIZONS = (1, 3, 6)
MIN_SEQUENCES = 20
MAX_EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 8
LSTM_UNITS = 32
DROPOUT = 0.2
LEARNING_RATE = 0.001


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:120]


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return {"rmse": None, "mae": None, "mape": None, "r2": None, "bias": None}
    error = predicted - actual
    nonzero = actual != 0
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
        if np.any(nonzero)
        else None,
        "r2": None
        if len(actual) < 2 or denominator == 0
        else float(1.0 - np.sum(error**2) / denominator),
        "bias": float(np.mean(error)),
    }


def _fit_scaler(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    scale = float(np.std(values))
    # A constant training window is valid; use unit scale rather than divide
    # by zero.  The fitted mean remains the only learned quantity.
    return {"mean": mean, "scale": scale if scale > 0 else 1.0}


def _scale(values: np.ndarray, scaler: dict[str, float]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - scaler["mean"]) / scaler["scale"]


def _inverse_scale(values: np.ndarray, scaler: dict[str, float]) -> np.ndarray:
    return np.asarray(values, dtype=float) * scaler["scale"] + scaler["mean"]


def _target_transform(prices: np.ndarray, lags: np.ndarray, method: str) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    lags = np.asarray(lags, dtype=float)
    if method == "original":
        return prices
    if method == "log1p":
        if np.any(prices < 0):
            raise ValueError("log1p target is invalid for negative prices")
        return np.log1p(prices)
    if method == "price_change":
        return prices - lags
    if method == "percentage_change":
        if np.any(lags == 0):
            raise ValueError("percentage-change target is invalid when a lag is zero")
        return prices / lags - 1.0
    raise ValueError(f"Unknown target transformation: {method}")


def _reconstruct(transformed: np.ndarray, lags: np.ndarray, method: str) -> np.ndarray:
    transformed = np.asarray(transformed, dtype=float)
    lags = np.asarray(lags, dtype=float)
    if method == "original":
        return transformed
    if method == "log1p":
        return np.maximum(np.expm1(transformed), 0.0)
    if method == "price_change":
        return lags + transformed
    if method == "percentage_change":
        return lags * (1.0 + transformed)
    raise ValueError(f"Unknown target transformation: {method}")


def _load_monthly_groups() -> dict[str, list[dict]]:
    if not MONTHLY_PATH.exists():
        raise FileNotFoundError(f"Missing prepared aggregate file: {MONTHLY_PATH}")
    approved_ids = {series_id(item) for item in APPROVED_AGGREGATE_SERIES}
    groups: dict[str, list[dict]] = {item: [] for item in sorted(approved_ids)}
    seen: set[tuple[str, str]] = set()
    for source in _read_csv(MONTHLY_PATH):
        current_id = source.get("series_id", "")
        if current_id not in approved_ids:
            continue
        month = source.get("month", "")
        key = (current_id, month)
        if key in seen:
            raise ValueError(f"Duplicate aggregate series/month key: {key}")
        seen.add(key)
        raw_price = source.get("mean_price", "")
        price = None if raw_price in ("", "None", "nan", "NaN") else float(raw_price)
        observed = str(source.get("observed", "")).strip().lower() == "true"
        if observed != (price is not None):
            raise ValueError(
                f"Observed/price mismatch for {current_id} at {month}: "
                f"observed={observed}, mean_price={raw_price!r}"
            )
        groups[current_id].append(
            {
                "series_id": current_id,
                "commodity": source.get("commodity", ""),
                "unit": source.get("unit", ""),
                "price_type": source.get("price_type", ""),
                "month": month,
                "date": source.get("date", ""),
                "mean_price": price,
                "n_markets": int(float(source.get("n_markets") or 0)),
                "n_records": int(float(source.get("n_records") or 0)),
                "observed": observed,
            }
        )
    missing = [item for item, rows in groups.items() if not rows]
    if missing:
        raise ValueError(f"Approved aggregate series missing: {missing}")
    for current_id in groups:
        groups[current_id].sort(key=lambda row: row["month"])
    return groups


def _build_sequences(rows: list[dict], window: int = WINDOW):
    observed = {
        row["month"]: float(row["mean_price"])
        for row in rows
        if row["observed"] and row["mean_price"] is not None
    }
    sequences: list[list[float]] = []
    targets: list[float] = []
    target_months: list[str] = []
    lags: list[float] = []
    for target_month in sorted(observed, key=month_index):
        target_index = month_index(target_month)
        history_months = [
            month_from_index(target_index - offset)
            for offset in range(window, 0, -1)
        ]
        history = [observed.get(month) for month in history_months]
        if any(value is None for value in history):
            continue
        sequences.append([float(value) for value in history])
        targets.append(float(observed[target_month]))
        target_months.append(target_month)
        lags.append(float(history[-1]))
    return (
        np.asarray(sequences, dtype=float),
        np.asarray(targets, dtype=float),
        target_months,
        np.asarray(lags, dtype=float),
    )


def _split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 3:
        return np.arange(n), np.asarray([], dtype=int), np.asarray([], dtype=int)
    train_end = max(1, int(n * 0.70))
    validation_end = max(train_end + 1, int(n * 0.85))
    validation_end = min(validation_end, n - 1)
    return (
        np.arange(0, train_end),
        np.arange(train_end, validation_end),
        np.arange(validation_end, n),
    )


def _build_model(tf, window: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, 1)),
            tf.keras.layers.LSTM(LSTM_UNITS),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(1),
        ]
    )
    # Explicitly state the Keras Adam default used by the completed run.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def _prepare_inputs(x: np.ndarray, y: np.ndarray, lags: np.ndarray, method: str):
    input_scaler = _fit_scaler(x.reshape(-1, 1))
    y_transformed = _target_transform(y, lags, method)
    target_scaler = _fit_scaler(y_transformed.reshape(-1, 1))
    x_scaled = _scale(x.reshape(-1, 1), input_scaler).reshape(len(x), x.shape[1], 1)
    y_scaled = _scale(y_transformed.reshape(-1, 1), target_scaler).ravel()
    return x_scaled, y_scaled, input_scaler, target_scaler


def _fit_candidate(
    tf,
    x_train: np.ndarray,
    y_train: np.ndarray,
    lag_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    lag_validation: np.ndarray,
    method: str,
):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    x_train_scaled, y_train_scaled, input_scaler, target_scaler = _prepare_inputs(
        x_train, y_train, lag_train, method
    )
    validation_target = _target_transform(y_validation, lag_validation, method)
    x_validation_scaled = _scale(
        x_validation.reshape(-1, 1), input_scaler
    ).reshape(len(x_validation), x_validation.shape[1], 1)
    y_validation_scaled = _scale(
        validation_target.reshape(-1, 1), target_scaler
    ).ravel()
    model = _build_model(tf, x_train.shape[1])
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    started = time.perf_counter()
    history = model.fit(
        x_train_scaled,
        y_train_scaled,
        validation_data=(x_validation_scaled, y_validation_scaled),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
        callbacks=[callback],
    )
    duration = time.perf_counter() - started
    predicted_scaled = model.predict(x_validation_scaled, verbose=0).ravel()
    predicted_transformed = _inverse_scale(predicted_scaled, target_scaler)
    predicted = _reconstruct(predicted_transformed, lag_validation, method)
    metrics = _metrics(y_validation, predicted)
    val_loss = history.history.get("val_loss", [])
    best_epoch = int(np.argmin(val_loss) + 1) if val_loss else len(history.history.get("loss", []))
    return {
        "model": model,
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "predicted": predicted,
        "metrics": metrics,
        "best_epoch": max(1, best_epoch),
        "epochs_run": len(history.history.get("loss", [])),
        "duration_seconds": float(duration),
        "history": history.history,
    }


def _fit_final(
    tf,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    lag_fit: np.ndarray,
    x_test: np.ndarray,
    method: str,
    epochs: int,
):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    x_fit_scaled, y_fit_scaled, input_scaler, target_scaler = _prepare_inputs(
        x_fit, y_fit, lag_fit, method
    )
    x_test_scaled = _scale(x_test.reshape(-1, 1), input_scaler).reshape(
        len(x_test), x_test.shape[1], 1
    )
    model = _build_model(tf, x_fit.shape[1])
    started = time.perf_counter()
    history = model.fit(
        x_fit_scaled,
        y_fit_scaled,
        epochs=max(1, int(epochs)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    duration = time.perf_counter() - started
    predicted_scaled = model.predict(x_test_scaled, verbose=0).ravel()
    return {
        "model": model,
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "predicted_transformed": _inverse_scale(predicted_scaled, target_scaler),
        "epochs_run": len(history.history.get("loss", [])),
        "duration_seconds": float(duration),
        "history": history.history,
    }


def _history_rows(series: str, transformation: str, stage: str, history: dict) -> list[dict]:
    rows: list[dict] = []
    loss_values = history.get("loss", [])
    val_values = history.get("val_loss", [])
    for index, loss in enumerate(loss_values, 1):
        rows.append(
            {
                "series_id": series,
                "transformation": transformation,
                "stage": stage,
                "epoch": index,
                "loss": float(loss),
                "val_loss": float(val_values[index - 1])
                if index <= len(val_values)
                else "",
            }
        )
    return rows


def _save_model_artifacts(series: str, selected: str, trained: dict, metadata: dict) -> None:
    target = MODELS_DIR / _slug(series)
    target.mkdir(parents=True, exist_ok=True)
    trained["model"].save(target / "lstm.keras")
    serializable = {
        **metadata,
        "input_scaler": trained["input_scaler"],
        "target_scaler": trained["target_scaler"],
        "transformation": selected,
    }
    (target / "lstm_preprocessing.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )


def _series_coverage(rows: list[dict]) -> dict:
    observed = [row for row in rows if row["observed"] and row["mean_price"] is not None]
    first = rows[0]["month"]
    last = rows[-1]["month"]
    calendar_count = month_index(last) - month_index(first) + 1
    observed_months = {row["month"] for row in observed}
    missing = [
        month_from_index(index)
        for index in range(month_index(first), month_index(last) + 1)
        if month_from_index(index) not in observed_months
    ]
    return {
        "original_monthly_observations": len(observed),
        "calendar_rows": calendar_count,
        "missing_calendar_months": len(missing),
        "missing_month_list": ";".join(missing),
        "calendar_coverage_percent": 100.0 * len(observed) / calendar_count,
        "first_observation_month": first,
        "last_observation_month": last,
    }


def _snapshot(paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for root in paths:
        if root.is_file():
            candidates = [root]
        elif root.exists():
            candidates = [item for item in root.rglob("*") if item.is_file()]
        else:
            candidates = []
        for item in candidates:
            digest = hashlib.sha256()
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result[str(item.relative_to(ROOT))] = digest.hexdigest()
    return result


def _protected_snapshot() -> dict[str, str]:
    paths = [
        ROOT / "data" / "raw",
        ROOT / "models" / "aggregate",
        ROOT / "models" / "final_validation",
        ROOT / "reports" / "final_validation",
        ROOT / "outputs" / "aggregate",
        ROOT / "outputs" / "final_validation",
    ]
    return _snapshot(paths)


def _load_final_comparison(lstm_test_months: dict[str, list[str]]) -> list[dict]:
    allowed = {"naive_last_observation", "random_forest", "xgboost", "gradient_boosting", "stacking"}
    rows: list[dict] = []
    if not FINAL_COMPARISON_PATH.exists():
        return rows
    for source in _read_csv(FINAL_COMPARISON_PATH):
        if source.get("series_id") not in lstm_test_months or source.get("model") not in allowed:
            continue
        source_months = [item for item in source.get("test_months", "").split(";") if item]
        target_months = lstm_test_months[source["series_id"]]
        overlap = len(set(source_months).intersection(target_months))
        rows.append(
            {
                "series_id": source.get("series_id", ""),
                "model": source.get("model", ""),
                "transformation": source.get("transformation", ""),
                "selection_split": source.get("selection_split", ""),
                "test_months": source.get("test_months", ""),
                "n_train": source.get("n_train", ""),
                "n_validation": source.get("n_validation", ""),
                "n_test": source.get("n_test", ""),
                "validation_rmse": source.get("validation_rmse", ""),
                "validation_mae": source.get("validation_mae", ""),
                "validation_mape": source.get("validation_mape", ""),
                "validation_r2": source.get("validation_r2", ""),
                "test_rmse": source.get("test_rmse", ""),
                "test_mae": source.get("test_mae", ""),
                "test_mape": source.get("test_mape", ""),
                "test_r2": source.get("test_r2", ""),
                "bias": source.get("bias", ""),
                "status": source.get("status", ""),
                "comparison_source": "final_validation",
                "test_date_alignment": "same" if source_months == target_months else "different",
                "test_date_overlap_count": overlap,
                "notes": source.get("notes", "")
                + " Existing final-validation metric; not recomputed.",
            }
        )
    return rows


def _comparison_row(
    series: str,
    model: str,
    test_months: list[str],
    n_train: int,
    n_validation: int,
    n_test: int,
    validation_metrics: dict | None,
    test_metrics: dict | None,
    transformation: str,
    status: str,
    source: str,
    notes: str,
    alignment: str = "same",
) -> dict:
    validation_metrics = validation_metrics or {}
    test_metrics = test_metrics or {}
    return {
        "series_id": series,
        "model": model,
        "transformation": transformation,
        "selection_split": "validation",
        "test_months": ";".join(test_months),
        "n_train": n_train,
        "n_validation": n_validation,
        "n_test": n_test,
        "validation_rmse": validation_metrics.get("rmse", ""),
        "validation_mae": validation_metrics.get("mae", ""),
        "validation_mape": validation_metrics.get("mape", ""),
        "validation_r2": validation_metrics.get("r2", ""),
        "test_rmse": test_metrics.get("rmse", ""),
        "test_mae": test_metrics.get("mae", ""),
        "test_mape": test_metrics.get("mape", ""),
        "test_r2": test_metrics.get("r2", ""),
        "bias": test_metrics.get("bias", ""),
        "status": status,
        "comparison_source": source,
        "test_date_alignment": alignment,
        "test_date_overlap_count": n_test if alignment == "same" else "",
        "notes": notes,
    }


def _recursive_forecast(
    trained: dict,
    rows: list[dict],
    series: str,
    transformation: str,
) -> list[dict]:
    observed = {
        row["month"]: float(row["mean_price"])
        for row in rows
        if row["observed"] and row["mean_price"] is not None
    }
    last_month = max(observed, key=month_index) if observed else ""
    base = {
        "series_id": series,
        "model": "lstm",
        "transformation": transformation,
        "origin_month": last_month,
        "method": "recursive; predicted prices fed back; no actual future prices",
    }
    if not last_month:
        return [
            {**base, "forecast_month": "", "horizon_months": horizon, "forecast_price": "", "status": "unavailable_no_history"}
            for horizon in HORIZONS
        ]
    history_months = [
        month_from_index(month_index(last_month) - offset)
        for offset in range(WINDOW - 1, -1, -1)
    ]
    if any(month not in observed for month in history_months):
        return [
            {
                **base,
                "forecast_month": month_from_index(month_index(last_month) + horizon),
                "horizon_months": horizon,
                "forecast_price": "",
                "status": "unavailable_missing_calendar_history",
            }
            for horizon in HORIZONS
        ]
    values = [observed[month] for month in history_months]
    output: list[dict] = []
    for step in range(1, max(HORIZONS) + 1):
        x = np.asarray(values[-WINDOW:], dtype=float).reshape(1, WINDOW, 1)
        scaled = _scale(x.reshape(-1, 1), trained["input_scaler"]).reshape(1, WINDOW, 1)
        transformed = float(
            _inverse_scale(
                trained["model"].predict(scaled, verbose=0).ravel(),
                trained["target_scaler"],
            )[0]
        )
        price = float(_reconstruct(np.asarray([transformed]), np.asarray([values[-1]]), transformation)[0])
        if not _finite(price):
            return [
                {
                    **base,
                    "forecast_month": month_from_index(month_index(last_month) + horizon),
                    "horizon_months": horizon,
                    "forecast_price": "",
                    "status": "unavailable_non_finite_prediction",
                }
                for horizon in HORIZONS
            ]
        price = max(price, 0.0)
        values.append(price)
        if step in HORIZONS:
            output.append(
                {
                    **base,
                    "forecast_month": month_from_index(month_index(last_month) + step),
                    "horizon_months": step,
                    "forecast_price": price,
                    "status": "recursive_forecast",
                }
            )
    return output


def _framework_info(tf) -> dict:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "tensorflow_version": tf.__version__,
        "physical_devices": [str(item) for item in tf.config.list_physical_devices()],
        "cpu_devices": [str(item) for item in tf.config.list_physical_devices("CPU")],
        "training_started": True,
    }


def run_isolated_lstm_experiment() -> dict:
    """Train and evaluate the isolated six-series TensorFlow experiment."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_snapshot()
    groups = _load_monthly_groups()

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - environment-specific
        tf = None
        tensorflow_error = str(exc)
    else:
        tensorflow_error = ""
        tf.keras.utils.set_random_seed(RANDOM_STATE)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    framework = (
        _framework_info(tf)
        if tf is not None
        else {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "tensorflow_version": "unavailable",
            "physical_devices": [],
            "cpu_devices": [],
            "training_started": False,
            "error": tensorflow_error,
        }
    )
    (REPORT_DIR / "framework_verification.json").write_text(
        json.dumps(framework, indent=2), encoding="utf-8"
    )

    config = {
        "series": list(sorted(groups)),
        "window_months": WINDOW,
        "split": "chronological 70% train / 15% validation / 15% test by sequence target month",
        "transformations": list(TRANSFORMS),
        "selection_metric": "validation RMSE on reconstructed prices",
        "architecture": {"lstm_units": LSTM_UNITS, "dropout": DROPOUT, "dense_units": 1},
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "mean_squared_error",
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_STATE,
        "shuffle": False,
        "future_price_imputation": False,
        "forecast_horizons_months": list(HORIZONS),
    }
    (REPORT_DIR / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    sequence_rows: list[dict] = []
    candidate_rows: list[dict] = []
    prediction_rows: list[dict] = []
    comparison_rows: list[dict] = []
    forecast_rows: list[dict] = []
    history_rows_by_series: dict[str, list[dict]] = {}
    series_results: dict[str, dict] = {}

    for current_series in sorted(groups):
        rows = groups[current_series]
        coverage = _series_coverage(rows)
        x, y, months, lags = _build_sequences(rows)
        train_i, validation_i, test_i = _split_indices(len(x))
        sequence_row = {
            "series_id": current_series,
            **coverage,
            "sequence_window_months": WINDOW,
            "complete_sequence_count": len(x),
            "train_sequences": len(train_i),
            "validation_sequences": len(validation_i),
            "test_sequences": len(test_i),
            "windows_overlap": True,
            "sequence_definition": "target plus 12 preceding observed calendar months",
            "status": "pending",
            "notes": "Window count is not an independent-observation count.",
        }
        if len(x) < MIN_SEQUENCES:
            sequence_row.update(status="insufficient_sequences", notes=f"Fewer than {MIN_SEQUENCES} complete sequences")
            sequence_rows.append(sequence_row)
            series_results[current_series] = {"status": "insufficient_sequences", "sequence_row": sequence_row}
            continue
        if tf is None:
            sequence_row.update(status="missing_tensorflow", notes=tensorflow_error)
            sequence_rows.append(sequence_row)
            series_results[current_series] = {"status": "missing_tensorflow", "sequence_row": sequence_row}
            continue
        if len(validation_i) == 0 or len(test_i) == 0:
            sequence_row.update(status="insufficient_split", notes="Validation or test partition is empty")
            sequence_rows.append(sequence_row)
            series_results[current_series] = {"status": "insufficient_split", "sequence_row": sequence_row}
            continue

        x_train, y_train, lag_train = x[train_i], y[train_i], lags[train_i]
        x_validation, y_validation, lag_validation = x[validation_i], y[validation_i], lags[validation_i]
        x_fit, y_fit, lag_fit = x[np.concatenate([train_i, validation_i])], y[np.concatenate([train_i, validation_i])], lags[np.concatenate([train_i, validation_i])]
        x_test, y_test, lag_test = x[test_i], y[test_i], lags[test_i]
        candidates: list[tuple[float, str, dict]] = []
        history_rows: list[dict] = []
        for transformation in TRANSFORMS:
            try:
                candidate = _fit_candidate(
                    tf,
                    x_train,
                    y_train,
                    lag_train,
                    x_validation,
                    y_validation,
                    lag_validation,
                    transformation,
                )
                candidates.append((float(candidate["metrics"]["rmse"]), transformation, candidate))
                candidate_rows.append(
                    {
                        "series_id": current_series,
                        "model": "lstm",
                        "transformation": transformation,
                        "selection_split": "validation",
                        "status": "evaluated",
                        "validation_rmse": candidate["metrics"]["rmse"],
                        "validation_mae": candidate["metrics"]["mae"],
                        "validation_mape": candidate["metrics"]["mape"],
                        "validation_r2": candidate["metrics"]["r2"],
                        "best_epoch": candidate["best_epoch"],
                        "epochs_run": candidate["epochs_run"],
                        "duration_seconds": candidate["duration_seconds"],
                        "notes": "Candidate evaluated on validation only; test unopened.",
                    }
                )
                history_rows.extend(_history_rows(current_series, transformation, "validation_candidate", candidate["history"]))
            except Exception as exc:
                candidate_rows.append(
                    {
                        "series_id": current_series,
                        "model": "lstm",
                        "transformation": transformation,
                        "selection_split": "validation",
                        "status": "failed",
                        "validation_rmse": "",
                        "validation_mae": "",
                        "validation_mape": "",
                        "validation_r2": "",
                        "best_epoch": "",
                        "epochs_run": "",
                        "duration_seconds": "",
                        "notes": str(exc),
                    }
                )
        if not candidates:
            sequence_row.update(status="failed", notes="All LSTM transformation candidates failed")
            sequence_rows.append(sequence_row)
            series_results[current_series] = {"status": "failed", "sequence_row": sequence_row}
            continue

        _, selected_transformation, selected_candidate = min(candidates, key=lambda item: (item[0], item[1]))
        final_fit = _fit_final(
            tf,
            x_fit,
            y_fit,
            lag_fit,
            x_test,
            selected_transformation,
            selected_candidate["best_epoch"],
        )
        history_rows.extend(_history_rows(current_series, selected_transformation, "final_train_validation", final_fit["history"]))
        test_pred = _reconstruct(
            final_fit["predicted_transformed"], lag_test, selected_transformation
        )
        test_metrics = _metrics(y_test, test_pred)
        validation_metrics = selected_candidate["metrics"]
        naive_test = _metrics(y_test, lag_test)
        test_months = [months[index] for index in test_i]
        validation_months = [months[index] for index in validation_i]
        train_months = [months[index] for index in train_i]
        sequence_row.update(status="trained", selected_transformation=selected_transformation, selected_best_epoch=selected_candidate["best_epoch"], final_epochs=final_fit["epochs_run"], final_fit_duration_seconds=final_fit["duration_seconds"])
        sequence_rows.append(sequence_row)
        history_rows_by_series[current_series] = history_rows
        for month, actual, predicted, baseline in zip(test_months, y_test, test_pred, lag_test):
            prediction_rows.append(
                {
                    "series_id": current_series,
                    "month": month,
                    "actual_price": float(actual),
                    "predicted_price": float(predicted),
                    "naive_price": float(baseline),
                    "model": "lstm",
                    "transformation": selected_transformation,
                }
            )
        slug = _slug(current_series)
        write_actual_predicted_svg(
            PLOTS_DIR / f"{slug}__lstm.svg",
            f"{current_series} - LSTM test predictions",
            test_months,
            y_test.tolist(),
            test_pred.tolist(),
            lag_test.tolist(),
        )
        metadata = {
            "series_id": current_series,
            "sequence_window_months": WINDOW,
            "selected_transformation": selected_transformation,
            "selection_split": "validation",
            "train_target_months": train_months,
            "validation_target_months": validation_months,
            "test_target_months": test_months,
            "best_validation_epoch": selected_candidate["best_epoch"],
            "final_fit_epochs": final_fit["epochs_run"],
            "final_fit_duration_seconds": final_fit["duration_seconds"],
            "architecture": config["architecture"],
            "random_seed": RANDOM_STATE,
        }
        _save_model_artifacts(current_series, selected_transformation, final_fit, metadata)
        series_results[current_series] = {
            "status": "trained",
            "sequence_row": sequence_row,
            "test_months": test_months,
            "validation_months": validation_months,
            "selected_transformation": selected_transformation,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "naive_test_metrics": naive_test,
            "model": final_fit["model"],
            "input_scaler": final_fit["input_scaler"],
            "target_scaler": final_fit["target_scaler"],
            "observed_rows": rows,
        }

    _write_csv(REPORT_DIR / "sequence_counts.csv", sequence_rows)
    _write_csv(REPORT_DIR / "target_selection.csv", candidate_rows)
    _write_csv(
        OUTPUT_DIR / "test_predictions.csv",
        prediction_rows,
        ["series_id", "month", "actual_price", "predicted_price", "naive_price", "model", "transformation"],
    )
    for current_series, history_rows in history_rows_by_series.items():
        _write_csv(
            REPORT_DIR / "history" / f"{_slug(current_series)}.csv",
            history_rows,
            ["series_id", "transformation", "stage", "epoch", "loss", "val_loss"],
        )

    comparison_rows.extend(
        _load_final_comparison(
            {
                current_series: result.get("test_months", [])
                for current_series, result in series_results.items()
                if result.get("status") == "trained"
            }
        )
    )
    for current_series, result in series_results.items():
        if result.get("status") != "trained":
            continue
        sequence_row = result["sequence_row"]
        test_months = result["test_months"]
        comparison_rows.append(
            _comparison_row(
                current_series,
                "naive_last_observation_lstm_test_dates",
                test_months,
                int(sequence_row["train_sequences"]),
                int(sequence_row["validation_sequences"]),
                int(sequence_row["test_sequences"]),
                None,
                result["naive_test_metrics"],
                "none",
                "evaluated",
                "lstm_experiment",
                "Naive previous calendar-month value on the exact LSTM sequence test dates.",
            )
        )
        comparison_rows.append(
            _comparison_row(
                current_series,
                "lstm",
                test_months,
                int(sequence_row["train_sequences"] + sequence_row["validation_sequences"]),
                int(sequence_row["validation_sequences"]),
                int(sequence_row["test_sequences"]),
                result["validation_metrics"],
                result["test_metrics"],
                result["selected_transformation"],
                "evaluated",
                "lstm_experiment",
                "Transformation selected on validation; refit on train+validation; test evaluated once.",
            )
        )
        forecast_rows.extend(
            _recursive_forecast(
                {
                    "model": result["model"],
                    "input_scaler": result["input_scaler"],
                    "target_scaler": result["target_scaler"],
                },
                result["observed_rows"],
                current_series,
                result["selected_transformation"],
            )
        )

    comparison_fields = [
        "series_id", "model", "transformation", "selection_split", "test_months",
        "n_train", "n_validation", "n_test", "validation_rmse", "validation_mae",
        "validation_mape", "validation_r2", "test_rmse", "test_mae", "test_mape",
        "test_r2", "bias", "status", "comparison_source", "test_date_alignment",
        "test_date_overlap_count", "notes",
    ]
    _write_csv(REPORT_DIR / "model_comparison.csv", comparison_rows, comparison_fields)
    _write_csv(
        FORECAST_DIR / "recursive_forecasts.csv",
        forecast_rows,
        ["series_id", "model", "transformation", "origin_month", "forecast_month", "horizon_months", "forecast_price", "status", "method"],
    )

    winners: list[dict] = []
    for current_series in sorted(series_results):
        available = []
        for row in comparison_rows:
            if row.get("series_id") != current_series or row.get("model") == "naive_last_observation_lstm_test_dates":
                continue
            if row.get("status") not in ("evaluated", "trained"):
                continue
            try:
                rmse = float(row.get("test_rmse", ""))
            except (TypeError, ValueError):
                continue
            available.append((rmse, row.get("model", ""), row.get("comparison_source", "")))
        if available:
            rmse, model_name, source = min(available, key=lambda item: (item[0], item[1]))
            winners.append({"series_id": current_series, "lowest_reported_test_rmse_model": model_name, "test_rmse": rmse, "comparison_source": source, "note": "Date alignment must be checked; LSTM sequence test dates may differ from tabular final-validation dates."})
        else:
            winners.append({"series_id": current_series, "lowest_reported_test_rmse_model": "", "test_rmse": "", "comparison_source": "", "note": "No successful metric rows."})
    _write_csv(REPORT_DIR / "winner_by_series.csv", winners)

    protected_after = _protected_snapshot()
    integrity = {
        "raw_and_existing_artifacts_unchanged": protected_before == protected_after,
        "protected_file_count_before": len(protected_before),
        "protected_file_count_after": len(protected_after),
        "changed_files": sorted(
            set(protected_before).union(protected_after)
            - {
                key
                for key in set(protected_before).intersection(protected_after)
                if protected_before[key] == protected_after[key]
            }
        ),
    }
    (REPORT_DIR / "integrity_check.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    trained_count = sum(result.get("status") == "trained" for result in series_results.values())
    report_lines = [
        "AgriPrice-NG isolated TensorFlow LSTM experiment",
        "",
        f"Interpreter: {sys.executable}",
        f"TensorFlow: {framework.get('tensorflow_version')}",
        f"CPU devices: {', '.join(framework.get('cpu_devices', [])) or 'none'}",
        f"Series considered: {len(groups)}; trained successfully: {trained_count}; unavailable/failed: {len(groups) - trained_count}.",
        "",
        "Methodology",
        "-----------",
        "Each approved aggregate commodity/unit/price-type series was trained independently.",
        "Only observed prices were used. Missing calendar months were not interpolated, forward-filled, or treated as previous observations.",
        "A sequence requires the target month and all 12 preceding calendar months to be observed.",
        "Sequences overlap and therefore are not independent historical observations.",
        "Sequence targets were split chronologically into 70% train, 15% validation and 15% test.",
        "Input and target scalers were fitted on training data for validation candidates. The selected transformation was refit on train+validation before the single test evaluation.",
        "Transformations were selected using validation RMSE on reconstructed prices only; test RMSE was not used for selection.",
        "",
        "Sequence counts",
        "---------------",
    ]
    for row in sequence_rows:
        report_lines.append(
            f"{row['series_id']}: observations={row['original_monthly_observations']}, calendar={row['calendar_rows']}, missing={row['missing_calendar_months']}, sequences={row['complete_sequence_count']} ({row['train_sequences']}/{row['validation_sequences']}/{row['test_sequences']} train/validation/test), status={row['status']}"
        )
    report_lines.extend(["", "Selected models and test metrics", "-------------------------------"])
    for current_series, result in sorted(series_results.items()):
        if result.get("status") != "trained":
            report_lines.append(f"{current_series}: {result.get('status')} - no metric fabricated")
            continue
        metric = result["test_metrics"]
        naive = result["naive_test_metrics"]
        report_lines.append(
            f"{current_series}: transformation={result['selected_transformation']}, test_dates={';'.join(result['test_months'])}, "
            f"LSTM RMSE={metric['rmse']:.6f}, MAE={metric['mae']:.6f}, MAPE={metric['mape']}, R2={metric['r2']}, bias={metric['bias']:.6f}; "
            f"same-date naive RMSE={naive['rmse']:.6f}"
        )
    report_lines.extend(
        [
            "",
            "Comparison caveat",
            "-----------------",
            "Existing Random Forest, Gradient Boosting, XGBoost and Stacking metrics were copied from reports/final_validation/final_model_comparison.csv and were not retrained or overwritten.",
            "The LSTM uses complete 12-month sequence target dates. Those dates can differ from the tabular final-validation test dates, so rows marked test_date_alignment=different are not a perfectly matched test-period comparison.",
            "The naive_last_observation_lstm_test_dates row is the fair same-date baseline for each LSTM result.",
            "No confidence intervals were generated.",
            "",
            "Recursive forecasting",
            "----------------------",
            "Forecasts for 1, 3 and 6 months were generated recursively. Each predicted price is fed back as history for later steps; actual future prices are never used.",
            "A forecast is marked unavailable when the latest observed origin does not have 12 consecutive calendar months of history.",
            f"Forecast file: {FORECAST_DIR / 'recursive_forecasts.csv'}",
            "",
            f"Protected raw/model/report/output artifacts unchanged: {integrity['raw_and_existing_artifacts_unchanged']}",
            "See sequence_counts.csv, target_selection.csv, model_comparison.csv, winner_by_series.csv, test_predictions.csv, framework_verification.json and integrity_check.json for auditable outputs.",
        ]
    )
    (REPORT_DIR / "training_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "series": series_results,
        "sequence_counts": sequence_rows,
        "candidates": candidate_rows,
        "comparison": comparison_rows,
        "forecasts": forecast_rows,
        "framework": framework,
        "integrity": integrity,
    }


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    result = run_isolated_lstm_experiment()
    print(
        f"LSTM experiment completed: {sum(item.get('status') == 'trained' for item in result['series'].values())}/"
        f"{len(result['series'])} series trained"
    )
