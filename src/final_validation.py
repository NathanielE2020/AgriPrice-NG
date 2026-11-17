"""Leakage-safe final validation for the approved aggregate experiments.

This module is deliberately separate from the Phase 3/3.2/3.3 runners.  It
reads the already-created monthly feature files and writes only below
``reports/final_validation``, ``models/final_validation`` and
``outputs/final_validation``.  The test period is opened exactly once, after
target transformation/model selection has been completed on the validation
period.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .aggregate_preprocessing import (
    AGGREGATE_FEATURE_COLUMNS,
    AGGREGATE_TARGET_COLUMN,
    chronological_split,
)
from .parquet_compat import month_from_index, month_index


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "final_validation"
MODEL_DIR = ROOT / "models" / "final_validation" / "aggregate"
OUTPUT_DIR = ROOT / "outputs" / "final_validation"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
RANDOM_STATE = 42
TRANSFORMS = ("original", "log1p", "price_change", "percentage_change")
MODEL_NAMES = ("random_forest", "xgboost", "gradient_boosting", "stacking")


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:120]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_feature_rows() -> list[dict]:
    rows = _read_csv(PROCESSED_DIR / "aggregate_training_features.csv")
    numeric = [*AGGREGATE_FEATURE_COLUMNS, AGGREGATE_TARGET_COLUMN, "mean_price", "n_markets", "n_records"]
    for row in rows:
        for column in numeric:
            row[column] = float(row[column])
    return rows


def _load_monthly_rows() -> list[dict]:
    rows = _read_csv(PROCESSED_DIR / "aggregate_monthly_reindexed.csv")
    for row in rows:
        row["mean_price"] = None if row["mean_price"] in ("", "None") else float(row["mean_price"])
        row["n_markets"] = int(float(row.get("n_markets") or 0))
        row["observed"] = str(row.get("observed", "")).lower() == "true"
    return rows


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["series_id"]].append(row)
    return {key: sorted(value, key=lambda item: item["month"]) for key, value in grouped.items()}


def _xgb_factory_status() -> tuple[dict, str | None, str]:
    """Return model factories and an explicit XGBoost import status."""

    factories = {
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=2,
            random_state=RANDOM_STATE,
            loss="squared_error",
        ),
    }
    try:
        from xgboost import XGBRegressor

        factories["xgboost"] = lambda: XGBRegressor(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        import xgboost

        return factories, None, f"available ({xgboost.__version__})"
    except Exception as exc:
        return factories, str(exc), f"unavailable: {exc}"


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[column]) for column in AGGREGATE_FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in rows], dtype=float)
    return x, y


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
        "mape": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0) if np.any(nonzero) else None,
        "r2": None if len(actual) < 2 or denominator == 0 else float(1.0 - np.sum(error**2) / denominator),
        "bias": float(np.mean(error)),
    }


def _target(rows: list[dict], method: str) -> tuple[np.ndarray, np.ndarray]:
    prices = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in rows], dtype=float)
    lag = np.asarray([float(row["price_lag_1"]) for row in rows], dtype=float)
    if method == "original":
        return prices, lag
    if method == "log1p":
        if np.any(prices < 0):
            raise ValueError("log1p cannot be used with negative prices")
        return np.log1p(prices), lag
    if method == "price_change":
        return prices - lag, lag
    if method == "percentage_change":
        if np.any(lag == 0):
            raise ValueError("percentage-change target has a zero lag price")
        return prices / lag - 1.0, lag
    raise ValueError(f"Unknown target transformation: {method}")


def _reconstruct(predicted: np.ndarray, lag: np.ndarray, method: str) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=float)
    if method == "original":
        return predicted
    if method == "log1p":
        return np.maximum(np.expm1(predicted), 0.0)
    if method == "price_change":
        return lag + predicted
    if method == "percentage_change":
        return lag * (1.0 + predicted)
    raise ValueError(method)


def _oof(factory, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_splits = 3 if len(y) >= 12 else 2
    result = np.full(len(y), np.nan, dtype=float)
    if len(y) <= n_splits + 1:
        return result
    for train_index, validation_index in TimeSeriesSplit(n_splits=n_splits).split(x):
        if len(train_index) < 3:
            continue
        model = factory()
        model.fit(x[train_index], y[train_index])
        result[validation_index] = model.predict(x[validation_index])
    return result


def _fit_model(name: str, factories: dict, x: np.ndarray, y: np.ndarray, x_eval: np.ndarray):
    if name != "stacking":
        if name not in factories:
            raise RuntimeError("model factory is unavailable")
        model = factories[name]()
        model.fit(x, y)
        return model, np.asarray(model.predict(x_eval), dtype=float), "direct fit"
    base_names = [item for item in ("random_forest", "gradient_boosting", "xgboost") if item in factories]
    if len(base_names) < 2:
        raise RuntimeError("fewer than two base models are available for stacking")
    oof = {item: _oof(factories[item], x, y) for item in base_names}
    valid = np.ones(len(y), dtype=bool)
    for values in oof.values():
        valid &= np.isfinite(values)
    if int(valid.sum()) < 4:
        raise RuntimeError("insufficient chronological out-of-fold predictions")
    meta = Ridge(alpha=1.0)
    meta.fit(np.column_stack([oof[item][valid] for item in base_names]), y[valid])
    bases = {}
    eval_predictions = []
    for item in base_names:
        fitted = factories[item]()
        fitted.fit(x, y)
        bases[item] = fitted
        eval_predictions.append(fitted.predict(x_eval))
    bundle = {"base_models": bases, "meta_model": meta, "base_names": base_names, "feature_columns": AGGREGATE_FEATURE_COLUMNS}
    return bundle, np.asarray(meta.predict(np.column_stack(eval_predictions)), dtype=float), "chronological OOF base predictions + Ridge meta-model"


def _predict_bundle(bundle, x: np.ndarray) -> np.ndarray:
    if isinstance(bundle, dict) and "base_models" in bundle:
        matrix = np.column_stack([bundle["base_models"][name].predict(x) for name in bundle["base_names"]])
        return np.asarray(bundle["meta_model"].predict(matrix), dtype=float)
    return np.asarray(bundle.predict(x), dtype=float)


def _save_model(series_id: str, model_name: str, model, metadata: dict) -> None:
    target = MODEL_DIR / _slug(series_id)
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, target / f"{model_name}.joblib")


def _series_split(rows: list[dict]) -> dict[str, list[dict]]:
    train, validation, test = chronological_split(rows)
    return {"train": train, "validation": validation, "test": test}


def _select_tree_models(series_id: str, split: dict, factories: dict) -> tuple[list[dict], dict, dict]:
    """Select target transformations on validation, then test once."""

    train_rows, validation_rows, test_rows = split["train"], split["validation"], split["test"]
    x_train, _ = _arrays(train_rows)
    x_validation, _ = _arrays(validation_rows)
    x_test, _ = _arrays(test_rows)
    actual_validation = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in validation_rows])
    lag_validation = np.asarray([float(row["price_lag_1"]) for row in validation_rows])
    actual_test = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in test_rows])
    lag_test = np.asarray([float(row["price_lag_1"]) for row in test_rows])
    audit: list[dict] = []
    chosen: dict = {}
    for model_name in MODEL_NAMES:
        if model_name == "xgboost" and "xgboost" not in factories:
            chosen[model_name] = {"status": "unavailable"}
            audit.append({"series_id": series_id, "model": model_name, "transformation": "", "selection_split": "validation", "status": "unavailable", "validation_rmse": "", "validation_mae": "", "notes": "XGBoost import unavailable"})
            continue
        candidates = []
        for method in TRANSFORMS:
            try:
                y_train, _ = _target(train_rows, method)
                fitted, transformed_pred, note = _fit_model(model_name, factories, x_train, y_train, x_validation)
                validation_pred = _reconstruct(transformed_pred, lag_validation, method)
                values = _metrics(actual_validation, validation_pred)
                candidates.append((float(values["rmse"]), method, fitted, values, note))
                audit.append({"series_id": series_id, "model": model_name, "transformation": method, "selection_split": "validation", "status": "evaluated", "validation_rmse": values["rmse"], "validation_mae": values["mae"], "validation_mape": values["mape"], "validation_r2": values["r2"], "notes": note})
            except Exception as exc:
                audit.append({"series_id": series_id, "model": model_name, "transformation": method, "selection_split": "validation", "status": "failed", "validation_rmse": "", "validation_mae": "", "validation_mape": "", "validation_r2": "", "notes": str(exc)})
        if not candidates:
            chosen[model_name] = {"status": "failed"}
            continue
        _, method, _, val_metrics, note = min(candidates, key=lambda item: (item[0], item[1]))
        # Refit the selected configuration on train + validation only.
        fit_rows = train_rows + validation_rows
        x_fit, _ = _arrays(fit_rows)
        y_fit, _ = _target(fit_rows, method)
        fitted, transformed_test, fit_note = _fit_model(model_name, factories, x_fit, y_fit, x_test)
        test_pred = _reconstruct(transformed_test, lag_test, method)
        test_metrics = _metrics(actual_test, test_pred)
        metadata = {"series_id": series_id, "model": model_name, "transformation": method, "selection_split": "validation", "fit_rows": len(fit_rows), "test_months": [row["month"] for row in test_rows], "notes": fit_note}
        _save_model(series_id, model_name, fitted, metadata)
        chosen[model_name] = {"status": "trained", "transformation": method, "validation_metrics": val_metrics, "test_metrics": test_metrics, "test_prediction": test_pred, "model": fitted, "notes": fit_note}
        audit.append({"series_id": series_id, "model": model_name, "transformation": method, "selection_split": "validation", "status": "selected_and_tested", "validation_rmse": val_metrics["rmse"], "validation_mae": val_metrics["mae"], "validation_mape": val_metrics["mape"], "validation_r2": val_metrics["r2"], "test_rmse": test_metrics["rmse"], "test_mae": test_metrics["mae"], "test_mape": test_metrics["mape"], "test_r2": test_metrics["r2"], "notes": "Test metrics recorded after validation-only selection"})
    return audit, chosen, {"actual_test": actual_test, "test_rows": test_rows}


def _build_sequences(monthly_rows: list[dict], window: int = 12):
    values = {row["month"]: row["mean_price"] for row in monthly_rows if row.get("observed") and row["mean_price"] is not None}
    sequences, targets, months, lags = [], [], [], []
    for month in sorted(values, key=month_index):
        current = month_index(month)
        history_months = [month_from_index(current - offset) for offset in range(window, 0, -1)]
        history = [values.get(item) for item in history_months]
        if any(item is None for item in history):
            continue
        sequences.append(history)
        targets.append(float(values[month]))
        months.append(month)
        lags.append(float(history[-1]))
    return np.asarray(sequences, dtype=float), np.asarray(targets, dtype=float), months, np.asarray(lags, dtype=float)


def _sequence_split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = max(1, int(n * 0.70))
    validation_end = max(train_end + 1, int(n * 0.85))
    validation_end = min(validation_end, n - 1)
    return np.arange(0, train_end), np.arange(train_end, validation_end), np.arange(validation_end, n)


def _lstm_fit_predict(tf, x_train, y_train, x_eval, transform: str, lag_train, lag_eval, validation_data=None, final_fit=False):
    from sklearn.preprocessing import StandardScaler

    y_train_t = _reconstruct(np.zeros(len(y_train)), lag_train, "original")  # shape-only branch below
    if transform == "original":
        y_train_t = y_train
    elif transform == "log1p":
        y_train_t = np.log1p(y_train)
    elif transform == "price_change":
        y_train_t = y_train - lag_train
    else:
        y_train_t = y_train / lag_train - 1.0
    input_scaler = StandardScaler().fit(x_train.reshape(-1, 1))
    target_scaler = StandardScaler().fit(y_train_t.reshape(-1, 1))
    x_scaled = input_scaler.transform(x_train.reshape(-1, 1)).reshape(x_train.shape[0], x_train.shape[1], 1)
    eval_scaled = input_scaler.transform(x_eval.reshape(-1, 1)).reshape(x_eval.shape[0], x_eval.shape[1], 1)
    y_scaled = target_scaler.transform(y_train_t.reshape(-1, 1)).ravel()
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(x_train.shape[1], 1)), tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    fit_kwargs = {"epochs": 60, "batch_size": 8, "verbose": 0, "shuffle": False, "callbacks": callbacks}
    if validation_data is not None and len(validation_data[0]):
        vx, vy, vlag = validation_data
        vx_scaled = input_scaler.transform(vx.reshape(-1, 1)).reshape(vx.shape[0], vx.shape[1], 1)
        if transform == "original":
            vy_t = vy
        elif transform == "log1p":
            vy_t = np.log1p(vy)
        elif transform == "price_change":
            vy_t = vy - vlag
        else:
            vy_t = vy / vlag - 1.0
        vy_scaled = target_scaler.transform(vy_t.reshape(-1, 1)).ravel()
        fit_kwargs["validation_data"] = (vx_scaled, vy_scaled)
    else:
        fit_kwargs["validation_split"] = 0.15 if len(x_train) >= 20 else 0.0
    model.fit(x_scaled, y_scaled, **fit_kwargs)
    predicted_t = target_scaler.inverse_transform(model.predict(eval_scaled, verbose=0).reshape(-1, 1)).ravel()
    predicted = _reconstruct(predicted_t, lag_eval, transform)
    return model, input_scaler, target_scaler, predicted


def _run_lstm(series_id: str, monthly_rows: list[dict], tf) -> tuple[dict, dict | None, list[dict]]:
    x, y, months, lags = _build_sequences(monthly_rows)
    result_base = {"series_id": series_id, "sequence_window": 12, "sequence_count": len(x), "status": "not_trained", "selected_transformation": "", "train_sequences": 0, "validation_sequences": 0, "test_sequences": 0}
    if len(x) < 20:
        result_base.update(status="insufficient_sequences", notes="Fewer than 20 complete sequences")
        return result_base, None, []
    train_i, val_i, test_i = _sequence_split(len(x))
    result_base.update(train_sequences=len(train_i), validation_sequences=len(val_i), test_sequences=len(test_i))
    if tf is None:
        result_base.update(status="unavailable", notes="TensorFlow import/install failed")
        return result_base, None, []
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    candidates = []
    candidate_rows = []
    for method in TRANSFORMS:
        try:
            _, _, _, pred = _lstm_fit_predict(tf, x[train_i], y[train_i], x[val_i], method, lags[train_i], lags[val_i], validation_data=(x[val_i], y[val_i], lags[val_i]))
            met = _metrics(y[val_i], pred)
            candidates.append((met["rmse"], method, met))
            candidate_rows.append({"series_id": series_id, "model": "lstm", "transformation": method, "selection_split": "validation", "status": "evaluated", "validation_rmse": met["rmse"], "validation_mae": met["mae"], "validation_mape": met["mape"], "validation_r2": met["r2"]})
        except Exception as exc:
            candidate_rows.append({"series_id": series_id, "model": "lstm", "transformation": method, "selection_split": "validation", "status": "failed", "validation_rmse": "", "validation_mae": "", "validation_mape": "", "validation_r2": "", "notes": str(exc)})
    if not candidates:
        result_base.update(status="failed", notes="All LSTM transformations failed")
        return result_base, None, candidate_rows
    _, selected, val_met = min(candidates, key=lambda item: (item[0], item[1]))
    fit_i = np.concatenate([train_i, val_i])
    model, input_scaler, target_scaler, test_pred = _lstm_fit_predict(tf, x[fit_i], y[fit_i], x[test_i], selected, lags[fit_i], lags[test_i], validation_data=None, final_fit=True)
    test_met = _metrics(y[test_i], test_pred)
    target = MODEL_DIR / _slug(series_id)
    target.mkdir(parents=True, exist_ok=True)
    model.save(target / "lstm.keras")
    joblib.dump({"input_scaler": input_scaler, "target_scaler": target_scaler, "metadata": {"series_id": series_id, "transformation": selected, "months": months, "test_months": [months[i] for i in test_i]}}, target / "lstm_scalers.joblib")
    result = {**result_base, "status": "trained", "selected_transformation": selected, "validation_rmse": val_met["rmse"], "validation_mae": val_met["mae"], "validation_mape": val_met["mape"], "validation_r2": val_met["r2"], "rmse": test_met["rmse"], "mae": test_met["mae"], "mape": test_met["mape"], "r2": test_met["r2"], "test_months": ";".join(months[i] for i in test_i), "notes": "Selected on validation; refit on train+validation; test evaluated once"}
    candidate_rows.append({"series_id": series_id, "model": "lstm", "transformation": selected, "selection_split": "validation", "status": "selected_and_tested", "validation_rmse": val_met["rmse"], "validation_mae": val_met["mae"], "validation_mape": val_met["mape"], "validation_r2": val_met["r2"], "test_rmse": test_met["rmse"], "test_mae": test_met["mae"], "test_mape": test_met["mape"], "test_r2": test_met["r2"], "notes": "Test metrics recorded after validation-only selection"})
    return result, {"model": model, "input_scaler": input_scaler, "target_scaler": target_scaler, "transformation": selected, "months": months, "history": [float(value) for value in y]}, candidate_rows


def _recursive_price_forecast(model, transform: str, history: list[float], start_month: str, horizon: int) -> list[float]:
    values = list(history)
    output = []
    for step in range(1, horizon + 1):
        current_month = month_from_index(month_index(start_month) + step)
        lag1, lag2, lag3 = values[-1], values[-2], values[-3]
        rolling = values[-3:]
        features = np.asarray([[lag1, lag2, lag3, float(np.mean(rolling)), float(np.std(rolling, ddof=1)), math.sin(2 * math.pi * (int(current_month[-2:]) - 1) / 12), math.cos(2 * math.pi * (int(current_month[-2:]) - 1) / 12), float(current_month[:4])]], dtype=float)
        transformed = float(_predict_bundle(model, features)[0])
        price = float(_reconstruct(np.asarray([transformed]), np.asarray([lag1]), transform)[0])
        price = max(price, 0.0)
        values.append(price)
        output.append(price)
    return output


def _recursive_lstm_forecast(tf, artifact: dict, history: list[float], start_month: str, horizon: int) -> list[float]:
    values = list(history)
    output = []
    transform = artifact["transformation"]
    for step in range(1, horizon + 1):
        x = np.asarray(values[-12:], dtype=float).reshape(1, 12, 1)
        scaled = artifact["input_scaler"].transform(x.reshape(-1, 1)).reshape(1, 12, 1)
        transformed = float(artifact["target_scaler"].inverse_transform(artifact["model"].predict(scaled, verbose=0).reshape(-1, 1))[0, 0])
        price = float(_reconstruct(np.asarray([transformed]), np.asarray([values[-1]]), transform)[0])
        price = max(price, 0.0)
        values.append(price)
        output.append(price)
    return output


def _environment(xgb_status: str, tf_status: str) -> dict:
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "architecture": platform.architecture()[0], "prefix": sys.prefix, "base_prefix": sys.base_prefix, "xgboost_status": xgb_status, "tensorflow_status": tf_status}


def run_final_validation() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    factories, xgb_error, xgb_status = _xgb_factory_status()
    try:
        import tensorflow as tf

        tf_status = f"available ({tf.__version__})"
    except Exception as exc:
        tf = None
        tf_status = f"unavailable: {exc}"
    _write_csv(REPORT_DIR / "aggregate_training_sample_counts.csv", [])
    feature_groups = _group(_load_feature_rows())
    monthly_groups = _group(_load_monthly_rows())
    all_audit: list[dict] = []
    comparison: list[dict] = []
    lstm_rows: list[dict] = []
    lstm_candidate_rows: list[dict] = []
    forecasts: list[dict] = []
    sample_rows: list[dict] = []
    for series_id in sorted(feature_groups):
        split = _series_split(feature_groups[series_id])
        selected_audit, chosen, test_info = _select_tree_models(series_id, split, factories)
        all_audit.extend(selected_audit)
        test_rows = test_info["test_rows"]
        actual_test = test_info["actual_test"]
        test_months = ";".join(row["month"] for row in test_rows)
        naive_pred = np.asarray([float(row["price_lag_1"]) for row in test_rows])
        naive_met = _metrics(actual_test, naive_pred)
        comparison.append({"series_id": series_id, "model": "naive_last_observation", "transformation": "none", "selection_split": "not_applicable", "test_months": test_months, "n_train": len(split["train"]), "n_validation": len(split["validation"]), "n_test": len(test_rows), "validation_rmse": "", "test_rmse": naive_met["rmse"], "test_mae": naive_met["mae"], "test_mape": naive_met["mape"], "test_r2": naive_met["r2"], "bias": naive_met["bias"], "status": "evaluated", "notes": "One-step previous calendar-month value; identical test dates"})
        for name in MODEL_NAMES:
            item = chosen.get(name, {"status": "unavailable"})
            if item.get("status") != "trained":
                comparison.append({"series_id": series_id, "model": name, "transformation": item.get("transformation", ""), "selection_split": "validation", "test_months": test_months, "n_train": len(split["train"]), "n_validation": len(split["validation"]), "n_test": len(test_rows), "validation_rmse": "", "test_rmse": "", "test_mae": "", "test_mape": "", "test_r2": "", "bias": "", "status": item.get("status", "unavailable"), "notes": "No metric fabricated"})
            else:
                vm = item["validation_metrics"]
                tm = item["test_metrics"]
                comparison.append({"series_id": series_id, "model": name, "transformation": item["transformation"], "selection_split": "validation", "test_months": test_months, "n_train": len(split["train"]) + len(split["validation"]), "n_validation": len(split["validation"]), "n_test": len(test_rows), "validation_rmse": vm["rmse"], "validation_mae": vm["mae"], "validation_mape": vm["mape"], "validation_r2": vm["r2"], "test_rmse": tm["rmse"], "test_mae": tm["mae"], "test_mape": tm["mape"], "test_r2": tm["r2"], "bias": tm["bias"], "status": "evaluated", "notes": item["notes"]})
        monthly = monthly_groups.get(series_id, [])
        observed_monthly = [row for row in monthly if row.get("observed") and row["mean_price"] is not None]
        sample_rows.append({"series_id": series_id, "feature_rows": len(feature_groups[series_id]), "train_rows": len(split["train"]), "validation_rows": len(split["validation"]), "test_rows": len(split["test"]), "observed_months": len(observed_monthly), "first_observed_month": observed_monthly[0]["month"] if observed_monthly else "", "last_observed_month": observed_monthly[-1]["month"] if observed_monthly else ""})
        if observed_monthly and tf is not None:
            lstm_result, lstm_artifact, candidates = _run_lstm(series_id, monthly, tf)
        else:
            lstm_result, lstm_artifact, candidates = _run_lstm(series_id, monthly, None)
        lstm_rows.append(lstm_result)
        lstm_candidate_rows.extend(candidates)
        if lstm_result.get("status") == "trained":
            comparison.append({"series_id": series_id, "model": "lstm", "transformation": lstm_result["selected_transformation"], "selection_split": "validation", "test_months": lstm_result["test_months"], "n_train": lstm_result["train_sequences"] + lstm_result["validation_sequences"], "n_validation": lstm_result["validation_sequences"], "n_test": lstm_result["test_sequences"], "validation_rmse": lstm_result["validation_rmse"], "validation_mae": lstm_result["validation_mae"], "validation_mape": lstm_result["validation_mape"], "validation_r2": lstm_result["validation_r2"], "test_rmse": lstm_result["rmse"], "test_mae": lstm_result["mae"], "test_mape": lstm_result["mape"], "test_r2": lstm_result["r2"], "bias": "", "status": "evaluated", "notes": lstm_result["notes"]})
        else:
            comparison.append({"series_id": series_id, "model": "lstm", "transformation": "", "selection_split": "validation", "test_months": test_months, "n_train": lstm_result.get("train_sequences", 0), "n_validation": lstm_result.get("validation_sequences", 0), "n_test": lstm_result.get("test_sequences", 0), "validation_rmse": "", "test_rmse": "", "test_mae": "", "test_mape": "", "test_r2": "", "bias": "", "status": lstm_result.get("status"), "notes": lstm_result.get("notes", "")})
        # Forecasts use only observed history up to the end of the feature series.
        # Forecast recursion starts from the observed monthly series, not from
        # the compressed feature table.  This prevents a previous available
        # observation from being mistaken for the previous calendar month.
        history = [float(row["mean_price"]) for row in observed_monthly]
        last_month = observed_monthly[-1]["month"] if observed_monthly else feature_groups[series_id][-1]["month"]
        last_three_contiguous = (
            len(observed_monthly) >= 3
            and month_index(observed_monthly[-1]["month"]) - month_index(observed_monthly[-3]["month"]) == 2
        )
        for horizon in (1, 3, 6):
            for model_name in ("naive_last_observation", *MODEL_NAMES):
                if model_name == "naive_last_observation":
                    values = [history[-1]] * horizon
                    transform = "none"
                elif chosen.get(model_name, {}).get("status") == "trained" and last_three_contiguous:
                    values = _recursive_price_forecast(chosen[model_name]["model"], chosen[model_name]["transformation"], history, last_month, horizon)
                    transform = chosen[model_name]["transformation"]
                else:
                    continue
                for offset, value in enumerate(values, 1):
                    forecasts.append({"series_id": series_id, "model": model_name, "transformation": transform, "origin_month": last_month, "forecast_month": month_from_index(month_index(last_month) + offset), "horizon_months": offset, "forecast_price": value, "method": "recursive; actual future prices not used"})
            lstm_history_contiguous = (
                len(observed_monthly) >= 12
                and month_index(observed_monthly[-1]["month"]) - month_index(observed_monthly[-12]["month"]) == 11
            )
            if lstm_artifact is not None and lstm_history_contiguous:
                values = _recursive_lstm_forecast(tf, lstm_artifact, history, last_month, horizon)
                for offset, value in enumerate(values, 1):
                    forecasts.append({"series_id": series_id, "model": "lstm", "transformation": lstm_artifact["transformation"], "origin_month": last_month, "forecast_month": month_from_index(month_index(last_month) + offset), "horizon_months": offset, "forecast_price": value, "method": "recursive; actual future prices not used"})
    _write_csv(REPORT_DIR / "selection_audit.csv", all_audit + lstm_candidate_rows)
    _write_csv(REPORT_DIR / "final_model_comparison.csv", comparison)
    _write_csv(REPORT_DIR / "aggregate_training_sample_counts.csv", sample_rows)
    _write_csv(REPORT_DIR / "lstm_sequence_counts.csv", lstm_rows)
    _write_csv(REPORT_DIR / "final_forecasts.csv", forecasts)
    _write_csv(OUTPUT_DIR / "forecasts" / "final_recursive_forecasts.csv", forecasts)
    _write_csv(REPORT_DIR / "xgboost_status.csv", [{"status": xgb_status, "error": xgb_error or "", "python": sys.executable}])
    _write_csv(REPORT_DIR / "lstm_experiment.csv", lstm_rows)
    environment = _environment(xgb_status, tf_status)
    (REPORT_DIR / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    xgb_lines = ["AgriPrice-NG final-validation XGBoost status", "", f"Interpreter: {sys.executable}", f"Status: {xgb_status}", f"Error: {xgb_error or 'none'}", "", "XGBoost target transformations were fitted on training data and selected by validation RMSE. The selected configuration was refit on train+validation and then evaluated once on the untouched test months.", "", "series_id | selected_transformation | validation_RMSE | test_RMSE | test_MAE | test_MAPE | status"]
    for item in comparison:
        if item.get("model") == "xgboost":
            xgb_lines.append(" | ".join(str(item.get(key, "")) for key in ("series_id", "transformation", "validation_rmse", "test_rmse", "test_mae", "test_mape", "status")))
    xgb_lines.extend(["", "Full candidate rows, including original-price and movement transformations, are in selection_audit.csv."])
    (REPORT_DIR / "xgboost_training_report.txt").write_text("\n".join(xgb_lines), encoding="utf-8")
    lstm_lines = ["AgriPrice-NG final-validation LSTM status", "", f"Interpreter: {sys.executable}", f"TensorFlow: {tf_status}", "", "series_id | sequences | train | validation | test | status | selected_transformation | test_RMSE"]
    for item in lstm_rows:
        lstm_lines.append(" | ".join(str(item.get(key, "")) for key in ("series_id", "sequence_count", "train_sequences", "validation_sequences", "test_sequences", "status", "selected_transformation", "rmse")))
    lstm_lines.extend(["", "The 12-month sequence window requires complete observed calendar history. Sequence counts and validation-selected transformations are also in lstm_sequence_counts.csv and selection_audit.csv."])
    (REPORT_DIR / "lstm_training_report.txt").write_text("\n".join(lstm_lines), encoding="utf-8")
    successful = [row for row in comparison if row.get("status") == "evaluated" and row.get("model") != "naive_last_observation"]
    report = [
        "AgriPrice-NG Phase 3.4 final validation",
        "",
        "Selection protocol: each transformation (original, log1p, price_change, percentage_change) was fitted on the chronological training period and selected using validation RMSE only. The selected configuration was refit on training plus validation rows, then evaluated once on the untouched test months.",
        "",
        "The naive baseline is the previous observed calendar-month value in the feature table. No target prices were interpolated or forward-filled. Movement-target predictions were reconstructed from the lag available before each target month.",
        "",
        f"Series evaluated: {len(feature_groups)}; successful non-naive metric rows: {len(successful)}.",
        f"XGBoost: {xgb_status}.",
        f"TensorFlow: {tf_status}.",
        "",
        "Test dates are comparable within each series. LSTM uses complete 12-month observed sequences; its sequence test dates can differ from the tabular feature test dates because the sequence window requires 12 preceding calendar months.",
        "Recursive forecasts use predicted prices as future history for all subsequent steps; they do not access actual future prices and therefore accumulate uncertainty, especially where historical coverage is irregular.",
        "",
        "See final_model_comparison.csv, selection_audit.csv, lstm_sequence_counts.csv, final_forecasts.csv and environment.json for auditable outputs.",
    ]
    (REPORT_DIR / "final_validation_report.txt").write_text("\n".join(report), encoding="utf-8")
    return {"comparison": comparison, "selection": all_audit, "lstm": lstm_rows, "forecasts": forecasts, "xgboost_status": xgb_status, "tensorflow_status": tf_status}
