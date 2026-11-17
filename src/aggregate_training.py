"""Separate aggregate-series training experiment for AgriPrice-NG.

This module writes only under models/aggregate, reports/aggregate, and
outputs/aggregate. The original market-level experiment is not imported for
training and its outputs are not overwritten.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .aggregate import build_commodity_monthly_aggregates
from .aggregate_preprocessing import (
    AGGREGATE_FEATURE_COLUMNS,
    AGGREGATE_TARGET_COLUMN,
    APPROVED_AGGREGATE_SERIES,
    chronological_split,
    prepare_aggregate_data,
)
from .data_loader import load_all_rows
from .plotting import write_actual_predicted_svg
from .parquet_compat import month_from_index, month_index


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "aggregate"
REPORT_DIR = ROOT / "reports" / "aggregate"
PLOT_DIR = ROOT / "outputs" / "aggregate" / "plots"
FORECAST_DIR = ROOT / "outputs" / "aggregate" / "forecasts"
RANDOM_STATE = 42


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:120]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if len(actual) == 0:
        return {"rmse": None, "mae": None, "mape": None, "r2": None}
    error = predicted - actual
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
        if np.any(nonzero)
        else None
    )
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "rmse": float(math.sqrt(float(np.mean(error**2)))),
        "mae": float(np.mean(np.abs(error))),
        "mape": mape,
        "r2": None if len(actual) < 2 or denominator == 0 else float(1.0 - np.sum(error**2) / denominator),
    }


def _metric_row(
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


def _factories() -> tuple[dict, str | None]:
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
        return factories, None
    except Exception as exc:  # ImportError and binary/version failures are both reportable.
        return factories, f"XGBoost unavailable or failed to import: {exc}"


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[column]) for column in AGGREGATE_FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in rows], dtype=float)
    return x, y


def _oof(factory, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_splits = 3 if len(y) >= 12 else 2
    if len(y) <= n_splits + 1:
        return np.full(len(y), np.nan)
    output = np.full(len(y), np.nan)
    for train_index, validation_index in TimeSeriesSplit(n_splits=n_splits).split(x):
        if len(train_index) < 3:
            continue
        model = factory()
        model.fit(x[train_index], y[train_index])
        output[validation_index] = model.predict(x[validation_index])
    return output


def _fit_stack(train_rows, validation_rows, test_rows, factories):
    base_names = [name for name in ("random_forest", "gradient_boosting", "xgboost") if name in factories]
    if len(base_names) < 2:
        return None, None, "fewer than two base models available"
    x_train, y_train = _arrays(train_rows)
    x_test, _ = _arrays(test_rows)
    oof_by_model = {name: _oof(factories[name], x_train, y_train) for name in base_names}
    valid = np.ones(len(y_train), dtype=bool)
    for values in oof_by_model.values():
        valid &= np.isfinite(values)
    if int(valid.sum()) < 4:
        return None, None, "insufficient chronological out-of-fold predictions"
    meta = Ridge(alpha=1.0)
    meta.fit(np.column_stack([oof_by_model[name][valid] for name in base_names]), y_train[valid])
    base_models = {}
    test_predictions = []
    validation_predictions = []
    x_validation = _arrays(validation_rows)[0] if validation_rows else None
    for name in base_names:
        model = factories[name]()
        model.fit(x_train, y_train)
        base_models[name] = model
        test_predictions.append(model.predict(x_test))
        if x_validation is not None:
            validation_predictions.append(model.predict(x_validation))
    bundle = {
        "base_models": base_models,
        "meta_model": meta,
        "feature_columns": AGGREGATE_FEATURE_COLUMNS,
        "target_column": AGGREGATE_TARGET_COLUMN,
        "base_names": base_names,
    }
    test_prediction = meta.predict(np.column_stack(test_predictions))
    validation_prediction = (
        meta.predict(np.column_stack(validation_predictions)) if validation_predictions else None
    )
    return bundle, {"test": test_prediction, "validation": validation_prediction}, (
        "trained with chronological out-of-fold base predictions; bases=" + ",".join(base_names)
    )


def _save_model(series_id: str, name: str, model, metadata: dict) -> Path:
    path = MODEL_DIR / _slug(series_id)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"{name}.joblib"
    joblib.dump({"model": model, "metadata": metadata}, output)
    return output


def _build_sequences(monthly_rows: list[dict], window: int = 12):
    values = {row["month"]: row["mean_price"] for row in monthly_rows}
    sequences = []
    targets = []
    months = []
    for month in sorted(values, key=month_index):
        current_index = month_index(month)
        history_months = [month_from_index(current_index - offset) for offset in range(window, 0, -1)]
        history = [values.get(history_month) for history_month in history_months]
        target = values.get(month)
        if target is None or any(value is None for value in history):
            continue
        sequences.append([float(value) for value in history])
        targets.append(float(target))
        months.append(month)
    return sequences, targets, months


def _sequence_split(n: int):
    train_end = max(1, int(n * 0.70))
    validation_end = max(train_end + 1, int(n * 0.85)) if n >= 3 else n
    validation_end = min(validation_end, n)
    return list(range(train_end)), list(range(train_end, validation_end)), list(range(validation_end, n))


def _run_lstm(prepared: dict) -> list[dict]:
    monthly_by_series: dict[str, list[dict]] = defaultdict(list)
    for row in prepared["monthly_rows"]:
        monthly_by_series[row["series_id"]].append(row)
    try:
        import tensorflow as tf

        tensorflow_note = "TensorFlow/Keras available"
    except Exception as exc:
        tf = None
        tensorflow_note = f"TensorFlow/Keras unavailable: {exc}"

    rows: list[dict] = []
    for series_id, monthly_rows in sorted(monthly_by_series.items()):
        sequences, targets, target_months = _build_sequences(monthly_rows)
        train_indices, validation_indices, test_indices = _sequence_split(len(sequences)) if sequences else ([], [], [])
        result = {
            "series_id": series_id,
            "sequence_window": 12,
            "sequence_count": len(sequences),
            "train_sequences": len(train_indices),
            "validation_sequences": len(validation_indices),
            "test_sequences": len(test_indices),
            "rmse": None,
            "mae": None,
            "mape": None,
            "r2": None,
            "status": "not_trained",
            "notes": "",
        }
        if len(sequences) < 20:
            result["status"] = "insufficient_sequences"
            result["notes"] = f"Fewer than 20 complete 12-month sequences. {tensorflow_note}"
            rows.append(result)
            continue
        if tf is None:
            result["status"] = "missing_tensorflow"
            result["notes"] = tensorflow_note
            rows.append(result)
            continue

        np.random.seed(RANDOM_STATE)
        tf.random.set_seed(RANDOM_STATE)
        x = np.asarray(sequences, dtype=float)
        y = np.asarray(targets, dtype=float)
        x_train, y_train = x[train_indices], y[train_indices]
        x_validation, y_validation = x[validation_indices], y[validation_indices]
        x_test, y_test = x[test_indices], y[test_indices]
        input_scaler = StandardScaler().fit(x_train.reshape(-1, 1))
        target_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        x_train_scaled = input_scaler.transform(x_train.reshape(-1, 1)).reshape(-1, 12, 1)
        x_validation_scaled = input_scaler.transform(x_validation.reshape(-1, 1)).reshape(-1, 12, 1)
        x_test_scaled = input_scaler.transform(x_test.reshape(-1, 1)).reshape(-1, 12, 1)
        y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_validation_scaled = target_scaler.transform(y_validation.reshape(-1, 1)).ravel()
        model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(12, 1)), tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            x_train_scaled,
            y_train_scaled,
            validation_data=(x_validation_scaled, y_validation_scaled),
            epochs=100,
            batch_size=8,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        )
        predicted_scaled = model.predict(x_test_scaled, verbose=0).ravel()
        predicted = target_scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()
        result.update(_metric(y_test, predicted))
        result["status"] = "trained"
        result["notes"] = "Small 12-month LSTM with early stopping; scalers fit on training sequences only."
        model_path = MODEL_DIR / _slug(series_id)
        model_path.mkdir(parents=True, exist_ok=True)
        model.save(model_path / "lstm.keras")
        joblib.dump(
            {"input_scaler": input_scaler, "target_scaler": target_scaler},
            model_path / "lstm_preprocessing.joblib",
        )
        write_actual_predicted_svg(
            PLOT_DIR / f"{_slug(series_id)}__lstm.svg",
            f"{series_id} — LSTM",
            [target_months[index] for index in test_indices],
            y_test.tolist(),
            predicted.tolist(),
        )
        rows.append(result)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(REPORT_DIR / "lstm_experiment.csv", rows)
    return rows


def _forecast_vector(history: dict[str, float], month: str) -> list[float] | None:
    current = month_index(month)
    lag_values = [history.get(month_from_index(current - lag)) for lag in (1, 2, 3)]
    rolling_values = [history.get(month_from_index(current - offset)) for offset in (1, 2, 3)]
    if any(value is None for value in lag_values + rolling_values):
        return None
    mean_value = sum(rolling_values) / 3.0
    std_value = math.sqrt(sum((value - mean_value) ** 2 for value in rolling_values) / 3.0)
    month_number = int(month.split("-")[1])
    year = int(month.split("-")[0])
    angle = 2.0 * math.pi * (month_number - 1) / 12.0
    return [
        float(lag_values[0]),
        float(lag_values[1]),
        float(lag_values[2]),
        mean_value,
        std_value,
        math.sin(angle),
        math.cos(angle),
        float(year),
    ]


def _recursive_forecast(model, monthly_rows: list[dict], series_id: str, model_name: str) -> list[dict]:
    observed = {row["month"]: float(row["mean_price"]) for row in monthly_rows if row["mean_price"] is not None}
    last_month = max(observed, key=month_index)
    last_observed_row = max(
        (row for row in monthly_rows if row["mean_price"] is not None), key=lambda row: month_index(row["month"])
    )
    predicted_by_step: dict[int, float] = {}
    failed = False
    history = dict(observed)
    for step in range(1, 7):
        target_month = month_from_index(month_index(last_month) + step)
        vector = _forecast_vector(history, target_month)
        if vector is None:
            failed = True
            break
        prediction = float(model.predict(np.asarray([vector], dtype=float))[0])
        history[target_month] = prediction
        predicted_by_step[step] = prediction
    output = []
    for horizon in (1, 3, 6):
        target_month = month_from_index(month_index(last_month) + horizon)
        output.append(
            {
                "series_id": series_id,
                "model": model_name,
                "horizon_months": horizon,
                "forecast_month": target_month,
                "forecast_price": predicted_by_step.get(horizon),
                "last_observed_month": last_month,
                "last_observed_n_markets": last_observed_row["n_markets"],
                "status": "recursive_forecast"
                if horizon in predicted_by_step
                else ("unavailable_missing_calendar_history" if failed else "unavailable_forecast_horizon"),
            }
        )
    return output


class _NaivePredictor:
    def predict(self, values):
        return np.asarray(values)[:, 0]


def _verify_source_data(prepared: dict) -> list[dict]:
    """Compare the saved aggregate rows with a fresh raw-row aggregation."""

    raw_rows = load_all_rows()
    recomputed = build_commodity_monthly_aggregates(raw_rows)
    approved = set(APPROVED_AGGREGATE_SERIES)
    raw_lookup = {
        (row["commodity"], row["unit"], row["price_type"], row["month"]): row
        for row in recomputed
        if (row["commodity"], row["unit"], row["price_type"]) in approved
    }
    source_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for row in prepared["source_rows"]:
        source_by_key[(row["commodity"], row["unit"], row["price_type"])].append(row)
    raw_market_sets: dict[tuple, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in raw_rows:
        key = (row["commodity"], row["unit"], row["pricetype"])
        if key in approved:
            raw_market_sets[key][row["date"].strftime("%Y-%m")].add(str(row["market"]))

    verification: list[dict] = []
    for key in APPROVED_AGGREGATE_SERIES:
        group = source_by_key[key]
        months = sorted((row["month"] for row in group), key=month_index)
        market_sets = raw_market_sets[key]
        ordered_market_sets = [market_sets[month] for month in sorted(market_sets, key=month_index)]
        composition_changes = sum(
            previous != current for previous, current in zip(ordered_market_sets, ordered_market_sets[1:])
        )
        unique_compositions = {tuple(sorted(markets)) for markets in ordered_market_sets}
        mismatch_count = 0
        for row in group:
            lookup = raw_lookup.get((row["commodity"], row["unit"], row["price_type"], row["month"]))
            if lookup is None:
                mismatch_count += 1
                continue
            if (
                abs(float(row["mean_price"]) - float(lookup["mean_price"])) > 1e-8
                or abs(float(row["median_price"]) - float(lookup["median_price"])) > 1e-8
                or int(row["n_markets"]) != int(lookup["n_markets"])
                or int(row["n_records"]) != int(lookup["n_records"])
            ):
                mismatch_count += 1
        counts = [int(row["n_markets"]) for row in group]
        verification.append(
            {
                "series_id": " | ".join(key),
                "commodity": key[0],
                "unit": key[1],
                "price_type": key[2],
                "observation_count": len(group),
                "first_month": months[0],
                "last_month": months[-1],
                "duplicate_group_months": len(group) - len(set(months)),
                "raw_aggregate_value_mismatches": mismatch_count,
                "min_markets_per_month": min(counts),
                "max_markets_per_month": max(counts),
                "average_markets_per_month": round(statistics.fmean(counts), 4),
                "unique_markets_over_period": len(set().union(*market_sets.values())),
                "market_composition_change_count": composition_changes,
                "unique_market_compositions": len(unique_compositions),
                "first_observed_market_set": ";".join(sorted(ordered_market_sets[0])),
                "last_observed_market_set": ";".join(sorted(ordered_market_sets[-1])),
                "currency_values": group[0].get("currency_values", ""),
                "calculation_check": "passed" if mismatch_count == 0 else "failed",
            }
        )
    _write_csv(REPORT_DIR / "data_verification.csv", verification)
    return verification


def _build_report(prepared, diagnostics, verification, metric_rows, lstm_rows, forecasts, xgb_note):
    lines = [
        "AgriPrice-NG Phase 3.2 aggregate-series training report",
        "=======================================================",
        "",
        "This is a separate experiment. Existing market-level models, reports, parameters, and outputs were preserved.",
        "Target: monthly unweighted aggregate mean_price, kept separate by commodity, exact unit, and price type.",
        "No missing target prices were interpolated or forward-filled.",
        "",
        "1. Aggregate source verification",
        "--------------------------------",
    ]
    for row in verification:
        lines.append(
            f"{row['series_id']}: n={row['observation_count']}, {row['first_month']} to {row['last_month']}, "
            f"markets/month={row['min_markets_per_month']}-{row['max_markets_per_month']} "
            f"(avg {row['average_markets_per_month']:.2f}), "
            f"composition_changes={row['market_composition_change_count']}, "
            f"calculation={row['calculation_check']}"
        )
    lines.extend(
        [
            "The aggregate calculation was independently recomputed from raw records: mean_price and median_price are",
            "record-level statistics, while n_markets is the number of distinct market names contributing in that month.",
            "Market composition changes are retained as verification metadata; n_markets is not used as a future-price feature.",
            "",
            "2. Preparation and sample counts",
            "--------------------------------",
            "Features use exact calendar-month lags 1, 2, and 3; rolling mean/std over the previous three calendar months;",
            "month sine/cosine; and calendar year_index. Rows with unavailable historical months are excluded.",
            f"Calendar rows: {len(prepared['monthly_rows']):,}; observed aggregate rows: {len(prepared['source_rows']):,}; usable feature rows: {len(prepared['feature_rows']):,}",
        ]
    )
    split_map = defaultdict(list)
    for row in prepared["feature_rows"]:
        split_map[row["series_id"]].append(row)
    lines.append("Exact test months by series:")
    for diagnostic in diagnostics:
        rows = split_map[diagnostic["series_id"]]
        train, validation, test = chronological_split(rows)
        lines.append(
            f"{diagnostic['series_id']}: observed={diagnostic['observed_rows']}, calendar={diagnostic['calendar_rows']}, "
            f"usable={diagnostic['usable_feature_rows']}, excluded={diagnostic['excluded_missing_history_rows']}, "
            f"split={len(train)}/{len(validation)}/{len(test)}"
        )
        lines.append(f"  test_months={','.join(row['month'] for row in test)}")
    lines.extend(
        [
            "",
            "3. Model results",
            "-----------------",
            "The split is chronological 70/15/15. Models within a series use identical test months.",
            "MAPE excludes zero actual prices. Metrics are actual run results; missing-dependency models have no fabricated metrics.",
        ]
    )
    for series_id in sorted({row["series_id"] for row in metric_rows}):
        lines.append(series_id)
        for row in metric_rows:
            if row["series_id"] != series_id or row["split"] != "test":
                continue
            if row["status"] == "trained":
                values = " | ".join(
                    "NA" if row[field] in (None, "") else f"{float(row[field]):.4f}"
                    for field in ("rmse", "mae", "mape", "r2")
                )
                lines.append(f"  {row['model']}: RMSE | MAE | MAPE% | R2 = {values}; n_test={row['n_test']}")
            else:
                lines.append(f"  {row['model']}: {row['status']}; {row['notes']}")
    lines.extend(
        [
            "",
            "4. XGBoost and LSTM status",
            "---------------------------",
            xgb_note or "XGBoost imported successfully and was trained.",
        ]
    )
    for row in lstm_rows:
        lines.append(
            f"{row['series_id']}: sequences={row['sequence_count']}, "
            f"train/validation/test={row['train_sequences']}/{row['validation_sequences']}/{row['test_sequences']}, "
            f"status={row['status']}; {row['notes']}"
        )
    lines.extend(
        [
            "LSTM uses a 12-month window, training-only StandardScaler objects, inverse transformation, and early stopping when TensorFlow is available.",
            "",
            "5. Recursive forecasts",
            "----------------------",
            "Forecasts for 1, 3, and 6 months are generated recursively. Each prediction is added to the working history",
            "for later horizons; actual future aggregate prices are never read. Missing historical calendar features stop recursion.",
        ]
    )
    status_counts = defaultdict(int)
    for row in forecasts:
        status_counts[row["status"]] += 1
    lines.append("Forecast statuses: " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))
    lines.extend(
        [
            "",
            "Interpretation limitations",
            "--------------------------",
            "Aggregation increases the number of usable monthly rows and can reduce market-specific noise, but it hides local market dispersion.",
            "The identity and number of contributing markets change over time, so an aggregate forecast is not a forecast for every individual market.",
            "The six series are evaluated independently; no units or retail/wholesale prices are combined.",
        ]
    )
    return "\n".join(lines) + "\n"


def train_aggregate() -> dict:
    prepared = prepare_aggregate_data()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    verification = _verify_source_data(prepared)
    _write_csv(REPORT_DIR / "training_sample_counts.csv", prepared["diagnostics"])
    diagnostics = prepared["diagnostics"]
    split_rows: dict[str, dict[str, list[dict]]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in prepared["feature_rows"]:
        grouped[row["series_id"]].append(row)
    for series_id, rows in grouped.items():
        train, validation, test = chronological_split(rows)
        split_rows[series_id] = {"train": train, "validation": validation, "test": test}

    factories, xgb_note = _factories()
    metric_rows: list[dict] = []
    prediction_records: list[dict] = []
    fitted_models: dict[str, dict[str, object]] = defaultdict(dict)
    for series_id in sorted(split_rows):
        split = split_rows[series_id]
        train_rows, validation_rows, test_rows = split["train"], split["validation"], split["test"]
        n_train, n_validation, n_test = len(train_rows), len(validation_rows), len(test_rows)
        if n_train < 8 or n_test < 2:
            note = "insufficient chronological rows for a train/test comparison"
            for name in ("naive_last_calendar_month", "random_forest", "xgboost", "gradient_boosting", "stacking"):
                metric_rows.append(_metric_row(series_id, name, "test", n_train, n_validation, n_test, {}, "insufficient_data", note))
            continue

        x_train, y_train = _arrays(train_rows)
        x_validation, y_validation = _arrays(validation_rows) if validation_rows else (None, None)
        x_test, y_test = _arrays(test_rows)
        months_test = [row["month"] for row in test_rows]
        naive_prediction = x_test[:, 0]
        metric_rows.append(
            _metric_row(
                series_id,
                "naive_last_calendar_month",
                "test",
                n_train,
                n_validation,
                n_test,
                _metric(y_test, naive_prediction),
                notes="Uses aggregate mean_price from the immediately preceding calendar month.",
            )
        )
        prediction_records.append({"series_id": series_id, "model": "naive_last_calendar_month", "months": months_test, "actual": y_test, "predicted": naive_prediction, "baseline": None})

        for model_name, factory in factories.items():
            model = factory()
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            model_path = _save_model(series_id, model_name, model, {"feature_columns": AGGREGATE_FEATURE_COLUMNS, "target_column": AGGREGATE_TARGET_COLUMN, "series_id": series_id})
            fitted_models[series_id][model_name] = model
            metric_rows.append(_metric_row(series_id, model_name, "test", n_train, n_validation, n_test, _metric(y_test, prediction), notes=f"Saved to {model_path}"))
            prediction_records.append({"series_id": series_id, "model": model_name, "months": months_test, "actual": y_test, "predicted": prediction, "baseline": naive_prediction})
            if validation_rows:
                validation_prediction = model.predict(x_validation)
                metric_rows.append(_metric_row(series_id, model_name, "validation", n_train, n_validation, n_test, _metric(y_validation, validation_prediction), notes="Chronological validation period."))

        if xgb_note:
            metric_rows.append(_metric_row(series_id, "xgboost", "test", n_train, n_validation, n_test, {}, "not_trained_missing_dependency", xgb_note))

        stack_bundle, stack_predictions, stack_note = _fit_stack(train_rows, validation_rows, test_rows, factories)
        if stack_bundle is None:
            metric_rows.append(_metric_row(series_id, "stacking", "test", n_train, n_validation, n_test, {}, "not_trained", stack_note))
        else:
            stack_path = _save_model(series_id, "stacking", stack_bundle, {"feature_columns": AGGREGATE_FEATURE_COLUMNS, "target_column": AGGREGATE_TARGET_COLUMN, "series_id": series_id, "meta_training": "chronological out-of-fold predictions"})
            fitted_models[series_id]["stacking"] = stack_bundle
            stack_prediction = stack_predictions["test"]
            metric_rows.append(_metric_row(series_id, "stacking", "test", n_train, n_validation, n_test, _metric(y_test, stack_prediction), notes=f"Saved to {stack_path}; {stack_note}"))
            prediction_records.append({"series_id": series_id, "model": "stacking", "months": months_test, "actual": y_test, "predicted": stack_prediction, "baseline": naive_prediction})

    lstm_rows = _run_lstm(prepared)
    for row in lstm_rows:
        diagnostic = next(item for item in diagnostics if item["series_id"] == row["series_id"])
        split = split_rows[row["series_id"]]
        metric_rows.append(_metric_row(row["series_id"], "lstm", "test", len(split["train"]), len(split["validation"]), len(split["test"]), row, row["status"], row["notes"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_columns": AGGREGATE_FEATURE_COLUMNS,
            "target_column": AGGREGATE_TARGET_COLUMN,
            "target_source": "mean_price",
            "approved_series": [list(key) for key in APPROVED_AGGREGATE_SERIES],
        },
        MODEL_DIR / "preprocessing_metadata.joblib",
    )

    forecast_rows: list[dict] = []
    monthly_by_series: dict[str, list[dict]] = defaultdict(list)
    for row in prepared["monthly_rows"]:
        monthly_by_series[row["series_id"]].append(row)
    for series_id, model_map in fitted_models.items():
        models_for_forecast = {"naive_last_calendar_month": _NaivePredictor(), **model_map}
        for model_name, model in models_for_forecast.items():
            if model_name == "stacking":
                bundle = model

                class _StackingPredictor:
                    def predict(self, values):
                        matrix = np.column_stack([base.predict(values) for base in bundle["base_models"].values()])
                        return bundle["meta_model"].predict(matrix)

                model = _StackingPredictor()
            forecast_rows.extend(_recursive_forecast(model, monthly_by_series[series_id], series_id, model_name))
    _write_csv(FORECAST_DIR / "recursive_forecasts.csv", forecast_rows)

    for record in prediction_records:
        write_actual_predicted_svg(
            PLOT_DIR / f"{_slug(record['series_id'])}__{record['model']}.svg",
            f"{record['series_id']} — {record['model']}",
            record["months"],
            record["actual"].tolist(),
            record["predicted"].tolist(),
            record["baseline"].tolist() if record["baseline"] is not None else None,
        )
    _write_csv(REPORT_DIR / "model_comparison.csv", metric_rows)
    _write_csv(REPORT_DIR / "individual_series_metrics.csv", metric_rows)
    _write_csv(
        REPORT_DIR / "xgboost_status.csv",
        [row for row in metric_rows if row["model"] == "xgboost" and row["split"] == "test"],
    )
    config = {
        "approved_series": [list(key) for key in APPROVED_AGGREGATE_SERIES],
        "target": AGGREGATE_TARGET_COLUMN,
        "aggregate_source_column": "mean_price",
        "feature_columns": AGGREGATE_FEATURE_COLUMNS,
        "chronological_split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "random_state": RANDOM_STATE,
        "xgboost_status": xgb_note or "available",
        "lstm_window": 12,
        "lstm_statuses": sorted({row["status"] for row in lstm_rows}),
    }
    (REPORT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = _build_report(prepared, diagnostics, verification, metric_rows, lstm_rows, forecast_rows, xgb_note or "XGBoost imported successfully and was trained.")
    (REPORT_DIR / "model_performance_report.txt").write_text(report, encoding="utf-8")
    return {"prepared": prepared, "metrics": metric_rows, "lstm": lstm_rows, "forecasts": forecast_rows}
