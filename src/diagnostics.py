"""Phase 3.3 aggregate forecasting diagnosis and small target experiments.

The module reads the saved aggregate experiment and writes new diagnostic
reports only. It does not overwrite models, plots, forecasts, or market-level
results.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

from .aggregate_preprocessing import AGGREGATE_FEATURE_COLUMNS, AGGREGATE_TARGET_COLUMN, chronological_split
from .aggregate_training import RANDOM_STATE, _factories
from .parquet_compat import month_index


ROOT = Path(__file__).resolve().parents[1]
AGG_PROCESSED = ROOT / "data" / "processed"
AGG_MODELS = ROOT / "models" / "aggregate"
REPORT_DIR = ROOT / "reports" / "diagnostics"


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


def _load_feature_rows() -> list[dict]:
    rows = _read_csv(AGG_PROCESSED / "aggregate_training_features.csv")
    numeric = [*AGGREGATE_FEATURE_COLUMNS, AGGREGATE_TARGET_COLUMN, "mean_price", "n_markets", "n_records"]
    for row in rows:
        for column in numeric:
            row[column] = float(row[column])
    return rows


def _load_monthly_rows() -> list[dict]:
    rows = _read_csv(AGG_PROCESSED / "aggregate_monthly_reindexed.csv")
    for row in rows:
        row["mean_price"] = None if row["mean_price"] == "" else float(row["mean_price"])
        row["n_markets"] = int(row["n_markets"])
    return rows


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if len(actual) == 0:
        return {"bias": None, "mae": None, "rmse": None, "mape": None, "r2": None}
    error = predicted - actual
    nonzero = actual != 0
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(float(np.mean(error**2)))),
        "mape": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0) if np.any(nonzero) else None,
        "r2": None if len(actual) < 2 or denominator == 0 else float(1.0 - np.sum(error**2) / denominator),
    }


def _as_float(value):
    if value in (None, ""):
        return None
    return float(value)


def _model_prediction(artifact_name: str, series_id: str, x_test: np.ndarray) -> np.ndarray:
    artifact = joblib.load(AGG_MODELS / _slug(series_id) / f"{artifact_name}.joblib")
    if artifact_name != "stacking":
        return np.asarray(artifact["model"].predict(x_test), dtype=float)
    bundle = artifact["model"]
    matrix = np.column_stack([model.predict(x_test) for model in bundle["base_models"].values()])
    return np.asarray(bundle["meta_model"].predict(matrix), dtype=float)


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:120]


def _group_and_split(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["series_id"]].append(row)
    output = {}
    for series_id, series_rows in sorted(grouped.items()):
        train, validation, test = chronological_split(series_rows)
        output[series_id] = {"train": train, "validation": validation, "test": test}
    return output


def _direction(value: float, previous: float, tolerance: float = 1e-12) -> str:
    difference = value - previous
    if difference > tolerance:
        return "increase"
    if difference < -tolerance:
        return "decrease"
    return "unchanged"


def _error_row(series_id, model, direction, actual, predicted, train_rows, naive_metrics, test_rows):
    values = _metrics(actual, predicted)
    train_prices = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in train_rows])
    pred_min = float(np.min(predicted)) if len(predicted) else None
    pred_max = float(np.max(predicted)) if len(predicted) else None
    out_count = int(np.sum((predicted < np.min(train_prices)) | (predicted > np.max(train_prices)))) if len(predicted) else 0
    return {
        "series_id": series_id,
        "model": model,
        "direction": direction,
        "n_test": len(actual),
        "test_months": ";".join(row["month"] for row in test_rows),
        "actual_values": ";".join(f"{float(value):.8g}" for value in actual),
        "predicted_values": ";".join(f"{float(value):.8g}" for value in predicted),
        "error_values_predicted_minus_actual": ";".join(
            f"{float(value):.8g}" for value in (predicted - actual)
        ),
        "bias_mean_predicted_minus_actual": values["bias"],
        "mae": values["mae"],
        "rmse": values["rmse"],
        "mape": values["mape"],
        "r2": values["r2"],
        "train_price_min": float(np.min(train_prices)),
        "train_price_max": float(np.max(train_prices)),
        "predicted_min": pred_min,
        "predicted_max": pred_max,
        "predictions_outside_train_range": out_count,
        "naive_rmse": naive_metrics["rmse"],
        "rmse_delta_vs_naive": None if values["rmse"] is None else values["rmse"] - naive_metrics["rmse"],
        "naive_mae": naive_metrics["mae"],
        "mae_delta_vs_naive": None if values["mae"] is None else values["mae"] - naive_metrics["mae"],
    }


def prediction_error_analysis(splits: dict) -> list[dict]:
    rows: list[dict] = []
    for series_id, split in splits.items():
        train_rows, test_rows = split["train"], split["test"]
        actual = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in test_rows])
        x_test = np.asarray([[float(row[column]) for column in AGGREGATE_FEATURE_COLUMNS] for row in test_rows])
        naive = x_test[:, 0]
        naive_metrics = _metrics(actual, naive)
        predictions = {"naive_last_observation": naive}
        for model_name in ("random_forest", "gradient_boosting", "stacking"):
            predictions[model_name] = _model_prediction(model_name, series_id, x_test)
        directions = [
            _direction(float(row[AGGREGATE_TARGET_COLUMN]), float(row["price_lag_1"]))
            for row in test_rows
        ]
        for model_name, predicted in predictions.items():
            rows.append(_error_row(series_id, model_name, "all", actual, predicted, train_rows, naive_metrics, test_rows))
            for direction in ("increase", "decrease", "unchanged"):
                mask = np.asarray([item == direction for item in directions], dtype=bool)
                if np.any(mask):
                    selected_rows = [row for row, keep in zip(test_rows, mask) if keep]
                    rows.append(
                        _error_row(
                            series_id,
                            model_name,
                            direction,
                            actual[mask],
                            predicted[mask],
                            train_rows,
                            naive_metrics,
                            selected_rows,
                        )
                    )
    _write_csv(REPORT_DIR / "prediction_error_analysis.csv", rows)
    return rows


def _fit_oof(factory, x_train, y_train):
    n_splits = 3 if len(y_train) >= 12 else 2
    output = np.full(len(y_train), np.nan)
    if len(y_train) <= n_splits + 1:
        return output
    for train_index, validation_index in TimeSeriesSplit(n_splits=n_splits).split(x_train):
        if len(train_index) < 3:
            continue
        model = factory()
        model.fit(x_train[train_index], y_train[train_index])
        output[validation_index] = model.predict(x_train[validation_index])
    return output


def _fit_target_models(x_train, y_train, x_test, factories):
    predictions = {}
    for name, factory in factories.items():
        model = factory()
        model.fit(x_train, y_train)
        predictions[name] = model.predict(x_test)
    base_names = [name for name in ("random_forest", "gradient_boosting", "xgboost") if name in factories]
    if len(base_names) >= 2:
        oof = {name: _fit_oof(factories[name], x_train, y_train) for name in base_names}
        valid = np.ones(len(y_train), dtype=bool)
        for values in oof.values():
            valid &= np.isfinite(values)
        if int(valid.sum()) >= 4:
            meta = Ridge(alpha=1.0)
            meta.fit(np.column_stack([oof[name][valid] for name in base_names]), y_train[valid])
            fitted_bases = []
            for name in base_names:
                model = factories[name]()
                model.fit(x_train, y_train)
                fitted_bases.append(model.predict(x_test))
            predictions["stacking"] = meta.predict(np.column_stack(fitted_bases))
        else:
            predictions["stacking"] = None
    else:
        predictions["stacking"] = None
    return predictions


def _transform_target(method: str, rows: list[dict]):
    actual = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in rows])
    lag1 = np.asarray([float(row["price_lag_1"]) for row in rows])
    if method == "original":
        return actual, lambda predicted: predicted
    if method == "log1p":
        if np.any(actual < 0):
            raise ValueError("negative prices cannot use log1p")
        return np.log1p(actual), lambda predicted: np.expm1(predicted)
    if method == "price_change":
        return actual - lag1, lambda predicted: predicted + lag1
    if method == "percentage_change":
        if np.any(lag1 == 0):
            raise ValueError("zero lag price prevents percentage-change reconstruction")
        return actual / lag1 - 1.0, lambda predicted: lag1 * (1.0 + predicted)
    raise ValueError(f"Unknown target method: {method}")


def target_transformation_comparison(splits: dict, factories: dict, xgb_note: str | None) -> list[dict]:
    rows: list[dict] = []
    methods = ("original", "log1p", "price_change", "percentage_change")
    for series_id, split in splits.items():
        train_rows, test_rows = split["train"], split["test"]
        x_train = np.asarray([[float(row[column]) for column in AGGREGATE_FEATURE_COLUMNS] for row in train_rows])
        x_test = np.asarray([[float(row[column]) for column in AGGREGATE_FEATURE_COLUMNS] for row in test_rows])
        actual_prices = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in test_rows])
        naive_rmse = _metrics(actual_prices, x_test[:, 0])["rmse"]
        for method in methods:
            try:
                y_train, _ = _transform_target(method, train_rows)
                y_test, inverse = _transform_target(method, test_rows)
                predictions = _fit_target_models(x_train, y_train, x_test, factories)
            except ValueError as exc:
                for model_name in ("random_forest", "gradient_boosting", "stacking"):
                    rows.append({"series_id": series_id, "target_transformation": method, "model": model_name, "status": "not_supported", "notes": str(exc)})
                continue
            for model_name in ("random_forest", "gradient_boosting", "stacking"):
                transformed_prediction = predictions.get(model_name)
                if transformed_prediction is None:
                    rows.append({"series_id": series_id, "target_transformation": method, "model": model_name, "status": "not_trained", "notes": "insufficient chronological OOF predictions"})
                    continue
                price_prediction = inverse(np.asarray(transformed_prediction, dtype=float))
                values = _metrics(y_test if method == "original" else actual_prices, price_prediction)
                rows.append(
                    {
                        "series_id": series_id,
                        "target_transformation": method,
                        "model": model_name,
                        "status": "trained",
                        "n_train": len(train_rows),
                        "n_test": len(test_rows),
                        "test_months": ";".join(row["month"] for row in test_rows),
                        "rmse": values["rmse"],
                        "naive_rmse": naive_rmse,
                        "rmse_delta_vs_naive": values["rmse"] - naive_rmse,
                        "mae": values["mae"],
                        "mape": values["mape"],
                        "r2": values["r2"],
                        "bias_mean_predicted_minus_actual": values["bias"],
                        "reconstruction": {
                            "original": "identity",
                            "log1p": "expm1",
                            "price_change": "predicted_change + observed_lag_1",
                            "percentage_change": "observed_lag_1 * (1 + predicted_percentage_change)",
                        }[method],
                        "notes": "Features and reconstruction use only the historical lag values present at each test date.",
                    }
                )
    _write_csv(REPORT_DIR / "target_transformation_comparison.csv", rows)
    return rows


def forecasting_baseline_comparison(splits: dict) -> list[dict]:
    rows: list[dict] = []
    for series_id, split in splits.items():
        train_rows, test_rows = split["train"], split["test"]
        actual = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in test_rows])
        x_test = np.asarray([[float(row[column]) for column in AGGREGATE_FEATURE_COLUMNS] for row in test_rows])
        naive = x_test[:, 0]
        train_x = np.asarray([[month_index(row["month"])] for row in train_rows], dtype=float)
        train_y = np.asarray([float(row[AGGREGATE_TARGET_COLUMN]) for row in train_rows])
        test_x = np.asarray([[month_index(row["month"])] for row in test_rows], dtype=float)
        trend = LinearRegression().fit(train_x, train_y).predict(test_x)
        predictions = {
            "naive_last_observation": naive,
            "linear_time_trend": trend,
            "random_forest_original": _model_prediction("random_forest", series_id, x_test),
            "gradient_boosting_original": _model_prediction("gradient_boosting", series_id, x_test),
            "stacking_original": _model_prediction("stacking", series_id, x_test),
        }
        naive_values = _metrics(actual, naive)
        for model_name, predicted in predictions.items():
            values = _metrics(actual, predicted)
            rows.append(
                {
                    "series_id": series_id,
                    "model": model_name,
                    "n_train": len(train_rows),
                    "n_test": len(test_rows),
                    "test_months": ";".join(row["month"] for row in test_rows),
                    "bias_mean_predicted_minus_actual": values["bias"],
                    "mae": values["mae"],
                    "rmse": values["rmse"],
                    "mape": values["mape"],
                    "r2": values["r2"],
                    "rmse_delta_vs_naive": values["rmse"] - naive_values["rmse"],
                    "status": "trained",
                    "notes": "Linear trend fits target_price against calendar month index using training rows only." if model_name == "linear_time_trend" else "Same saved aggregate model/test dates as Phase 3.2.",
                }
            )
    _write_csv(REPORT_DIR / "forecasting_baseline_comparison.csv", rows)
    return rows


def _build_report(error_rows, transform_rows, baseline_rows, xgb_note):
    lines = [
        "AgriPrice-NG Phase 3.3 forecasting performance diagnosis",
        "========================================================",
        "",
        "This diagnosis preserves all existing market-level and aggregate outputs. New calculations are written only under reports/diagnostics/.",
        "",
        "1. Alignment and feature checks",
        "--------------------------------",
        "Predictions were reconstructed from the saved aggregate model artifacts and the saved aggregate feature rows.",
        "Each model uses the same chronological test rows as the naive baseline within its series.",
        "The target is aggregate mean_price; units and price types remain separate.",
        "Existing aggregate features are calendar lags 1-3, previous-three-month rolling statistics, month sine/cosine, and year_index.",
        "Log, change, and percentage-change reconstruction uses the historical lag_1 available at each test date; no future target is used.",
        "",
        "2. Error direction and baseline comparison",
        "--------------------------------------------",
    ]
    for series_id in sorted({row["series_id"] for row in error_rows}):
        all_rows = [row for row in error_rows if row["series_id"] == series_id and row["direction"] == "all"]
        lines.append(series_id)
        for row in all_rows:
            bias = float(row["bias_mean_predicted_minus_actual"])
            direction = "overestimates" if bias > 0 else "underestimates" if bias < 0 else "is unbiased"
            lines.append(
                f"  {row['model']}: bias={bias:.2f} ({direction}), RMSE={float(row['rmse']):.2f}, "
                f"MAE={float(row['mae']):.2f}, MAPE={float(row['mape']):.2f}%, "
                f"RMSE delta vs naive={float(row['rmse_delta_vs_naive']):.2f}, "
                f"outside training range={row['predictions_outside_train_range']}"
            )
        direction_rows = [row for row in error_rows if row["series_id"] == series_id and row["direction"] != "all"]
        for row in direction_rows:
            lines.append(
                f"  direction={row['direction']} model={row['model']}: n={row['n_test']}, "
                f"bias={float(row['bias_mean_predicted_minus_actual']):.2f}, RMSE={float(row['rmse']):.2f}"
            )
    lines.extend(
        [
            "",
            "Positive bias means systematic overprediction; negative bias means underprediction.",
            "Predictions outside the training price range are counted against the train-only historical range, avoiding use of test actuals for that check.",
            "",
            "3. Target transformation comparison",
            "------------------------------------",
        ]
    )
    for series_id in sorted({row["series_id"] for row in transform_rows}):
        lines.append(series_id)
        candidates = [row for row in transform_rows if row["series_id"] == series_id and row.get("status") == "trained"]
        best = sorted(candidates, key=lambda row: float(row["rmse"]))[:3]
        for row in best:
            lines.append(
                f"  {row['target_transformation']} + {row['model']}: RMSE={float(row['rmse']):.2f}, "
                f"MAE={float(row['mae']):.2f}, MAPE={float(row['mape']):.2f}%, bias={float(row['bias_mean_predicted_minus_actual']):.2f}, "
                f"RMSE delta vs naive={float(row['rmse_delta_vs_naive']):.2f}"
            )
    lines.extend(
        [
            "",
            "The original, log1p, price-change, and percentage-change rows are all evaluated after reconstructing prices.",
            "Change and percentage-change models are one-step test-date experiments; a multi-step deployment would recursively feed predicted changes and must not substitute future actual lags.",
            "",
            "4. Forecasting baselines",
            "------------------------",
        ]
    )
    for series_id in sorted({row["series_id"] for row in baseline_rows}):
        series_rows = [row for row in baseline_rows if row["series_id"] == series_id]
        best = min(series_rows, key=lambda row: float(row["rmse"]))
        lines.append(
            f"{series_id}: lowest RMSE in this fixed test period was {best['model']} ({float(best['rmse']):.2f}); "
            "this is not treated as a universal winner."
        )
    lines.extend(
        [
            "",
            "5. Diagnosis and recommendation",
            "--------------------------------",
            "The saved tree/stacking models often produce large errors and negative R2 because the test regimes contain price levels and jumps outside what the small training samples represent.",
            "The naive last-observation rule is a strong persistence benchmark for these monthly series and remains difficult to beat without additional explanatory variables or a model designed for level/trend extrapolation.",
            "A log target can stabilize scale, while change or percentage-change targets directly model movement; their results should be judged against the naive baseline per series, not pooled across units.",
            "The simple linear trend is an intentionally limited benchmark and should only be retained where its fixed test-period results justify it.",
            f"XGBoost remains unavailable in this environment: {xgb_note}",
            "TensorFlow/Keras remains unavailable for LSTM training; no LSTM metric was fabricated.",
            "No extensive hyperparameter tuning was performed and no model was forced to beat the naive baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_diagnostics() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    feature_rows = _load_feature_rows()
    _load_monthly_rows()  # Ensure the calendar source used for Phase 3.2 still exists.
    splits = _group_and_split(feature_rows)
    factories, xgb_note = _factories()
    errors = prediction_error_analysis(splits)
    transformations = target_transformation_comparison(splits, factories, xgb_note)
    baselines = forecasting_baseline_comparison(splits)
    report = _build_report(errors, transformations, baselines, xgb_note or "imported successfully")
    (REPORT_DIR / "performance_diagnosis.txt").write_text(report, encoding="utf-8")
    (REPORT_DIR / "experiment_config.json").write_text(
        __import__("json").dumps(
            {
                "source": "saved aggregate_training_features.csv and aggregate model artifacts",
                "target_transformations": ["original", "log1p", "price_change", "percentage_change"],
                "models": ["random_forest", "gradient_boosting", "stacking"],
                "baseline": ["naive_last_observation", "linear_time_trend"],
                "xgboost_status": xgb_note or "available",
                "future_actuals_used_for_reconstruction": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"errors": errors, "transformations": transformations, "baselines": baselines}


if __name__ == "__main__":
    result = run_diagnostics()
    print(
        f"Diagnostics completed: error rows={len(result['errors'])}, "
        f"transformation rows={len(result['transformations'])}, "
        f"baseline rows={len(result['baselines'])}"
    )
