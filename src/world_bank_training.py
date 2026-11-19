"""Retired historical World Bank model experiment.

This runner trains only on the normalized World Bank release under
``data/processed/world_bank`` and writes only below ``models/world_bank``,
``reports/world_bank`` and ``outputs/world_bank``.  The WFP experiments are
not imported or overwritten.
"""

from __future__ import annotations

import csv
import json
import math
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .parquet_compat import month_from_index, month_index
from .plotting import write_actual_predicted_svg
from .world_bank_ingestion import NORMALIZED_FILE, prepare_world_bank_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "world_bank"
REPORT_ROOT = ROOT / "reports" / "world_bank"
OUTPUT_ROOT = ROOT / "outputs" / "world_bank"
FORECAST_ROOT = OUTPUT_ROOT / "forecasts"
CHAMPION_MODEL_ROOT = MODEL_ROOT / "champions"
CHAMPION_FORECAST_ROOT = FORECAST_ROOT / "champions"
CHAMPION_REGISTRY_PATH = REPORT_ROOT / "champion_models.json"
CHAMPION_EXPERIMENT_VERSION = "world-bank-champion-v1"
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
TRANSFORMS = ("original", "log1p", "price_change", "percentage_change")
# A defensible first external-data experiment should remain reviewable and
# runnable on a student laptop.  The normalized table still contains every
# commodity/location; this list controls only the initial trained panel.
TRAIN_TICKERS = ("beans", "rice", "millet", "maize_fao", "sorghum_fao", "yam")
MODEL_NAMES = ("random_forest", "xgboost", "gradient_boosting", "stacking")
RANDOM_STATE = 42
# Lower rank means a simpler model when validation scores tie exactly.  The
# registry stores this rule so a future rerun cannot make an arbitrary choice.
MODEL_SIMPLICITY_ORDER = (
    "naive_last_observation",
    "gradient_boosting",
    "random_forest",
    "xgboost",
    "stacking",
)
TRANSFORM_SIMPLICITY_ORDER = ("original", "log1p", "price_change", "percentage_change")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:120]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_if_missing(path: Path, rows: list[dict[str, Any]]) -> None:
    """Preserve an existing research output while allowing first-run setup."""

    if not path.exists():
        _write_csv(path, rows)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not len(actual):
        return {"rmse": None, "mae": None, "mape": None, "r2": None, "bias": None}
    error = predicted - actual
    valid = np.abs(actual) > 1e-12
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error[valid] / actual[valid])) * 100) if np.any(valid) else None,
        "r2": None if len(actual) < 2 or denominator == 0 else float(1 - np.sum(error**2) / denominator),
        "bias": float(np.mean(error)),
    }


def _candidate_rmse(candidate: dict[str, Any]) -> float | None:
    metrics = candidate.get("validation_metrics") or {}
    value = candidate.get("validation_rmse", metrics.get("rmse"))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def select_model_transform(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select one transformation for a model using validation RMSE only.

    This small pure helper is intentionally public for audit tests.  Candidate
    rows may contain either ``validation_rmse`` or a ``validation_metrics``
    mapping.  Failed/unavailable rows are ignored, but are retained by the
    caller's audit output.
    """

    valid = [row for row in candidates if row.get("status", "evaluated") in {"evaluated", "trained", "selected"} and _candidate_rmse(row) is not None]
    if not valid:
        return None
    transform_rank = {name: index for index, name in enumerate(TRANSFORM_SIMPLICITY_ORDER)}
    return min(
        valid,
        key=lambda row: (
            _candidate_rmse(row),
            transform_rank.get(str(row.get("transformation", "")), len(transform_rank)),
            str(row.get("transformation", "")),
        ),
    )


def select_champion(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select exactly one series champion from validation candidates.

    Test metrics are deliberately never read here.  Ties are deterministic:
    validation RMSE first, then the documented model simplicity order, then
    model name and transformation for complete reproducibility.
    """

    valid = [row for row in candidates if row.get("status", "evaluated") in {"evaluated", "trained", "selected"} and _candidate_rmse(row) is not None]
    if not valid:
        return None
    model_rank = {name: index for index, name in enumerate(MODEL_SIMPLICITY_ORDER)}
    transform_rank = {name: index for index, name in enumerate(TRANSFORM_SIMPLICITY_ORDER)}
    return min(
        valid,
        key=lambda row: (
            _candidate_rmse(row),
            model_rank.get(str(row.get("model", "")), len(model_rank)),
            transform_rank.get(str(row.get("transformation", "")), len(transform_rank)),
            str(row.get("model", "")),
            str(row.get("transformation", "")),
        ),
    )


def _atomic_json_write(path: Path, payload: Any) -> None:
    """Write JSON beside the destination, replacing it only when complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
        Path(temp_name).replace(path)
    finally:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _target(rows: list[dict], transform: str) -> tuple[np.ndarray, np.ndarray]:
    prices = np.asarray([float(row["target_price"]) for row in rows])
    lags = np.asarray([float(row["price_lag_1"]) for row in rows])
    if transform == "original":
        return prices, lags
    if transform == "log1p":
        if np.any(prices < 0):
            raise ValueError("log1p cannot represent negative prices")
        return np.log1p(prices), lags
    if transform == "price_change":
        return prices - lags, lags
    if transform == "percentage_change":
        if np.any(np.abs(lags) <= 1e-12):
            raise ValueError("percentage-change target has a zero lag")
        return prices / lags - 1.0, lags
    raise ValueError(transform)


def _reconstruct(values: np.ndarray, lags: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "original":
        return values
    if transform == "log1p":
        return np.maximum(np.expm1(values), 0.0)
    if transform == "price_change":
        return lags + values
    if transform == "percentage_change":
        return lags * (1 + values)
    raise ValueError(transform)


def _factories() -> tuple[dict[str, Callable[[], Any]], str | None]:
    factories: dict[str, Callable[[], Any]] = {
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=60, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=60, learning_rate=0.04, max_depth=2, random_state=RANDOM_STATE
        ),
    }
    try:
        from xgboost import XGBRegressor
        import xgboost

        factories["xgboost"] = lambda: XGBRegressor(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        return factories, f"available ({xgboost.__version__})"
    except Exception as exc:
        return factories, f"unavailable: {exc}"


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.asarray([float(row["target_price"]) for row in rows], dtype=float)
    return x, y


def _oof(factory: Callable[[], Any], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    splits = 3 if len(y) >= 18 else 2
    output = np.full(len(y), np.nan)
    if len(y) <= splits + 1:
        return output
    for train_idx, valid_idx in TimeSeriesSplit(n_splits=splits).split(x):
        model = factory()
        model.fit(x[train_idx], y[train_idx])
        output[valid_idx] = model.predict(x[valid_idx])
    return output


def _fit_model(name: str, factories: dict[str, Callable[[], Any]], x: np.ndarray, y: np.ndarray, x_eval: np.ndarray):
    if name != "stacking":
        model = factories[name]()
        model.fit(x, y)
        return model, np.asarray(model.predict(x_eval), dtype=float), "direct fit"
    base_names = [item for item in ("random_forest", "gradient_boosting", "xgboost") if item in factories]
    if len(base_names) < 2:
        raise RuntimeError("fewer than two base models available")
    oof = {name_: _oof(factories[name_], x, y) for name_ in base_names}
    valid = np.ones(len(y), dtype=bool)
    for values in oof.values():
        valid &= np.isfinite(values)
    if int(valid.sum()) < 8:
        raise RuntimeError("insufficient chronological out-of-fold predictions")
    meta = Ridge(alpha=1.0)
    meta.fit(np.column_stack([oof[name_][valid] for name_ in base_names]), y[valid])
    bases: dict[str, Any] = {}
    predictions = []
    for name_ in base_names:
        fitted = factories[name_]()
        fitted.fit(x, y)
        bases[name_] = fitted
        predictions.append(fitted.predict(x_eval))
    bundle = {
        "base_models": bases,
        "meta_model": meta,
        "base_names": base_names,
        "feature_columns": FEATURE_COLUMNS,
    }
    return bundle, np.asarray(meta.predict(np.column_stack(predictions)), dtype=float), "chronological OOF base predictions + Ridge meta-model"


def _save_model(series_id: str, model_name: str, model: Any, metadata: dict[str, Any]) -> Path:
    path = MODEL_ROOT / _slug(series_id)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"{model_name}.joblib"
    # Existing research artifacts are immutable inputs to the audit.  A
    # rerun may create a missing artifact, but never replaces one produced by
    # the earlier experiment.
    if not output.exists():
        joblib.dump({"model": model, "metadata": metadata}, output)
    return output


def _save_champion_model(series_id: str, model: Any, metadata: dict[str, Any]) -> Path:
    CHAMPION_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    output = CHAMPION_MODEL_ROOT / f"{_slug(series_id)}.joblib"
    joblib.dump({"model": model, "metadata": metadata}, output)
    return output


def _feature_rows(normalized: pd.DataFrame, levels: tuple[str, ...] = ("national", "market_average")) -> tuple[list[dict], list[dict]]:
    filtered = normalized[normalized["geography_level"].isin(levels)].copy()
    filtered = filtered[filtered["commodity_ticker"].isin(TRAIN_TICKERS)]
    # The food-price index has an index unit rather than a commodity price unit;
    # it is retained in normalized data but excluded from this price experiment.
    filtered = filtered[filtered["commodity_ticker"] != "food_price_index"]
    grouped = filtered.groupby(["commodity_ticker", "geography_level", "geography_name", "commodity", "unit", "price_type"], sort=True)
    features: list[dict] = []
    diagnostics: list[dict] = []
    for key, group in grouped:
        group = group.sort_values("month")
        by_month = {row["month"]: row for row in group.to_dict("records")}
        months = sorted(by_month, key=month_index)
        usable = 0
        for month in months:
            current = by_month[month]
            idx = month_index(month)
            history = [by_month.get(month_from_index(idx - lag), {}).get("price") for lag in (1, 2, 3)]
            rolling = [by_month.get(month_from_index(idx - lag), {}).get("price") for lag in (1, 2, 3)]
            if any(value is None or pd.isna(value) for value in history + rolling):
                continue
            numeric = [float(value) for value in rolling]
            mean = float(np.mean(numeric))
            std = float(np.std(numeric))
            angle = 2 * math.pi * (int(month[5:7]) - 1) / 12
            series_id = " | ".join([str(key[3]), str(key[2]), str(key[4]), str(key[5])])
            features.append({
                "series_id": series_id,
                "source": "world_bank",
                "commodity_ticker": key[0],
                "commodity": key[3],
                "geography_level": key[1],
                "geography_name": key[2],
                "unit": key[4],
                "price_type": key[5],
                "month": month,
                "date": current["date"],
                "price": float(current["price"]),
                "low_estimate": current.get("low_estimate"),
                "high_estimate": current.get("high_estimate"),
                "trust_score": current.get("trust_score"),
                "n_markets": None,
                "price_lag_1": float(history[0]),
                "price_lag_2": float(history[1]),
                "price_lag_3": float(history[2]),
                "rolling_mean_3": mean,
                "rolling_std_3": std,
                "month_sin": math.sin(angle),
                "month_cos": math.cos(angle),
                "trend_index": float(idx),
                "target_price": float(current["price"]),
            })
            usable += 1
        diagnostics.append({
            "series_id": series_id,
            "source": "world_bank",
            "commodity": key[3],
            "commodity_ticker": key[0],
            "geography_level": key[1],
            "geography_name": key[2],
            "unit": key[4],
            "price_type": key[5],
            "original_observations": len(group),
            "first_date": str(group["date"].min()),
            "last_date": str(group["date"].max()),
            "calendar_months": len(months),
            "missing_calendar_months": 0,
            "usable_feature_rows": usable,
            "train_rows": int(usable * 0.70),
            "validation_rows": int(usable * 0.15),
            "test_rows": usable - int(usable * 0.70) - int(usable * 0.15),
        })
    return features, diagnostics


def _split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: month_index(row["month"]))
    n = len(ordered)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    return ordered[:train_end], ordered[train_end:valid_end], ordered[valid_end:]


def _predict_stored(bundle: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(bundle, dict) and "base_models" in bundle:
        matrix = np.column_stack([bundle["base_models"][name].predict(x) for name in bundle["base_names"]])
        return np.asarray(bundle["meta_model"].predict(matrix), dtype=float)
    return np.asarray(bundle.predict(x), dtype=float)


def _recursive_forecast(model: Any, rows: list[dict], transform: str, horizons=(1, 3, 6)) -> list[dict]:
    history = {row["month"]: float(row["price"]) for row in rows}
    origin = max(history, key=month_index)
    predictions: dict[int, float] = {}
    for step in range(1, max(horizons) + 1):
        month = month_from_index(month_index(origin) + step)
        idx = month_index(month)
        lags = [history.get(month_from_index(idx - lag)) for lag in (1, 2, 3)]
        if any(value is None for value in lags):
            break
        numeric = [float(value) for value in lags]
        angle = 2 * math.pi * (int(month[5:7]) - 1) / 12
        x = np.asarray([[numeric[0], numeric[1], numeric[2], np.mean(numeric), np.std(numeric), math.sin(angle), math.cos(angle), float(idx)]])
        transformed = float(_predict_stored(model, x)[0])
        lag = np.asarray([numeric[0]])
        value = float(_reconstruct(np.asarray([transformed]), lag, transform)[0])
        value = max(0.0, value)
        history[month] = value
        predictions[step] = value
    return [
        {"horizon_months": horizon, "forecast_month": month_from_index(month_index(origin) + horizon), "forecast_price": predictions.get(horizon), "status": "recursive_forecast" if horizon in predictions else "unavailable_missing_calendar_history"}
        for horizon in horizons
    ]


def _legacy_train_world_bank(rebuild_data: bool = False) -> dict[str, Any]:
    raise RuntimeError(
        "This training pipeline is retired. Model fitting must use the approved WFP/NBS migration scope."
    )

    if rebuild_data or not NORMALIZED_FILE.exists():
        prepared = prepare_world_bank_data()
        normalized = prepared["normalized"]
    else:
        normalized = pd.read_csv(NORMALIZED_FILE)
    features, diagnostics = _feature_rows(normalized)
    _write_csv(REPORT_ROOT / "training_sample_counts.csv", diagnostics)
    _write_csv(OUTPUT_ROOT / "world_bank_training_features.csv", features)
    factories, xgb_status = _factories()
    metrics: list[dict] = []
    predictions: list[dict] = []
    forecasts: list[dict] = []
    selections: list[dict] = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in features:
        grouped[row["series_id"]].append(row)
    # Keep one model per selected commodity/geography/unit series. The full
    # normalized release remains available for later experiments; this first
    # panel is deliberately small enough to rerun and defend.
    for series_id, series_rows in sorted(grouped.items()):
        train_rows, valid_rows, test_rows = _split(series_rows)
        x_train, _ = _arrays(train_rows)
        x_valid, _ = _arrays(valid_rows)
        x_test, _ = _arrays(test_rows)
        actual_valid = np.asarray([row["target_price"] for row in valid_rows], dtype=float)
        actual_test = np.asarray([row["target_price"] for row in test_rows], dtype=float)
        lag_valid = np.asarray([row["price_lag_1"] for row in valid_rows], dtype=float)
        lag_test = np.asarray([row["price_lag_1"] for row in test_rows], dtype=float)
        baseline = lag_test.copy()
        baseline_metrics = _metrics(actual_test, baseline)
        common = {
            "source": "world_bank",
            "series_id": series_id,
            "model": "naive_last_observation",
            "transformation": "none",
            "split": "test",
            "test_months": ";".join(row["month"] for row in test_rows),
            "n_train": len(train_rows),
            "n_validation": len(valid_rows),
            "n_test": len(test_rows),
            **baseline_metrics,
            "status": "evaluated",
            "notes": "Previous calendar-month close estimate; identical test dates used for all models in this series.",
        }
        metrics.append(common)
        predictions.extend({"source": "world_bank", "series_id": series_id, "model": "naive_last_observation", "month": row["month"], "actual": float(actual), "predicted": float(pred)} for row, actual, pred in zip(test_rows, actual_test, baseline))
        last_model_rows: list[dict] = []
        for model_name in MODEL_NAMES:
            if model_name == "xgboost" and "xgboost" not in factories:
                metrics.append({**common, "model": model_name, "status": "unavailable", "rmse": None, "mae": None, "mape": None, "r2": None, "bias": None, "notes": xgb_status})
                continue
            candidates = []
            for transform in TRANSFORMS:
                try:
                    y_train, _ = _target(train_rows, transform)
                    fitted, transformed, note = _fit_model(model_name, factories, x_train, y_train, x_valid)
                    predicted_valid = _reconstruct(transformed, lag_valid, transform)
                    candidate_metrics = _metrics(actual_valid, predicted_valid)
                    candidates.append((candidate_metrics["rmse"], transform, candidate_metrics, note))
                except Exception:
                    continue
            if not candidates:
                metrics.append({**common, "model": model_name, "status": "failed", "rmse": None, "mae": None, "mape": None, "r2": None, "bias": None, "notes": "No valid validation transformation."})
                continue
            _, transform, validation_metrics, _ = min(candidates, key=lambda item: (item[0], item[1]))
            fit_rows = train_rows + valid_rows
            x_fit, _ = _arrays(fit_rows)
            y_fit, _ = _target(fit_rows, transform)
            fitted, transformed_test, note = _fit_model(model_name, factories, x_fit, y_fit, x_test)
            predicted_test = _reconstruct(transformed_test, lag_test, transform)
            test_metrics = _metrics(actual_test, predicted_test)
            metadata = {
                "source": "world_bank",
                "series_id": series_id,
                "feature_columns": FEATURE_COLUMNS,
                "target_column": "price",
                "transformation": transform,
                "selection_split": "validation",
                "fit_rows": len(fit_rows),
                "test_months": [row["month"] for row in test_rows],
                "notes": note,
            }
            path = _save_model(series_id, model_name, fitted, metadata)
            metrics.append({**common, "model": model_name, "transformation": transform, **test_metrics, "status": "evaluated", "notes": f"Validation-selected transformation; saved artifact {path.relative_to(ROOT)}."})
            predictions.extend({"source": "world_bank", "series_id": series_id, "model": model_name, "month": row["month"], "actual": float(actual), "predicted": float(pred)} for row, actual, pred in zip(test_rows, actual_test, predicted_test))
            selections.append({"series_id": series_id, "model": model_name, "transformation": transform, "validation_rmse": validation_metrics["rmse"], "test_rmse": test_metrics["rmse"], "status": "selected_on_validation"})
            last_model_rows.append((model_name, fitted, transform))
            for forecast in _recursive_forecast(fitted, series_rows, transform):
                forecasts.append({"source": "world_bank", "series_id": series_id, "model": model_name, **forecast})
        # The naive forecast is defined explicitly and does not require a
        # model artifact: persist the last available calendar-month estimate.
        last_price = float(series_rows[-1]["price"])
        origin = max(row["month"] for row in series_rows)
        for horizon in (1, 3, 6):
            forecasts.append({
                "source": "world_bank",
                "series_id": series_id,
                "model": "naive_last_observation",
                "horizon_months": horizon,
                "forecast_month": month_from_index(month_index(origin) + horizon),
                "forecast_price": last_price,
                "status": "recursive_forecast",
            })
    _write_csv(REPORT_ROOT / "model_comparison.csv", metrics)
    _write_csv(REPORT_ROOT / "predictions.csv", predictions)
    _write_csv(REPORT_ROOT / "selection_audit.csv", selections)
    _write_csv(FORECAST_ROOT / "world_bank_forecasts.csv", forecasts)
    # Keep plots separate from all WFP/aggregate charts.
    grouped_predictions: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in predictions:
        grouped_predictions[(row["series_id"], row["model"])].append(row)
    for (series_id, model_name), rows in grouped_predictions.items():
        rows = sorted(rows, key=lambda row: month_index(row["month"]))
        baseline_rows = grouped_predictions.get((series_id, "naive_last_observation"))
        baseline_values = None
        if model_name != "naive_last_observation" and baseline_rows:
            baseline_values = [float(row["predicted"]) for row in sorted(baseline_rows, key=lambda row: month_index(row["month"]))]
        write_actual_predicted_svg(
            OUTPUT_ROOT / "plots" / f"{_slug(series_id)}_{model_name}.svg",
            f"World Bank {series_id} — {model_name}",
            [row["month"] for row in rows],
            [float(row["actual"]) for row in rows],
            [float(row["predicted"]) for row in rows],
            baseline_values,
        )
    report = _build_report(normalized, diagnostics, metrics, xgb_status)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "training_report.txt").write_text(report, encoding="utf-8")
    (REPORT_ROOT / "experiment_config.json").write_text(json.dumps({"source": "world_bank", "geography_levels": ["national", "market_average"], "models": MODEL_NAMES, "transforms": TRANSFORMS, "feature_columns": FEATURE_COLUMNS, "xgboost_status": xgb_status}, indent=2), encoding="utf-8")
    return {"metrics": metrics, "predictions": predictions, "forecasts": forecasts, "diagnostics": diagnostics}


def train_world_bank(rebuild_data: bool = False) -> dict[str, Any]:
    """Run the World Bank research audit and create a production registry.

    The historical files produced by the earlier runner are preserved.  New
    champion artifacts and reports live under explicit ``champion_*`` paths.
    """

    raise RuntimeError(
        "This training pipeline is retired. Model fitting must use the approved WFP/NBS migration scope."
    )

    if rebuild_data or not NORMALIZED_FILE.exists():
        normalized = prepare_world_bank_data()["normalized"]
    else:
        normalized = pd.read_csv(NORMALIZED_FILE)
    features, diagnostics = _feature_rows(normalized)
    _write_csv_if_missing(REPORT_ROOT / "training_sample_counts.csv", diagnostics)
    _write_csv_if_missing(OUTPUT_ROOT / "world_bank_training_features.csv", features)
    factories, xgb_status = _factories()
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    champion_audit: list[dict[str, Any]] = []
    champion_predictions: list[dict[str, Any]] = []
    champion_forecasts: list[dict[str, Any]] = []
    registry_series: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        grouped[row["series_id"]].append(row)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    for series_id, series_rows in sorted(grouped.items()):
        train_rows, valid_rows, test_rows = _split(series_rows)
        x_train, _ = _arrays(train_rows)
        x_valid, _ = _arrays(valid_rows)
        x_test, _ = _arrays(test_rows)
        actual_valid = np.asarray([row["target_price"] for row in valid_rows], dtype=float)
        actual_test = np.asarray([row["target_price"] for row in test_rows], dtype=float)
        lag_valid = np.asarray([row["price_lag_1"] for row in valid_rows], dtype=float)
        lag_test = np.asarray([row["price_lag_1"] for row in test_rows], dtype=float)
        valid_baseline = _metrics(actual_valid, lag_valid)
        test_baseline = _metrics(actual_test, lag_test)
        common = {
            "source": "world_bank", "series_id": series_id,
            "model": "naive_last_observation", "transformation": "none",
            "split": "test", "test_months": ";".join(row["month"] for row in test_rows),
            "n_train": len(train_rows), "n_validation": len(valid_rows), "n_test": len(test_rows),
            **test_baseline, "status": "evaluated",
            "notes": "Previous calendar-month close estimate; identical test dates used for all models in this series.",
        }
        metrics.append(common)
        predictions.extend(
            {"source": "world_bank", "series_id": series_id, "model": "naive_last_observation", "month": row["month"], "actual": float(actual), "predicted": float(pred)}
            for row, actual, pred in zip(test_rows, actual_test, lag_test)
        )
        naive_candidate = {"series_id": series_id, "model": "naive_last_observation", "transformation": "none", "status": "evaluated", "validation_metrics": valid_baseline, "validation_rmse": valid_baseline["rmse"], "test_metrics": test_baseline}
        candidates = [naive_candidate]
        fitted_models: dict[str, tuple[Any, str, dict[str, Any]]] = {}
        champion_audit.append({"series_id": series_id, "model": "naive_last_observation", "transformation": "none", "selection_split": "validation", "validation_rmse": valid_baseline["rmse"], "validation_mae": valid_baseline["mae"], "validation_mape": valid_baseline["mape"], "validation_r2": valid_baseline["r2"], "test_rmse": test_baseline["rmse"], "status": "evaluated", "notes": "Naive baseline evaluated on validation and untouched test dates."})

        for model_name in MODEL_NAMES:
            if model_name == "xgboost" and "xgboost" not in factories:
                metrics.append({**common, "model": model_name, "status": "unavailable", "rmse": None, "mae": None, "mape": None, "r2": None, "bias": None, "notes": xgb_status})
                champion_audit.append({"series_id": series_id, "model": model_name, "transformation": "", "selection_split": "validation", "status": "unavailable", "notes": xgb_status})
                continue
            transform_candidates: list[dict[str, Any]] = []
            for transform in TRANSFORMS:
                try:
                    y_train, _ = _target(train_rows, transform)
                    _, transformed, note = _fit_model(model_name, factories, x_train, y_train, x_valid)
                    validation_metrics = _metrics(actual_valid, _reconstruct(transformed, lag_valid, transform))
                    transform_candidates.append({"series_id": series_id, "model": model_name, "transformation": transform, "status": "evaluated", "validation_metrics": validation_metrics, "validation_rmse": validation_metrics["rmse"], "note": note})
                except Exception as exc:
                    transform_candidates.append({"series_id": series_id, "model": model_name, "transformation": transform, "status": "failed", "notes": str(exc)})
            best = select_model_transform(transform_candidates)
            for row in transform_candidates:
                vm = row.get("validation_metrics") or {}
                champion_audit.append({"series_id": series_id, "model": model_name, "transformation": row.get("transformation", ""), "selection_split": "validation", "validation_rmse": vm.get("rmse", row.get("validation_rmse")), "validation_mae": vm.get("mae"), "validation_mape": vm.get("mape"), "validation_r2": vm.get("r2"), "status": row.get("status"), "notes": row.get("note") or row.get("notes", "")})
            if best is None:
                metrics.append({**common, "model": model_name, "status": "failed", "rmse": None, "mae": None, "mape": None, "r2": None, "bias": None, "notes": "No valid validation transformation."})
                continue
            transform = str(best["transformation"])
            fit_rows = train_rows + valid_rows
            x_fit, _ = _arrays(fit_rows)
            y_fit, _ = _target(fit_rows, transform)
            try:
                fitted, transformed_test, note = _fit_model(model_name, factories, x_fit, y_fit, x_test)
                predicted_test = _reconstruct(transformed_test, lag_test, transform)
                test_metrics = _metrics(actual_test, predicted_test)
            except Exception as exc:
                metrics.append({**common, "model": model_name, "status": "failed", "rmse": None, "mae": None, "mape": None, "r2": None, "bias": None, "notes": f"Refit failed: {exc}"})
                continue
            metadata = {"source": "world_bank", "series_id": series_id, "feature_columns": FEATURE_COLUMNS, "target_column": "price", "transformation": transform, "selection_split": "validation", "fit_rows": len(fit_rows), "test_months": [row["month"] for row in test_rows], "notes": note}
            old_path = _save_model(series_id, model_name, fitted, metadata)
            metrics.append({**common, "model": model_name, "transformation": transform, **test_metrics, "status": "evaluated", "notes": f"Validation-selected transformation; saved artifact {old_path.relative_to(ROOT)}."})
            predictions.extend({"source": "world_bank", "series_id": series_id, "model": model_name, "month": row["month"], "actual": float(actual), "predicted": float(pred)} for row, actual, pred in zip(test_rows, actual_test, predicted_test))
            selected_candidate = {"series_id": series_id, "model": model_name, "transformation": transform, "status": "evaluated", "validation_metrics": best.get("validation_metrics") or {}, "validation_rmse": best.get("validation_rmse"), "test_metrics": test_metrics}
            candidates.append(selected_candidate)
            fitted_models[model_name] = (fitted, transform, metadata)
            selections.append({"series_id": series_id, "model": model_name, "transformation": transform, "validation_rmse": selected_candidate["validation_rmse"], "test_rmse": test_metrics["rmse"], "status": "selected_on_validation"})
            for forecast in _recursive_forecast(fitted, series_rows, transform):
                forecasts.append({"source": "world_bank", "series_id": series_id, "model": model_name, **forecast})

        champion = select_champion(candidates)
        if champion is None:
            raise RuntimeError(f"No valid champion candidate for {series_id}")
        selected_model = str(champion["model"])
        selected_transform = str(champion.get("transformation", "none"))
        artifact_path: str | None = None
        if selected_model == "naive_last_observation":
            production_forecasts = [{"source": "world_bank", "series_id": series_id, "model": selected_model, "horizon_months": horizon, "forecast_month": month_from_index(month_index(series_rows[-1]["month"]) + horizon), "forecast_price": float(series_rows[-1]["price"]), "status": "recursive_forecast"} for horizon in (1, 3, 6)]
        else:
            fitted, selected_transform, metadata = fitted_models[selected_model]
            champion_metadata = {**metadata, "selected_model": selected_model, "production_champion": True, "experiment_version": CHAMPION_EXPERIMENT_VERSION, "selection_criterion": "minimum validation RMSE; deterministic simplicity tie-break"}
            champion_path = _save_champion_model(series_id, fitted, champion_metadata)
            artifact_path = champion_path.relative_to(ROOT).as_posix()
            production_forecasts = [{"source": "world_bank", "series_id": series_id, "model": selected_model, **forecast} for forecast in _recursive_forecast(fitted, series_rows, selected_transform)]
        champion_forecasts.extend(production_forecasts)
        test_prediction_lookup = {(row["model"], row["month"]): row["predicted"] for row in predictions if row["series_id"] == series_id}
        for row, naive_pred in zip(test_rows, lag_test):
            predicted = float(naive_pred) if selected_model == "naive_last_observation" else test_prediction_lookup.get((selected_model, row["month"]))
            if predicted is not None:
                champion_predictions.append({"source": "world_bank", "series_id": series_id, "model": selected_model, "month": row["month"], "actual": float(row["target_price"]), "predicted": float(predicted)})
        registry_series.append({
            "series_id": series_id, "selected_model": selected_model, "selected_transformation": selected_transform,
            "validation_metrics": champion.get("validation_metrics") or {},
            "validation_period": {"start": valid_rows[0]["month"], "end": valid_rows[-1]["month"], "observations": len(valid_rows)},
            "training_period": {"start": train_rows[0]["month"], "end": train_rows[-1]["month"], "observations": len(train_rows)},
            "test_period": {"start": test_rows[0]["month"], "end": test_rows[-1]["month"], "observations": len(test_rows)},
            "test_metrics": champion.get("test_metrics") or {},
            "selection_criterion": "minimum validation RMSE; deterministic simplicity tie-break; test metrics not used for selection",
            "source": "World Bank Nigeria Real Time Food Prices; c_ close estimate", "target_column": "price", "is_model_estimate": True, "source_data_is_direct_observation": False,
            "artifact_path": artifact_path, "experiment_version": CHAMPION_EXPERIMENT_VERSION, "selection_timestamp": generated_at, "status": "available",
        })

    # Never overwrite the earlier research outputs; create them only when a
    # checkout does not already contain them.
    _write_csv_if_missing(REPORT_ROOT / "model_comparison.csv", metrics)
    _write_csv_if_missing(REPORT_ROOT / "predictions.csv", predictions)
    _write_csv_if_missing(REPORT_ROOT / "selection_audit.csv", selections)
    _write_csv_if_missing(FORECAST_ROOT / "world_bank_forecasts.csv", forecasts)
    _write_csv(REPORT_ROOT / "champion_predictions.csv", champion_predictions)
    _write_csv(REPORT_ROOT / "champion_selection_audit.csv", champion_audit)
    _write_csv(CHAMPION_FORECAST_ROOT / "champion_forecasts.csv", champion_forecasts)
    _write_csv(REPORT_ROOT / "champion_model_comparison.csv", [{**row, "production_champion": any(item["series_id"] == row.get("series_id") and item["selected_model"] == row.get("model") for item in registry_series)} for row in metrics])
    _atomic_json_write(CHAMPION_REGISTRY_PATH, {"schema_version": 1, "experiment_version": CHAMPION_EXPERIMENT_VERSION, "source": "World Bank Nigeria Real Time Food Prices", "target_column": "price", "price_semantics": "c_ close estimate; not a direct survey observation", "generated_at": generated_at, "selection_rule": "70/15/15 chronological split; transformations selected on validation RMSE; one champion per series; untouched test metrics are audit-only", "series": registry_series})
    # Existing plot files are research artifacts; only create missing plots.
    grouped_predictions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped_predictions[(row["series_id"], row["model"])].append(row)
    for (series_id, model_name), rows in grouped_predictions.items():
        plot_path = OUTPUT_ROOT / "plots" / f"{_slug(series_id)}_{model_name}.svg"
        if plot_path.exists():
            continue
        rows = sorted(rows, key=lambda row: month_index(row["month"]))
        baseline_rows = grouped_predictions.get((series_id, "naive_last_observation"))
        baseline_values = [float(row["predicted"]) for row in sorted(baseline_rows, key=lambda row: month_index(row["month"]))] if model_name != "naive_last_observation" and baseline_rows else None
        write_actual_predicted_svg(plot_path, f"World Bank {series_id} — {model_name}", [row["month"] for row in rows], [float(row["actual"]) for row in rows], [float(row["predicted"]) for row in rows], baseline_values)
    if not (REPORT_ROOT / "training_report.txt").exists():
        (REPORT_ROOT / "training_report.txt").write_text(_build_report(normalized, diagnostics, metrics, xgb_status), encoding="utf-8")
    champion_lines = ["AgriPrice-NG World Bank champion–challenger training", "", "Target: World Bank c_ close estimate (modelled/imputed; not a direct survey observation).", "Selection: 70/15/15 chronological split; validation RMSE only; untouched test metrics are audit-only.", "Tie-break: naive, Gradient Boosting, Random Forest, XGBoost, stacking.", "", "series_id | champion | transformation | validation_RMSE | test_RMSE | artifact"]
    for item in registry_series:
        champion_lines.append(f"{item['series_id']} | {item['selected_model']} | {item['selected_transformation']} | {item['validation_metrics'].get('rmse')} | {item['test_metrics'].get('rmse')} | {item.get('artifact_path') or 'none (naive)'}")
    champion_lines += ["", "Forecasts are recursive for 1, 3 and 6 months; actual future prices are never used.", "No confidence intervals are claimed."]
    (REPORT_ROOT / "champion_training_report.txt").write_text("\n".join(champion_lines) + "\n", encoding="utf-8")
    return {"metrics": metrics, "predictions": predictions, "forecasts": forecasts, "diagnostics": diagnostics, "champions": registry_series, "champion_forecasts": champion_forecasts}


def _build_report(normalized: pd.DataFrame, diagnostics: list[dict], metrics: list[dict], xgb_status: str | None) -> str:
    lines = [
        "AgriPrice-NG World Bank training experiment",
        "============================================",
        "",
        "Source: World Bank Nigeria Real Time Food Prices release; c_ close estimates.",
        "These values are modelled/imputed estimates and are not described as direct market surveys.",
        "No interpolation or forward fill was used. Original WFP files and experiment outputs were not modified.",
        "",
        f"Normalized source rows: {len(normalized):,}",
        f"Date range: {normalized['date'].min()} to {normalized['date'].max()}",
        f"Trained commodity/location series: {len(diagnostics):,}",
        "The normalized release contains additional commodities and 67 market labels; this first panel trains six commodities at National Average and Market Average only.",
        f"XGBoost: {xgb_status}",
        "LSTM: not retrained in this World Bank run; the validated WFP aggregate LSTM remains source-specific and is not reused for World Bank estimates.",
        "",
        "Chronology and leakage control",
        "------------------------------",
        "Each series uses 70% training, 15% validation, and 15% untouched test rows in calendar order.",
        "Transformations are selected only on validation RMSE, then refit on train+validation before one test evaluation.",
        "Stacking meta-models are fitted from chronological out-of-fold base predictions, never in-sample predictions.",
        "The naive baseline uses the immediately previous calendar-month close estimate on the same test dates.",
        "",
        "Model comparison",
        "-----------------",
    ]
    for row in metrics:
        if row["model"] in {"naive_last_observation", "random_forest", "xgboost", "gradient_boosting", "stacking"} and row["status"] == "evaluated":
            lines.append(f"{row['series_id']} | {row['model']} | transform={row.get('transformation','none')} | RMSE={row.get('rmse'):.4f} | MAE={row.get('mae'):.4f} | MAPE={row.get('mape'):.3f}% | R2={row.get('r2')}")
    lines += [
        "",
        "Geography note",
        "--------------",
        "The source has 15 actual market labels, five geopolitical-zone records, one National Average and one Market Average.",
        "Only national and Market Average series were selected for this first World Bank model run to avoid double-counting and keep the experiment manageable.",
        "All normalized geography levels remain queryable for future experiments.",
        "",
        "Forecasts",
        "---------",
        "Forecasts are recursive for 1, 3 and 6 months: predicted values are fed back as future lags and actual future prices are never used.",
        "Confidence intervals are not provided because they have not been statistically validated for these modelled estimates.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    result = train_world_bank()
    print(f"World Bank training completed: {len(result['metrics'])} metric rows, {len(result['forecasts'])} forecast rows")
