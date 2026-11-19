"""Request-time WFP market forecasting with chronological validation.

This module is intentionally separate from the approved historical research
pipeline. It uses only one exact WFP ``priceflag=actual`` market series at a
time, never fills missing prices, and writes reusable artifacts only to a
runtime cache (Render's filesystem is ephemeral).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = (
    "lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_std_3",
    "month_sin", "month_cos", "trend_month",
)
MIN_SUPERVISED_ROWS = 30
MIN_TRAIN_ROWS = 20
MIN_VALIDATION_ROWS = 6
CACHE_FORMAT_VERSION = 1
MODEL_CONFIGURATION = {
    "random_forest": "300 trees; min_samples_leaf=2; max_features=1.0; seed=42; n_jobs=1",
    "gradient_boosting": "120 estimators; depth=2; learning_rate=0.03; Huber loss; seed=42",
}


class ForecastingError(Exception):
    """A specific, safe forecasting eligibility or runtime problem."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _month_ordinal(month: str) -> int:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", str(month)):
        raise ValueError(f"Invalid month: {month}")
    year, month_number = (int(part) for part in month.split("-"))
    return year * 12 + month_number - 1


def _month_from_ordinal(ordinal: int) -> str:
    year, month_zero = divmod(int(ordinal), 12)
    return f"{year:04d}-{month_zero + 1:02d}"


def _collect_actual_history(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and reduce source rows without bridging gaps or changing prices."""
    values: dict[str, float] = {}
    dates: dict[str, str] = {}
    missing_price_rows = 0
    invalid_price_rows = 0
    invalid_month_rows = 0
    duplicate_rows_removed = 0
    duplicate_conflicts: set[str] = set()
    seen: dict[str, float | None] = {}

    for row in record.get("history", []):
        month = str(row.get("month") or "").strip()
        try:
            _month_ordinal(month)
        except ValueError:
            invalid_month_rows += 1
            continue

        raw = row.get("price")
        value: float | None
        if raw in (None, ""):
            missing_price_rows += 1
            value = None
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError):
                value = None
                invalid_price_rows += 1
            else:
                if not math.isfinite(value) or value <= 0:
                    value = None
                    invalid_price_rows += 1

        if month in seen:
            duplicate_rows_removed += 1
            if seen[month] != value:
                duplicate_conflicts.add(month)
            continue
        seen[month] = value
        if value is not None:
            values[month] = value
            observed_date = str(row.get("last_observation_date") or row.get("first_observation_date") or "").strip()
            if observed_date:
                dates[month] = observed_date

    if duplicate_conflicts:
        month_list = ", ".join(sorted(duplicate_conflicts))
        raise ForecastingError(
            "conflicting_monthly_observations",
            f"The selected WFP series has conflicting source prices for {month_list}. The records were not combined.",
        )

    return {
        "prices": values,
        "dates": dates,
        "missing_price_rows": missing_price_rows,
        "invalid_price_rows": invalid_price_rows,
        "invalid_month_rows": invalid_month_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
    }


def _feature_row(prices: dict[str, float], target_month: str, trend_origin: int) -> dict[str, float] | None:
    target_ordinal = _month_ordinal(target_month)
    prior = [prices.get(_month_from_ordinal(target_ordinal - offset)) for offset in (1, 2, 3)]
    if any(value is None for value in prior):
        return None
    numeric = [float(value) for value in prior]
    mean = sum(numeric) / 3.0
    month_number = target_ordinal % 12 + 1
    angle = 2.0 * math.pi * month_number / 12.0
    return {
        "lag_1": numeric[0],
        "lag_2": numeric[1],
        "lag_3": numeric[2],
        "rolling_mean_3": mean,
        "rolling_std_3": math.sqrt(sum((value - mean) ** 2 for value in numeric) / 3.0),
        "month_sin": math.sin(angle),
        "month_cos": math.cos(angle),
        "trend_month": float(target_ordinal - trend_origin),
    }


def _build_feature_rows(prices: dict[str, float]) -> list[dict[str, Any]]:
    if not prices:
        return []
    ordinals = sorted(_month_ordinal(month) for month in prices)
    trend_origin = ordinals[0]
    rows: list[dict[str, Any]] = []
    for ordinal in range(ordinals[0], ordinals[-1] + 1):
        month = _month_from_ordinal(ordinal)
        target = prices.get(month)
        if target is None:
            continue
        features = _feature_row(prices, month, trend_origin)
        if features is not None:
            rows.append({"month": month, "target": float(target), "features": features})
    return rows


def assess_wfp_series(record: dict[str, Any]) -> dict[str, Any]:
    """Return request-time eligibility facts for one exact series, without fitting."""
    if record.get("source_key") != "wfp" and record.get("source") != "WFP":
        return {"eligible": False, "code": "source_not_supported", "reason": "On-demand forecasting is currently supported for WFP market data only."}
    if str(record.get("price_flag", "")) != "actual":
        return {
            "eligible": False,
            "code": "not_direct_actual",
            "reason": "This WFP series is aggregate or mixed-price-flag data. Forecast training requires a pure priceflag=actual market series.",
        }
    try:
        collected = _collect_actual_history(record)
    except ForecastingError as error:
        return {"eligible": False, "code": error.code, "reason": error.message}
    prices = collected["prices"]
    actual_count = len(prices)
    if not prices:
        return {
            "eligible": False, "code": "no_valid_actual_prices",
            "reason": "No finite, positive actual monthly prices remain after validating this series.",
            "actual_observations": 0,
            **{key: value for key, value in collected.items() if key != "prices"},
        }
    months = sorted(prices, key=_month_ordinal)
    feature_rows = _build_feature_rows(prices)
    split = int(math.floor(len(feature_rows) * 0.80))
    train_count = split
    validation_count = len(feature_rows) - split
    origin = months[-1]
    origin_ordinal = _month_ordinal(origin)
    missing_recent = [
        _month_from_ordinal(value)
        for value in range(origin_ordinal - 2, origin_ordinal + 1)
        if _month_from_ordinal(value) not in prices
    ]
    calendar_span = _month_ordinal(months[-1]) - _month_ordinal(months[0]) + 1
    reasons: list[str] = []
    if len(feature_rows) < MIN_SUPERVISED_ROWS:
        reasons.append(
            f"only {len(feature_rows)} complete calendar-lag training rows are available; at least {MIN_SUPERVISED_ROWS} are required"
        )
    if train_count < MIN_TRAIN_ROWS:
        reasons.append(f"the chronological training partition has {train_count} rows; at least {MIN_TRAIN_ROWS} are required")
    if validation_count < MIN_VALIDATION_ROWS:
        reasons.append(f"the chronological validation partition has {validation_count} rows; at least {MIN_VALIDATION_ROWS} are required")
    if missing_recent:
        reasons.append(
            "the latest actual observation is not preceded by three consecutive actual calendar months; missing required months: "
            + ", ".join(missing_recent)
        )
    return {
        "eligible": not reasons,
        "code": "eligible" if not reasons else "insufficient_or_gapped_history",
        "reason": "; ".join(reasons),
        "actual_observations": actual_count,
        "first_actual_month": months[0],
        "latest_actual_month": origin,
        "usable_training_rows": len(feature_rows),
        "training_rows": train_count,
        "validation_rows": validation_count,
        "validation_first_month": feature_rows[split]["month"] if validation_count else None,
        "validation_last_month": feature_rows[-1]["month"] if validation_count else None,
        "missing_calendar_months": calendar_span - actual_count,
        "invalid_price_rows": collected["invalid_price_rows"],
        "missing_price_rows": collected["missing_price_rows"],
        "invalid_month_rows": collected["invalid_month_rows"],
        "duplicate_rows_removed": collected["duplicate_rows_removed"],
        "feature_rows": feature_rows,
        "price_map": prices,
        "observation_dates": collected["dates"],
    }


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float | int | None]:
    errors = [prediction - truth for truth, prediction in zip(actual, predicted)]
    count = len(errors)
    if not count:
        return {"rmse": None, "mae": None, "mape": None, "r2": None, "n": 0}
    rmse = math.sqrt(sum(error * error for error in errors) / count)
    mae = sum(abs(error) for error in errors) / count
    mape = sum(abs(error / truth) for truth, error in zip(actual, errors) if truth != 0)
    mape_count = sum(truth != 0 for truth in actual)
    mean_actual = sum(actual) / count
    total_sum_squares = sum((truth - mean_actual) ** 2 for truth in actual)
    residual_sum_squares = sum(error * error for error in errors)
    r2 = 1.0 - residual_sum_squares / total_sum_squares if count >= 2 and total_sum_squares > 0 else None
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": 100.0 * mape / mape_count if mape_count else None,
        "r2": r2,
        "n": count,
    }


def _make_estimators() -> dict[str, Any]:
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    except ImportError as exc:
        raise ForecastingError(
            "forecast_dependencies_missing",
            "The server is missing the scikit-learn forecasting dependency. Contact the site administrator.",
            503,
        ) from exc
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=1.0,
            random_state=42, n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.03, max_depth=2,
            min_samples_leaf=2, loss="huber", random_state=42,
        ),
    }


def _matrix(rows: list[dict[str, Any]]) -> Any:
    import numpy as np

    return np.asarray(
        [[row["features"][name] for name in FEATURE_COLUMNS] for row in rows],
        dtype=float,
    )


def _predict(estimator: Any, feature: dict[str, float]) -> float:
    import numpy as np

    with warnings.catch_warnings():
        # Registered research artifacts were fitted with named DataFrame
        # columns. The validated feature schema/order is checked before load.
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but .* was fitted with feature names",
            category=UserWarning,
        )
        value = float(estimator.predict(np.asarray([[feature[name] for name in FEATURE_COLUMNS]], dtype=float))[0])
    if not math.isfinite(value):
        raise ValueError("model produced a non-finite price")
    return max(0.0, value)


def _cache_key(series_id: str, assessment: dict[str, Any]) -> str:
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "series_id": series_id,
        "features": FEATURE_COLUMNS,
        "minimums": [MIN_SUPERVISED_ROWS, MIN_TRAIN_ROWS, MIN_VALIDATION_ROWS],
        "models": MODEL_CONFIGURATION,
        "prices": [
            [month, format(float(price), ".17g")]
            for month, price in sorted(assessment["price_map"].items(), key=lambda item: _month_ordinal(item[0]))
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _paths(cache_dir: Path, series_id: str) -> tuple[Path, Path]:
    basename = hashlib.sha256(series_id.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{basename}.joblib", cache_dir / f"{basename}.json"


def _load_runtime_cache(cache_dir: Path, series_id: str, cache_key: str) -> dict[str, Any] | None:
    artifact_path, metadata_path = _paths(cache_dir, series_id)
    if not artifact_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key") != cache_key or metadata.get("feature_columns") != list(FEATURE_COLUMNS):
            return None
        import joblib

        bundle = joblib.load(artifact_path)
        if not isinstance(bundle, dict) or bundle.get("cache_key") != cache_key:
            return None
        if bundle.get("series_id") != series_id or bundle.get("feature_columns") != list(FEATURE_COLUMNS):
            return None
        if bundle.get("model_name") not in {"naive_last_observation", *MODEL_CONFIGURATION}:
            return None
        if bundle.get("model_name") != "naive_last_observation" and bundle.get("estimator") is None:
            return None
        if not isinstance(bundle.get("validation_scores"), dict) or bundle["model_name"] not in bundle["validation_scores"]:
            return None
        return {**bundle, "metadata": metadata}
    except Exception:
        return None


def _save_runtime_cache(cache_dir: Path, series_id: str, bundle: dict[str, Any], metadata: dict[str, Any]) -> bool:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        artifact_path, metadata_path = _paths(cache_dir, series_id)
        import joblib

        with tempfile.NamedTemporaryFile(prefix="agriprice-", suffix=".joblib.tmp", dir=cache_dir, delete=False) as handle:
            temporary_artifact = Path(handle.name)
        temporary_metadata = temporary_artifact.with_suffix(".json.tmp")
        try:
            joblib.dump(bundle, temporary_artifact, compress=1)
            temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            os.replace(temporary_artifact, artifact_path)
            os.replace(temporary_metadata, metadata_path)
        finally:
            temporary_artifact.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
        return True
    except Exception:
        # Cache persistence is an optimization only. A valid forecast does not
        # depend on the Render instance's ephemeral filesystem being writable.
        return False


def _registered_estimator(
    root: Path,
    registry_entry: dict[str, Any] | None,
    series_id: str,
    model_name: str,
    latest_month: str,
) -> Any | None:
    if not registry_entry or registry_entry.get("series_id") != series_id:
        return None
    artifact_relative = (registry_entry.get("artifact_paths") or {}).get(model_name)
    if not artifact_relative:
        return None
    artifact_path = (root / str(artifact_relative)).resolve()
    root_resolved = root.resolve()
    if not artifact_path.is_relative_to(root_resolved) or not artifact_path.is_file():
        return None
    try:
        import joblib

        payload = joblib.load(artifact_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("series_id") != series_id or payload.get("model_name") != model_name:
        return None
    if payload.get("feature_columns") != list(FEATURE_COLUMNS):
        return None
    # Only reuse a final refit that includes the latest valid source month.
    if str(payload.get("training_window_end") or "") != latest_month:
        return None
    estimator = payload.get("model")
    if estimator is None or not callable(getattr(estimator, "predict", None)):
        return None
    expected_class = {
        "random_forest": "RandomForestRegressor",
        "gradient_boosting": "GradientBoostingRegressor",
    }.get(model_name)
    if expected_class is None or type(estimator).__name__ != expected_class:
        return None
    if getattr(estimator, "n_features_in_", len(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
        return None
    feature_names = getattr(estimator, "feature_names_in_", None)
    if feature_names is not None and list(feature_names) != list(FEATURE_COLUMNS):
        return None
    return estimator


def _forecast_prices(
    model_name: str,
    estimator: Any | None,
    prices: dict[str, float],
    origin: str,
    trend_origin: int,
    horizon: int,
) -> list[dict[str, Any]]:
    origin_ordinal = _month_ordinal(origin)
    working = dict(prices)
    result: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        target_month = _month_from_ordinal(origin_ordinal + step)
        if model_name == "naive_last_observation":
            prediction = working[origin]
        else:
            feature = _feature_row(working, target_month, trend_origin)
            if feature is None:
                raise ForecastingError(
                    "forecast_history_gap",
                    f"The forecast cannot construct calendar lag features for {target_month} without inventing missing monthly prices.",
                )
            try:
                prediction = _predict(estimator, feature)
            except Exception as exc:
                raise ForecastingError(
                    "forecast_prediction_failed",
                    "The selected model failed while generating a recursive prediction. No forecast was returned.",
                    500,
                ) from exc
        working[target_month] = prediction
        result.append({
            "date": target_month,
            "month": target_month,
            "price": prediction,
            "step": step,
            "quality_flag": "model_estimate",
        })
    return result


def generate_wfp_forecast(
    record: dict[str, Any],
    series_id: str,
    horizon: int,
    registry_entry: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
    root: Path = ROOT,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Validate, select, train/reuse, cache, and recursively forecast one series."""
    if horizon not in (1, 3, 6):
        raise ForecastingError("invalid_horizon", "horizon must be 1, 3, or 6 months.", 400)
    assessment = assess_wfp_series(record)
    if not assessment["eligible"]:
        raise ForecastingError(assessment["code"], assessment["reason"])

    prices = assessment["price_map"]
    feature_rows = assessment["feature_rows"]
    split = assessment["training_rows"]
    train_rows = feature_rows[:split]
    validation_rows = feature_rows[split:]
    actual_validation = [float(row["target"]) for row in validation_rows]
    baseline_predictions = [float(row["features"]["lag_1"]) for row in validation_rows]
    cache_override = os.environ.get("AGRIPRICE_MODEL_CACHE_DIR", "").strip()
    cache_root = Path(cache_dir) if cache_dir is not None else (
        Path(cache_override) if cache_override else Path(tempfile.gettempdir()) / "agriprice-ng" / "runtime-models"
    )
    cache_key = _cache_key(series_id, assessment)
    cached = _load_runtime_cache(cache_root, series_id, cache_key)
    validation_scores: dict[str, dict[str, float | int | None]]
    model_failures: dict[str, str]
    selected_model: str
    selected_estimator: Any | None
    model_reuse: str
    train_started = time.perf_counter()

    if cached is not None:
        selected_model = str(cached["model_name"])
        selected_estimator = cached.get("estimator")
        validation_scores = cached["validation_scores"]
        model_failures = cached.get("model_failures", {})
        model_reuse = "runtime_cache_hit"
        fit_seconds = float(cached.get("fit_seconds", 0.0) or 0.0)
    else:
        validation_scores = {
            "naive_last_observation": _metrics(actual_validation, baseline_predictions),
        }
        model_failures = {}
        try:
            estimators = _make_estimators()
        except ForecastingError:
            raise
        except ImportError as exc:
            raise ForecastingError(
                "forecast_dependencies_missing",
                "The server is missing a required forecasting dependency. Contact the site administrator.",
                503,
            ) from exc

        train_matrix = _matrix(train_rows)
        validation_matrix = _matrix(validation_rows)
        for name, estimator in estimators.items():
            try:
                estimator.fit(train_matrix, [row["target"] for row in train_rows])
                prediction = [max(0.0, float(value)) for value in estimator.predict(validation_matrix)]
                if len(prediction) != len(actual_validation) or not all(math.isfinite(value) for value in prediction):
                    raise ValueError("model returned invalid validation predictions")
                validation_scores[name] = _metrics(actual_validation, prediction)
            except Exception:
                model_failures[name] = "Validation fitting or prediction failed; this model was excluded from selection."

        selected_model = "naive_last_observation"
        selected_estimator = None
        model_reuse = "baseline_selected"
        ranked_models = sorted(validation_scores, key=lambda name: float(validation_scores[name]["rmse"]))
        for candidate_model in ranked_models:
            if candidate_model == "naive_last_observation":
                selected_model = candidate_model
                selected_estimator = None
                model_reuse = "baseline_selected"
                break
            latest_month = assessment["latest_actual_month"]
            registered = _registered_estimator(root, registry_entry, series_id, candidate_model, latest_month)
            if registered is not None:
                selected_model = candidate_model
                selected_estimator = registered
                model_reuse = "registered_artifact_reused"
                break
            else:
                try:
                    selected_estimator = _make_estimators()[candidate_model]
                    selected_estimator.fit(_matrix(feature_rows), [row["target"] for row in feature_rows])
                    selected_model = candidate_model
                    model_reuse = "trained_on_demand"
                    break
                except Exception:
                    model_failures[candidate_model] = "Final fitting failed; this model was excluded from selection."
                    validation_scores.pop(candidate_model, None)

        fit_seconds = time.perf_counter() - train_started

    month_dates = assessment["observation_dates"]
    origin = assessment["latest_actual_month"]
    origin_date = month_dates.get(origin)
    forecast_rows = _forecast_prices(
        selected_model,
        selected_estimator,
        prices,
        origin,
        min(_month_ordinal(month) for month in prices),
        horizon,
    )
    validation_dates = [row["month"] for row in validation_rows]
    selected_metrics = validation_scores[selected_model]

    comparison = [
        {"model": name, **metrics}
        for name, metrics in sorted(validation_scores.items(), key=lambda item: item[0])
    ]
    today = as_of or date.today()
    current_month = f"{today.year:04d}-{today.month:02d}"
    for row in forecast_rows:
        row["temporal_status"] = "future_as_of_generation" if row["month"] > current_month else "after_origin_not_current"
    future_flags = [row["month"] > current_month for row in forecast_rows]
    all_dates_future = all(future_flags)
    any_date_future = any(future_flags)
    forecast_classification = (
        "future_as_of_generation" if all_dates_future else
        "mixed_horizon_relative_to_generation" if any_date_future else
        "historical_projection_from_latest_actual"
    )
    origin_age = max(0, (today.year - int(origin[:4])) * 12 + today.month - int(origin[5:7]))
    warning = (
        f"Generated from the latest valid WFP actual observation ({origin}{f', {origin_date}' if origin_date else ''}). "
        f"That observation is {origin_age} month(s) old as of {today.isoformat()}; this is not a live or guaranteed market price."
    )
    if not any_date_future:
        warning += " The requested forecast dates are after their data origin but are not future dates as of today; treat them only as historical projections."
    elif not all_dates_future:
        warning += " Some requested forecast dates are already past as of today; inspect each row's period status and do not interpret these as current prices."

    metadata = {
        "commodity": record.get("commodity"),
        "market": record.get("market"),
        "state": record.get("state"),
        "unit": record.get("unit"),
        "price_type": record.get("price_type"),
        "currency": record.get("currency"),
        "model": selected_model,
        "transformation": "original WFP actual price in source currency per reported unit",
        "method": "recursive calendar-month forecast; previous model predictions feed later steps; missing actual prices are never filled",
        "forecast_origin_month": origin,
        "latest_actual_month_in_source": origin,
        "latest_actual_date_in_source": origin_date,
        "forecast_generation_date": today.isoformat(),
        "forecast_classification": forecast_classification,
        "confidence_intervals": False,
        "selection_basis": "lowest chronological validation RMSE, including the naive last-observation baseline; no random split or test-set selection",
        "validation_metrics": selected_metrics,
        "model_comparison_validation": comparison,
        "validation_months": validation_dates,
        "training_first_month": feature_rows[0]["month"],
        "training_last_month": feature_rows[-1]["month"],
        "actual_observation_count": assessment["actual_observations"],
        "usable_training_rows": len(feature_rows),
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "missing_calendar_months": assessment["missing_calendar_months"],
        "missing_price_rows": assessment["missing_price_rows"],
        "invalid_price_rows_excluded": assessment["invalid_price_rows"],
        "invalid_month_rows_excluded": assessment["invalid_month_rows"],
        "exact_duplicate_rows_removed": assessment["duplicate_rows_removed"],
        "input_window_months": 3,
        "model_configurations": MODEL_CONFIGURATION,
        "model_failures": model_failures,
        "model_reuse": model_reuse,
        "fit_seconds": fit_seconds,
        "runtime_artifact_persistence": "OS temporary directory; ephemeral on Render and rebuilt after service restart",
    }

    bundle = {
        "cache_key": cache_key,
        "series_id": series_id,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_name": selected_model,
        "estimator": selected_estimator,
        "validation_scores": validation_scores,
        "model_failures": model_failures,
        "fit_seconds": fit_seconds,
    }
    cache_metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "cache_key": cache_key,
        "series_id": series_id,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_name": selected_model,
        "target": "WFP priceflag=actual monthly price",
        "target_transformation": metadata["transformation"],
        "forecast_origin_month": origin,
        "training_first_month": feature_rows[0]["month"],
        "training_last_month": feature_rows[-1]["month"],
        "created_at": today.isoformat(),
        "validation_metrics": selected_metrics,
        "selection_basis": metadata["selection_basis"],
        "model_configurations": MODEL_CONFIGURATION,
    }
    cache_persisted = _save_runtime_cache(cache_root, series_id, bundle, cache_metadata) if cached is None else True
    if model_reuse != "runtime_cache_hit" and cache_persisted:
        metadata["model_reuse"] = model_reuse
    metadata["runtime_cache_status"] = "hit" if cached is not None else ("saved" if cache_persisted else "not_persisted")

    origin_ordinal = _month_ordinal(origin)
    context_start = _month_from_ordinal(origin_ordinal - 23)
    history = [
        row for row in record.get("history", [])
        if row.get("month") and context_start <= str(row["month"]) <= origin
    ]
    return {
        "series": record,
        "model": selected_model,
        "horizon": horizon,
        "historical": history,
        "forecast": forecast_rows,
        "metadata": metadata,
        "warning": warning,
    }
