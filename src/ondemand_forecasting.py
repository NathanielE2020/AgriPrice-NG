"""Request-time WFP/NBS forecasting with gap-aware chronological validation.

This module is intentionally separate from the approved historical research
pipeline. It handles one exact WFP price-flag series or one exact NBS
item/geography/methodology-regime series at a time. WFP flags are never
combined, missing prices are never filled, and reusable artifacts are written
only to a runtime cache (Render's filesystem is ephemeral).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = (
    "observation_lag_1", "observation_lag_2", "observation_lag_3", "rolling_mean_3", "rolling_std_3",
    "months_to_target", "gap_lag_1_to_lag_2", "gap_lag_2_to_lag_3",
    "month_sin", "month_cos", "trend_month",
)
MIN_SUPERVISED_ROWS = 30
MIN_TRAIN_ROWS = 20
MIN_VALIDATION_ROWS = 6
CACHE_FORMAT_VERSION = 6
DEFAULT_MAX_SOURCE_AGE_MONTHS = 3
DEFAULT_MAX_FORECAST_BRIDGE_MONTHS = 3
DEFAULT_BASELINE_MIN_HISTORY = 12
DEFAULT_BASELINE_MIN_TRAIN = 8
DEFAULT_BASELINE_MIN_VALIDATION = 4
DEFAULT_BASELINE_MAX_VALIDATION = 6
DEFAULT_BASELINE_MAX_TARGET_GAP_MONTHS = 3
MODEL_CONFIGURATION = {
    "random_forest": "300 trees; min_samples_leaf=2; max_features=1.0; seed=42; n_jobs=1",
    "gradient_boosting": "120 estimators; depth=2; learning_rate=0.03; Huber loss; seed=42",
    "trend_baseline": "extrapolated log-price trend over the last three actual observations; gap-adjusted",
    "robust_recent_movement": "median of up to six recent log price changes per elapsed calendar month, shrunk toward zero by sample support and directional agreement",
    "seasonal_naive_12": "same-calendar-month observation 12 months earlier; available only when every validation/future lag is observed or generated recursively",
}
BASELINE_MODELS = {"naive_last_observation", "trend_baseline", "robust_recent_movement", "seasonal_naive_12", "last_price_carry_forward"}


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


def _future_forecast_window(origin: str, horizon: int, as_of: date) -> tuple[int, int, int]:
    """Return the displayed future window and internal bridge length."""
    origin_ordinal = _month_ordinal(origin)
    generation_ordinal = _month_ordinal(f"{as_of.year:04d}-{as_of.month:02d}")
    first_future = max(origin_ordinal, generation_ordinal) + 1
    return first_future, first_future + horizon - 1, max(0, generation_ordinal - origin_ordinal)


def _classify_forecast_months(months: list[str], as_of: date, evaluation_only: bool = False) -> str:
    """Distinguish historical/current-month projections from future months."""
    if evaluation_only:
        return "validation_only"
    generation = _month_ordinal(f"{as_of.year:04d}-{as_of.month:02d}")
    ordinals = [_month_ordinal(month) for month in months]
    if not ordinals:
        return "unavailable"
    has_past = any(value < generation for value in ordinals)
    has_current = any(value == generation for value in ordinals)
    has_future = any(value > generation for value in ordinals)
    if has_future and (has_past or has_current):
        return "mixed_horizon_relative_to_generation"
    if has_future:
        return "future_as_of_generation"
    if has_current and has_past:
        return "mixed_horizon_relative_to_generation"
    if has_current:
        return "current_month_projection"
    return "historical_projection_from_latest_observation"


def forecast_policy() -> dict[str, int]:
    """Read conservative forecast limits from environment, with safe defaults."""
    defaults = {
        "max_source_age_months": DEFAULT_MAX_SOURCE_AGE_MONTHS,
        "max_bridge_months": DEFAULT_MAX_FORECAST_BRIDGE_MONTHS,
        "baseline_min_history": DEFAULT_BASELINE_MIN_HISTORY,
        "baseline_min_train": DEFAULT_BASELINE_MIN_TRAIN,
        "baseline_min_validation": DEFAULT_BASELINE_MIN_VALIDATION,
        "baseline_max_validation": DEFAULT_BASELINE_MAX_VALIDATION,
        "baseline_max_target_gap_months": DEFAULT_BASELINE_MAX_TARGET_GAP_MONTHS,
    }
    names = {
        "max_source_age_months": "AGRIPRICE_MAX_SOURCE_AGE_MONTHS",
        "max_bridge_months": "AGRIPRICE_MAX_FORECAST_BRIDGE_MONTHS",
        "baseline_min_history": "AGRIPRICE_BASELINE_MIN_HISTORY",
        "baseline_min_train": "AGRIPRICE_BASELINE_MIN_TRAIN_OBSERVATIONS",
        "baseline_min_validation": "AGRIPRICE_BASELINE_MIN_VALIDATION_OBSERVATIONS",
        "baseline_max_validation": "AGRIPRICE_BASELINE_MAX_VALIDATION_OBSERVATIONS",
        "baseline_max_target_gap_months": "AGRIPRICE_BASELINE_MAX_TARGET_GAP_MONTHS",
    }
    values: dict[str, int] = {}
    for key, default in defaults.items():
        raw = os.environ.get(names[key], "").strip()
        try:
            value = int(raw) if raw else default
        except ValueError as exc:
            raise ForecastingError(
                "forecast_configuration_invalid",
                f"{names[key]} must be a non-negative whole number.", 503,
            ) from exc
        if value < 0:
            raise ForecastingError(
                "forecast_configuration_invalid",
                f"{names[key]} must be a non-negative whole number.", 503,
            )
        values[key] = value
    if values["baseline_min_train"] < 3 or values["baseline_min_validation"] < 2:
        raise ForecastingError(
            "forecast_configuration_invalid",
            "Baseline validation requires at least three training observations and two validation observations.", 503,
        )
    if values["baseline_max_validation"] < values["baseline_min_validation"]:
        raise ForecastingError(
            "forecast_configuration_invalid",
            "The maximum baseline validation count cannot be below its minimum.", 503,
        )
    return values


def _source_age_months(source_key: str, origin_month: str, origin_date: str | None, as_of: date) -> int:
    """Calculate source age; use exact observation days only for dated WFP rows."""
    current_month = f"{as_of.year:04d}-{as_of.month:02d}"
    age = max(0, _month_ordinal(current_month) - _month_ordinal(origin_month))
    if source_key == "wfp" and origin_date:
        try:
            actual_date = date.fromisoformat(str(origin_date)[:10])
        except ValueError:
            actual_date = None
        if actual_date is not None:
            age -= int(as_of.day < actual_date.day and age > 0)
    return max(0, age)


def _check_source_freshness(
    source_key: str, origin_month: str, origin_date: str | None, as_of: date,
) -> tuple[int, int, dict[str, int]]:
    """Reject future observations; source age is reported, not an eligibility gate."""
    policy = forecast_policy()
    current_month = f"{as_of.year:04d}-{as_of.month:02d}"
    current_ordinal = _month_ordinal(current_month)
    origin_ordinal = _month_ordinal(origin_month)
    if origin_ordinal > current_ordinal:
        raise ForecastingError(
            "actual_data_after_generation_date",
            f"The latest source observation is in {origin_month}, after the forecast generation month {current_month}; future data were not used.",
        )
    if source_key == "wfp" and origin_date:
        try:
            observed_day = date.fromisoformat(str(origin_date)[:10])
        except ValueError:
            observed_day = None
        if observed_day is not None and observed_day > as_of:
            raise ForecastingError(
                "actual_data_after_generation_date",
                f"The latest source observation is dated {observed_day.isoformat()}, after the forecast generation date {as_of.isoformat()}; future data were not used.",
            )
    age = _source_age_months(source_key, origin_month, origin_date, as_of)
    # Staleness is surfaced as a warning, not an eligibility gate. Forecast
    # generation recursively estimates intervening months when needed, without
    # treating them as actual observations.
    return age, 0, policy


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
            f"The selected series has conflicting source prices for {month_list}. The records were not combined.",
        )

    return {
        "prices": values,
        "dates": dates,
        "missing_price_rows": missing_price_rows,
        "invalid_price_rows": invalid_price_rows,
        "invalid_month_rows": invalid_month_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
    }


def _feature_row(
    observations: list[tuple[str, float]],
    target_month: str,
    trend_origin: int,
) -> dict[str, float] | None:
    """Build features from the three prior observations, retaining real month gaps."""
    if len(observations) < 3:
        return None
    previous = observations[-3:]
    target_ordinal = _month_ordinal(target_month)
    previous_ordinals = [_month_ordinal(month) for month, _ in previous]
    values = [float(value) for _, value in reversed(previous)]
    latest_ordinal, second_ordinal, third_ordinal = reversed(previous_ordinals)
    months_to_target = target_ordinal - latest_ordinal
    if months_to_target <= 0:
        return None
    mean = sum(values) / 3.0
    month_number = target_ordinal % 12 + 1
    angle = 2.0 * math.pi * month_number / 12.0
    return {
        "observation_lag_1": values[0],
        "observation_lag_2": values[1],
        "observation_lag_3": values[2],
        "rolling_mean_3": mean,
        "rolling_std_3": math.sqrt(sum((value - mean) ** 2 for value in values) / 3.0),
        "months_to_target": float(months_to_target),
        "gap_lag_1_to_lag_2": float(latest_ordinal - second_ordinal),
        "gap_lag_2_to_lag_3": float(second_ordinal - third_ordinal),
        "month_sin": math.sin(angle),
        "month_cos": math.cos(angle),
        "trend_month": float(target_ordinal - trend_origin),
    }


def _build_feature_rows(prices: dict[str, float]) -> list[dict[str, Any]]:
    """Create next-observed-month training rows; absent months are never filled."""
    if not prices:
        return []
    ordered_months = sorted(prices, key=_month_ordinal)
    trend_origin = _month_ordinal(ordered_months[0])
    rows: list[dict[str, Any]] = []
    for target_index in range(3, len(ordered_months)):
        target_month = ordered_months[target_index]
        prior_months = ordered_months[target_index - 3:target_index]
        observations = [(month, float(prices[month])) for month in prior_months]
        features = _feature_row(observations, target_month, trend_origin)
        if features is not None:
            rows.append({
                "month": target_month,
                "target": float(prices[target_month]),
                "features": features,
                "prior_observation_months": prior_months,
            })
    return rows


def _assess_series(record: dict[str, Any], source_key: str) -> dict[str, Any]:
    source_label = "WFP" if source_key == "wfp" else "NBS"
    record_source = str(record.get("source_key") or "").lower()
    if record_source != source_key and str(record.get("source") or "").upper() != source_label:
        return {"eligible": False, "code": "source_not_supported", "reason": f"This series is not from the selected {source_label} source."}
    # WFP priceflag is part of the exact series identity. Non-actual groups
    # may be forecast only as their own reported series; they are never relabeled
    # as direct actuals or pooled with actual-flag markets.
    try:
        collected = _collect_actual_history(record)
    except ForecastingError as error:
        return {"eligible": False, "code": error.code, "reason": error.message}
    prices = collected["prices"]
    actual_count = len(prices)
    if not prices:
        return {
            "eligible": False, "code": "no_valid_actual_prices",
            "reason": f"No finite, positive reported monthly prices remain in this {source_label} series.",
            "actual_observations": 0,
            **{key: value for key, value in collected.items() if key != "prices"},
        }
    months = sorted(prices, key=_month_ordinal)
    feature_rows = _build_feature_rows(prices)
    baseline_validation = _rolling_baseline_validation(prices)
    split = int(math.floor(len(feature_rows) * 0.80))
    train_count = split
    validation_count = len(feature_rows) - split
    calendar_span = _month_ordinal(months[-1]) - _month_ordinal(months[0]) + 1
    reasons: list[str] = []
    if len(feature_rows) < MIN_SUPERVISED_ROWS:
        reasons.append(
            f"only {len(feature_rows)} gap-aware supervised rows can be formed from {actual_count} valid reported observations; at least {MIN_SUPERVISED_ROWS} are required"
        )
    if train_count < MIN_TRAIN_ROWS:
        reasons.append(f"the chronological training partition has {train_count} rows; at least {MIN_TRAIN_ROWS} are required")
    if validation_count < MIN_VALIDATION_ROWS:
        reasons.append(f"the chronological validation partition has {validation_count} rows; at least {MIN_VALIDATION_ROWS} are required")
    return {
        "eligible": not reasons,
        "code": "eligible" if not reasons else "insufficient_or_gapped_history",
        "reason": "; ".join(reasons),
        "actual_observations": actual_count,
        "first_actual_month": months[0],
        "latest_actual_month": months[-1],
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
        "baseline_validation": baseline_validation,
        "feature_rows": feature_rows,
        "price_map": prices,
        "observation_dates": collected["dates"],
        "source": source_key,
    }


def assess_wfp_series(record: dict[str, Any]) -> dict[str, Any]:
    """Assess one exact WFP price-flag series without fitting models."""
    return _assess_series(record, "wfp")


def assess_nbs_series(record: dict[str, Any]) -> dict[str, Any]:
    """Assess one exact NBS item/geography series without combining it with WFP."""
    return _assess_series(record, "nbs")


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


def _robust_monthly_movement(prices: dict[str, float], max_changes: int = 6) -> dict[str, Any]:
    """Estimate recent log movement per calendar month without filling gaps.

    The median of at most six adjacent-observation log changes is divided by
    each pair's elapsed calendar months. The result is shrunk toward zero by
    both sample support (n/(n+1)) and directional agreement (absolute signed
    balance). This is an empirical, conservative movement estimate, not a fixed
    percentage uplift. With no usable change, the resulting forecast is flat.
    """
    ordered = sorted(prices.items(), key=lambda item: _month_ordinal(item[0]))
    rates: list[float] = []
    for (previous_month, previous_price), (month, price) in zip(ordered, ordered[1:]):
        elapsed = _month_ordinal(month) - _month_ordinal(previous_month)
        if elapsed <= 0 or previous_price <= 0 or price <= 0:
            continue
        rate = math.log(float(price) / float(previous_price)) / elapsed
        if math.isfinite(rate):
            rates.append(rate)
    recent_rates = rates[-max_changes:]
    if not recent_rates:
        return {
            "monthly_log_rate": 0.0, "median_monthly_log_rate": 0.0,
            "change_count": 0, "directional_agreement": 0.0,
            "method": "last-price carry-forward; no usable historical change is available",
        }
    median_rate = float(statistics.median(recent_rates))
    nonzero_signs = [1 if rate > 0 else -1 if rate < 0 else 0 for rate in recent_rates]
    agreement = abs(sum(nonzero_signs)) / len(nonzero_signs)
    support_weight = len(recent_rates) / (len(recent_rates) + 1.0)
    monthly_rate = median_rate * support_weight * agreement
    if not math.isfinite(monthly_rate):
        monthly_rate = 0.0
    return {
        "monthly_log_rate": monthly_rate,
        "median_monthly_log_rate": median_rate,
        "change_count": len(recent_rates),
        "directional_agreement": agreement,
        "method": "robust recent movement; median calendar-month log changes with evidence-based shrinkage",
    }


def _movement_prediction(prices: dict[str, float], target_month: str) -> float:
    """Predict one target from the latest known value and robust monthly movement."""
    if not prices:
        raise ValueError("a movement prediction requires at least one positive observation")
    latest_month = max(prices, key=_month_ordinal)
    latest_price = float(prices[latest_month])
    elapsed = _month_ordinal(target_month) - _month_ordinal(latest_month)
    if elapsed <= 0:
        raise ValueError("movement target must follow the latest known observation")
    rate = _robust_monthly_movement(prices)["monthly_log_rate"]
    try:
        prediction = latest_price * math.exp(float(rate) * elapsed)
    except (OverflowError, ValueError):
        return latest_price
    return prediction if math.isfinite(prediction) and prediction > 0 else latest_price


def _rolling_baseline_validation(prices: dict[str, float]) -> dict[str, Any]:
    """Evaluate simple baselines on a short, recent rolling-origin window.

    Each target is the next *observed* month. Origins advance only after an
    actual observation; no missing month is filled and no validation target is
    used before its forecast is made.
    """
    policy = forecast_policy()
    months = sorted(prices, key=_month_ordinal)
    n = len(months)
    validation_count = min(policy["baseline_max_validation"], n - policy["baseline_min_train"])
    if n < policy["baseline_min_history"] or validation_count < policy["baseline_min_validation"]:
        return {
            "eligible": False,
            "reason": (
                f"short-history baseline validation requires at least {policy['baseline_min_history']} actual observations, "
                f"with {policy['baseline_min_train']} training and {policy['baseline_min_validation']} validation observations; found {n}."
            ),
            "actual_observations": n,
            "validation_rows": 0,
            "validation_months": [],
            "scores": {},
            "predictions": [],
        }

    start = n - validation_count
    validation_months = months[start:]
    for index in range(start, n):
        step_gap = _month_ordinal(months[index]) - _month_ordinal(months[index - 1])
        if step_gap > policy["baseline_max_target_gap_months"]:
            return {
                "eligible": False,
                "reason": (
                    f"short-history baseline validation was not run because the next observed target is {step_gap} calendar months "
                    f"after its rolling origin; the configured maximum is {policy['baseline_max_target_gap_months']} months."
                ),
                "actual_observations": n,
                "validation_rows": 0,
                "validation_months": [],
                "scores": {},
                "predictions": [],
            }

    all_seasonal_available = all(
        _month_from_ordinal(_month_ordinal(month) - 12) in set(months[:index])
        for index, month in enumerate(months[start:], start=start)
    )
    predictions: list[dict[str, Any]] = []
    values_by_model: dict[str, list[float]] = {
        "naive_last_observation": [],
        "trend_baseline": [],
        "robust_recent_movement": [],
    }
    if all_seasonal_available:
        values_by_model["seasonal_naive_12"] = []

    for index in range(start, n):
        month = months[index]
        past = months[:index]
        previous = [float(prices[key]) for key in past[-3:]]
        naive = float(prices[past[-1]])
        gap_to_target = _month_ordinal(month) - _month_ordinal(past[-1])
        elapsed = _month_ordinal(past[-1]) - _month_ordinal(past[-3])
        slope = math.log(previous[0] / previous[-1]) / elapsed if elapsed > 0 else 0.0
        trend = previous[0] * math.exp(slope * gap_to_target)
        model_predictions = {
            "naive_last_observation": naive,
            "trend_baseline": trend,
            "robust_recent_movement": _movement_prediction(
                {key: float(prices[key]) for key in past}, month,
            ),
        }
        if all_seasonal_available:
            seasonal_month = _month_from_ordinal(_month_ordinal(month) - 12)
            model_predictions["seasonal_naive_12"] = float(prices[seasonal_month])
        actual = float(prices[month])
        for name, value in model_predictions.items():
            if not math.isfinite(value) or value <= 0:
                return {
                    "eligible": False,
                    "reason": f"short-history baseline validation produced an invalid {name} prediction for {month}.",
                    "actual_observations": n,
                    "validation_rows": 0,
                    "validation_months": [],
                    "scores": {},
                    "predictions": [],
                }
            values_by_model[name].append(value)
        predictions.append({"month": month, "actual": actual, "models": model_predictions})

    actuals = [float(prices[month]) for month in validation_months]
    scores = {name: _metrics(actuals, values) for name, values in values_by_model.items()}
    return {
        "eligible": True,
        "reason": "Simple baselines passed rolling-origin validation on recent observed months.",
        "actual_observations": n,
        "training_observations": start,
        "validation_rows": validation_count,
        "validation_months": validation_months,
        "scores": scores,
        "predictions": predictions,
        "validation_design": "rolling-origin one-step-ahead forecasts of the next actual observation; no target values were used before their origin",
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
        if bundle.get("model_name") not in (BASELINE_MODELS | {"random_forest", "gradient_boosting"}):
            return None
        if bundle.get("model_name") not in BASELINE_MODELS and bundle.get("estimator") is None:
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
    as_of: date,
    source_key: str,
    origin_date: str | None,
) -> tuple[list[dict[str, Any]], int]:
    origin_ordinal = _month_ordinal(origin)
    _check_source_freshness(source_key, origin, origin_date, as_of)
    generation_ordinal = _month_ordinal(f"{as_of.year:04d}-{as_of.month:02d}")
    final_ordinal = generation_ordinal + horizon
    working = dict(prices)
    starting_price = float(prices[origin])
    movement = _robust_monthly_movement(prices)
    result: list[dict[str, Any]] = []
    bridge_count = max(0, generation_ordinal - origin_ordinal)
    for target_ordinal in range(origin_ordinal + 1, final_ordinal + 1):
        target_month = _month_from_ordinal(target_ordinal)
        previous_month = max(working, key=_month_ordinal)
        previous_price = float(working[previous_month])
        if model_name in {"naive_last_observation", "last_price_carry_forward"}:
            prediction = previous_price
            step_method = "last available price carried forward"
        elif model_name == "robust_recent_movement":
            try:
                prediction = previous_price * math.exp(float(movement["monthly_log_rate"]))
            except (OverflowError, ValueError):
                prediction = previous_price
            step_method = movement["method"]
        else:
            observations = sorted(working.items(), key=lambda item: _month_ordinal(item[0]))
            feature = _feature_row(observations, target_month, trend_origin)
            if feature is None:
                raise ForecastingError(
                    "forecast_history_gap",
                    f"The selected model cannot construct a valid historical input for {target_month}; no missing price was filled.",
                )
            if model_name == "trend_baseline":
                gaps = feature["gap_lag_1_to_lag_2"] + feature["gap_lag_2_to_lag_3"]
                slope = math.log(feature["observation_lag_1"] / feature["observation_lag_3"]) / gaps if gaps > 0 else 0.0
                prediction = feature["observation_lag_1"] * math.exp(slope * feature["months_to_target"])
                step_method = MODEL_CONFIGURATION[model_name]
            elif model_name == "seasonal_naive_12":
                seasonal_month = _month_from_ordinal(target_ordinal - 12)
                if seasonal_month not in working:
                    raise ForecastingError(
                        "seasonal_history_unavailable",
                        f"A same-calendar-month value is unavailable for {seasonal_month}; no missing price was substituted.",
                    )
                prediction = working[seasonal_month]
                step_method = MODEL_CONFIGURATION[model_name]
            else:
                prediction = _predict(estimator, feature)
                step_method = f"recursive {model_name.replace('_', ' ')} prediction"
        if not math.isfinite(float(prediction)) or float(prediction) <= 0:
            prediction = previous_price
            step_method = "last available price carried forward because the model output was not finite and positive"
        prediction = float(prediction)
        working[target_month] = prediction
        percentage_change = (prediction / starting_price - 1.0) * 100.0
        if not math.isfinite(percentage_change):
            percentage_change = None
        if target_ordinal > generation_ordinal:
            result.append({
                "date": target_month,
                "month": target_month,
                "price": prediction,
                "step": len(result) + 1,
                "percentage_change_from_origin": percentage_change,
                "quality_flag": "model_estimate",
                "method": step_method,
                "temporal_status": "future_as_of_generation",
            })
    return result, bridge_count


def _movement_fallback_result(
    record: dict[str, Any],
    series_id: str,
    horizon: int,
    source_key: str,
    assessment: dict[str, Any],
    as_of: date,
    reason: str,
    evaluation_only: bool = False,
) -> dict[str, Any]:
    """Guarantee a qualified estimate for an exact series with positive history."""
    prices = assessment.get("price_map") or {}
    if not prices:
        raise ForecastingError("no_valid_actual_prices", "No finite positive price is available for the exact selected series.")
    origin = max(prices, key=_month_ordinal)
    origin_date = (assessment.get("observation_dates") or {}).get(origin)
    age, _calendar_age, policy = _check_source_freshness(source_key, origin, origin_date, as_of)
    movement = _robust_monthly_movement(prices)
    model_name = "robust_recent_movement" if movement["change_count"] else "last_price_carry_forward"
    if abs(float(movement["monthly_log_rate"])) < 1e-15:
        model_name = "last_price_carry_forward"
    forecasts: list[dict[str, Any]] = []
    bridge_count = 0
    if not evaluation_only:
        forecasts, bridge_count = _forecast_prices(
            model_name, None, prices, origin,
            min(_month_ordinal(month) for month in prices), horizon,
            as_of, source_key, origin_date,
        )
    classification = _classify_forecast_months([str(row["month"]) for row in forecasts], as_of, evaluation_only)
    freshness = "fresh" if age <= policy["max_source_age_months"] else "stale"
    count = len(prices)
    sparse_calendar = assessment.get("missing_calendar_months", 0) > count
    confidence = "very low" if count < 3 else "low" if count < 12 or freshness == "stale" or sparse_calendar else "limited"
    is_wfp_actual = source_key == "wfp" and str(record.get("price_flag") or "") == "actual"
    flag = record.get("price_flag") if source_key == "wfp" else None
    if source_key == "wfp":
        history_label = "WFP direct actual observations" if is_wfp_actual else f"WFP reported priceflag={flag or 'not stated'} observations; not labelled as direct actuals"
    else:
        history_label = "NBS reported monthly observations"
    method = movement["method"] if model_name == "robust_recent_movement" else "last available price carried forward; no reliable change was observed"
    warning = (
        f"An exact-series fallback estimate was generated because model/validation requirements were not met ({reason}). "
        f"It starts from the latest reported value in {origin}{f' ({origin_date})' if origin_date else ''}. "
        f"Source age is {age} calendar month(s); freshness limit is {policy['max_source_age_months']} month(s). "
        "Treat this as a low-confidence estimate, not a live or guaranteed market price."
    )
    if freshness == "stale":
        warning += " The source is stale for current-market interpretation; these future dates are projections from historical information."
    if bridge_count:
        warning += f" {bridge_count} intervening calendar month(s) were recursively estimated internally to reach the future window; they are not actual observations."
    metadata = {
        "source": source_key.upper(),
        "source_key": source_key,
        "commodity": record.get("commodity"),
        "market": record.get("market"),
        "market_id": record.get("market_id"),
        "state": record.get("state"),
        "geography_level": record.get("geography_level"),
        "geography_name": record.get("geography_name"),
        "methodology_regime": record.get("methodology_regime"),
        "research_commodity_mapping": record.get("research_commodity_mapping"),
        "mapping_status": record.get("mapping_status"),
        "unit": record.get("unit"),
        "unit_basis": record.get("unit_basis"),
        "price_type": record.get("price_type"),
        "currency": record.get("currency"),
        "price_flag": flag,
        "model": model_name,
        "model_type": "unvalidated conservative baseline",
        "strategy_level": "exact_series_fallback",
        "transformation": "source-reported price; no transformation across commodities, units, geographies, regimes, or price flags",
        "method": method,
        "movement_estimate": movement,
        "forecast_origin_month": origin,
        "forecast_start_month": forecasts[0]["month"] if forecasts else None,
        "forecast_end_month": forecasts[-1]["month"] if forecasts else None,
        "latest_actual_month_in_source": origin if is_wfp_actual else None,
        "latest_actual_date_in_source": origin_date if is_wfp_actual else None,
        "latest_observation_month_in_source": origin,
        "latest_observation_date_in_source": origin_date or origin,
        "source_age_months": age,
        "freshness_limit_months": policy["max_source_age_months"],
        "source_freshness_classification": f"fresh_within_{policy['max_source_age_months']}_months" if freshness == "fresh" else "stale_for_current_market",
        "is_current_market_forecast": not evaluation_only and freshness == "fresh" and classification == "future_as_of_generation",
        "forecast_date_scope": "validation_only" if evaluation_only else "future_calendar_months_after_generation_month",
        "forecast_bridge_months": bridge_count,
        "forecast_generation_date": as_of.isoformat(),
        "forecast_classification": classification,
        "confidence_level": confidence,
        "confidence_note": "Qualitative confidence only; no calibrated prediction interval is available.",
        "historical_price_label": history_label,
        "confidence_intervals": False,
        "validation_metrics": {"rmse": None, "mae": None, "mape": None, "r2": None, "n": 0},
        "model_comparison_validation": [],
        "validation_rows": 0,
        "source_observation_count": count,
        "valid_positive_observations": count,
        "missing_calendar_months": assessment.get("missing_calendar_months"),
        "eligibility_reason": reason,
        "selection_basis": "No model-selection claim: insufficient supervised history; a transparent exact-series robust movement/carry-forward fallback was used.",
    }
    history = [] if evaluation_only else [
        row for row in record.get("history", [])
        if row.get("month") and _month_ordinal(row["month"]) >= _month_ordinal(origin) - 23 and row.get("month") <= origin
    ]
    return {"series": record, "model": model_name, "horizon": horizon, "historical": history, "forecast": forecasts, "metadata": metadata, "warning": warning}


def _short_history_baseline_result(
    record: dict[str, Any],
    series_id: str,
    horizon: int,
    source_key: str,
    assessment: dict[str, Any],
    as_of: date,
    evaluation_only: bool = False,
) -> dict[str, Any]:
    """Return a transparently qualified baseline result for short exact series."""
    baseline = assessment.get("baseline_validation") or {}
    if not baseline.get("eligible"):
        raise ForecastingError(
            "insufficient_history",
            baseline.get("reason") or assessment.get("reason") or "The selected exact series has insufficient history for validation.",
        )
    prices = assessment["price_map"]
    origin = assessment["latest_actual_month"]
    origin_date = assessment.get("observation_dates", {}).get(origin)
    age, bridge, policy = _check_source_freshness(source_key, origin, origin_date, as_of) if not evaluation_only else (
        _source_age_months(source_key, origin, origin_date, as_of),
        0,
        forecast_policy(),
    )
    scores = baseline["scores"]
    selected_model = min(scores, key=lambda name: float(scores[name]["rmse"]))
    comparison = [{"model": name, **metrics} for name, metrics in sorted(scores.items())]
    selected_metrics = scores[selected_model]
    forecasts: list[dict[str, Any]] = []
    bridge_count = 0
    if not evaluation_only:
        forecasts, bridge_count = _forecast_prices(
            selected_model, None, prices, origin, _month_ordinal(min(prices, key=_month_ordinal)),
            horizon, as_of, source_key, origin_date,
        )
    latest_row = next((row for row in reversed(record.get("history", [])) if row.get("month") == origin), {})
    classification = _classify_forecast_months([str(row["month"]) for row in forecasts], as_of, evaluation_only)
    is_wfp_actual = source_key == "wfp" and str(record.get("price_flag") or "") == "actual"
    metadata = {
        "source": source_key.upper(), "source_key": source_key,
        "commodity": record.get("commodity"),
        "research_commodity_mapping": record.get("research_commodity_mapping"),
        "mapping_status": record.get("mapping_status"),
        "market": record.get("market"), "market_id": record.get("market_id"), "state": record.get("state"),
        "geography_level": record.get("geography_level"), "geography_name": record.get("geography_name"),
        "methodology_regime": record.get("methodology_regime"),
        "unit": record.get("unit"), "unit_basis": record.get("unit_basis"),
        "price_type": record.get("price_type"), "currency": record.get("currency"),
        "price_flag": record.get("price_flag") if source_key == "wfp" else None,
        "model": selected_model, "model_type": "validated baseline",
        "strategy_level": "individual_series_short_history_baseline",
        "transformation": "original reported monthly price in source currency per exact source-defined series",
        "method": "simple baseline forecast selected from a rolling-origin comparison; no model extrapolation or missing-price filling",
        "validation_design": baseline["validation_design"],
        "forecast_origin_month": origin,
        "forecast_start_month": forecasts[0]["month"] if forecasts else None,
        "forecast_end_month": forecasts[-1]["month"] if forecasts else None,
        "latest_actual_month_in_source": origin if is_wfp_actual else None,
        "latest_actual_date_in_source": origin_date if is_wfp_actual else None,
        "latest_observation_month_in_source": origin,
        "latest_observation_date_in_source": origin_date or origin,
        "latest_source_report_date": latest_row.get("source_report_date"),
        "source_age_months": age,
        "freshness_limit_months": policy["max_source_age_months"],
        "max_forecast_bridge_months": policy["max_bridge_months"],
        "source_freshness_classification": f"fresh_within_{policy['max_source_age_months']}_months" if age <= policy["max_source_age_months"] else "stale_for_current_market",
        "is_current_market_forecast": not evaluation_only and age <= policy["max_source_age_months"],
        "forecast_date_scope": "validation_only" if evaluation_only else "future_calendar_months_after_generation_month",
        "forecast_bridge_months": bridge_count,
        "forecast_generation_date": as_of.isoformat(),
        "forecast_classification": classification,
        "is_current_market_forecast": not evaluation_only and age <= policy["max_source_age_months"] and classification == "future_as_of_generation",
        "price_flag": record.get("price_flag") if source_key == "wfp" else None,
        "historical_price_label": "WFP direct actual observations" if is_wfp_actual else (f"WFP reported priceflag={record.get('price_flag') or 'not stated'} observations; not direct actuals" if source_key == "wfp" else "NBS reported monthly observations"),
        "confidence_level": "low" if age > policy["max_source_age_months"] or assessment["actual_observations"] < 12 or assessment.get("missing_calendar_months", 0) > assessment["actual_observations"] else "limited",
        "confidence_note": "Qualitative confidence only; no calibrated prediction interval is available.",
        "confidence_intervals": False,
        "selection_basis": "lowest RMSE among simple baselines on the same recent rolling-origin validation dates; no test data or advanced-model results were used",
        "validation_metrics": selected_metrics,
        "model_comparison_validation": comparison,
        "validation_predictions": baseline["predictions"],
        "validation_months": baseline["validation_months"],
        "training_rows": baseline["training_observations"],
        "validation_rows": baseline["validation_rows"],
        "validation_first_month": baseline["validation_months"][0],
        "validation_last_month": baseline["validation_months"][-1],
        "source_observation_count": assessment["actual_observations"],
        "usable_training_rows": assessment["actual_observations"],
        "missing_calendar_months": assessment["missing_calendar_months"],
        "input_window_observations": "last observed price; trend baseline uses last three actual observations",
        "model_configurations": {key: value for key, value in MODEL_CONFIGURATION.items() if key in scores},
        "evaluation_note": "Short-series validation is not independent across rolling origins; it is a limited one-step-ahead diagnostic, not evidence of long-horizon accuracy.",
    }
    history = [] if evaluation_only else [
        row for row in record.get("history", [])
        if row.get("month") and _month_ordinal(row["month"]) >= _month_ordinal(origin) - 23
        and row.get("month") <= origin
    ]
    warning = (
        "Validation-only model comparison; no forecast was generated."
        if evaluation_only else
        f"A simple validated baseline was selected for this short exact series. The latest observation is {origin}"
        f"{f' ({origin_date})' if origin_date else ''}; it is {age} calendar month(s) old. "
        "Displayed forecasts cover calendar months after the generation month; any intervening months are recursively estimated internally, not actual observations. This is an estimate, not a live or guaranteed price."
    )
    return {
        "series": record, "model": selected_model, "horizon": horizon,
        "historical": history, "forecast": forecasts, "metadata": metadata, "warning": warning,
    }


def _generate_forecast(
    record: dict[str, Any],
    series_id: str,
    horizon: int,
    source_key: str,
    registry_entry: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
    root: Path = ROOT,
    as_of: date | None = None,
    pool_records: list[dict[str, Any]] | None = None,
    evaluation_only: bool = False,
) -> dict[str, Any]:
    """Fit/validate one exact series; optionally return validation only."""
    if horizon not in range(1, 7):
        raise ForecastingError("invalid_horizon", "horizon must be an integer from 1 through 6 months.", 400)
    today = as_of or date.today()
    assessment = assess_wfp_series(record) if source_key == "wfp" else assess_nbs_series(record)
    origin = assessment.get("latest_actual_month")
    origin_date = (assessment.get("observation_dates") or {}).get(origin) if origin else None
    if origin and not evaluation_only:
        _check_source_freshness(source_key, origin, origin_date, today)
    if not assessment["eligible"]:
        # Keep compatible pooling available for sparse actual/NBS series; the
        # exact-series movement fallback guarantees a result if pooling cannot
        # validate. Non-actual WFP flags are never borrowed into a pool.
        if assessment["code"] == "insufficient_or_gapped_history":
            cache_override = os.environ.get("AGRIPRICE_MODEL_CACHE_DIR", "").strip()
            cache_root = Path(cache_dir) if cache_dir is not None else (
                Path(cache_override) if cache_override else Path(tempfile.gettempdir()) / "agriprice-ng" / "runtime-models"
            )
            can_pool = source_key == "nbs" or str(record.get("price_flag") or "") == "actual"
            if can_pool:
                from .pooled_forecasting import generate_pooled_forecast
                try:
                    pooled = generate_pooled_forecast(
                        record, series_id, horizon, source_key,
                        pool_records or [record], cache_root, today,
                        evaluation_only=evaluation_only,
                    )
                    return pooled
                except ForecastingError as pooled_error:
                    if pooled_error.code in {"actual_data_after_generation_date", "conflicting_monthly_observations"}:
                        raise
                    pool_reason = pooled_error.message
                except Exception:
                    pool_reason = "compatible pooled validation or prediction was unavailable"
            else:
                pool_reason = "WFP non-actual price flags are not pooled with other flags or actual-price markets"
            baseline = assessment.get("baseline_validation") or {}
            if baseline.get("eligible"):
                return _short_history_baseline_result(
                    record, series_id, horizon, source_key, assessment, today, evaluation_only,
                )
            return _movement_fallback_result(
                record, series_id, horizon, source_key, assessment, today,
                f"{assessment.get('reason') or 'insufficient supervised history'}; {pool_reason}",
                evaluation_only,
            )
        raise ForecastingError(assessment["code"], assessment["reason"])

    prices = assessment["price_map"]
    origin = str(assessment["latest_actual_month"])
    origin_date = assessment["observation_dates"].get(origin)
    current_month = f"{today.year:04d}-{today.month:02d}"
    origin_age, policy = _source_age_months(source_key, origin, origin_date, today), forecast_policy()

    feature_rows = assessment["feature_rows"]
    split = assessment["training_rows"]
    train_rows = feature_rows[:split]
    validation_rows = feature_rows[split:]
    actual_validation = [float(row["target"]) for row in validation_rows]
    baseline_predictions = [float(row["features"]["observation_lag_1"]) for row in validation_rows]
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
    validation_predictions: dict[str, list[float]]
    train_started = time.perf_counter()

    if cached is not None:
        selected_model = str(cached["model_name"])
        selected_estimator = cached.get("estimator")
        validation_scores = cached["validation_scores"]
        validation_predictions = cached.get("validation_predictions", {})
        model_failures = cached.get("model_failures", {})
        model_reuse = "runtime_cache_hit"
        fit_seconds = float(cached.get("fit_seconds", 0.0) or 0.0)
    else:
        validation_scores = {
            "naive_last_observation": _metrics(actual_validation, baseline_predictions),
        }
        validation_predictions = {"naive_last_observation": baseline_predictions}
        trend_predictions = []
        for row in validation_rows:
            features = row["features"]
            gaps = features["gap_lag_1_to_lag_2"] + features["gap_lag_2_to_lag_3"]
            slope = math.log(features["observation_lag_1"] / features["observation_lag_3"]) / gaps if gaps > 0 else 0.0
            trend_predictions.append(features["observation_lag_1"] * math.exp(slope * features["months_to_target"]))
        validation_scores["trend_baseline"] = _metrics(actual_validation, trend_predictions)
        validation_predictions["trend_baseline"] = trend_predictions
        movement_predictions = []
        for row in validation_rows:
            past_prices = {
                month: float(price) for month, price in prices.items()
                if _month_ordinal(month) < _month_ordinal(row["month"])
            }
            movement_predictions.append(_movement_prediction(past_prices, row["month"]))
        validation_scores["robust_recent_movement"] = _metrics(actual_validation, movement_predictions)
        validation_predictions["robust_recent_movement"] = movement_predictions

        # A seasonal-naive metric is reported only on the exact same complete
        # validation dates; otherwise it is not a fair comparator.
        seasonal_predictions = []
        seasonal_complete = True
        for row in validation_rows:
            seasonal_month = _month_from_ordinal(_month_ordinal(row["month"]) - 12)
            if seasonal_month not in assessment["price_map"]:
                seasonal_complete = False
                break
            seasonal_predictions.append(float(assessment["price_map"][seasonal_month]))
        if seasonal_complete and len(seasonal_predictions) == len(validation_rows):
            validation_scores["seasonal_naive_12"] = _metrics(actual_validation, seasonal_predictions)
            validation_predictions["seasonal_naive_12"] = seasonal_predictions
        model_failures = {}
        try:
            estimators = _make_estimators()
        except ForecastingError as exc:
            estimators = {}
            model_failures["machine_learning_models"] = exc.message
        except ImportError as exc:
            estimators = {}
            model_failures["machine_learning_models"] = "scikit-learn is unavailable; baseline candidates remain active."

        train_matrix = _matrix(train_rows)
        validation_matrix = _matrix(validation_rows)
        for name, estimator in estimators.items():
            try:
                estimator.fit(train_matrix, [row["target"] for row in train_rows])
                prediction = [max(0.0, float(value)) for value in estimator.predict(validation_matrix)]
                if len(prediction) != len(actual_validation) or not all(math.isfinite(value) for value in prediction):
                    raise ValueError("model returned invalid validation predictions")
                validation_scores[name] = _metrics(actual_validation, prediction)
                validation_predictions[name] = prediction
            except Exception:
                model_failures[name] = "Validation fitting or prediction failed; this model was excluded from selection."

        selected_model = "naive_last_observation"
        selected_estimator = None
        model_reuse = "baseline_selected"
        ranked_models = sorted(validation_scores, key=lambda name: float(validation_scores[name]["rmse"]))
        for candidate_model in ranked_models:
            if candidate_model in BASELINE_MODELS:
                selected_model = candidate_model
                selected_estimator = None
                model_reuse = f"{candidate_model}_selected"
                break
            if evaluation_only:
                selected_model = candidate_model
                selected_estimator = None
                model_reuse = "validation_only_no_final_refit"
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

    # A seasonal winner is admissible only if its recursive lag can be
    # constructed for each requested month after this exact series' origin.
    if selected_model == "seasonal_naive_12" and not evaluation_only:
        working_months = set(prices)
        seasonal_supported = True
        _first_future, final_ordinal, _bridge_count = _future_forecast_window(origin, horizon, today)
        for target_ordinal in range(_month_ordinal(origin) + 1, final_ordinal + 1):
            target_month = _month_from_ordinal(target_ordinal)
            seasonal_month = _month_from_ordinal(target_ordinal - 12)
            if seasonal_month not in working_months:
                seasonal_supported = False
                break
            working_months.add(target_month)
        if not seasonal_supported:
            alternatives = [name for name in validation_scores if name != "seasonal_naive_12"]
            selected_model = min(alternatives, key=lambda name: float(validation_scores[name]["rmse"]))
            if selected_model in BASELINE_MODELS:
                selected_estimator = None
                model_reuse = f"{selected_model}_selected"
            else:
                selected_estimator = _make_estimators()[selected_model]
                selected_estimator.fit(_matrix(feature_rows), [row["target"] for row in feature_rows])
                model_reuse = "trained_on_demand"
    if evaluation_only:
        forecast_rows, bridge_month_count = [], 0
    else:
        try:
            forecast_rows, bridge_month_count = _forecast_prices(
                selected_model, selected_estimator, prices, origin,
                min(_month_ordinal(month) for month in prices), horizon,
                today, source_key, origin_date,
            )
        except Exception as model_error:
            if isinstance(model_error, ForecastingError) and model_error.code == "actual_data_after_generation_date":
                raise
            selected_model = "robust_recent_movement" if _robust_monthly_movement(prices)["change_count"] else "last_price_carry_forward"
            selected_estimator = None
            model_reuse = "safe_exact_series_fallback_after_model_error"
            forecast_rows, bridge_month_count = _forecast_prices(
                selected_model, None, prices, origin,
                min(_month_ordinal(month) for month in prices), horizon,
                today, source_key, origin_date,
            )
    validation_dates = [row["month"] for row in validation_rows]
    selected_metrics = validation_scores.get(selected_model) or {
        "rmse": None, "mae": None, "mape": None, "r2": None, "n": 0,
    }

    comparison = [
        {"model": name, **metrics}
        for name, metrics in sorted(validation_scores.items(), key=lambda item: item[0])
    ]
    forecast_start_month = forecast_rows[0]["month"] if forecast_rows else None
    forecast_end_month = forecast_rows[-1]["month"] if forecast_rows else None
    forecast_classification = _classify_forecast_months([str(row["month"]) for row in forecast_rows], today, evaluation_only)
    is_wfp_actual = source_key == "wfp" and str(record.get("price_flag") or "") == "actual"
    temporal_phrase = (
        "latest WFP direct actual observation" if is_wfp_actual else
        f"latest WFP priceflag={record.get('price_flag') or 'not stated'} observation" if source_key == "wfp" else
        "latest NBS reported observation"
    )
    warning = (
        "Validation-only historical model comparison; no forecast was generated."
        if evaluation_only else
        f"Generated using the {temporal_phrase} through {origin}{f', {origin_date}' if origin_date else ''}. "
        f"The source is {origin_age} calendar month(s) behind the generation month {current_month}. "
        "Displayed forecast months begin after the generation month. Any intervening months were recursively estimated internally; they are estimates, not source observations or historical actual prices. "
        "This is an estimate, not a live or guaranteed market price."
    )
    if not evaluation_only and origin_age > policy["max_source_age_months"]:
        warning += " The latest source observation is stale for current-market interpretation; treat this as a model estimate based on historical information."

    metadata = {
        "source": source_key.upper(),
        "source_key": source_key,
        "commodity": record.get("commodity"),
        "market": record.get("market"),
        "market_id": record.get("market_id"),
        "state": record.get("state"),
        "geography_level": record.get("geography_level"),
        "geography_name": record.get("geography_name"),
        "methodology_regime": record.get("methodology_regime"),
        "research_commodity_mapping": record.get("research_commodity_mapping"),
        "mapping_status": record.get("mapping_status"),
        "unit": record.get("unit"),
        "unit_basis": record.get("unit_basis"),
        "price_type": record.get("price_type"),
        "currency": record.get("currency"),
        "price_flag": record.get("price_flag") if source_key == "wfp" else None,
        "model": selected_model,
        "model_type": "individual-series model" if selected_model not in BASELINE_MODELS else "validated baseline",
        "strategy_level": "individual_series",
        "transformation": "original reported monthly price in source currency per source-defined series",
        "method": (
            "recursive robust recent movement from exact-series calendar-month-normalized changes; previous predictions feed later steps"
            if selected_model == "robust_recent_movement" else
            "last available exact-series price carried forward recursively"
            if selected_model in {"naive_last_observation", "last_price_carry_forward"} else
            "recursive calendar-month model forecast; previous model predictions feed later steps"
        ),
        "validation_design": "chronological 80/20 split of observed-month supervised rows; validation is rolling-origin one-step prediction using only history available before each target month",
        "forecast_origin_month": origin,
        "forecast_start_month": forecast_start_month,
        "forecast_end_month": forecast_end_month,
        "latest_actual_month_in_source": origin if is_wfp_actual else None,
        "latest_actual_date_in_source": origin_date if is_wfp_actual else None,
        "latest_observation_month_in_source": origin,
        "latest_observation_date_in_source": origin_date or origin,
        "latest_source_report_date": next((row.get("source_report_date") for row in reversed(record.get("history", [])) if row.get("month") == origin), None),
        "source_age_months": origin_age,
        "freshness_limit_months": policy["max_source_age_months"],
        "max_forecast_bridge_months": policy["max_bridge_months"],
        "source_freshness_classification": f"fresh_within_{policy['max_source_age_months']}_months" if origin_age <= policy["max_source_age_months"] else "stale_for_current_market",
        "is_current_market_forecast": not evaluation_only and origin_age <= policy["max_source_age_months"] and forecast_classification == "future_as_of_generation",
        "forecast_date_scope": "validation_only" if evaluation_only else "future_calendar_months_after_generation_month",
        "forecast_bridge_months": bridge_month_count,
        "forecast_generation_date": today.isoformat(),
        "forecast_classification": forecast_classification,
        "confidence_intervals": False,
        "selection_basis": "lowest chronological validation RMSE among Random Forest, Gradient Boosting, naive, robust recent movement and trend baselines; seasonal-naive is included only on complete shared validation months; no random split or test-set selection",
        "validation_metrics": selected_metrics,
        "model_comparison_validation": comparison,
        "validation_predictions": [
            {
                "month": row["month"],
                "actual": float(row["target"]),
                "models": {
                    name: float(values[index])
                    for name, values in validation_predictions.items()
                    if index < len(values) and math.isfinite(float(values[index]))
                },
            }
            for index, row in enumerate(validation_rows)
        ],
        "validation_months": validation_dates,
        "training_first_month": train_rows[0]["month"],
        "training_last_month": train_rows[-1]["month"],
        "validation_first_month": validation_rows[0]["month"],
        "validation_last_month": validation_rows[-1]["month"],
        "final_model_fit_rows": len(feature_rows),
        "final_model_fit_first_month": feature_rows[0]["month"],
        "final_model_fit_last_month": feature_rows[-1]["month"],
        "actual_observation_count": assessment["actual_observations"] if is_wfp_actual else None,
        "source_observation_count": assessment["actual_observations"],
        "valid_positive_observations": assessment["actual_observations"],
        "historical_price_label": "WFP direct actual observations" if is_wfp_actual else (f"WFP reported priceflag={record.get('price_flag') or 'not stated'} observations; not direct actuals" if source_key == "wfp" else "NBS reported monthly observations"),
        "confidence_level": "low" if origin_age > policy["max_source_age_months"] or assessment["actual_observations"] < 12 or assessment.get("missing_calendar_months", 0) > assessment["actual_observations"] else "limited",
        "confidence_note": "Qualitative confidence only; no calibrated prediction interval is available.",
        "usable_training_rows": len(feature_rows),
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "missing_calendar_months": assessment["missing_calendar_months"],
        "missing_price_rows": assessment["missing_price_rows"],
        "invalid_price_rows_excluded": assessment["invalid_price_rows"],
        "invalid_month_rows_excluded": assessment["invalid_month_rows"],
        "exact_duplicate_rows_removed": assessment["duplicate_rows_removed"],
        "input_window_observations": 3,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_configurations": MODEL_CONFIGURATION,
        "model_failures": model_failures,
        "model_reuse": model_reuse,
        "fit_seconds": fit_seconds,
        "runtime_artifact_persistence": "OS temporary directory; ephemeral on Render and rebuilt after service restart",
        "evaluation_note": "Validation metrics describe the exact chronological validation dates; test metrics were not used for model selection.",
    }

    bundle = {
        "cache_key": cache_key,
        "series_id": series_id,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_name": selected_model,
        "estimator": selected_estimator,
        "validation_scores": validation_scores,
        "validation_predictions": validation_predictions,
        "model_failures": model_failures,
        "fit_seconds": fit_seconds,
    }
    cache_metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "cache_key": cache_key,
        "series_id": series_id,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_name": selected_model,
        "target": f"{source_key.upper()} reported monthly series price",
        "target_transformation": metadata["transformation"],
        "forecast_origin_month": origin,
        "training_first_month": feature_rows[0]["month"],
        "training_last_month": feature_rows[-1]["month"],
        "created_at": today.isoformat(),
        "validation_metrics": selected_metrics,
        "selection_basis": metadata["selection_basis"],
        "model_configurations": MODEL_CONFIGURATION,
    }
    cache_persisted = _save_runtime_cache(cache_root, series_id, bundle, cache_metadata) if cached is None and not evaluation_only else bool(cached)
    if model_reuse != "runtime_cache_hit" and cache_persisted:
        metadata["model_reuse"] = model_reuse
    metadata["runtime_cache_status"] = "validation_only" if evaluation_only else ("hit" if cached is not None else ("saved" if cache_persisted else "not_persisted"))

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


def generate_wfp_forecast(
    record: dict[str, Any],
    series_id: str,
    horizon: int,
    registry_entry: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
    root: Path = ROOT,
    as_of: date | None = None,
    pool_records: list[dict[str, Any]] | None = None,
    evaluation_only: bool = False,
) -> dict[str, Any]:
    """Forecast or evaluate an exact pure-actual WFP market series."""
    return _generate_forecast(
        record, series_id, horizon, "wfp", registry_entry, cache_dir, root, as_of, pool_records, evaluation_only,
    )


def generate_nbs_forecast(
    record: dict[str, Any],
    series_id: str,
    horizon: int,
    cache_dir: Path | None = None,
    as_of: date | None = None,
    pool_records: list[dict[str, Any]] | None = None,
    evaluation_only: bool = False,
) -> dict[str, Any]:
    """Forecast or evaluate one exact NBS item/geography series, without WFP data."""
    return _generate_forecast(
        record, series_id, horizon, "nbs", None, cache_dir, ROOT, as_of, pool_records, evaluation_only,
    )
