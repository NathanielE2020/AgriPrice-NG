"""Request-time WFP/NBS forecasting with gap-aware chronological validation.

This module is intentionally separate from the approved historical research
pipeline. It handles one exact WFP ``priceflag=actual`` market series or one
exact NBS item/geography/methodology-regime series at a time. It never fills
missing prices and writes reusable artifacts only to a runtime cache (Render's
filesystem is ephemeral).
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
    "observation_lag_1", "observation_lag_2", "observation_lag_3", "rolling_mean_3", "rolling_std_3",
    "months_to_target", "gap_lag_1_to_lag_2", "gap_lag_2_to_lag_3",
    "month_sin", "month_cos", "trend_month",
)
MIN_SUPERVISED_ROWS = 30
MIN_TRAIN_ROWS = 20
MIN_VALIDATION_ROWS = 6
CACHE_FORMAT_VERSION = 5
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
    "seasonal_naive_12": "same-calendar-month observation 12 months earlier; available only when every validation/future lag is observed or generated recursively",
}
BASELINE_MODELS = {"naive_last_observation", "trend_baseline", "seasonal_naive_12"}


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
    """Reject stale or long-gap forecast requests before fitting or bridging."""
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
    bridge = max(0, current_ordinal - origin_ordinal)
    if age > policy["max_source_age_months"]:
        raise ForecastingError(
            "stale_source_data",
            f"The latest {source_key.upper()} observation is {origin_month}"
            f"{f' ({origin_date})' if origin_date else ''}, {age} calendar month(s) old. "
            f"The configured maximum source age is {policy['max_source_age_months']} month(s); "
            "this series is available for historical analysis but cannot be forecast as current market conditions.",
        )
    if bridge > policy["max_bridge_months"]:
        raise ForecastingError(
            "forecast_bridge_too_long",
            f"The latest actual observation is {origin_month}; reaching the current forecast window would require {bridge} unobserved bridge month(s), above the configured limit of {policy['max_bridge_months']}. No bridge values were generated.",
        )
    return age, bridge, policy


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
    if source_key == "wfp" and str(record.get("price_flag", "")) != "actual":
        return {
            "eligible": False,
            "code": "not_direct_actual",
            "reason": "This WFP series is aggregate or mixed-price-flag data. Forecasting requires a pure priceflag=actual market series.",
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
            f"only {len(feature_rows)} gap-aware supervised rows can be formed from {actual_count} valid actual observations; at least {MIN_SUPERVISED_ROWS} are required"
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
    """Assess one exact pure-actual WFP market series without fitting models."""
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
        if bundle.get("model_name") not in {"naive_last_observation", "trend_baseline", "seasonal_naive_12", "random_forest", "gradient_boosting"}:
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
    _age, max_bridge, _policy = _check_source_freshness(source_key, origin, origin_date, as_of)
    current_month = f"{as_of.year:04d}-{as_of.month:02d}"
    current_ordinal = _month_ordinal(current_month)
    bridge_months = max(0, current_ordinal - origin_ordinal)
    if bridge_months > max_bridge:
        raise ForecastingError(
            "forecast_bridge_too_long",
            f"Reaching the current forecast window requires {bridge_months} unobserved bridge month(s), above the configured limit of {max_bridge}. No bridge values were generated.",
        )
    working = dict(prices)
    result: list[dict[str, Any]] = []
    bridge_month_count = 0
    final_ordinal = current_ordinal + horizon
    for target_ordinal in range(origin_ordinal + 1, final_ordinal + 1):
        target_month = _month_from_ordinal(target_ordinal)
        if model_name == "naive_last_observation":
            previous_month = max(working, key=_month_ordinal)
            prediction = working[previous_month]
        else:
            observations = sorted(working.items(), key=lambda item: _month_ordinal(item[0]))
            feature = _feature_row(observations, target_month, trend_origin)
            if feature is None:
                raise ForecastingError(
                    "forecast_history_gap",
                    f"The forecast cannot construct gap-aware historical features for {target_month}; no missing actual values were filled.",
                )
            if model_name == "trend_baseline":
                gaps = feature["gap_lag_1_to_lag_2"] + feature["gap_lag_2_to_lag_3"]
                slope = math.log(feature["observation_lag_1"] / feature["observation_lag_3"]) / gaps if gaps > 0 else 0.0
                try:
                    prediction = feature["observation_lag_1"] * math.exp(slope * feature["months_to_target"])
                except OverflowError as exc:
                    raise ForecastingError("forecast_prediction_failed", "The gap-adjusted trend baseline overflowed; no forecast was returned.", 500) from exc
            elif model_name == "seasonal_naive_12":
                seasonal_month = _month_from_ordinal(target_ordinal - 12)
                if seasonal_month not in working:
                    raise ForecastingError(
                        "seasonal_history_unavailable",
                        f"A same-calendar-month observation or recursively generated estimate is not available for {seasonal_month}; the seasonal baseline was not substituted.",
                    )
                prediction = working[seasonal_month]
            else:
                try:
                    prediction = _predict(estimator, feature)
                except Exception as exc:
                    raise ForecastingError(
                        "forecast_prediction_failed",
                        "The selected model failed while generating a recursive prediction. No forecast was returned.",
                        500,
                    ) from exc
        if not math.isfinite(prediction) or prediction <= 0:
            raise ForecastingError("forecast_prediction_failed", "The selected strategy produced a non-positive or non-finite price; no forecast was returned.", 500)
        working[target_month] = prediction
        if target_ordinal > current_ordinal:
            step = len(result) + 1
            result.append({
                "date": target_month,
                "month": target_month,
                "price": prediction,
                "step": step,
                "quality_flag": "model_estimate",
                "temporal_status": "future_as_of_generation",
            })
        else:
            # These recursively predicted bridge values let the model reach the
            # current future calendar. They are never returned as actual history.
            bridge_month_count += 1
    return result, bridge_month_count


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
        max(0, _month_ordinal(f"{as_of.year:04d}-{as_of.month:02d}") - _month_ordinal(origin)),
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
    metadata = {
        "source": source_key.upper(),
        "commodity": record.get("commodity"),
        "research_commodity_mapping": record.get("research_commodity_mapping"),
        "mapping_status": record.get("mapping_status"),
        "market": record.get("market"), "state": record.get("state"),
        "geography_level": record.get("geography_level"), "geography_name": record.get("geography_name"),
        "methodology_regime": record.get("methodology_regime"),
        "unit": record.get("unit"), "unit_basis": record.get("unit_basis"),
        "price_type": record.get("price_type"), "currency": record.get("currency"),
        "model": selected_model, "model_type": "validated baseline",
        "strategy_level": "individual_series_short_history_baseline",
        "transformation": "original reported monthly price in source currency per exact source-defined series",
        "method": "simple baseline forecast selected from a rolling-origin comparison; no model extrapolation or missing-price filling",
        "validation_design": baseline["validation_design"],
        "forecast_origin_month": origin,
        "forecast_start_month": forecasts[0]["month"] if forecasts else None,
        "forecast_end_month": forecasts[-1]["month"] if forecasts else None,
        "latest_actual_month_in_source": origin if source_key == "wfp" else None,
        "latest_actual_date_in_source": origin_date if source_key == "wfp" else None,
        "latest_observation_month_in_source": origin,
        "latest_observation_date_in_source": origin_date or origin,
        "latest_source_report_date": latest_row.get("source_report_date"),
        "source_age_months": age,
        "freshness_limit_months": policy["max_source_age_months"],
        "max_forecast_bridge_months": policy["max_bridge_months"],
        "source_freshness_classification": f"fresh_within_{policy['max_source_age_months']}_months" if age <= policy["max_source_age_months"] else "stale_for_current_market",
        "is_current_market_forecast": not evaluation_only and age <= policy["max_source_age_months"],
        "forecast_date_scope": "validation_only" if evaluation_only else "future_only_after_current_calendar_month",
        "forecast_bridge_months": bridge_count,
        "forecast_generation_date": as_of.isoformat(),
        "forecast_classification": "validation_only" if evaluation_only else "future_as_of_generation",
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
        f"{bridge_count} unobserved bridge month(s) were estimated internally, below the configured limit of {policy['max_bridge_months']}. This is an estimate, not a live or guaranteed price."
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
        # Sparse or irregular exact series may use a definition-compatible
        # pooled model, but aggregates, mixed WFP flags, duplicate conflicts,
        # and records without usable local inputs are never substituted.
        if assessment["code"] == "insufficient_or_gapped_history":
            cache_override = os.environ.get("AGRIPRICE_MODEL_CACHE_DIR", "").strip()
            cache_root = Path(cache_dir) if cache_dir is not None else (
                Path(cache_override) if cache_override else Path(tempfile.gettempdir()) / "agriprice-ng" / "runtime-models"
            )
            from .pooled_forecasting import generate_pooled_forecast

            try:
                pooled = generate_pooled_forecast(
                    record, series_id, horizon, source_key,
                    pool_records or [record], cache_root, today,
                    evaluation_only=evaluation_only,
                )
                return pooled
            except ForecastingError as pooled_error:
                if pooled_error.code == "pooled_validation_unavailable":
                    baseline = assessment.get("baseline_validation") or {}
                    if baseline.get("eligible"):
                        return _short_history_baseline_result(
                            record, series_id, horizon, source_key, assessment, today, evaluation_only,
                        )
                    raise ForecastingError(
                        pooled_error.code,
                        f"Individual-series method: {assessment['reason']}. Pooled/global method unavailable: {pooled_error.message}",
                        pooled_error.status_code,
                    ) from pooled_error
                raise
            except Exception as exc:
                raise ForecastingError(
                    "forecast_generation_failed",
                    "The compatible pooled strategy could not complete safely; no forecast was returned.",
                    500,
                ) from exc
        raise ForecastingError(assessment["code"], assessment["reason"])

    prices = assessment["price_map"]
    origin = str(assessment["latest_actual_month"])
    origin_date = assessment["observation_dates"].get(origin)
    current_month = f"{today.year:04d}-{today.month:02d}"
    current_ordinal = _month_ordinal(current_month)
    origin_age, _bridge_candidate, policy = _source_age_months(source_key, origin, origin_date, today), max(
        0, current_ordinal - _month_ordinal(origin)
    ), forecast_policy()

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
    # constructed for every requested bridge and future month.
    if selected_model == "seasonal_naive_12" and not evaluation_only:
        working_months = set(prices)
        seasonal_supported = True
        for target_ordinal in range(_month_ordinal(origin) + 1, current_ordinal + horizon + 1):
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
        forecast_rows, bridge_month_count = _forecast_prices(
            selected_model,
            selected_estimator,
            prices,
            origin,
            min(_month_ordinal(month) for month in prices),
            horizon,
            today,
            source_key,
            origin_date,
        )
    validation_dates = [row["month"] for row in validation_rows]
    selected_metrics = validation_scores[selected_model]

    comparison = [
        {"model": name, **metrics}
        for name, metrics in sorted(validation_scores.items(), key=lambda item: item[0])
    ]
    forecast_start_month = _month_from_ordinal(current_ordinal + 1) if not evaluation_only else None
    forecast_end_month = _month_from_ordinal(current_ordinal + horizon) if not evaluation_only else None
    forecast_classification = "validation_only" if evaluation_only else "future_as_of_generation"
    temporal_phrase = "latest valid WFP actual observation" if source_key == "wfp" else "latest NBS reported observation"
    warning = (
        "Validation-only historical model comparison; no forecast was generated."
        if evaluation_only else
        f"Generated using the {temporal_phrase} through {origin}{f', {origin_date}' if origin_date else ''}. "
        f"The source is {origin_age} calendar month(s) behind the generation month {current_month}. "
        f"{bridge_month_count} unobserved intervening month(s) were generated internally to reach the future window; they are not returned as source observations. "
        "This is an estimate, not a live or guaranteed market price."
    )
    if not evaluation_only and origin_age > policy["max_source_age_months"]:
        warning += " The latest source observation is stale for current-market interpretation; treat this as a model estimate based on historical information."

    metadata = {
        "source": source_key.upper(),
        "commodity": record.get("commodity"),
        "market": record.get("market"),
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
        "model": selected_model,
        "model_type": "individual-series model" if selected_model not in BASELINE_MODELS else "validated baseline",
        "strategy_level": "individual_series",
        "transformation": "original reported monthly price in source currency per source-defined series",
        "method": "recursive calendar-month forecast; previous model predictions feed later steps; missing actual prices are never filled",
        "validation_design": "chronological 80/20 split of observed-month supervised rows; validation is rolling-origin one-step prediction using only history available before each target month",
        "forecast_origin_month": origin,
        "forecast_start_month": forecast_start_month,
        "forecast_end_month": forecast_end_month,
        "latest_actual_month_in_source": origin if source_key == "wfp" else None,
        "latest_actual_date_in_source": origin_date if source_key == "wfp" else None,
        "latest_observation_month_in_source": origin,
        "latest_observation_date_in_source": origin_date or origin,
        "latest_source_report_date": next((row.get("source_report_date") for row in reversed(record.get("history", [])) if row.get("month") == origin), None),
        "source_age_months": origin_age,
        "freshness_limit_months": policy["max_source_age_months"],
        "max_forecast_bridge_months": policy["max_bridge_months"],
        "source_freshness_classification": f"fresh_within_{policy['max_source_age_months']}_months" if origin_age <= policy["max_source_age_months"] else "stale_for_current_market",
        "is_current_market_forecast": not evaluation_only and origin_age <= policy["max_source_age_months"],
        "forecast_date_scope": "validation_only" if evaluation_only else "future_only_after_current_calendar_month",
        "forecast_bridge_months": bridge_month_count,
        "forecast_generation_date": today.isoformat(),
        "forecast_classification": forecast_classification,
        "confidence_intervals": False,
        "selection_basis": "lowest chronological validation RMSE among Random Forest, Gradient Boosting, last-observation and gap-adjusted trend baselines; seasonal-naive is included only on complete shared validation months; no random split or test-set selection",
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
        "actual_observation_count": assessment["actual_observations"] if source_key == "wfp" else None,
        "source_observation_count": assessment["actual_observations"],
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
