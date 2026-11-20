"""Source- and definition-constrained pooled forecasts for sparse series.

Pooling is intentionally narrow: WFP series share only the same commodity,
unit, price type and currency; NBS series share the exact source item,
methodology regime, geography level and reported unit. Validation holds out
markets/geographies and uses only pre-cutoff target months for fitting.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import copy
import calendar
from datetime import date
from pathlib import Path
from typing import Any


POOL_FORMAT_VERSION = 4
POOL_MODELS = ("pooled_random_forest", "pooled_gradient_boosting")
POOL_MODEL_CONFIGURATION = {
    "pooled_random_forest": "150 trees; min_samples_leaf=3; max_features=0.8; seed=42; n_jobs=1; log monthly price movement target",
    "pooled_gradient_boosting": "100 estimators; depth=2; min_samples_leaf=2; learning_rate=0.03; Huber loss; seed=42; log monthly price movement target",
}
MIN_POOL_GEOGRAPHIES = 5
MIN_POOL_TRAIN_ROWS = 30
MIN_POOL_VALIDATION_ROWS = 12


def compatibility_key(record: dict[str, Any], source_key: str) -> tuple[str, ...]:
    """Return the exact definition boundary across which borrowing is allowed."""
    if source_key == "wfp":
        return (
            "wfp", str(record.get("commodity") or ""),
            str(record.get("unit") or ""), str(record.get("price_type") or ""),
            str(record.get("currency") or ""), str(record.get("price_flag") or ""),
        )
    return (
        "nbs", str(record.get("commodity") or ""),
        str(record.get("methodology_regime") or ""),
        str(record.get("geography_level") or ""), str(record.get("unit") or ""),
        str(record.get("price_type") or ""), str(record.get("currency") or ""),
    )


def _geography_group(record: dict[str, Any], source_key: str) -> str:
    if source_key == "wfp":
        return str(record.get("market_id") or record.get("market") or "")
    return str(record.get("geography_name") or "")


def _categorical_features(record: dict[str, Any], features: dict[str, float], target_month: str) -> dict[str, Any]:
    latest = float(features["observation_lag_1"])
    if latest <= 0:
        raise ValueError("pooled feature history has a non-positive price")
    year, month = (int(part) for part in target_month.split("-"))
    values: dict[str, Any] = {
        "relative_lag_2": float(features["observation_lag_2"]) / latest,
        "relative_lag_3": float(features["observation_lag_3"]) / latest,
        "relative_rolling_mean_3": float(features["rolling_mean_3"]) / latest,
        "relative_rolling_std_3": float(features["rolling_std_3"]) / latest,
        "months_to_target": float(features["months_to_target"]),
        "gap_lag_1_to_lag_2": float(features["gap_lag_1_to_lag_2"]),
        "gap_lag_2_to_lag_3": float(features["gap_lag_2_to_lag_3"]),
        "month_sin": float(features["month_sin"]),
        "month_cos": float(features["month_cos"]),
        "target_year_since_2000": float(year - 2000),
        "commodity": str(record.get("commodity") or "not stated"),
        "unit": str(record.get("unit") or "not stated"),
        "price_type": str(record.get("price_type") or "not stated"),
        "currency": str(record.get("currency") or "not stated"),
    }
    if record.get("source_key") == "wfp":
        values["state"] = str(record.get("state") or "not stated")
        values["place"] = str(record.get("market_id") or record.get("market") or "not stated")
    else:
        values["methodology_regime"] = str(record.get("methodology_regime") or "not stated")
        values["geography_level"] = str(record.get("geography_level") or "not stated")
        values["place"] = str(record.get("geography_name") or "not stated")
    return values


def _model_factories() -> dict[str, Any]:
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for pooled forecasting") from exc
    return {
        "pooled_random_forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, max_features=0.8,
            random_state=42, n_jobs=1,
        ),
        "pooled_gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.03, max_depth=2,
            min_samples_leaf=2, loss="huber", random_state=42,
        ),
    }


def _price_metrics(actual: list[float], predicted: list[float]) -> dict[str, float | int | None]:
    from .ondemand_forecasting import _metrics

    return _metrics(actual, predicted)


def _row_baseline_prediction(row: dict[str, Any], method: str) -> float:
    features = row["raw_features"]
    last = float(features["observation_lag_1"])
    if method == "naive_last_observation":
        return last
    gap = float(features["gap_lag_1_to_lag_2"] + features["gap_lag_2_to_lag_3"])
    if gap <= 0:
        return last
    older = float(features["observation_lag_3"])
    slope = math.log(last / older) / gap
    return last * math.exp(slope * float(features["months_to_target"]))


def _series_rows(records: list[dict[str, Any]], source_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from .ondemand_forecasting import ForecastingError, _build_feature_rows, _collect_actual_history

    prepared: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in records:
        if source_key == "wfp" and str(record.get("price_flag") or "") != "actual":
            continue
        try:
            collected = _collect_actual_history(record)
        except ForecastingError as exc:
            skipped.append({"series_id": record.get("series_id"), "reason": exc.message})
            continue
        price_map = collected["prices"]
        rows = _build_feature_rows(price_map)
        if not rows:
            continue
        row_objects = []
        for row in rows:
            features = _categorical_features(record, row["features"], row["month"])
            latest = float(row["features"]["observation_lag_1"])
            months = float(row["features"]["months_to_target"])
            target = math.log(float(row["target"]) / latest) / months
            if not math.isfinite(target):
                continue
            row_objects.append({
                "series_id": str(record.get("series_id") or ""),
                "group": _geography_group(record, source_key),
                "month": row["month"],
                "target": target,
                "actual_price": float(row["target"]),
                "features": features,
                "raw_features": row["features"],
            })
        if row_objects:
            prepared.append({
                "record": record,
                "series_id": str(record.get("series_id") or ""),
                "group": _geography_group(record, source_key),
                "price_map": price_map,
                "rows": row_objects,
                "actual_observations": len(price_map),
            })
    return prepared, skipped


def _fingerprint(group_key: tuple[str, ...], prepared: list[dict[str, Any]]) -> str:
    payload = {
        "version": POOL_FORMAT_VERSION,
        "group": group_key,
        "configuration": POOL_MODEL_CONFIGURATION,
        "minimums": [MIN_POOL_GEOGRAPHIES, MIN_POOL_TRAIN_ROWS, MIN_POOL_VALIDATION_ROWS],
        "series": [
            {
                "series_id": item["series_id"],
                "prices": [[month, format(price, ".17g")] for month, price in sorted(item["price_map"].items())],
            }
            for item in sorted(prepared, key=lambda part: part["series_id"])
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _pool_cache_path(cache_dir: Path, fingerprint: str) -> Path:
    return cache_dir / "pooled" / f"{fingerprint}.joblib"


def _load_bundle(cache_dir: Path, fingerprint: str) -> dict[str, Any] | None:
    path = _pool_cache_path(cache_dir, fingerprint)
    if not path.is_file():
        return None
    try:
        import joblib

        payload = joblib.load(path)
        if payload.get("fingerprint") == fingerprint and payload.get("format_version") == POOL_FORMAT_VERSION:
            return payload
    except Exception:
        return None
    return None


def _save_bundle(cache_dir: Path, fingerprint: str, payload: dict[str, Any]) -> bool:
    path = _pool_cache_path(cache_dir, fingerprint)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        with tempfile.NamedTemporaryFile(prefix="agriprice-pool-", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        joblib.dump(payload, temporary, compress=1)
        os.replace(temporary, path)
        return True
    except Exception:
        if temporary:
            temporary.unlink(missing_ok=True)
        return False


def _fit_pool(
    records: list[dict[str, Any]], source_key: str, group_key: tuple[str, ...],
    cache_dir: Path,
) -> dict[str, Any]:
    from sklearn.feature_extraction import DictVectorizer
    from .ondemand_forecasting import _month_from_ordinal, _month_ordinal

    prepared, skipped = _series_rows(records, source_key)
    distinct_groups = sorted({item["group"] for item in prepared if item["group"]})
    if len(distinct_groups) < MIN_POOL_GEOGRAPHIES:
        return {
            "eligible": False, "code": "pooled_validation_unavailable",
            "reason": (
                f"The definition-compatible pool contains {len(distinct_groups)} usable markets/geographies; "
                f"group-held-out validation requires at least {MIN_POOL_GEOGRAPHIES}."
            ),
            "series_count": len(prepared), "group_count": len(distinct_groups),
        }
    all_rows = [row for item in prepared for row in item["rows"]]
    target_months = sorted({row["month"] for row in all_rows})
    if len(target_months) < 6:
        return {
            "eligible": False, "code": "pooled_validation_unavailable",
            "reason": f"The compatible pool has only {len(target_months)} supervised target months; at least 6 are needed for chronological validation.",
            "series_count": len(prepared), "group_count": len(distinct_groups),
        }
    cutoff_index = max(1, min(len(target_months) - 2, int(math.floor(0.8 * len(target_months)))))
    cutoff = target_months[cutoff_index]
    fingerprint = _fingerprint(group_key, prepared)
    cached = _load_bundle(cache_dir, fingerprint)
    if cached is not None:
        return cached["assessment"]

    folds = [distinct_groups[index::min(5, len(distinct_groups))] for index in range(min(5, len(distinct_groups)))]
    folds = [set(fold) for fold in folds if fold]
    baseline_actual: list[float] = []
    baseline_predictions: dict[str, list[float]] = {"naive_last_observation": [], "trend_baseline": []}
    model_predictions: dict[str, list[float]] = {name: [] for name in POOL_MODELS}
    model_failed: set[str] = set()
    series_validation: dict[str, dict[str, dict[str, list[float]]]] = {}
    validation_months: set[str] = set()
    validation_groups: set[str] = set()

    for held_out_groups in folds:
        train_rows = [
            row for row in all_rows
            if row["group"] not in held_out_groups and row["month"] < cutoff
        ]
        validation_rows = [
            row for row in all_rows
            if row["group"] in held_out_groups and row["month"] >= cutoff
        ]
        if len(train_rows) < MIN_POOL_TRAIN_ROWS or not validation_rows:
            continue
        train_dicts = [row["features"] for row in train_rows]
        validation_dicts = [row["features"] for row in validation_rows]
        vectorizer = DictVectorizer(sparse=False)
        x_train = vectorizer.fit_transform(train_dicts)
        x_valid = vectorizer.transform(validation_dicts)
        fitted_fold_models: dict[str, Any] = {}
        for name, model in _model_factories().items():
            try:
                model.fit(x_train, [row["target"] for row in train_rows])
                rates = model.predict(x_valid)
                prices = [
                    float(row["raw_features"]["observation_lag_1"])
                    * math.exp(float(rate) * float(row["raw_features"]["months_to_target"]))
                    for row, rate in zip(validation_rows, rates)
                ]
                if len(prices) != len(validation_rows) or not all(math.isfinite(value) and value > 0 for value in prices):
                    raise ValueError("invalid pooled validation price")
                fitted_fold_models[name] = prices
            except Exception:
                model_failed.add(name)
        for index, row in enumerate(validation_rows):
            truth = row["actual_price"]
            baseline_actual.append(truth)
            baseline_predictions["naive_last_observation"].append(_row_baseline_prediction(row, "naive_last_observation"))
            baseline_predictions["trend_baseline"].append(_row_baseline_prediction(row, "trend_baseline"))
            validation_months.add(row["month"])
            validation_groups.add(row["group"])
            detail = series_validation.setdefault(row["series_id"], {
                "months": [], "actual": [], "naive_last_observation": [], "trend_baseline": [],
                **{name: [] for name in POOL_MODELS},
            })
            detail["months"].append(row["month"])
            detail["actual"].append(truth)
            detail["naive_last_observation"].append(baseline_predictions["naive_last_observation"][-1])
            detail["trend_baseline"].append(baseline_predictions["trend_baseline"][-1])
            for name in POOL_MODELS:
                if name in fitted_fold_models:
                    prediction = fitted_fold_models[name][index]
                    model_predictions[name].append(prediction)
                    detail[name].append(prediction)
                else:
                    model_predictions[name].append(float("nan"))

    if len(baseline_actual) < MIN_POOL_VALIDATION_ROWS or len(validation_groups) < 4:
        return {
            "eligible": False, "code": "pooled_validation_unavailable",
            "reason": (
                f"Group-held-out chronological validation produced {len(baseline_actual)} observations across "
                f"{len(validation_groups)} markets/geographies; at least {MIN_POOL_VALIDATION_ROWS} observations "
                "across 4 held-out groups are required."
            ),
            "series_count": len(prepared), "group_count": len(distinct_groups),
            "validation_rows": len(baseline_actual), "skipped_peer_series": skipped,
        }

    validation_scores: dict[str, dict[str, Any]] = {
        name: _price_metrics(baseline_actual, values)
        for name, values in baseline_predictions.items()
    }
    for name, values in model_predictions.items():
        if name in model_failed or len(values) != len(baseline_actual) or not all(math.isfinite(value) for value in values):
            continue
        validation_scores[name] = _price_metrics(baseline_actual, values)
    selected_model = min(validation_scores, key=lambda name: float(validation_scores[name]["rmse"]))
    winning_local_metrics: dict[str, Any] | None = None
    selected_record = None
    # The exact selected series' metrics are supplied at forecast time from
    # its group-held-out fold; global metrics remain separately labelled.
    assessment = {
        "eligible": True,
        "code": "pooled_eligible",
        "reason": "A same-definition, same-source pooled model passed group-held-out chronological validation.",
        "selected_model": selected_model,
        "validation_scores": validation_scores,
        "series_count": len(prepared),
        "group_count": len(distinct_groups),
        "training_rows": len(all_rows),
        "validation_rows": len(baseline_actual),
        "validation_group_count": len(validation_groups),
        "validation_cutoff_month": cutoff,
        "validation_training_target_last_month": _month_from_ordinal(_month_ordinal(cutoff) - 1),
        "validation_first_month": min(validation_months),
        "validation_last_month": max(validation_months),
        "validation_months": sorted(validation_months),
        "series_validation": {
            series_id: {
                name: _price_metrics(detail["actual"], detail[name])
                for name in ("naive_last_observation", "trend_baseline", *POOL_MODELS)
                if detail.get(name) and len(detail[name]) == len(detail["actual"])
            }
            for series_id, detail in series_validation.items()
        },
        "series_validation_predictions": {
            series_id: [
                {
                    "month": month,
                    "actual": detail["actual"][index],
                    "models": {
                        name: detail[name][index]
                        for name in ("naive_last_observation", "trend_baseline", *POOL_MODELS)
                        if len(detail.get(name, [])) == len(detail["actual"])
                    },
                }
                for index, month in enumerate(detail["months"])
            ]
            for series_id, detail in series_validation.items()
        },
        "skipped_peer_series": skipped,
        "failed_models": sorted(model_failed),
    }

    # Fit the chosen pooled model on all compatible observed targets. The
    # validation process above never trained on held-out group/month labels.
    bundle: dict[str, Any] = {
        "format_version": POOL_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "assessment": assessment,
        "selected_model": selected_model,
        "vectorizer": None,
        "estimator": None,
        "feature_names": [],
        "training_rows": len(all_rows),
        "training_series_ids": sorted({row["series_id"] for row in all_rows}),
        "training_first_month": min(row["month"] for row in all_rows),
        "training_last_month": max(row["month"] for row in all_rows),
    }
    if selected_model in POOL_MODELS:
        vectorizer = DictVectorizer(sparse=False)
        matrix = vectorizer.fit_transform([row["features"] for row in all_rows])
        model = _model_factories()[selected_model]
        model.fit(matrix, [row["target"] for row in all_rows])
        bundle.update({"vectorizer": vectorizer, "estimator": model, "feature_names": list(vectorizer.feature_names_)})
    _save_bundle(cache_dir, fingerprint, bundle)
    return assessment


def _load_or_fit_pool(
    records: list[dict[str, Any]], source_key: str, selected_record: dict[str, Any], cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    key = compatibility_key(selected_record, source_key)
    compatible = [record for record in records if compatibility_key(record, source_key) == key]
    if source_key == "wfp":
        compatible = [record for record in compatible if str(record.get("price_flag") or "") == "actual"]
    prepared, _ = _series_rows(compatible, source_key)
    if len({item["series_id"] for item in prepared}) == 0:
        return {"eligible": False, "code": "pooled_validation_unavailable", "reason": "No valid series share this exact source and definition."}, None
    fingerprint = _fingerprint(key, prepared)
    cached = _load_bundle(cache_dir, fingerprint)
    if cached is None:
        assessment = _fit_pool(compatible, source_key, key, cache_dir)
        cached = _load_bundle(cache_dir, fingerprint)
        if cached is None:
            # Ineligible pool results are also not persisted. This branch
            # reruns the assessment once to return its detailed reason.
            return assessment, None
    return cached["assessment"], cached


def _records_as_of(records: list[dict[str, Any]], as_of: date) -> list[dict[str, Any]]:
    """Return shallow record copies with only observations known by as-of."""
    from .ondemand_forecasting import _month_ordinal

    current_month = f"{as_of.year:04d}-{as_of.month:02d}"
    bounded_records = []
    for record in records:
        bounded = copy.copy(record)
        bounded["history"] = []
        for row in record.get("history", []):
            month = str(row.get("month") or "")
            try:
                if _month_ordinal(month) > _month_ordinal(current_month):
                    continue
            except ValueError:
                continue
            observation_date = str(row.get("last_observation_date") or row.get("first_observation_date") or "")
            if observation_date:
                try:
                    if date.fromisoformat(observation_date[:10]) > as_of:
                        continue
                except ValueError:
                    pass
            bounded["history"].append(row)
        bounded_records.append(bounded)
    return bounded_records


def generate_pooled_forecast(
    record: dict[str, Any],
    series_id: str,
    horizon: int,
    source_key: str,
    pool_records: list[dict[str, Any]],
    cache_dir: Path,
    as_of: date,
    evaluation_only: bool = False,
) -> dict[str, Any]:
    """Validate a compatible pool and optionally omit future prediction steps."""
    from .ondemand_forecasting import (
        ForecastingError, _collect_actual_history, _feature_row,
        _classify_forecast_months, _month_from_ordinal, _month_ordinal, _source_age_months,
        _check_source_freshness, _future_forecast_window, forecast_policy,
    )

    if source_key == "wfp" and str(record.get("price_flag") or "") != "actual":
        raise ForecastingError(
            "not_direct_actual",
            "WFP aggregate and mixed-price-flag records remain available for historical analysis, but forecasting requires an exact pure priceflag=actual market series.",
        )
    if horizon not in range(1, 7):
        raise ForecastingError("invalid_horizon", "horizon must be an integer from 1 through 6 months.", 400)
    try:
        collected = _collect_actual_history(record)
    except ForecastingError:
        raise
    prices = collected["prices"]
    if not prices:
        raise ForecastingError(
            "insufficient_local_history",
            "This exact series has no valid reported observations. No other market's price was substituted.",
        )
    months = sorted(prices, key=_month_ordinal)
    origin = months[-1]
    current_month = f"{as_of.year:04d}-{as_of.month:02d}"
    origin_date = collected["dates"].get(origin)
    if not evaluation_only:
        _check_source_freshness(source_key, origin, origin_date, as_of)
        if len(prices) < 3:
            raise ForecastingError(
                "insufficient_local_history",
                f"This exact series has {len(prices)} valid reported observations; at least 3 are required to form a gap-aware prediction input. No other market's price was substituted.",
            )

    # Peer records must not carry information from after the selected series'
    # forecast origin. Use the precise WFP observation day when present; NBS
    # monthly summaries have a month-only origin and are bounded to month end.
    if origin_date:
        try:
            pool_cutoff = date.fromisoformat(str(origin_date)[:10])
        except ValueError:
            pool_cutoff = None
    else:
        pool_cutoff = None
    if pool_cutoff is None:
        origin_year, origin_month_number = (int(part) for part in origin.split("-"))
        pool_cutoff = date(origin_year, origin_month_number, calendar.monthrange(origin_year, origin_month_number)[1])
    bounded_peers = _records_as_of(pool_records, pool_cutoff)

    assessment, bundle = _load_or_fit_pool(bounded_peers, source_key, record, cache_dir)
    if not assessment.get("eligible") or bundle is None:
        reason = assessment.get("reason") or "A compatible pooled model could not be validated for this series."
        raise ForecastingError(assessment.get("code", "pooled_validation_unavailable"), reason)

    chosen = str(bundle["selected_model"])
    estimator = bundle.get("estimator")
    vectorizer = bundle.get("vectorizer")
    origin_ordinal = _month_ordinal(origin)
    current_ordinal = _month_ordinal(current_month)
    trend_origin = _month_ordinal(months[0])
    working = dict(prices)
    forecasts: list[dict[str, Any]] = []
    bridge_count = 0
    starting_price = float(prices[origin])
    _first_future, final_ordinal, bridge_count = _future_forecast_window(origin, horizon, as_of)
    for target_ordinal in range(origin_ordinal + 1, final_ordinal + 1) if not evaluation_only else []:
        target_month = _month_from_ordinal(target_ordinal)
        observations = sorted(working.items(), key=lambda item: _month_ordinal(item[0]))
        feature = _feature_row(observations, target_month, trend_origin)
        if feature is None:
            raise ForecastingError("insufficient_local_history", "The selected series does not contain three prior observations for a gap-aware pooled input; no values were filled.")
        if chosen == "naive_last_observation":
            prediction = float(observations[-1][1])
        elif chosen == "trend_baseline":
            gap = feature["gap_lag_1_to_lag_2"] + feature["gap_lag_2_to_lag_3"]
            slope = math.log(feature["observation_lag_1"] / feature["observation_lag_3"]) / gap if gap > 0 else 0.0
            try:
                prediction = feature["observation_lag_1"] * math.exp(slope * feature["months_to_target"])
            except OverflowError as exc:
                raise ForecastingError("forecast_prediction_failed", "The pooled trend baseline overflowed; no forecast was returned.", 500) from exc
        else:
            from .ondemand_forecasting import _month_from_ordinal as month_from_ordinal

            model_features = _categorical_features(record, feature, target_month)
            matrix = vectorizer.transform([model_features])
            monthly_rate = float(estimator.predict(matrix)[0])
            if not math.isfinite(monthly_rate) or abs(monthly_rate) > 5:
                raise ForecastingError("forecast_prediction_failed", "The validated pooled model produced an implausible numeric movement and no forecast was returned.", 500)
            prediction = feature["observation_lag_1"] * math.exp(monthly_rate * feature["months_to_target"])
        if not math.isfinite(prediction) or prediction <= 0:
            raise ForecastingError("forecast_prediction_failed", "The selected pooled strategy produced a non-positive or non-finite prediction; no forecast was returned.", 500)
        working[target_month] = float(prediction)
        percentage_change = (float(prediction) / starting_price - 1.0) * 100.0
        if not math.isfinite(percentage_change):
            percentage_change = None
        if target_ordinal > current_ordinal:
            forecasts.append({
                "date": target_month, "month": target_month, "price": float(prediction),
                "step": len(forecasts) + 1, "quality_flag": "model_estimate",
                "percentage_change_from_origin": percentage_change,
                "method": f"recursive pooled {chosen.replace('_', ' ')} prediction",
                "temporal_status": "future_as_of_generation",
            })

    origin_age = _source_age_months(source_key, origin, origin_date, as_of)
    policy = forecast_policy()
    selected_series_validation = assessment.get("series_validation", {}).get(series_id, {})
    selected_series_validation_predictions = assessment.get("series_validation_predictions", {}).get(series_id, [])
    validation_metrics = assessment["validation_scores"][chosen]
    training_series_ids = set(bundle.get("training_series_ids", []))
    peers = []
    for item in bounded_peers:
        if item.get("series_id") == series_id or item.get("series_id") not in training_series_ids or compatibility_key(item, source_key) != compatibility_key(record, source_key):
            continue
        if source_key == "wfp" and str(item.get("price_flag") or "") != "actual":
            continue
        peers.append({
            "series_id": item.get("series_id"), "commodity": item.get("commodity"),
            "place": item.get("market") if source_key == "wfp" else item.get("geography_name"),
            "state_or_level": item.get("state") if source_key == "wfp" else item.get("geography_level"),
            "observations": int(item.get("observed_months", 0) or 0),
        })
    history = [] if evaluation_only else [row for row in record.get("history", []) if row.get("month") and _month_ordinal(row["month"]) >= origin_ordinal - 23]
    classification = _classify_forecast_months([str(row["month"]) for row in forecasts], as_of, evaluation_only)
    is_wfp_actual = source_key == "wfp" and str(record.get("price_flag") or "") == "actual"
    metadata = {
        "source": source_key.upper(), "commodity": record.get("commodity"),
        "source_key": source_key,
        "research_commodity_mapping": record.get("research_commodity_mapping"),
        "mapping_status": record.get("mapping_status"),
        "market": record.get("market"), "state": record.get("state"),
        "geography_level": record.get("geography_level"), "geography_name": record.get("geography_name"),
        "methodology_regime": record.get("methodology_regime"), "unit": record.get("unit"),
        "unit_basis": record.get("unit_basis"),
        "price_type": record.get("price_type"), "currency": record.get("currency"),
        "price_flag": record.get("price_flag") if source_key == "wfp" else None,
        "model": chosen,
        "model_type": "validated pooled/global model" if chosen in POOL_MODELS else "validated baseline",
        "strategy_level": "pooled_global",
        "transformation": "dimensionless log monthly price change; reconstructed on exact selected-series price scale",
        "method": "recursive calendar-month forecast; prior model estimates feed later steps; missing actual prices are never filled",
        "validation_design": "5-fold group-held-out validation across compatible markets/geographies, with training target months strictly before the chronological 80% cutoff; validation inputs use only prior observations",
        "validation_scope": "same-source, same-commodity/item, compatible-definition pool; not an independent evaluation of only the selected series",
        "forecast_origin_month": origin, "forecast_start_month": forecasts[0]["month"] if forecasts else None,
        "forecast_end_month": forecasts[-1]["month"] if forecasts else None,
        "latest_actual_month_in_source": origin if is_wfp_actual else None,
        "latest_actual_date_in_source": origin_date if is_wfp_actual else None,
        "latest_observation_month_in_source": origin, "latest_observation_date_in_source": origin_date or origin,
        "latest_source_report_date": next((row.get("source_report_date") for row in reversed(record.get("history", [])) if row.get("month") == origin), None),
        "source_age_months": origin_age,
        "freshness_limit_months": policy["max_source_age_months"],
        "max_forecast_bridge_months": policy["max_bridge_months"],
        "source_freshness_classification": f"fresh_within_{policy['max_source_age_months']}_months" if origin_age <= policy["max_source_age_months"] else "stale_for_current_market",
        "is_current_market_forecast": not evaluation_only and origin_age <= policy["max_source_age_months"] and classification == "future_as_of_generation",
        "forecast_date_scope": "validation_only" if evaluation_only else "future_calendar_months_after_generation_month",
        "forecast_bridge_months": bridge_count if not evaluation_only else 0,
        "forecast_generation_date": as_of.isoformat(), "forecast_classification": classification,
        "historical_price_label": "WFP direct actual observations" if is_wfp_actual else (f"WFP reported priceflag={record.get('price_flag') or 'not stated'} observations; not direct actuals" if source_key == "wfp" else "NBS reported monthly observations"),
        "confidence_level": "low" if origin_age > policy["max_source_age_months"] or len(prices) < 12 or max(0, _month_ordinal(origin) - _month_ordinal(months[0]) + 1 - len(prices)) > len(prices) else "limited",
        "confidence_note": "Qualitative confidence only; no calibrated prediction interval is available.",
        "confidence_intervals": False,
        "selection_basis": "lowest RMSE across validation of naive last observation, historical trend, pooled random forest and pooled gradient boosting; all predictions are group-held-out and chronological",
        "validation_metrics": validation_metrics,
        "model_comparison_validation": [{"model": name, **metrics} for name, metrics in sorted(assessment["validation_scores"].items())],
        "selected_series_validation_metrics": selected_series_validation.get(chosen),
        "selected_series_validation_predictions": selected_series_validation_predictions,
        "validation_predictions": selected_series_validation_predictions,
        "evaluation_note": (
            "Metrics are group-held-out pool-level validation; selected-series predictions are shown only if the exact series contributed held-out rows."
        ),
        "validation_rows": assessment["validation_rows"],
        "validation_months": assessment.get("validation_months", []),
        "validation_first_month": assessment["validation_first_month"],
        "validation_last_month": assessment["validation_last_month"],
        "validation_cutoff_month": assessment["validation_cutoff_month"],
        "validation_training_target_last_month": assessment["validation_training_target_last_month"],
        "validation_group_count": assessment["validation_group_count"],
        "training_series_count": assessment["series_count"],
        "training_contributing_series_count": len(training_series_ids),
        "training_group_count": assessment["group_count"],
        "training_observation_rows": bundle["training_rows"],
        "training_first_month": bundle["training_first_month"],
        "training_last_month": bundle["training_last_month"],
        "pool_definition": "same exact commodity, unit, price type, currency and WFP price flag" if source_key == "wfp" else "same exact NBS item label, methodology regime, geography level, unit, price type and currency",
        "other_series_contributors": peers,
        "source_observation_count": len(prices), "missing_calendar_months": max(0, _month_ordinal(origin) - _month_ordinal(months[0]) + 1 - len(prices)),
        "model_configurations": POOL_MODEL_CONFIGURATION,
        "runtime_cache_status": "hit_or_saved" if bundle else "not_persisted",
    }
    warning = (
        "Validation-only compatible-pool comparison; no future forecast was generated."
        if evaluation_only else
        f"A pooled strategy trained on {len(training_series_ids)} compatible series, including {len(peers)} other market/geography series where they had usable training targets. "
        f"The latest observation is {origin}{f' ({origin_date})' if origin_date else ''}; it is {origin_age} month(s) behind {current_month}. "
        f"{bridge_count} unobserved intervening month(s) were estimated internally. This is an estimate, not a live or guaranteed price."
    )
    if not evaluation_only and origin_age > policy["max_source_age_months"]:
        warning += " The source is stale for current-market interpretation."
    return {
        "series": record, "model": chosen, "horizon": horizon,
        "historical": history, "forecast": forecasts, "metadata": metadata,
        "warning": warning,
    }


def assess_pooled_group(
    record: dict[str, Any], pool_records: list[dict[str, Any]], source_key: str,
    cache_dir: Path, as_of: date | None = None,
) -> dict[str, Any]:
    """Return pooled eligibility and validation metrics without a forecast."""
    bounded = _records_as_of(pool_records, as_of or date.today())
    assessment, _bundle = _load_or_fit_pool(bounded, source_key, record, cache_dir)
    return assessment

