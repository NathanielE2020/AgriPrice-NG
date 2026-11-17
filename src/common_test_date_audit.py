"""Audit fair common-test-date comparisons for the six aggregate series.

This is a read-only audit of the existing final-validation tabular artifacts and
the isolated TensorFlow LSTM artifacts.  It does not retrain models or modify
the raw data or previous experiment reports.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from .aggregate_preprocessing import AGGREGATE_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
LSTM_REPORT_DIR = ROOT / "reports" / "lstm_experiment"
LSTM_OUTPUT_DIR = ROOT / "outputs" / "lstm_experiment"
TAB_REPORT_DIR = ROOT / "reports" / "final_validation"
TAB_MODEL_DIR = ROOT / "models" / "final_validation" / "aggregate"
AUDIT_DIR = ROOT / "reports" / "common_test_date_audit"

MODEL_NAMES = (
    "naive_last_observation",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "stacking",
    "lstm",
)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:120]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _parse_months(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(";") if item]


def _finite_float(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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


def _reconstruct(predicted: np.ndarray, lag: np.ndarray, transformation: str) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=float)
    lag = np.asarray(lag, dtype=float)
    if transformation in ("", "original", "none"):
        return predicted
    if transformation == "log1p":
        return np.maximum(np.expm1(predicted), 0.0)
    if transformation == "price_change":
        return lag + predicted
    if transformation == "percentage_change":
        return lag * (1.0 + predicted)
    raise ValueError(f"Unsupported transformation: {transformation}")


def _predict_bundle(bundle: object, x: np.ndarray) -> np.ndarray:
    """Predict from a saved direct model or stacking dictionary."""

    if isinstance(bundle, dict) and "base_models" in bundle:
        values = np.column_stack(
            [bundle["base_models"][name].predict(x) for name in bundle["base_names"]]
        )
        return np.asarray(bundle["meta_model"].predict(values), dtype=float)
    return np.asarray(bundle.predict(x), dtype=float)


def _load_feature_rows() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _read_csv(PROCESSED_DIR / "aggregate_training_features.csv"):
        for column in [*AGGREGATE_FEATURE_COLUMNS, "target_price", "mean_price", "price_lag_1"]:
            row[column] = float(row[column])
        grouped[row["series_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["month"])
    return dict(grouped)


def _load_monthly_actuals() -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for row in _read_csv(PROCESSED_DIR / "aggregate_monthly_reindexed.csv"):
        if row.get("mean_price") not in (None, "", "None"):
            result[row["series_id"]][row["month"]] = float(row["mean_price"])
    return dict(result)


def _load_lstm_partitions() -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for row in _read_csv(LSTM_REPORT_DIR / "partition_dates.csv"):
        result[row["series_id"]][row["partition"]] = _parse_months(row["target_months"])
    return dict(result)


def _load_lstm_predictions() -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in _read_csv(LSTM_OUTPUT_DIR / "test_predictions.csv"):
        result[row["series_id"]][row["month"]] = {
            "actual_price": float(row["actual_price"]),
            "predicted_price": float(row["predicted_price"]),
            "naive_price": float(row["naive_price"]),
        }
    return dict(result)


def _load_tabular_rows() -> list[dict]:
    return _read_csv(TAB_REPORT_DIR / "final_model_comparison.csv")


def _load_tabular_sample_counts() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    path = TAB_REPORT_DIR / "aggregate_training_sample_counts.csv"
    for row in _read_csv(path):
        result[row["series_id"]] = {
            "train_rows": int(float(row["train_rows"])),
            "validation_rows": int(float(row["validation_rows"])),
            "test_rows": int(float(row["test_rows"])),
            "feature_rows": int(float(row["feature_rows"])),
        }
    return result


def _validate_saved_tabular_metrics(
    predictions: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, object]:
    """Recalculate original tabular test metrics as an artifact-alignment check."""

    differences: list[float] = []
    checked = 0
    failed: list[str] = []
    for row in _load_tabular_rows():
        model_name = row.get("model", "")
        if model_name not in {"random_forest", "gradient_boosting", "xgboost", "stacking"}:
            continue
        series_id = row["series_id"]
        model_predictions = predictions.get(series_id, {}).get(model_name, {})
        if not model_predictions:
            failed.append(f"{series_id}/{model_name}: no reproduced predictions")
            continue
        actual = np.asarray([item["actual_price"] for item in model_predictions.values()], dtype=float)
        predicted = np.asarray([item["predicted_price"] for item in model_predictions.values()], dtype=float)
        metrics = _metrics(actual, predicted)
        reported = row.get("test_rmse", "")
        if reported in (None, ""):
            failed.append(f"{series_id}/{model_name}: missing reported test RMSE")
            continue
        difference = abs(float(reported) - float(metrics["rmse"]))
        differences.append(difference)
        checked += 1
        if difference > 1e-6:
            failed.append(f"{series_id}/{model_name}: RMSE difference {difference}")
    return {
        "rows_checked": checked,
        "max_rmse_absolute_difference": max(differences) if differences else None,
        "all_match": not failed and checked == 24,
        "failures": failed,
    }


def _load_tabular_predictions(
    feature_groups: dict[str, list[dict]],
) -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], dict[str, dict[str, object]], list[dict]]:
    """Reproduce predictions using fitted artifacts on their saved test months."""

    predictions: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    metadata_by_series_model: dict[str, dict[str, object]] = defaultdict(dict)
    eligibility: list[dict] = []
    tab_comparison = _load_tabular_rows()
    sample_counts = _load_tabular_sample_counts()
    model_rows = {
        (row["series_id"], row["model"]): row
        for row in tab_comparison
        if row.get("model") in {"random_forest", "gradient_boosting", "xgboost", "stacking"}
    }

    for (series_id, model_name), report_row in sorted(model_rows.items()):
        artifact = TAB_MODEL_DIR / _slug(series_id) / f"{model_name}.joblib"
        base = {
            "series_id": series_id,
            "model": model_name,
            "artifact_path": str(artifact.relative_to(ROOT)) if artifact.exists() else "",
            "artifact_exists": artifact.exists(),
            "saved_test_months": report_row.get("test_months", ""),
            "transformation": report_row.get("transformation", ""),
            "selection_split": report_row.get("selection_split", ""),
            "prediction_status": "not_evaluated",
            "prediction_error": "",
        }
        if not artifact.exists():
            base.update(prediction_status="unavailable", prediction_error="saved model artifact is missing")
            eligibility.append(base)
            continue
        try:
            saved = joblib.load(artifact)
            model = saved["model"]
            metadata = saved.get("metadata", {})
            transformation = str(metadata.get("transformation") or report_row.get("transformation") or "original")
            test_months = list(metadata.get("test_months") or _parse_months(report_row.get("test_months")))
            rows_by_month = {row["month"]: row for row in feature_groups.get(series_id, [])}
            missing = [month for month in test_months if month not in rows_by_month]
            if missing:
                raise ValueError(f"required feature rows missing for {','.join(missing)}")
            rows = [rows_by_month[month] for month in test_months]
            counts = sample_counts.get(series_id, {})
            n_train = int(counts.get("train_rows", 0))
            n_validation = int(counts.get("validation_rows", 0))
            n_test = int(counts.get("test_rows", len(test_months)))
            ordered_feature_rows = feature_groups.get(series_id, [])
            train_partition = ordered_feature_rows[:n_train]
            validation_partition = ordered_feature_rows[n_train : n_train + n_validation]
            expected_test_partition = ordered_feature_rows[n_train + n_validation : n_train + n_validation + n_test]
            partition_test_months = [item["month"] for item in expected_test_partition]
            partition_consistent = partition_test_months == test_months
            x = np.asarray([[row[column] for column in AGGREGATE_FEATURE_COLUMNS] for row in rows], dtype=float)
            lag = np.asarray([row["price_lag_1"] for row in rows], dtype=float)
            actual = np.asarray([row["target_price"] for row in rows], dtype=float)
            transformed_prediction = _predict_bundle(model, x)
            predicted = _reconstruct(transformed_prediction, lag, transformation)
            if not np.all(np.isfinite(predicted)):
                raise ValueError("saved model produced non-finite predictions")
            for month, actual_value, predicted_value, lag_value in zip(test_months, actual, predicted, lag):
                predictions[series_id][model_name][month] = {
                    "actual_price": float(actual_value),
                    "predicted_price": float(predicted_value),
                    "naive_price": float(lag_value),
                }
            fit_end = ""
            prior = [row["month"] for row in feature_groups.get(series_id, []) if row["month"] not in set(test_months)]
            if prior:
                fit_end = max(prior)
            metadata_by_series_model[series_id][model_name] = {
                "test_months": test_months,
                "transformation": transformation,
                "selection_split": str(metadata.get("selection_split") or report_row.get("selection_split") or ""),
                "fit_end_month": fit_end,
                "train_months": [item["month"] for item in train_partition],
                "validation_months": [item["month"] for item in validation_partition],
                "partition_consistent": partition_consistent,
                "no_test_in_fit": not (set(item["month"] for item in train_partition + validation_partition) & set(test_months)),
                "fit_rows": metadata.get("fit_rows", ""),
            }
            base.update(
                prediction_status="reproduced",
                prediction_count=len(predicted),
                first_prediction_month=test_months[0] if test_months else "",
                last_prediction_month=test_months[-1] if test_months else "",
                fit_end_month=fit_end,
                train_first_month=train_partition[0]["month"] if train_partition else "",
                train_last_month=train_partition[-1]["month"] if train_partition else "",
                validation_first_month=validation_partition[0]["month"] if validation_partition else "",
                validation_last_month=validation_partition[-1]["month"] if validation_partition else "",
                partition_consistent=partition_consistent,
                no_test_in_fit=not (set(item["month"] for item in train_partition + validation_partition) & set(test_months)),
            )
        except Exception as exc:  # report the failure; never fabricate a metric
            base.update(prediction_status="failed", prediction_error=str(exc))
        eligibility.append(base)
    return dict(predictions), dict(metadata_by_series_model), eligibility


def _lstm_metadata(
    series_id: str,
    partitions: dict[str, dict[str, list[str]]],
) -> dict[str, object]:
    split = partitions.get(series_id, {})
    train_months = split.get("train", [])
    validation_months = split.get("validation", [])
    test_months = split.get("test", [])
    return {
        "test_months": test_months,
        "transformation": next(
            (row["selected_transformation"] for row in _read_csv(LSTM_REPORT_DIR / "reproducibility_audit.csv") if row["series_id"] == series_id),
            "",
        ),
        "selection_split": "validation",
        "fit_end_month": (validation_months[-1] if validation_months else (train_months[-1] if train_months else "")),
        "train_months": train_months,
        "validation_months": validation_months,
        "partition_consistent": bool(set(train_months + validation_months).isdisjoint(test_months)),
        "no_test_in_fit": bool(set(train_months + validation_months).isdisjoint(test_months)),
    }


def _all_series(lstm_predictions: dict, feature_groups: dict) -> list[str]:
    return sorted(set(lstm_predictions) & set(feature_groups))


def _model_date_eligibility(
    series_id: str,
    common_dates: list[str],
    lstm_meta: dict[str, object],
    tab_meta: dict[str, object],
    lstm_predictions: dict[str, dict[str, float]],
    tab_eligibility: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    by_model = {row["model"]: row for row in tab_eligibility if row.get("series_id") == series_id}
    for model_name in MODEL_NAMES:
        if model_name == "lstm":
            meta = lstm_meta
            available = list(lstm_predictions)
            artifact_exists = True
            status = "reproduced_from_saved_predictions"
            error = ""
        elif model_name == "naive_last_observation":
            # The naive line is available for both saved prediction periods.
            tab_test: set[str] = set()
            for item in tab_meta.values():
                tab_test.update(item.get("test_months", []))
            available = sorted(tab_test)
            source_meta = next(iter(tab_meta.values()), {})
            meta = {
                **source_meta,
                "test_months": available,
                "transformation": "none",
                "selection_split": "not_applicable",
                "fit_end_month": "",
            }
            artifact_exists = True
            status = "reproduced_from_calendar_lag"
            error = ""
        else:
            meta = tab_meta.get(model_name, {})
            available = list(meta.get("test_months", []))
            source = by_model.get(model_name, {})
            artifact_exists = bool(source.get("artifact_exists"))
            status = str(source.get("prediction_status", "unavailable"))
            error = str(source.get("prediction_error", ""))
        available_set = set(available)
        eligible_set = set(common_dates) & available_set
        fit_end = str(meta.get("fit_end_month", ""))
        cutoff_ok = all((not fit_end) or fit_end < month for month in eligible_set)
        selection_ok = model_name in {"naive_last_observation", "lstm"} or meta.get("selection_split") == "validation"
        partition_ok = bool(meta.get("partition_consistent", True) and meta.get("no_test_in_fit", True))
        valid = bool(artifact_exists and status not in {"failed", "unavailable"} and eligible_set and cutoff_ok and selection_ok and partition_ok)
        reason_parts = []
        if not artifact_exists:
            reason_parts.append("saved artifact missing")
        if status in {"failed", "unavailable"}:
            reason_parts.append(error or status)
        if not eligible_set:
            reason_parts.append("no common eligible test dates")
        if not cutoff_ok:
            reason_parts.append("fit period reaches or exceeds a common target date")
        if not selection_ok:
            reason_parts.append("transformation selection is not recorded as validation-only")
        if not partition_ok:
            reason_parts.append("partition metadata is inconsistent or includes test months in fitting")
        rows.append(
            {
                "series_id": series_id,
                "model": model_name,
                "artifact_or_prediction_available": artifact_exists,
                "selection_split": meta.get("selection_split", ""),
                "transformation": meta.get("transformation", ""),
                "fit_end_month": fit_end,
                "train_first_month": (meta.get("train_months") or [""])[0],
                "train_last_month": (meta.get("train_months") or [""])[-1],
                "validation_first_month": (meta.get("validation_months") or [""])[0],
                "validation_last_month": (meta.get("validation_months") or [""])[-1],
                "test_first_month": (meta.get("test_months") or [""])[0],
                "test_last_month": (meta.get("test_months") or [""])[-1],
                "partition_consistent": meta.get("partition_consistent", True),
                "no_test_in_fit": meta.get("no_test_in_fit", True),
                "available_test_count": len(available),
                "available_test_dates": ";".join(sorted(available)),
                "common_eligible_count": len(eligible_set),
                "common_eligible_dates": ";".join(sorted(eligible_set)),
                "cutoff_check_passed": cutoff_ok,
                "selection_check_passed": selection_ok,
                "partition_check_passed": partition_ok,
                "methodologically_eligible": valid,
                "reason": "; ".join(reason_parts) if reason_parts else "eligible on common dates",
            }
        )
    return rows


def run_audit() -> dict[str, object]:
    feature_groups = _load_feature_rows()
    actuals = _load_monthly_actuals()
    partitions = _load_lstm_partitions()
    lstm_predictions = _load_lstm_predictions()
    tab_predictions, tab_meta, tab_prediction_status = _load_tabular_predictions(feature_groups)
    tabular_metric_reproduction = _validate_saved_tabular_metrics(tab_predictions)

    common_date_rows: list[dict] = []
    comparison_rows: list[dict] = []
    coverage_rows: list[dict] = []
    eligibility_rows: list[dict] = []
    report_lines = [
        "AgriPrice-NG common-test-date model comparison audit",
        "",
        "This read-only audit reuses saved tabular model artifacts and saved LSTM predictions. No model was retrained and no original report was overwritten.",
        "",
        "Decision priority: Option A (common eligible prediction dates) was applied per series when all six models had legitimate predictions. Each model is evaluated on exactly the same target months and actual prices within that series.",
        "Saved tabular models were allowed to predict only their recorded final-validation test months. Dates outside those periods were not extrapolated. A prediction is eligible only when its fitting period ends before the target date and transformation selection is recorded as validation-only.",
        f"Reproduction check: {tabular_metric_reproduction['rows_checked']} saved tabular test rows were recalculated; all reported RMSE values match within 1e-6: {tabular_metric_reproduction['all_match']} (maximum absolute difference: {tabular_metric_reproduction['max_rmse_absolute_difference']}).",
        "",
    ]

    for series_id in _all_series(lstm_predictions, feature_groups):
        lstm_meta = _lstm_metadata(series_id, partitions)
        series_tab_meta = tab_meta.get(series_id, {})
        model_maps: dict[str, dict[str, dict[str, float]]] = {"lstm": lstm_predictions.get(series_id, {})}
        tab_series_predictions = tab_predictions.get(series_id, {})
        for model_name in ("random_forest", "gradient_boosting", "xgboost", "stacking"):
            model_maps[model_name] = tab_series_predictions.get(model_name, {})
        # All six maps must contain the target month. This strict intersection
        # prevents silently dropping a model with missing predictions.
        tab_date_sets = [set(model_maps[name]) for name in ("random_forest", "gradient_boosting", "xgboost", "stacking")]
        tab_dates = set().union(*tab_date_sets) if tab_date_sets else set()
        lstm_dates = set(lstm_predictions.get(series_id, {}))
        common_dates = sorted(set.intersection(*tab_date_sets, lstm_dates)) if tab_date_sets else []
        common_dates = [
            month
            for month in common_dates
            if _finite_float(actuals.get(series_id, {}).get(month))
            and abs(float(actuals[series_id][month]) - float(lstm_predictions[series_id][month]["actual_price"])) <= 1e-8
            and all(abs(float(actuals[series_id][month]) - float(model_maps[name][month]["actual_price"])) <= 1e-8 for name in ("random_forest", "gradient_boosting", "xgboost", "stacking"))
        ]
        # Calendar-lag naive values are reconstructed from the same pre-target
        # month for every common target, not from a previous available row.
        naive_map = {
            month: {
                "actual_price": actuals[series_id][month],
                "predicted_price": model_maps["random_forest"][month]["naive_price"],
                "naive_price": model_maps["random_forest"][month]["naive_price"],
            }
            for month in common_dates
        }
        model_maps["naive_last_observation"] = naive_map
        eligibility = _model_date_eligibility(
            series_id,
            common_dates,
            lstm_meta,
            series_tab_meta,
            lstm_predictions.get(series_id, {}),
            tab_prediction_status,
        )
        eligibility_rows.extend(eligibility)
        valid_models = {row["model"] for row in eligibility if row["methodologically_eligible"]}
        all_six_valid = valid_models == set(MODEL_NAMES) and bool(common_dates)
        common_dates_for_metrics = common_dates if all_six_valid else []

        # One row per common target month makes actual-value equality and all
        # prediction alignment independently inspectable.
        for month in common_dates:
            row = {
                "series_id": series_id,
                "month": month,
                "actual_price": actuals[series_id][month],
                "actual_matches_lstm": abs(actuals[series_id][month] - lstm_predictions[series_id][month]["actual_price"]) <= 1e-8,
                "actual_matches_tabular": all(abs(actuals[series_id][month] - model_maps[name][month]["actual_price"]) <= 1e-8 for name in ("random_forest", "gradient_boosting", "xgboost", "stacking")),
                "included_in_all_model_comparison": month in common_dates_for_metrics,
            }
            for model_name in MODEL_NAMES:
                prediction = model_maps.get(model_name, {}).get(month, {})
                value = prediction.get("predicted_price")
                row[f"{model_name}_prediction"] = value if value is not None else ""
                row[f"{model_name}_prediction_finite"] = _finite_float(value)
            common_date_rows.append(row)

        for model_name in MODEL_NAMES:
            available = list(model_maps.get(model_name, {}))
            if model_name == "naive_last_observation":
                available = list(tab_dates)
            included = sorted(set(available) & set(common_dates_for_metrics))
            excluded = sorted(set(available) - set(included))
            finite = all(_finite_float(model_maps.get(model_name, {}).get(month, {}).get("predicted_price")) for month in included)
            coverage_rows.append(
                {
                    "series_id": series_id,
                    "model": model_name,
                    "available_prediction_count": len(available),
                    "available_first_test_month": min(available) if available else "",
                    "available_last_test_month": max(available) if available else "",
                    "common_comparison_count": len(included),
                    "common_first_test_month": min(included) if included else "",
                    "common_last_test_month": max(included) if included else "",
                    "available_test_dates": ";".join(sorted(available)),
                    "included_test_dates": ";".join(included),
                    "excluded_test_dates": ";".join(excluded),
                    "finite_predictions": finite,
                    "dates_excluded_reason": "none" if not excluded else ("outside all-model common intersection" if all_six_valid else "all-model comparison unavailable for this series"),
                }
            )
            if included and all_six_valid and finite:
                actual = np.asarray([actuals[series_id][month] for month in included], dtype=float)
                predicted = np.asarray([model_maps[model_name][month]["predicted_price"] for month in included], dtype=float)
                metrics = _metrics(actual, predicted)
                status = "valid_common_test_date_comparison"
                reason = "all six models use identical target dates and actual values"
            else:
                metrics = {"rmse": None, "mae": None, "mape": None, "r2": None, "bias": None}
                status = "not_comparable"
                reason = "no valid all-six common-date evaluation set"
            comparison_rows.append(
                {
                    "series_id": series_id,
                    "model": model_name,
                    "common_test_observations": len(included),
                    "first_test_date": min(included) if included else "",
                    "last_test_date": max(included) if included else "",
                    "test_dates": ";".join(included),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "mape": metrics["mape"],
                    "r2": metrics["r2"],
                    "bias": metrics["bias"],
                    "methodologically_valid": all_six_valid and bool(included) and finite,
                    "status": status,
                    "reason": reason,
                }
            )

        report_lines.extend(
            [
                f"Series: {series_id}",
                f"  LSTM train/validation/test: {lstm_meta.get('train_months', [''])[0] if lstm_meta.get('train_months') else ''} to {lstm_meta.get('train_months', [''])[-1] if lstm_meta.get('train_months') else ''}; {lstm_meta.get('validation_months', [''])[0] if lstm_meta.get('validation_months') else ''} to {lstm_meta.get('validation_months', [''])[-1] if lstm_meta.get('validation_months') else ''}; {lstm_meta.get('test_months', [''])[0] if lstm_meta.get('test_months') else ''} to {lstm_meta.get('test_months', [''])[-1] if lstm_meta.get('test_months') else ''}",
                f"  Tabular train/validation/test: {(next(iter(series_tab_meta.values()), {}).get('train_months') or [''])[0] if series_tab_meta else ''} to {(next(iter(series_tab_meta.values()), {}).get('train_months') or [''])[-1] if series_tab_meta else ''}; {(next(iter(series_tab_meta.values()), {}).get('validation_months') or [''])[0] if series_tab_meta else ''} to {(next(iter(series_tab_meta.values()), {}).get('validation_months') or [''])[-1] if series_tab_meta else ''}; {min(tab_dates) if tab_dates else ''} to {max(tab_dates) if tab_dates else ''}",
                f"  Common eligible target dates before all-model checks: {len(common_dates)} ({common_dates[0] if common_dates else ''} to {common_dates[-1] if common_dates else ''})",
                f"  All-six common-date comparison: {'YES' if all_six_valid else 'NO'}",
                f"  LSTM transformation: {lstm_meta.get('transformation', '')}; tabular transformations: " + ", ".join(f"{name}={series_tab_meta.get(name, {}).get('transformation', '')}" for name in ("random_forest", "gradient_boosting", "xgboost", "stacking")),
                "",
            ]
        )

    valid_comparison_rows = [row for row in comparison_rows if row["methodologically_valid"]]
    valid_series = sorted({row["series_id"] for row in valid_comparison_rows})
    report_lines.extend(
        [
            "Overall decision",
            f"  Series with a valid all-six common-date comparison: {len(valid_series)} of {len(_all_series(lstm_predictions, feature_groups))}.",
            f"  Valid model metric rows: {len(valid_comparison_rows)}.",
            "  Existing reports, models, raw files, and forecasts were not overwritten.",
            "  If a series has no all-six common dates, a unified chronological retraining experiment is required before claiming a complete six-model comparison.",
        ]
    )

    _write_csv(AUDIT_DIR / "common_test_dates.csv", common_date_rows)
    _write_csv(AUDIT_DIR / "common_test_date_model_comparison.csv", comparison_rows)
    _write_csv(AUDIT_DIR / "prediction_coverage.csv", coverage_rows)
    _write_csv(AUDIT_DIR / "model_date_eligibility.csv", eligibility_rows)
    (AUDIT_DIR / "common_test_date_audit_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    decision = "Option A: common eligible prediction dates"
    methodology = f"""# Common-test-date comparison decision

## Decision

The audit selected **{decision}** wherever all six models had saved, leakage-safe predictions on the same target months. Metrics are calculated separately for each commodity/unit/price-type series; units and price types are never pooled into an unqualified score.

The existing tabular artifacts were not retrained. Their saved model bundles were loaded, their recorded validation-selected transformations were applied, and predictions were reproduced only on the tabular models' recorded final-validation test months. The LSTM predictions were loaded from `outputs/lstm_experiment/test_predictions.csv`. The naive baseline uses the actual previous calendar-month lag for each target month.

## Leakage and alignment checks

- A target month was included only when every compared model had a prediction and the actual price matched the aggregate monthly source.
- A tabular prediction was accepted only within its saved test period and when the fitted period ended before the target month.
- Exact train, validation, and test boundaries for the tabular feature split and the LSTM sequence split are recorded in `model_date_eligibility.csv` and the text report.
- Tabular transformations are marked validation-selected in the saved metadata and final-validation report; LSTM transformation selection is marked `validation` in its target-selection and reproducibility artifacts.
- Predictions were finite, dates were deduplicated, and all excluded dates are listed in `prediction_coverage.csv`.
- The audit did not use test RMSE to select a model or transformation.

## Important limitation

Same-date evaluation removes the largest comparison defect, but the historical experiments were not trained from one identical split: the tabular models and LSTM can have different training cutoffs because the LSTM requires a complete 12-month sequence. Therefore, the audit supports a defensible same-observation comparison for series with a non-empty all-model intersection, but a fully controlled six-model experiment would still retrain all models using one common chronological train/validation/test boundary.

## Reproducibility files

- `common_test_date_audit_report.txt`: date boundaries, decisions, and exclusions.
- `common_test_dates.csv`: one row per common target month with actual and model predictions.
- `common_test_date_model_comparison.csv`: per-series metrics on identical dates.
- `prediction_coverage.csv`: available, included, and excluded dates for every model.
- `model_date_eligibility.csv`: artifact, cutoff, transformation-selection, and eligibility checks.
"""
    (AUDIT_DIR / "methodology_decision.md").write_text(methodology, encoding="utf-8")

    summary = {
        "series_count": len(_all_series(lstm_predictions, feature_groups)),
        "valid_common_date_series": valid_series,
        "valid_metric_rows": len(valid_comparison_rows),
        "tabular_metric_reproduction": tabular_metric_reproduction,
        "decision": decision,
        "artifacts": [
            "common_test_date_audit_report.txt",
            "common_test_dates.csv",
            "common_test_date_model_comparison.csv",
            "prediction_coverage.csv",
            "model_date_eligibility.csv",
            "methodology_decision.md",
        ],
    }
    (AUDIT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2))
