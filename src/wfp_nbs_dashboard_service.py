"""Read-only dashboard adapter for the active WFP and NBS source scope."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WFP_DATA = ROOT / "data" / "processed" / "wfp_monthly.csv"
NBS_DATA = ROOT / "data" / "processed" / "nbs_monthly.csv"
TRAINING_REPORT_DIR = ROOT / "reports" / "training"
TRAINING_REGISTRY = TRAINING_REPORT_DIR / "model_registry.json"
TRAINING_METRICS = TRAINING_REPORT_DIR / "model_comparison.csv"
TRAINING_PREDICTIONS = TRAINING_REPORT_DIR / "test_predictions.csv"
TRAINING_ELIGIBILITY = TRAINING_REPORT_DIR / "series_eligibility.csv"
TRAINING_FORECASTS = ROOT / "outputs" / "training" / "forecasts.csv"
SUPPORTED_DATASETS = ("wfp", "nbs")


class DashboardError(Exception):
    """A safe, user-facing dashboard request error."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError, OverflowError):
        return None


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _month(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m")
        return len(value) == 7
    except (TypeError, ValueError):
        return False


def _shift_month(value: str, offset: int) -> str:
    year, month = (int(part) for part in value.split("-", 1))
    ordinal = year * 12 + month - 1 + offset
    return f"{ordinal // 12:04d}-{ordinal % 12 + 1:02d}"


def _series_id(source: str, fields: tuple[str, ...]) -> str:
    label = "-".join(re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-") for part in fields)
    digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:12]
    return f"{source}-{label[:70]}-{digest}"


def _read_dict_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError("The normalized WFP/NBS data tables are not available. Run the source preparation command first.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_optional_dict_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, "", "nan", "NaN"):
            return None
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError, OverflowError):
        return None


class DashboardService:
    """Serve only normalized WFP and NBS records; no training at startup."""

    deploy_scope = "wfp_nbs"
    default_dataset = "wfp"

    def __init__(self, reference_date: date | None = None):
        self.reference_date = reference_date or date.today()
        self._training_registry = self._load_training_registry()
        self._training_metrics = _read_optional_dict_rows(TRAINING_METRICS)
        self._training_predictions = _read_optional_dict_rows(TRAINING_PREDICTIONS)
        self._training_eligibility = _read_optional_dict_rows(TRAINING_ELIGIBILITY)
        self._training_forecasts = _read_optional_dict_rows(TRAINING_FORECASTS)
        self._training_series = self._training_registry.get("series", {})
        self._eligibility_by_series = {
            row.get("series_id", ""): row for row in self._training_eligibility
        }
        self._series: dict[str, list[dict[str, Any]]] = {name: [] for name in SUPPORTED_DATASETS}
        self._series_index: dict[tuple[str, str], dict[str, Any]] = {}
        self._dataset_stats: dict[str, dict[str, Any]] = {}
        self._load_wfp()
        self._load_nbs()

    @staticmethod
    def _load_training_registry() -> dict[str, Any]:
        if not TRAINING_REGISTRY.is_file():
            return {}
        try:
            with TRAINING_REGISTRY.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("source_scope") != ["WFP", "NBS"] or not isinstance(payload.get("series"), dict):
                return {}
            return payload
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}

    def _load_wfp(self) -> None:
        grouped: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in _read_dict_rows(WFP_DATA):
            fields = (
                row.get("commodity", ""), row.get("market_id", ""), row.get("market", ""),
                row.get("admin1", ""), row.get("unit", ""), row.get("pricetype", ""),
                row.get("currency", ""), row.get("priceflag", ""),
            )
            record = grouped.get(fields)
            if record is None:
                sid = _series_id("wfp", fields)
                record = {
                    "series_id": sid,
                    "source": "WFP",
                    "source_key": "wfp",
                    "commodity": fields[0],
                    "market": fields[2],
                    "market_id": fields[1],
                    "state": fields[3],
                    "unit": fields[4] or "Not stated",
                    "price_type": fields[5],
                    "currency": fields[6],
                    "price_flag": fields[7],
                    "category": "",
                    "geography_level": "market",
                    "geography_name": fields[2],
                    "history": [],
                }
                grouped[fields] = record
            month = row.get("observation_month", "")
            record["history"].append({
                "month": month,
                "date": month,
                "price": _number(row.get("price")),
                "observed": _number(row.get("price")) is not None,
                "source_observation_count": _integer(row.get("source_observation_count"), 1),
                "first_observation_date": row.get("first_observation_date", ""),
                "last_observation_date": row.get("last_observation_date", ""),
                "market": fields[2],
                "state": fields[3],
                "price_flag": fields[7],
            })

        rows = list(grouped.values())
        for record in rows:
            record["history"].sort(key=lambda item: item["month"])
            self._finalize_series("wfp", record)
        self._series["wfp"] = sorted(rows, key=lambda item: (item["commodity"], item["market"], item["unit"], item["price_type"], item["price_flag"]))

        raw_count = sum(item["source_observation_count"] for record in rows for item in record["history"])
        all_monthly = [item for record in rows for item in record["history"]]
        first_dates = [item["first_observation_date"] for item in all_monthly if item["first_observation_date"]]
        last_dates = [item["last_observation_date"] for item in all_monthly if item["last_observation_date"]]
        actual_dates = [
            item["last_observation_date"]
            for record in rows if record["price_flag"] == "actual"
            for item in record["history"]
            if item["price"] is not None and item["last_observation_date"]
        ]
        self._dataset_stats["wfp"] = {
            "historical_observations": raw_count,
            "normalized_monthly_rows": len(all_monthly),
            "series_count": len(rows),
            "commodity_count": len({record["commodity"] for record in rows}),
            "market_count": len({record["market_id"] for record in rows if record["market_id"]}),
            "state_count": len({record["state"] for record in rows if record["state"]}),
            "date_start": min(first_dates) if first_dates else None,
            "date_end": max(last_dates) if last_dates else None,
            "earliest_observation_date": min(first_dates) if first_dates else None,
            "latest_observation_date": max(last_dates) if last_dates else None,
            "latest_actual_observation_date": max(actual_dates) if actual_dates else None,
            "frequency": "monthly reporting; source observation dates retained",
            "currency": "NGN",
            "price_flags": sorted({record["price_flag"] for record in rows}),
            "price_types": sorted({record["price_type"] for record in rows}),
            "source_description": "WFP Nigeria Food Prices; price flags remain separate series dimensions.",
        }

    def _load_nbs(self) -> None:
        grouped: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in _read_dict_rows(NBS_DATA):
            fields = (
                row.get("item_label", ""), row.get("geography_level", ""),
                row.get("geography_name", ""), row.get("unit", ""),
                row.get("price_type", ""), row.get("currency", ""),
            )
            record = grouped.get(fields)
            if record is None:
                sid = _series_id("nbs", fields)
                record = {
                    "series_id": sid,
                    "source": "NBS",
                    "source_key": "nbs",
                    "commodity": fields[0],
                    "market": "",
                    "state": "",
                    "unit": fields[3] or "Not stated in workbook",
                    "price_type": fields[4],
                    "currency": fields[5] or "NGN (confirmed by paired PDF)",
                    "price_flag": "NBS reported average",
                    "category": "",
                    "geography_level": fields[1],
                    "geography_name": fields[2],
                    "history": [],
                }
                grouped[fields] = record
            month = row.get("observation_month", "")
            record["history"].append({
                "month": month,
                "date": month,
                "price": _number(row.get("price")),
                "observed": _number(row.get("price")) is not None,
                "source_observation_count": 1,
                "geography_level": fields[1],
                "geography_name": fields[2],
                "highest_state_and_price_raw": row.get("highest_state_and_price_raw", ""),
                "lowest_state_and_price_raw": row.get("lowest_state_and_price_raw", ""),
            })

        rows = list(grouped.values())
        for record in rows:
            record["history"].sort(key=lambda item: item["month"])
            self._finalize_series("nbs", record)
        self._series["nbs"] = sorted(rows, key=lambda item: (item["commodity"], item["geography_level"], item["geography_name"]))

        periods = [item["month"] for record in rows for item in record["history"] if item["month"]]
        history = [item for record in rows for item in record["history"]]
        self._dataset_stats["nbs"] = {
            "historical_observations": len(history),
            "normalized_monthly_rows": len(history),
            "series_count": len(rows),
            "commodity_count": len({record["commodity"] for record in rows}),
            "market_count": None,
            "state_count": None,
            "geography_count": len({record["geography_name"] for record in rows}),
            "geography_levels": sorted({record["geography_level"] for record in rows}),
            "date_start": min(periods) if periods else None,
            "date_end": max(periods) if periods else None,
            "earliest_observation_date": min(periods) if periods else None,
            "latest_observation_date": max(periods) if periods else None,
            "frequency": "monthly report periods",
            "currency": "NGN (confirmed by paired official PDF reports)",
            "unit_note": "No standardized unit column; the source item label is preserved verbatim.",
            "source_description": "NBS Selected Food Price Watch; national and zone summary averages, not market-level observations.",
        }

    def _finalize_series(self, dataset: str, record: dict[str, Any]) -> None:
        history = record["history"]
        valid = [row for row in history if row["price"] is not None]
        record["observed_months"] = len(valid)
        record["first_observed_month"] = history[0]["month"] if history else ""
        record["last_observed_month"] = history[-1]["month"] if history else ""
        record["latest_observation_date"] = (
            history[-1].get("last_observation_date") or history[-1]["month"] if history else ""
        )
        record["latest_price"] = valid[-1]["price"] if valid else None
        try:
            latest_date = date.fromisoformat(str(record["latest_observation_date"])[:10])
        except (TypeError, ValueError):
            try:
                latest_date = datetime.strptime(str(record["latest_observation_date"])[:7], "%Y-%m").date()
            except (TypeError, ValueError):
                latest_date = None
        if latest_date is None:
            record["observation_age_months"] = None
            record["freshness"] = "unavailable"
            record["freshness_explanation"] = "No parseable observation date is available."
        else:
            age = max(0, (self.reference_date.year - latest_date.year) * 12 + self.reference_date.month - latest_date.month - int(self.reference_date.day < latest_date.day))
            record["observation_age_months"] = age
            if age <= 3:
                record["freshness"] = "fresh"
                record["freshness_explanation"] = "The latest recorded value is within the last three months."
            elif age <= 6:
                record["freshness"] = "recent"
                record["freshness_explanation"] = "The latest recorded value is more than three months old."
            elif age <= 12:
                record["freshness"] = "aging"
                record["freshness_explanation"] = "The latest recorded value is more than six months old; interpret it as historical context."
            else:
                record["freshness"] = "stale"
                record["freshness_explanation"] = "The latest recorded value is more than twelve months old and may not reflect current conditions."
        record["models"] = []
        self._series_index[(dataset, record["series_id"])] = record

    def _dataset(self, dataset: str) -> str:
        if dataset not in SUPPORTED_DATASETS:
            raise DashboardError(
                "dataset_unavailable",
                "Only the normalized WFP and NBS datasets are available in this application.",
                404,
            )
        return dataset

    def _require_series(self, dataset: str, series_id: str) -> dict[str, Any]:
        dataset = self._dataset(dataset)
        record = self._series_index.get((dataset, series_id))
        if record is None:
            raise DashboardError("series_not_found", "The selected series is not available for this data source.", 404)
        return record

    def _stats(self, dataset: str) -> dict[str, Any]:
        dataset = self._dataset(dataset)
        return dict(self._dataset_stats[dataset])

    def overview(self) -> dict[str, Any]:
        stats = {dataset: self._stats(dataset) for dataset in SUPPORTED_DATASETS}
        selected = stats[self.default_dataset]
        registry_summary = self._training_registry.get("summary", {})
        trained_model_count = int(registry_summary.get("model_series_fits", 0) or 0)
        trained_series_count = int(registry_summary.get("historical_wfp_series_trained", len(self._training_series)) or 0)
        historical_forecast_count = sum(
            bool(series.get("forecast_available"))
            and series.get("forecast_classification") == "historical_only"
            for series in self._training_series.values()
        )
        return {
            "application": "AgriPrice-NG",
            "deploy_scope": self.deploy_scope,
            "default_dataset": self.default_dataset,
            "available_datasets": list(SUPPORTED_DATASETS),
            "dataset": selected,
            "datasets": stats,
            "sources": {
                "wfp": {"available": True, "description": stats["wfp"]["source_description"]},
                "nbs": {"available": True, "description": stats["nbs"]["source_description"]},
            },
            "model_status": {
                "trained_on_current_sources": bool(self._training_series),
                "available_model_count": trained_model_count,
                "trained_series_count": trained_series_count,
                "historical_forecast_series_count": historical_forecast_count,
                "current_market_forecast_series_count": 0,
                "registry_available": bool(self._training_series),
                "message": (
                    "Historical WFP experiments are available; generated forecasts are not current-market prices."
                    if self._training_series else
                    "No model has been trained on the migrated WFP/NBS source scope."
                ),
            },
            "training_note": (
                "Historical WFP test results and recursive forecasts are available. NBS models remain unavailable because each stable post-rebase series is too short."
                if self._training_series else
                "No model trained on the migrated WFP/NBS source scope is currently registered."
            ),
        }

    def series(self, dataset: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        if dataset is not None:
            dataset = self._dataset(dataset)
            return [self._public_series(item) for item in self._series[dataset]]
        return {key: [self._public_series(item) for item in self._series[key]] for key in SUPPORTED_DATASETS}

    def _public_series(self, record: dict[str, Any]) -> dict[str, Any]:
        public = {key: value for key, value in record.items() if key != "history"}
        training = self._training_series.get(record["series_id"], {})
        eligibility = self._eligibility_by_series.get(record["series_id"], {})
        public.update({
            "training_status": "historical_experiment_available" if training else eligibility.get("eligibility_status", "not_trained"),
            "available_models": training.get("available_models", []),
            "champion_model": training.get("champion"),
            "forecast_available": bool(training.get("forecast_available", False)),
            "forecast_origin_month": training.get("forecast_origin_month"),
            "forecast_classification": training.get("forecast_classification"),
            "training_note": training.get("historical_forecast_note") or eligibility.get("eligibility_reason", ""),
        })
        return public

    def history(
        self,
        dataset: str,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        record = self._require_series(dataset, series_id)
        for name, value in (("start_date", start_date), ("end_date", end_date)):
            if value and not _month(value):
                raise DashboardError("invalid_date", f"{name} must use YYYY-MM format.", 400)
        if start_date and end_date and start_date > end_date:
            raise DashboardError("invalid_date_range", "start_date must not be later than end_date.", 400)
        rows = [row for row in record["history"] if (not start_date or row["month"] >= start_date) and (not end_date or row["month"] <= end_date)]
        return {
            "dataset": dataset,
            "series": self._public_series(record),
            "history": rows,
            "requested_range": {"start_month": start_date, "end_month": end_date},
            "returned_observations": len(rows),
            "gaps_filled": False,
        }

    def performance(self, dataset: str, series_id: str | None = None) -> dict[str, Any]:
        dataset = self._dataset(dataset)
        selected = self._require_series(dataset, series_id) if series_id else None
        series_public = self._public_series(selected) if selected else None
        rows = [row for row in self._training_metrics if row.get("series_id") == series_id] if series_id else []
        predictions = [
            row for row in self._training_predictions
            if row.get("series_id") == series_id and row.get("partition") == "test"
        ] if series_id else []
        if rows:
            records = []
            for row in rows:
                try:
                    test_n = int(float(row["test_n"])) if row.get("test_n") else None
                except (TypeError, ValueError, OverflowError):
                    test_n = None
                records.append({
                    "model": row.get("model"), "status": row.get("status"),
                    "rmse": _float_or_none(row.get("test_rmse")),
                    "mae": _float_or_none(row.get("test_mae")),
                    "mape": _float_or_none(row.get("test_mape")),
                    "r2": _float_or_none(row.get("test_r2")),
                    "bias": _float_or_none(row.get("test_bias")),
                    "n": test_n,
                    "validation_rmse": _float_or_none(row.get("validation_rmse")),
                    "champion_selected_on_validation": str(row.get("champion_selected_on_validation", "")).lower() == "true",
                    "test_dates": row.get("test_dates", ""),
                    "unavailable_reason": row.get("unavailable_reason", ""),
                })
            chart: dict[str, dict[str, dict[str, float]]] = {}
            for row in predictions:
                actual = _float_or_none(row.get("actual_price"))
                predicted = _float_or_none(row.get("predicted_price"))
                if actual is None or predicted is None:
                    continue
                chart.setdefault(row.get("target_month", ""), {})[row.get("model", "")] = {
                    "actual": actual, "predicted": predicted,
                }
            return {
                "dataset": dataset, "series": series_public, "records": records,
                "summary": {
                    "champion_model": self._training_series.get(series_id, {}).get("champion"),
                    "selection_basis": "validation RMSE; test metrics were not used for selection",
                    "test_date_count": len(chart),
                    "test_first_month": min(chart) if chart else None,
                    "test_last_month": max(chart) if chart else None,
                    "forecast_classification": self._training_series.get(series_id, {}).get("forecast_classification"),
                },
                "predictions_by_date": [
                    {"month": month, "models": values}
                    for month, values in sorted(chart.items()) if month
                ],
                "status": "evaluated",
                "message": "Test metrics are audit results only; the champion was chosen on validation data.",
                "legacy_results_included": False,
            }
        eligibility = self._eligibility_by_series.get(series_id or "", {})
        message = eligibility.get("eligibility_reason") or "No model evaluation is registered for this series."
        return {
            "dataset": dataset,
            "series": series_public,
            "records": [],
            "summary": None,
            "predictions_by_date": [],
            "status": "not_eligible" if eligibility else "not_trained",
            "message": message,
            "legacy_results_included": False,
        }

    def forecast(self, dataset: str, series_id: str, model: str, horizon: int) -> dict[str, Any]:
        record = self._require_series(dataset, series_id)
        if horizon not in (1, 3, 6):
            raise DashboardError("invalid_horizon", "horizon must be 1, 3, or 6 months.", 400)
        if not model:
            raise DashboardError("missing_model", "A model selection is required.", 400)
        if dataset != "wfp":
            raise DashboardError("forecast_unavailable", "No trained forecast is available for this NBS series.", 422)
        registry = self._training_series.get(series_id)
        if not registry or not registry.get("forecast_available"):
            raise DashboardError("forecast_unavailable", "No validated recursive forecast is available for this series.", 422)
        if model != registry.get("champion"):
            raise DashboardError(
                "model_unavailable",
                "Only the model selected using validation data has a saved recursive forecast for this series.",
                422,
            )
        rows = [
            row for row in self._training_forecasts
            if row.get("series_id") == series_id
            and row.get("model") == model
            and str(row.get("requested_horizon")) == str(horizon)
        ]
        rows.sort(key=lambda row: int(float(row.get("forecast_step", 0) or 0)))
        if len(rows) != horizon:
            raise DashboardError(
                "forecast_unavailable",
                "The requested horizon is not present in the saved forecast outputs.",
                422,
            )
        origin = rows[0].get("forecast_origin_month", "")
        context_start = _shift_month(origin, -11) if origin else ""
        historical = [
            item for item in record["history"]
            if (not context_start or item["month"] >= context_start)
            and (not origin or item["month"] <= origin)
        ]
        forecast_rows = [{
            "date": row.get("forecast_month"), "month": row.get("forecast_month"),
            "price": _float_or_none(row.get("forecast_price")),
            "step": int(float(row.get("forecast_step", 0) or 0)),
            "quality_flag": row.get("quality_flag", "model_estimate"),
        } for row in rows]
        return {
            "dataset": dataset, "series": self._public_series(record),
            "model": model, "horizon": horizon,
            "historical": historical, "forecast": forecast_rows,
            "metadata": {
                "commodity": registry.get("commodity"), "market": registry.get("market"),
                "state": registry.get("state"), "unit": registry.get("unit"),
                "price_type": registry.get("price_type"), "model": model,
                "transformation": "original price in source NGN/unit",
                "method": "recursive; previous model predictions are fed back; no actual future prices are used",
                "forecast_origin_month": origin,
                "latest_actual_month_in_source": registry.get("latest_actual_month"),
                "latest_actual_date_in_source": registry.get("latest_actual_date"),
                "forecast_generation_date": rows[0].get("forecast_generation_date"),
                "forecast_classification": registry.get("forecast_classification", "historical_only"),
                "confidence_intervals": False,
            },
            "warning": "Historical-only projection; this is not a current or guaranteed market price.",
        }


def create_dashboard_service() -> DashboardService:
    return DashboardService()
