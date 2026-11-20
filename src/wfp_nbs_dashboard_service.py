"""Read-only dashboard adapter for the active WFP and NBS source scope."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
import statistics
import threading
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
COVERAGE_AUDIT = ROOT / "reports" / "final_commodity_coverage_audit.csv"
COVERAGE_AUDIT_REPORT = ROOT / "reports" / "final_commodity_coverage_audit.txt"
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

    def __init__(self, reference_date: date | None = None, forecast_cache_dir: Path | None = None):
        self.reference_date = reference_date or date.today()
        self.forecast_cache_dir = forecast_cache_dir
        self._forecast_lock = threading.RLock()
        self._training_registry = self._load_training_registry()
        self._training_metrics = _read_optional_dict_rows(TRAINING_METRICS)
        self._training_predictions = _read_optional_dict_rows(TRAINING_PREDICTIONS)
        self._training_eligibility = _read_optional_dict_rows(TRAINING_ELIGIBILITY)
        self._training_forecasts = _read_optional_dict_rows(TRAINING_FORECASTS)
        self._coverage_audit = _read_optional_dict_rows(COVERAGE_AUDIT)
        self._coverage_audit_as_of = self._load_coverage_audit_date()
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
    def _load_coverage_audit_date() -> str | None:
        if not COVERAGE_AUDIT_REPORT.is_file():
            return None
        try:
            report = COVERAGE_AUDIT_REPORT.read_text(encoding="utf-8-sig")
        except OSError:
            return None
        match = re.search(r"^As-of date:\s*(\d{4}-\d{2}-\d{2})\s*$", report, re.MULTILINE)
        return match.group(1) if match else None

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
                row.get("methodology_regime", ""),
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
                    "methodology_regime": fields[6],
                    "research_commodity_mapping": row.get("research_commodity_mapping", ""),
                    "mapping_status": row.get("mapping_status", ""),
                    "mapping_rationale": row.get("mapping_rationale", ""),
                    "unit_basis": row.get("unit_basis", ""),
                    "price_measure": row.get("price_measure", ""),
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
                "methodology_regime": fields[6],
                "source_file": row.get("source_file", ""),
                "source_report_date": row.get("source_report_date", ""),
                "source_publication_date": row.get("source_publication_date", ""),
                "highest_state_and_price_raw": row.get("highest_state_and_price_raw", ""),
                "lowest_state_and_price_raw": row.get("lowest_state_and_price_raw", ""),
            })

        rows = list(grouped.values())
        for record in rows:
            record["history"].sort(key=lambda item: item["month"])
            self._finalize_series("nbs", record)
        self._series["nbs"] = sorted(rows, key=lambda item: (
            item["commodity"], item["geography_level"], item["geography_name"],
            item["methodology_regime"], item["unit"], item["price_type"],
        ))

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
        record["first_observed_month"] = valid[0]["month"] if valid else ""
        record["last_observed_month"] = valid[-1]["month"] if valid else ""
        record["latest_observation_date"] = (
            valid[-1].get("last_observation_date") or valid[-1]["month"] if valid else ""
        )
        record["latest_source_report_date"] = valid[-1].get("source_report_date", "") if valid else ""
        record["latest_source_publication_date"] = valid[-1].get("source_publication_date", "") if valid else ""
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
                "on_demand_wfp_forecasting_enabled": True,
                "registry_available": bool(self._training_series),
                "message": (
                    "Historical model artifacts are available. Exact WFP and NBS series can be assessed independently for on-demand forecasting; origins may be stale and are not live prices."
                    if self._training_series else
                    "Each exact WFP and NBS series with at least one valid positive reported price can receive a forecast; validated models are used where supported, otherwise a conservative movement or carry-forward estimate is disclosed."
                ),
            },
            "training_note": (
                "Historical WFP test results remain audit-only. WFP and NBS source series are forecast independently only when they pass gap-aware chronological training and validation checks."
                if self._training_series else
                "No historical model is registered. Exact WFP and NBS series with a valid positive reported price can still receive a qualified movement or carry-forward estimate."
            ),
        }

    def series(self, dataset: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        if dataset is not None:
            dataset = self._dataset(dataset)
            return [self._public_series(item) for item in self._series[dataset]]
        return {key: [self._public_series(item) for item in self._series[key]] for key in SUPPORTED_DATASETS}

    def _public_series(self, record: dict[str, Any]) -> dict[str, Any]:
        public = {key: value for key, value in record.items() if key != "history"}
        try:
            from .ondemand_forecasting import ForecastingError, _source_age_months, forecast_policy

            policy = forecast_policy()
            latest_month = record.get("last_observed_month")
            if latest_month:
                age = _source_age_months(
                    str(record.get("source_key") or ""),
                    str(latest_month),
                    str(record.get("latest_observation_date") or "") or None,
                    self.reference_date,
                )
                public["forecast_source_age_months"] = age
                public["forecast_freshness_limit_months"] = policy["max_source_age_months"]
                public["forecast_freshness"] = "fresh" if age <= policy["max_source_age_months"] else "stale"
            else:
                public["forecast_source_age_months"] = None
                public["forecast_freshness_limit_months"] = policy["max_source_age_months"]
                public["forecast_freshness"] = "unavailable"
        except ForecastingError:
            public["forecast_source_age_months"] = None
            public["forecast_freshness_limit_months"] = None
            public["forecast_freshness"] = "unavailable"
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
            "forecast_request_available": bool(
                record.get("source_key") in SUPPORTED_DATASETS
                and int(record.get("observed_months", 0) or 0) > 0
            ),
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

    def _research_validation_rows(self, dataset: str) -> list[dict[str, str]]:
        """Adapt saved validation-only research metrics when the coverage audit is absent."""
        if dataset != "wfp":
            return []
        rows = []
        for row in self._training_metrics:
            if row.get("source", "WFP").upper() != "WFP":
                continue
            if _integer(row.get("validation_n"), 0) <= 0 or not row.get("validation_dates"):
                continue
            rows.append({
                "source": "WFP",
                "series_id": row.get("series_id", ""),
                "forecast_attempt_status": "generated",
                "forecast_available": "true",
                "selected_strategy": "individual_series",
                "selected_model_or_baseline": row.get("model", ""),
                "validation_rmse": row.get("validation_rmse", ""),
                "validation_mae": row.get("validation_mae", ""),
                "validation_mape": row.get("validation_mape", ""),
                "validation_r2": row.get("validation_r2", ""),
                "validation_observations": row.get("validation_n", ""),
            })
        return rows

    def _overall_performance(
        self,
        dataset: str,
        selected: dict[str, Any] | None,
        unavailable_message: str,
    ) -> dict[str, Any]:
        """Build source-separated, validation-only summaries without cross-scale errors."""
        source = dataset.upper()
        audit_rows = [row for row in self._coverage_audit if row.get("source", "").upper() == source]
        has_saved_audit_evaluation = any(
            row.get("forecast_attempt_status") == "generated"
            and str(row.get("forecast_available", "")).lower() == "true"
            and row.get("selected_strategy") in {"individual_series", "pooled_global"}
            and _integer(row.get("validation_observations"), 0) > 0
            for row in audit_rows
        )
        evaluation_rows = audit_rows if has_saved_audit_evaluation else self._research_validation_rows(dataset)
        evaluation_source = (
            "saved exhaustive one-month forecast validation audit"
            if has_saved_audit_evaluation else
            "saved registered WFP chronological validation metrics"
            if evaluation_rows else
            "no saved validation evaluation"
        )
        evaluation_date = self._coverage_audit_as_of if has_saved_audit_evaluation else None
        if not evaluation_date and evaluation_rows and dataset == "wfp":
            generated_at = str(self._training_registry.get("generated_at") or "")
            evaluation_date = generated_at[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", generated_at) else None

        def finite_metric(row: dict[str, str], field: str) -> float | None:
            value = _float_or_none(row.get(field))
            return value if value is not None and math.isfinite(value) else None

        # Each key is one exact series or one compatible group-held-out pool.
        # Audit rows repeat the same pooled score for member series; collapse
        # exact duplicates and exclude inconsistent duplicates rather than
        # counting one pool as many independent evaluations.
        unit_metrics: dict[tuple[str, str, tuple[Any, ...]], set[tuple[float | None, float | None]]] = defaultdict(set)
        unit_observations: dict[tuple[str, tuple[Any, ...]], set[int]] = defaultdict(set)
        units_by_scope: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
        units_by_model_scope: dict[tuple[str, str], set[tuple[Any, ...]]] = defaultdict(set)
        models_by_scope: dict[str, set[str]] = defaultdict(set)
        for row in evaluation_rows:
            if row.get("source", source).upper() != source:
                continue
            if row.get("forecast_attempt_status") != "generated" or str(row.get("forecast_available", "")).lower() != "true":
                continue
            strategy = row.get("selected_strategy", "")
            if strategy not in {"individual_series", "pooled_global"}:
                continue
            model = row.get("selected_model_or_baseline", "").strip()
            series_id = row.get("series_id", "").strip()
            observations = _integer(row.get("validation_observations"), 0)
            if not model or observations <= 0:
                continue
            if strategy == "individual_series":
                if not series_id:
                    continue
                unit = ("series", series_id)
            else:
                # These are the same dimensions used by pooled compatibility
                # rules; NBS geography and CPI regime remain separate.
                unit = (
                    "pool", row.get("commodity", ""), row.get("unit", ""),
                    row.get("price_type", ""), row.get("price_flag", ""),
                    row.get("methodology_regime", ""), row.get("geography_level", ""),
                    observations, _integer(row.get("pooled_training_series_count"), 0),
                    _integer(row.get("pooled_training_rows"), 0),
                )
            units_by_scope[strategy].add(unit)
            unit_observations[(strategy, unit)].add(observations)
            models_by_scope[strategy].add(model)
            units_by_model_scope[(strategy, model)].add(unit)
            unit_metrics[(strategy, model, unit)].add((
                finite_metric(row, "validation_mape"),
                finite_metric(row, "validation_r2"),
            ))

        grouped_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"mape": [], "r2": []})
        metric_counts: dict[tuple[str, str], int] = defaultdict(int)
        conflicted_units = 0
        for (strategy, model, _unit), signatures in unit_metrics.items():
            if len(signatures) != 1:
                conflicted_units += 1
                continue
            mape, r2 = next(iter(signatures))
            key = (strategy, model)
            if mape is not None or r2 is not None:
                metric_counts[key] += 1
            if mape is not None:
                grouped_values[key]["mape"].append(mape)
            if r2 is not None:
                grouped_values[key]["r2"].append(r2)

        # Build a compact, explicitly descriptive set of cards. Absolute
        # errors are not pooled across different price scales. Scale-free
        # medians are calculated once per unambiguous evaluation unit, and a
        # repeated/ambiguous model result never increases the observation
        # count. Prefer exact-series scope; use held-out pools only where exact
        # validation scores are unavailable (as is common for NBS).
        card_scope_units: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        card_mape: dict[str, list[float]] = defaultdict(list)
        card_r2: dict[str, list[float]] = defaultdict(list)
        card_scored_units: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
        ambiguous_card_units: dict[str, int] = defaultdict(int)
        for strategy in ("individual_series", "pooled_global"):
            scope_unit_models: dict[tuple[Any, ...], dict[str, set[tuple[float | None, float | None]]]] = defaultdict(lambda: defaultdict(set))
            for (unit_strategy, model, unit), signatures in unit_metrics.items():
                if unit_strategy == strategy:
                    scope_unit_models[unit][model].update(signatures)
            for unit in units_by_scope[strategy]:
                observation_counts = unit_observations.get((strategy, unit), set())
                if len(observation_counts) != 1:
                    continue
                card_scope_units[strategy].append(unit)
                model_signatures = scope_unit_models.get(unit, {})
                if len(model_signatures) != 1:
                    if model_signatures:
                        ambiguous_card_units[strategy] += 1
                    continue
                signatures = next(iter(model_signatures.values()))
                if len(signatures) != 1:
                    ambiguous_card_units[strategy] += 1
                    continue
                mape_value, r2_value = next(iter(signatures))
                if mape_value is not None or r2_value is not None:
                    card_scored_units[strategy].add(unit)
                if mape_value is not None:
                    card_mape[strategy].append(mape_value)
                if r2_value is not None:
                    card_r2[strategy].append(r2_value)

        scored_card_scopes = [
            strategy for strategy in ("individual_series", "pooled_global")
            if card_mape[strategy] or card_r2[strategy]
        ]
        card_strategy = (
            "individual_series" if "individual_series" in scored_card_scopes else
            "pooled_global" if "pooled_global" in scored_card_scopes else
            "individual_series" if card_scope_units["individual_series"] else
            "pooled_global"
        )
        card_observations = sum(
            next(iter(unit_observations[(card_strategy, unit)]))
            for unit in card_scope_units[card_strategy]
            if len(unit_observations[(card_strategy, unit)]) == 1
        )
        card_scope_label = (
            "exact-series chronological validation" if card_strategy == "individual_series"
            else "compatible-pool group-held-out validation"
        )
        card_unit_label = "series" if card_strategy == "individual_series" else "pools"
        card_mape_value = statistics.median(card_mape[card_strategy]) if card_mape[card_strategy] else None
        card_r2_value = statistics.median(card_r2[card_strategy]) if card_r2[card_strategy] else None
        summary_cards = {
            "label": "Overall validation performance",
            "source": source,
            "evaluation_scope": card_scope_label,
            "rmse": None,
            "mae": None,
            "rmse_unavailable_reason": "Absolute errors are not combined across different commodities, units, and price scales; aligned raw validation predictions are not available for one common-scale calculation.",
            "mae_unavailable_reason": "Absolute errors are not combined across different commodities, units, and price scales; aligned raw validation predictions are not available for one common-scale calculation.",
            "mape": card_mape_value,
            "mape_label": f"Median per-{card_unit_label} MAPE",
            "mape_method": f"Median of genuine per-{card_unit_label} MAPE values for unambiguous selected-model validation units in {card_scope_label}; the existing evaluator omits zero actuals from each MAPE denominator.",
            "r2": card_r2_value,
            "r2_label": f"Median per-{card_unit_label} R²",
            "r2_method": f"Median of genuine per-{card_unit_label} R² values in {card_scope_label}; this is a descriptive median, not a pooled R².",
            "validation_observations": card_observations,
            "evaluation_unit_count": len(card_scope_units[card_strategy]),
            "metric_unit_count": len(card_scored_units[card_strategy]),
            "observation_count_method": f"Sum of validation observations once per distinct {card_unit_label} in the displayed {card_scope_label}; duplicate model rows do not multiply the count.",
            "ambiguous_metric_units_excluded": ambiguous_card_units[card_strategy],
        }
        if card_mape_value is None:
            summary_cards["mape_unavailable_reason"] = "No unambiguous, finite saved per-series or per-pool MAPE is available for this scope."
        if card_r2_value is None:
            summary_cards["r2_unavailable_reason"] = "No unambiguous, finite saved per-series or per-pool R² is available for this scope."

        records = []
        scope_names = {
            "individual_series": "Exact-series chronological validation",
            "pooled_global": "Compatible-pool group-held-out validation",
        }
        for strategy, model in sorted(units_by_model_scope):
            values = grouped_values[(strategy, model)]
            mape = statistics.median(values["mape"]) if values["mape"] else None
            r2 = statistics.median(values["r2"]) if values["r2"] else None
            count = len(units_by_model_scope[(strategy, model)])
            scope_label = scope_names[strategy]
            unit_label = "series" if strategy == "individual_series" else "pools"
            records.append({
                "model": model,
                "status": "overall_validation" if mape is not None or r2 is not None else "overall_coverage",
                # Raw price-scale errors cannot be combined across commodities,
                # source units, and series. Only per-unit relative/scale-free
                # metrics are summarized here.
                "rmse": None, "mae": None, "mape": mape, "r2": r2, "bias": None,
                "n": None,
                "evaluation_unit_count": count,
                "metric_unit_count": metric_counts[(strategy, model)],
                "evaluation_scope": f"{scope_label}; median per-{unit_label} MAPE/R² across {metric_counts[(strategy, model)]} scored validation {unit_label}",
                "champion_selected_on_validation": False,
                "unavailable_reason": "",
            })

        exact_count = len(units_by_scope["individual_series"])
        pool_count = len(units_by_scope["pooled_global"])
        available_models = sorted({model for scope_models in models_by_scope.values() for model in scope_models})
        selected_name = " / ".join(filter(None, [selected.get("commodity", ""), selected.get("market") or selected.get("geography_name", "")])) if selected else "the current selection"
        has_overall_scores = any(row["status"] == "overall_validation" for row in records)
        if has_overall_scores:
            detail = (
                f"This {source} summary uses {exact_count} exact-series validation results and "
                f"{pool_count} distinct compatible-pool group-held-out evaluations"
            )
            if evaluation_date:
                detail += f" from the saved audit as of {evaluation_date}"
            detail += ". It is not a result for the selected commodity. Scores are medians of per-series or per-pool MAPE and R²; RMSE, MAE, and bias are omitted because raw errors are not comparable across different price scales and units. Model rows may cover different evaluation units and are descriptive, not a head-to-head comparison."
        else:
            detail = (
                f"No valid overall validation scores are available for {source}. "
                f"Saved evaluation coverage: {exact_count} exact series, {pool_count} compatible pools, "
                f"and {len(available_models)} available models. No scores or prediction rows have been added."
            )
        reason = f" Individual result unavailable: {unavailable_message.strip()}" if unavailable_message else ""
        message = f"No individual validation results are available for {selected_name}.{reason} {detail}"
        return {
            "dataset": dataset,
            "series": self._public_series(selected) if selected else None,
            "records": records,
            "summary": {
                "evaluation_period": "validation",
                "evaluation_scope": f"source-wide {source} saved chronological validation; exact-series and compatible-pool scopes remain separate",
                "evaluation_source": evaluation_source,
                "validation_audit_as_of": evaluation_date,
                "evaluated_series_count": exact_count,
                "evaluated_pool_count": pool_count,
                "available_models": available_models,
                "scope_counts": {"exact_series": exact_count, "pooled_global": pool_count},
                "metric_summary_method": "median of per-series or per-pool MAPE and R²; no raw-scale metric pooling",
                "raw_scale_metrics_omitted": True,
                "scored_evaluation_unit_count": sum(metric_counts.values()),
                "conflicting_metric_units_excluded": conflicted_units,
            },
            "summary_cards": summary_cards,
            "predictions_by_date": [],
            "status": "overall_evaluated" if any(row["status"] == "overall_validation" for row in records) else "overall_coverage",
            "overall_evaluation": True,
            "message": message,
            "legacy_results_included": False,
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
            for result in records:
                if result["status"] != "trained_and_evaluated":
                    continue
                prediction_count = sum(
                    result["model"] in day for day in chart.values()
                )
                if not prediction_count or prediction_count != (result["n"] or 0):
                    result.update({
                        "status": "unavailable",
                        "rmse": None, "mae": None, "mape": None,
                        "r2": None, "bias": None,
                        "unavailable_reason": (
                            f"Saved metrics report {result['n']} test observations but only "
                            f"{prediction_count} aligned actual/prediction rows are available; metrics are not presented as evaluated."
                        ),
                    })
            evaluated_records = [
                result for result in records if result["status"] == "trained_and_evaluated"
            ]
            return {
                "dataset": dataset, "series": series_public, "records": records,
                "summary": {
                    "champion_model": self._training_series.get(series_id, {}).get("champion"),
                    "selection_basis": "validation RMSE; test metrics were not used for selection",
                    "evaluation_period": "test",
                    "test_date_count": len(chart),
                    "test_first_month": min(chart) if chart else None,
                    "test_last_month": max(chart) if chart else None,
                    "forecast_classification": self._training_series.get(series_id, {}).get("forecast_classification"),
                },
                "predictions_by_date": [
                    {"month": month, "models": values}
                    for month, values in sorted(chart.items()) if month
                ],
                "status": "evaluated" if evaluated_records else "not_evaluated",
                "message": (
                    "Test metrics are audit results only; the champion was chosen on validation data."
                    if evaluated_records else
                    "No model has aligned saved test predictions and metrics for this exact series."
                ),
                "legacy_results_included": False,
            }
        if selected is not None:
            try:
                evaluated = self.forecast(
                    dataset, series_id, model="auto", horizon=1,
                    mode="on_demand", evaluation_only=True,
                )
            except DashboardError as error:
                return self._overall_performance(dataset, selected, error.message)
            metadata = evaluated.get("metadata", {})
            comparisons = metadata.get("model_comparison_validation") or []
            records = [{
                "model": row.get("model"), "status": "validation_evaluated",
                "rmse": _float_or_none(row.get("rmse")),
                "mae": _float_or_none(row.get("mae")),
                "mape": _float_or_none(row.get("mape")),
                "r2": _float_or_none(row.get("r2")),
                "bias": _float_or_none(row.get("bias")),
                "n": _integer(row.get("n"), 0),
                "validation_rmse": _float_or_none(row.get("rmse")),
                "validation_n": _integer(row.get("n"), 0),
                "evaluation_scope": metadata.get("validation_scope", "exact selected series"),
                "champion_selected_on_validation": row.get("model") == evaluated.get("model"),
                "unavailable_reason": "",
            } for row in comparisons]
            failed = metadata.get("model_failures") or metadata.get("failed_models") or []
            if isinstance(failed, dict):
                failed = list(failed)
            for name in failed:
                if any(row["model"] == name for row in records):
                    continue
                records.append({
                    "model": name, "status": "unavailable",
                    "rmse": None, "mae": None, "mape": None, "r2": None,
                    "bias": None, "n": 0, "validation_rmse": None,
                    "validation_n": 0, "evaluation_scope": metadata.get("validation_scope", "exact selected series"),
                    "champion_selected_on_validation": False,
                    "unavailable_reason": "Model validation fitting or prediction failed; no metric is reported.",
                })
            if not any(
                row.get("status") == "validation_evaluated"
                and _float_or_none(row.get("rmse")) is not None
                for row in records
            ):
                return self._overall_performance(
                    dataset, selected,
                    metadata.get("evaluation_note") or "No valid selected-series or compatible-pool validation metrics are available.",
                )
            validation_predictions = metadata.get("validation_predictions") or []
            chart = []
            for item in validation_predictions:
                actual = _float_or_none(item.get("actual"))
                if actual is None:
                    continue
                model_values = item.get("models") or {}
                chart.append({
                    "month": item.get("month"),
                    "models": {
                        name: {"actual": actual, "predicted": value}
                        for name, raw_value in model_values.items()
                        if (value := _float_or_none(raw_value)) is not None
                    },
                })
            dates = metadata.get("validation_months") or [row.get("month") for row in chart]
            global_n = max((_integer(row.get("n"), 0) for row in comparisons), default=0)
            selected_series_metrics = metadata.get("selected_series_validation_metrics") or metadata.get("validation_metrics") or {}
            selected_series_n = _integer(selected_series_metrics.get("n"), 0)
            evaluation_scope = metadata.get("validation_scope", "exact selected series")
            message = (
                "On-demand models and baselines were compared on chronological validation data; no test period was used for selection."
                if metadata.get("strategy_level") != "pooled_global" else
                "Metrics are group-held-out, chronological results for the compatible source/definition pool. They are not an independent evaluation of the selected series; exact-series validation is shown only when that series contributed held-out predictions."
            )
            if metadata.get("strategy_level") == "pooled_global" and selected_series_n == 0:
                message += " The selected exact series has no held-out validation predictions in this pool."
            return {
                "dataset": dataset, "series": series_public, "records": records,
                "summary": {
                    "evaluation_period": "validation",
                    "champion_model": evaluated.get("model"),
                    "selection_basis": metadata.get("selection_basis"),
                    "validation_scope": evaluation_scope,
                    "validation_date_count": len(dates),
                    "validation_first_month": min(dates) if dates else metadata.get("validation_first_month"),
                    "validation_last_month": max(dates) if dates else metadata.get("validation_last_month"),
                    "validation_observations": global_n,
                    "selected_series_validation_observations": selected_series_n,
                    "forecast_classification": "validation_only",
                },
                "predictions_by_date": chart,
                "status": "validation_evaluated" if records else "not_eligible",
                "message": message if records else (metadata.get("evaluation_note") or "No validated predictions are available for this selection."),
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

    def forecast(
        self,
        dataset: str,
        series_id: str,
        model: str = "auto",
        horizon: int = 1,
        mode: str = "",
        evaluation_only: bool = False,
    ) -> dict[str, Any]:
        record = self._require_series(dataset, series_id)
        if horizon not in range(1, 7):
            raise DashboardError("invalid_horizon", "horizon must be an integer from 1 through 6 months.", 400)
        if mode not in ("", "on_demand", "historical"):
            raise DashboardError("invalid_forecast_mode", "mode must be on_demand or historical.", 400)
        model = (model or "auto").strip()
        # Explicit model-name requests preserve the historical saved-forecast
        # API. The live dashboard requests model=auto&mode=on_demand.
        if mode == "historical" or (mode == "" and model not in ("", "auto")):
            return self._saved_historical_forecast(dataset, series_id, record, model, horizon)
        if model not in ("", "auto"):
            raise DashboardError("invalid_model_selection", "On-demand forecasts select the model from chronological validation; use model=auto.", 400)
        from .ondemand_forecasting import ForecastingError, generate_nbs_forecast, generate_wfp_forecast
        from .pooled_forecasting import compatibility_key
        pool_records = [
            item for item in self._series[dataset]
            if compatibility_key(item, dataset) == compatibility_key(record, dataset)
            and (dataset != "wfp" or item.get("price_flag") == "actual")
        ]
        try:
            with self._forecast_lock:
                if dataset == "wfp":
                    generated = generate_wfp_forecast(
                        record,
                        series_id,
                        horizon,
                        registry_entry=self._training_series.get(series_id),
                        cache_dir=self.forecast_cache_dir,
                        root=ROOT,
                        as_of=self.reference_date,
                        pool_records=pool_records,
                        evaluation_only=evaluation_only,
                    )
                else:
                    generated = generate_nbs_forecast(
                        record,
                        series_id,
                        horizon,
                        cache_dir=self.forecast_cache_dir,
                        as_of=self.reference_date,
                        pool_records=pool_records,
                        evaluation_only=evaluation_only,
                    )
        except ForecastingError as exc:
            raise DashboardError(exc.code, exc.message, exc.status_code) from exc
        except Exception as exc:
            logging.getLogger(__name__).exception("On-demand %s forecast failed for %s", dataset.upper(), series_id)
            raise DashboardError(
                "forecast_generation_failed",
                "Forecast training or prediction failed. No forecast was returned; the source data were not changed.",
                500,
            ) from exc
        generated["dataset"] = dataset
        generated["series"] = self._public_series(record)
        return generated

    def _saved_historical_forecast(
        self,
        dataset: str,
        series_id: str,
        record: dict[str, Any],
        model: str,
        horizon: int,
    ) -> dict[str, Any]:
        if not model:
            raise DashboardError("missing_model", "A model selection is required for a saved historical forecast.", 400)
        if dataset != "wfp":
            raise DashboardError("forecast_unavailable", "No historical saved forecast is available for this NBS series.", 422)
        registry = self._training_series.get(series_id)
        if not registry or not registry.get("forecast_available"):
            raise DashboardError("forecast_unavailable", "No saved historical forecast is registered for this series. Use model=auto for on-demand eligibility checks.", 422)
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
                "selection_basis": registry.get("champion_selection_basis"),
                "validation_metrics": {"rmse": registry.get("champion_validation_rmse")},
                "confidence_intervals": False,
            },
            "warning": "Historical-only projection; this is not a current or guaranteed market price.",
        }


def create_dashboard_service() -> DashboardService:
    return DashboardService()
