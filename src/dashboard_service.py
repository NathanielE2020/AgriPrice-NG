"""Retired dashboard adapters retained for historical research reference.

The active Flask application imports ``wfp_nbs_dashboard_service`` instead.
This legacy module is no longer used by the dashboard or ``train.py``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import warnings
from collections import Counter
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from .data_loader import APPROVED_SERIES
from .parquet_compat import month_from_index, month_index, read_parquet_rows, to_date


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
FINAL_MODEL_DIR = MODEL_DIR / "final_validation" / "aggregate"
WORLD_BANK_MODEL_DIR = MODEL_DIR / "world_bank"
WORLD_BANK_CHAMPION_REGISTRY = REPORT_DIR / "world_bank" / "champion_models.json"
WORLD_BANK_REGISTRY_MAX_AGE_DAYS = 180
LSTM_MODEL_DIR = MODEL_DIR / "lstm_experiment"
LSTM_FORECAST_PATH = ROOT / "outputs" / "lstm_experiment" / "forecasts" / "recursive_forecasts.csv"
COMMON_AUDIT_DIR = REPORT_DIR / "common_test_date_audit"
RAW_PARQUET_PATH = ROOT / "data" / "raw" / "train.parquet"
WINDOWS_LSTM_PYTHON = ROOT / ".venv-lstm" / "Scripts" / "python.exe"

# Freshness is a description of observation age only.  It does not indicate
# whether a recorded price is accurate or representative of today's market.
FRESHNESS_THRESHOLDS_MONTHS = {
    "fresh": 3,
    "recent": 6,
    "aging": 12,
}

VALID_DATASETS = ("market", "aggregate", "world_bank")
DEPLOY_SCOPE_ENV = "AGRIPRICE_DEPLOY_SCOPE"
APPROVED_MARKET_SCOPE = "approved_market"
WORLD_BANK_SCOPE = "world_bank"
WORLD_BANK_CHAMPION_COUNT = 12
WORLD_BANK_COLUMNS = (
    "source", "date", "month", "commodity_ticker", "commodity", "unit", "price_type",
    "price", "low_estimate", "high_estimate", "trust_score", "base_observation_price",
    "is_model_estimate", "geography_level", "geography_name", "state", "market",
    "data_coverage", "data_coverage_recent", "index_confidence_score", "last_survey_point",
    "source_data_is_direct_observation",
)
VALID_HORIZONS = (1, 3, 6)
MODEL_ORDER = (
    "naive_last_observation",
    "random_forest",
    "xgboost",
    "gradient_boosting",
    "stacking",
    "lstm",
)
MODEL_LABELS = {
    "naive_last_observation": "Naive last observation",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradient Boosting",
    "stacking": "Hybrid Stacking",
    "lstm": "LSTM",
}


class DashboardError(Exception):
    """Expected, user-facing dashboard error."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class _StackingPredictor:
    """Expose a saved stacking bundle through the predictor interface."""

    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle

    def predict(self, values):
        import numpy as np

        bundle = self.bundle
        matrix = np.column_stack(
            [bundle["base_models"][name].predict(values) for name in bundle["base_names"]]
        )
        return bundle["meta_model"].predict(matrix)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise DashboardError(
            "missing_data",
            f"Required dashboard data is unavailable: {path.name}.",
            503,
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:120]


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _as_float(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    numeric = float(value)
    return None if math.isnan(numeric) else numeric


def _as_int(value: Any) -> int | None:
    if value in (None, "", "nan", "NaN"):
        return None
    return int(float(value))


def _parse_month_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    return [item for item in str(value).split(";") if item]


def _parse_date_filter(value: Any, parameter_name: str) -> tuple[date | None, str | None]:
    """Parse an API date filter and return its date and calendar month.

    The dashboard is monthly, so both ``YYYY-MM`` and ISO ``YYYY-MM-DD`` are
    accepted.  A month filter includes the complete calendar month rather than
    silently excluding observations because their snapshot day is the 15th.
    """

    if value in (None, ""):
        return None, None
    text = str(value).strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}", text):
            parsed = datetime.strptime(text, "%Y-%m").date()
        else:
            parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise DashboardError(
            "invalid_date",
            f"{parameter_name} must use YYYY-MM or YYYY-MM-DD format.",
            400,
        ) from exc
    return parsed, f"{parsed.year:04d}-{parsed.month:02d}"


def _date_from_row(row: dict[str, Any]) -> date | None:
    value = row.get("date")
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            pass
    month = str(row.get("month") or "")
    if re.fullmatch(r"\d{4}-\d{2}", month):
        return datetime.strptime(month, "%Y-%m").date().replace(day=15)
    return None


def _month_age(observation_date: date, reference_date: date) -> int:
    """Return elapsed complete calendar months between two dates."""

    months = (reference_date.year - observation_date.year) * 12
    months += reference_date.month - observation_date.month
    if reference_date.day < observation_date.day:
        months -= 1
    return max(0, months)


def _freshness(age_months: int | None) -> tuple[str, str]:
    if age_months is None:
        return "unavailable", "No real recorded observation is available for this series."
    if age_months <= FRESHNESS_THRESHOLDS_MONTHS["fresh"]:
        return "fresh", "The latest available recorded observation is within the last three months."
    if age_months <= FRESHNESS_THRESHOLDS_MONTHS["recent"]:
        return "recent", "The latest available recorded observation is more than three months old."
    if age_months <= FRESHNESS_THRESHOLDS_MONTHS["aging"]:
        return "aging", "The latest available recorded observation is more than six months old; interpret it as historical context rather than today's market price."
    return "stale", "The latest available recorded observation is more than twelve months old; this series may not reflect current market conditions."


def _month_contiguous(months: list[str], count: int) -> bool:
    if len(months) < count:
        return False
    tail = sorted(months, key=month_index)[-count:]
    return month_index(tail[-1]) - month_index(tail[0]) == count - 1


def _public_note(value: Any) -> str:
    """Remove local artifact paths from report notes before returning JSON."""

    note = str(value or "")
    note = re.sub(r"Saved to [^;]+;?", "Trained artifact available. ", note)
    note = re.sub(r"[A-Za-z]:\\[^\s;]+", "artifact", note)
    return " ".join(note.split())


def _public_metric(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _standard_model_name(name: str) -> str:
    if name == "naive_last_calendar_month":
        return "naive_last_observation"
    return name


def _model_path(dataset: str, series_id: str, model_name: str) -> Path:
    slug = _slug(series_id)
    if dataset == "market":
        return MODEL_DIR / slug / f"{model_name}.joblib"
    if dataset == "world_bank":
        return WORLD_BANK_MODEL_DIR / slug / f"{model_name}.joblib"
    return FINAL_MODEL_DIR / slug / f"{model_name}.joblib"


def _lstm_model_path(series_id: str) -> Path:
    return LSTM_MODEL_DIR / _slug(series_id) / "lstm.keras"


def _lstm_preprocessing_path(series_id: str) -> Path:
    return LSTM_MODEL_DIR / _slug(series_id) / "lstm_preprocessing.json"


def _lstm_interpreter() -> Path | None:
    """Select the interpreter used for isolated TensorFlow inference."""

    configured = os.environ.get("LSTM_PYTHON", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else (ROOT / path).resolve()
    if os.name == "nt":
        return WINDOWS_LSTM_PYTHON if WINDOWS_LSTM_PYTHON.exists() else None
    return Path(sys.executable)


def _load_artifact(path: Path) -> tuple[Any, dict[str, Any]]:
    if not path.exists():
        raise DashboardError(
            "model_unavailable",
            "The requested model artifact is not available for this series.",
            422,
        )
    try:
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            artifact = joblib.load(path)
    except Exception as exc:  # Do not expose deserialization details to users.
        raise DashboardError(
            "model_unavailable",
            "The requested model artifact could not be loaded.",
            503,
        ) from exc
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        metadata = artifact.get("metadata") or {}
    else:
        model = artifact
        metadata = {}
    if isinstance(model, dict) and "base_models" in model:
        model = _StackingPredictor(model)
    return model, metadata


def _validate_world_bank_champion_artifact(
    metadata: dict[str, Any],
    series_id: str,
    champion_record: dict[str, Any],
) -> None:
    """Reject a World Bank artifact that does not match its champion registry."""
    expected = {
        "source": "world_bank",
        "series_id": series_id,
        "selected_model": champion_record.get("selected_model"),
        "transformation": champion_record.get("selected_transformation"),
        "experiment_version": champion_record.get("experiment_version"),
    }

    if (
        not isinstance(metadata, dict)
        or any(metadata.get(key) != value for key, value in expected.items())
        or metadata.get("production_champion") is not True
    ):
        raise DashboardError(
            "model_unavailable",
            "The registered World Bank champion artifact failed integrity validation.",
            503,
        )


class DashboardService:
    """Cached read-only dashboard data service."""

    def __init__(self, reference_date: date | None = None):
        self.reference_date = reference_date or date.today()
        self.deploy_scope = os.environ.get(DEPLOY_SCOPE_ENV, "").strip().lower()
        self.market_only = self.deploy_scope == APPROVED_MARKET_SCOPE
        self.world_bank_only = self.deploy_scope == WORLD_BANK_SCOPE
        self.default_dataset = "market" if self.market_only else ("world_bank" if self.world_bank_only else "aggregate")

        # The registry determines the only World Bank series allowed in the
        # dedicated production scope. Load and validate it before reading the
        # large normalized CSV. An invalid registry yields an empty filter.
        self._world_bank_champions = self._load_world_bank_champion_registry(
            require_expected_count=self.world_bank_only
        )
        champion_ids = tuple(self._world_bank_champions.get("records", {}).keys()) if self._world_bank_champions.get("available") else ()

        self.market_rows = [] if self.world_bank_only else self._load_market_rows()
        self.aggregate_rows = [] if self.market_only or self.world_bank_only else self._load_aggregate_rows()
        self._world_bank_data_error: str | None = None
        if self.market_only:
            self.world_bank_rows = []
        elif self.world_bank_only and not champion_ids:
            self.world_bank_rows = []
        else:
            try:
                self.world_bank_rows = self._load_world_bank_rows(champion_ids if self.world_bank_only else None)
            except Exception:
                self.world_bank_rows = []
                self._world_bank_data_error = "The normalized World Bank champion data could not be loaded."
        # Keep an index of references to the already-loaded World Bank row
        # dictionaries.  The index makes series lookups O(1) without creating
        # another copy of the 300k-row normalized table.
        self.world_bank_rows_by_series: dict[str, list[dict[str, Any]]] = {}
        for row in self.world_bank_rows:
            self.world_bank_rows_by_series.setdefault(row["series_id"], []).append(row)
        # The deployment scope deliberately avoids a full raw Parquet scan.
        # The inspection report is the authoritative source summary in that
        # mode; local development still calculates the summary dynamically.
        self.source_summary = {} if self.world_bank_only else self._load_source_summary(skip_raw=self.market_only)
        self._market_metrics = [] if self.world_bank_only else self._load_metrics("market")
        self._aggregate_metrics = [] if self.market_only or self.world_bank_only else self._load_metrics("aggregate")
        if self.market_only:
            self._world_bank_metrics = []
        elif self.world_bank_only and not champion_ids:
            # An invalid registry must not trigger a broad metrics read either.
            self._world_bank_metrics = []
        else:
            try:
                self._world_bank_metrics = self._load_metrics(
                    "world_bank", series_ids=set(champion_ids) if self.world_bank_only else None
                )
            except Exception:
                self._world_bank_metrics = []
                if self.world_bank_only:
                    self._world_bank_data_error = "World Bank champion evaluation metrics are unavailable."
        if self.world_bank_only and champion_ids:
            loaded_ids = {row["series_id"] for row in self.world_bank_rows}
            if loaded_ids != set(champion_ids):
                self.world_bank_rows = []
                self._world_bank_data_error = "The normalized World Bank data does not contain every registered champion series."
            metric_ids = {row.get("series_id") for row in self._world_bank_metrics}
            if not set(champion_ids).issubset(metric_ids):
                self._world_bank_data_error = "World Bank champion evaluation metrics are incomplete."
                self.world_bank_rows = []
        self._artifact_compatibility: dict[tuple[str, str, str], bool] = {}
        self._lstm_forecast_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self.market_series = self._build_series("market", self.market_rows)
        self.aggregate_series = self._build_series("aggregate", self.aggregate_rows)
        self.world_bank_series = self._build_series("world_bank", self.world_bank_rows_by_series)

    def _load_world_bank_champion_registry(self, require_expected_count: bool = False) -> dict[str, Any]:
        """Load the production registry and fail closed on any integrity issue."""

        unavailable = {
            "available": False,
            "reason": "World Bank champion registry is unavailable.",
            "records": {},
            "generated_at": None,
            "champion_count": 0,
        }
        if self.market_only:
            return unavailable
        if not WORLD_BANK_CHAMPION_REGISTRY.exists():
            unavailable["reason"] = "World Bank champion registry is missing; production forecasts are disabled."
            return unavailable
        try:
            payload = json.loads(WORLD_BANK_CHAMPION_REGISTRY.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("series"), list):
                raise ValueError("registry must contain a series list")
            generated_at = str(payload.get("generated_at") or "")
            parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
            age_seconds = (datetime.now(parsed.tzinfo) - parsed).total_seconds()
            if age_seconds < -300:
                unavailable["reason"] = "World Bank champion registry timestamp is in the future."
                unavailable["generated_at"] = generated_at
                return unavailable
            age_days = age_seconds / 86400
            if age_days > WORLD_BANK_REGISTRY_MAX_AGE_DAYS:
                unavailable["reason"] = "World Bank champion registry is stale; rerun the World Bank validation before forecasting."
                unavailable["generated_at"] = generated_at
                return unavailable
            records: dict[str, dict[str, Any]] = {}
            champion_dir = (MODEL_DIR / "world_bank" / "champions").resolve()
            for record in payload["series"]:
                if not isinstance(record, dict) or not record.get("series_id"):
                    raise ValueError("registry contains an invalid series record")
                series_id = str(record["series_id"]).strip()
                if not series_id:
                    raise ValueError("registry contains an empty series ID")
                if series_id in records:
                    raise ValueError(f"duplicate champion series ID: {series_id}")
                model = str(record.get("selected_model") or "")
                if model not in MODEL_ORDER or model == "lstm":
                    raise ValueError(f"unsupported champion model for {series_id}")
                transform = str(record.get("selected_transformation") or "none")
                if transform not in {"none", "original", "log1p", "price_change", "percentage_change"}:
                    raise ValueError(f"unsupported champion transformation for {series_id}")
                artifact = record.get("artifact_path")
                if model == "naive_last_observation":
                    if artifact not in (None, ""):
                        raise ValueError(f"naive champion must not require an artifact for {series_id}")
                else:
                    if not isinstance(artifact, str) or not artifact:
                        raise ValueError(f"missing artifact path for {series_id}")
                    artifact_path = (ROOT / artifact).resolve()
                    if champion_dir not in artifact_path.parents or not artifact_path.is_file():
                        raise ValueError(
                            f"champion artifact must be a file inside the champion directory for {series_id}"
                        )
                records[series_id] = record
            if not records:
                raise ValueError("registry contains no series records")
            if require_expected_count and len(records) != WORLD_BANK_CHAMPION_COUNT:
                raise ValueError(f"registry must contain exactly {WORLD_BANK_CHAMPION_COUNT} champion series")
            return {"available": True, "reason": "Validated World Bank champion registry.", "records": records, "generated_at": generated_at, "champion_count": len(records), "experiment_version": payload.get("experiment_version")}
        except Exception as exc:
            unavailable["reason"] = f"World Bank champion registry is invalid; production forecasts are disabled ({type(exc).__name__})."
            return unavailable

    @staticmethod
    def _load_source_summary(skip_raw: bool = False) -> dict[str, Any]:
        """Summarize the actual supplied source dataset without writing to it."""

        if not skip_raw:
            try:
                raw_rows = read_parquet_rows(RAW_PARQUET_PATH)
                dates = [to_date(row["date"]) for row in raw_rows if row.get("date") is not None]
                commodities = {str(row.get("commodity")) for row in raw_rows if row.get("commodity") is not None}
                markets = {str(row.get("market")) for row in raw_rows if row.get("market") is not None}
                states = {str(row.get("admin1")) for row in raw_rows if row.get("admin1") is not None}
                currencies = {str(row.get("currency")) for row in raw_rows if row.get("currency")}
                earliest = min(dates) if dates else None
                latest = max(dates) if dates else None
                return {
                    "historical_observations": len(raw_rows),
                    "commodity_count": len(commodities),
                    "market_count": len(markets),
                    "state_count": len(states),
                    "date_start": f"{earliest.year:04d}-{earliest.month:02d}" if earliest else None,
                    "date_end": f"{latest.year:04d}-{latest.month:02d}" if latest else None,
                    "earliest_observation_date": earliest.isoformat() if earliest else None,
                    "latest_observation_date": latest.isoformat() if latest else None,
                    "frequency": "monthly",
                    "currency": next(iter(currencies)) if len(currencies) == 1 else "NGN",
                    "source_summary": "data/raw/train.parquet",
                    "statistics_source": "dynamic_raw_parquet",
                    "fallback_used": False,
                }
            except Exception:
                fallback_reason = "The source dataset could not be loaded by the available Parquet reader."
        else:
            fallback_reason = "Deployment scope intentionally skipped the full raw Parquet scan."

        # The active Flask environment may not have a Parquet engine, or may
        # intentionally skip the raw scan to stay within the deployment memory
        # budget.  The checked-in inspection report is the recorded result of
        # reading this exact raw file and is explicitly marked as a fallback.
        report_path = REPORT_DIR / "dataset_inspection.txt"
        if not report_path.exists():
            return {}
        report = report_path.read_text(encoding="utf-8", errors="replace")
        row_match = re.search(r"Rows:\s*([\d,]+)", report)
        commodity_match = re.search(r"There are\s+(\d+)\s+distinct commodity", report)
        market_match = re.search(r"There are\s+(\d+)\s+distinct markets", report)
        state_match = re.search(r"There are\s+(\d+)\s+distinct states", report)
        coverage_match = re.search(r"(\d+) commodities,\s*(\d+) observed markets in\s*(\d+) states", report)
        if coverage_match:
            market_match = market_match or coverage_match
            state_match = state_match or coverage_match
        first_match = re.search(r"Earliest date:\s*(\d{4}-\d{2}-\d{2})", report)
        last_match = re.search(r"Latest date:\s*(\d{4}-\d{2}-\d{2})", report)
        earliest = first_match.group(1) if first_match else None
        latest = last_match.group(1) if last_match else None
        return {
            "historical_observations": int(row_match.group(1).replace(",", "")) if row_match else None,
            "commodity_count": int(commodity_match.group(1)) if commodity_match else None,
            "market_count": int(coverage_match.group(2)) if coverage_match else (int(market_match.group(1)) if market_match else None),
            "state_count": int(coverage_match.group(3)) if coverage_match else (int(state_match.group(1)) if state_match else None),
            "date_start": earliest[:7] if earliest else None,
            "date_end": latest[:7] if latest else None,
            "earliest_observation_date": earliest,
            "latest_observation_date": latest,
            "frequency": "monthly",
            "currency": "NGN",
            "source_summary": "reports/dataset_inspection.txt (raw Parquet scan skipped)" if skip_raw else "reports/dataset_inspection.txt (raw Parquet reader unavailable)",
            "statistics_source": "verified_inspection_report_fallback",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
        }

    @staticmethod
    def _load_market_rows() -> list[dict[str, Any]]:
        rows = _read_csv(PROCESSED_DIR / "approved_monthly_reindexed.csv")
        approved = set(APPROVED_SERIES)
        output: list[dict[str, Any]] = []
        for row in rows:
            key = (row.get("commodity", ""), row.get("market", ""), row.get("unit", ""), row.get("price_type", ""))
            if key not in approved:
                continue
            series_id = " | ".join(
                (row["commodity"], row["market"], row["unit"], row["price_type"])
            )
            output.append(
                {
                    **row,
                    "series_id": series_id,
                    "market_id": _as_int(row.get("market_id")),
                    "price": _as_float(row.get("price")),
                    "observed": _as_bool(row.get("observed")),
                }
            )
        return sorted(output, key=lambda item: (item["series_id"], item["month"]))

    @staticmethod
    def _load_aggregate_rows() -> list[dict[str, Any]]:
        rows = _read_csv(PROCESSED_DIR / "aggregate_monthly_reindexed.csv")
        output: list[dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    **row,
                    "mean_price": _as_float(row.get("mean_price")),
                    "median_price": _as_float(row.get("median_price")),
                    "n_markets": _as_int(row.get("n_markets")) or 0,
                    "n_records": _as_int(row.get("n_records")) or 0,
                    "observed": _as_bool(row.get("observed")),
                }
            )
        return sorted(output, key=lambda item: (item["series_id"], item["month"]))

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_world_bank_rows(champion_series_ids: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Load normalized World Bank estimates without mixing WFP rows."""

        # Keep the World Bank ingestion dependency (and its pandas import)
        # outside the deployment-only startup path.
        from .world_bank_ingestion import NORMALIZED_FILE

        if not NORMALIZED_FILE.exists():
            return []
        import pandas as pd

        if champion_series_ids is None:
            # Full local mode intentionally preserves the existing catalogue
            # view and loading behaviour.
            rows = pd.read_csv(NORMALIZED_FILE).to_dict("records")
        else:
            # Restricted deployment reads bounded chunks and materializes
            # dictionaries only for the exact IDs validated from the registry.
            wanted = set(champion_series_ids)
            if not wanted:
                return []
            selected_rows: list[dict[str, Any]] = []
            for chunk in pd.read_csv(
                NORMALIZED_FILE,
                usecols=list(WORLD_BANK_COLUMNS),
                chunksize=25_000,
                low_memory=False,
            ):
                commodity = chunk["commodity"].fillna("").astype(str)
                geography = chunk["geography_name"].fillna("").astype(str)
                unit = chunk["unit"].fillna("").astype(str)
                price_type = chunk["price_type"].fillna("").astype(str)
                series_ids = commodity + " | " + geography + " | " + unit + " | " + price_type
                mask = series_ids.isin(wanted)
                if not mask.any():
                    continue
                kept = chunk.loc[mask].copy()
                kept["series_id"] = series_ids.loc[mask]
                selected_rows.extend(kept.to_dict("records"))
            rows = selected_rows
        for row in rows:
            commodity = row.get("commodity", "")
            location = row.get("geography_name", "")
            # Parquet/CSV values may be NaN outside the approved panel; keep
            # the established series ID convention without stringifying NaN.
            def clean_text(value: Any) -> str:
                return "" if pd.isna(value) else str(value)

            series_id = " | ".join((clean_text(commodity), clean_text(location), clean_text(row.get("unit", "")), clean_text(row.get("price_type", ""))))
            # Mutate each record returned by pandas in place.  This preserves
            # the existing row schema and conversions while avoiding a second
            # dictionary allocation for every normalized observation.
            price = _as_float(row.get("price"))
            row["series_id"] = series_id
            row["price"] = price
            row["low_estimate"] = _as_float(row.get("low_estimate"))
            row["high_estimate"] = _as_float(row.get("high_estimate"))
            row["trust_score"] = _as_float(row.get("trust_score"))
            row["base_observation_price"] = _as_float(row.get("base_observation_price"))
            row["data_coverage"] = _as_float(row.get("data_coverage"))
            row["data_coverage_recent"] = _as_float(row.get("data_coverage_recent"))
            row["index_confidence_score"] = _as_float(row.get("index_confidence_score"))
            row["observed"] = price is not None
            row["is_model_estimate"] = _as_bool(row.get("is_model_estimate"))
            row["geography_level"] = row.get("geography_level", "")
            row["state"] = None if pd.isna(row.get("state")) else row.get("state")
            row["market"] = None if pd.isna(row.get("market")) else row.get("market")
            row["source_data_is_direct_observation"] = _as_bool(row.get("source_data_is_direct_observation"))
        rows.sort(key=lambda item: (item["series_id"], item["month"]))
        return rows

    def _rows(self, dataset: str) -> list[dict[str, Any]]:
        self._ensure_dataset_enabled(dataset)
        if dataset == "market":
            return self.market_rows
        if dataset == "aggregate":
            return self.aggregate_rows
        if dataset == "world_bank":
            return self.world_bank_rows
        raise DashboardError(
            "invalid_dataset",
            "dataset must be 'market', 'aggregate', or 'world_bank'.",
            400,
        )

    def _ensure_dataset_enabled(self, dataset: str) -> None:
        if dataset not in VALID_DATASETS:
            raise DashboardError(
                "invalid_dataset",
                "dataset must be 'market', 'aggregate', or 'world_bank'.",
                400,
            )
        if self.market_only and dataset != "market":
            raise DashboardError(
                "dataset_unavailable",
                "This approved_market deployment exposes only the approved market-level WFP dataset.",
                404,
            )
        if self.world_bank_only and dataset != "world_bank":
            raise DashboardError(
                "dataset_unavailable",
                "This world_bank deployment exposes only registered World Bank champion series.",
                404,
            )

    def _rows_for_series(self, dataset: str, series_id: str) -> list[dict[str, Any]]:
        """Return rows for one series without scanning World Bank rows.

        World Bank rows use the startup index.  WFP market and aggregate rows
        retain their existing filtering behavior because their datasets are
        small and their loaders already expose the approved series rows.
        """

        self._ensure_dataset_enabled(dataset)
        if dataset == "world_bank":
            return self.world_bank_rows_by_series.get(series_id, [])
        return [row for row in self._rows(dataset) if row["series_id"] == series_id]

    def _series_map(self, dataset: str) -> dict[str, dict[str, Any]]:
        self._ensure_dataset_enabled(dataset)
        if dataset == "market":
            values = self.market_series
        elif dataset == "aggregate":
            values = self.aggregate_series
        elif dataset == "world_bank":
            values = self.world_bank_series
        else:
            raise DashboardError(
                "invalid_dataset",
                "dataset must be 'market', 'aggregate', or 'world_bank'.",
                400,
            )
        return {item["series_id"]: item for item in values}

    def _build_series(
        self,
        dataset: str,
        rows: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        if isinstance(rows, dict):
            # Reuse the caller-provided index directly; do not regroup or copy
            # the World Bank row dictionaries.
            grouped = rows
        else:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(row["series_id"], []).append(row)
        result: list[dict[str, Any]] = []
        for series_id, group in sorted(grouped.items()):
            observed = [row for row in group if row["observed"] and self._price(row) is not None]
            first = min(observed, key=lambda row: month_index(row["month"])) if observed else None
            last = max(observed, key=lambda row: month_index(row["month"])) if observed else None
            first_date = _date_from_row(first) if first else None
            last_date = _date_from_row(last) if last else None
            age_months = _month_age(last_date, self.reference_date) if last_date else None
            freshness_status, freshness_explanation = _freshness(age_months)
            base = {
                "dataset": dataset,
                "series_id": series_id,
                "commodity": group[0]["commodity"],
                "unit": group[0]["unit"],
                "price_type": group[0]["price_type"],
                "calendar_months": len(group),
                "observed_months": len(observed),
                "first_observed_month": first["month"] if first else None,
                "last_observed_month": last["month"] if last else None,
                "first_observation_date": first_date.isoformat() if first_date else None,
                "last_observation_date": last_date.isoformat() if last_date else None,
                "latest_observation_date": last_date.isoformat() if last_date else None,
                "latest_observation_month": last["month"] if last else None,
                "latest_actual_price": self._price(last) if last else None,
                "latest_available_recorded_price": self._price(last) if last else None,
                "observation_age_months": age_months,
                "freshness_status": freshness_status,
                "freshness_explanation": freshness_explanation,
            }
            if dataset == "market":
                base.update(
                    {
                        "market": group[0]["market"],
                        "market_id": group[0].get("market_id"),
                        "state": group[0].get("state"),
                    }
                )
            elif dataset == "aggregate":
                counts = [row["n_markets"] for row in observed]
                base.update(
                    {
                        "market": None,
                        "market_id": None,
                        "state": None,
                        "min_contributing_markets": min(counts) if counts else 0,
                        "max_contributing_markets": max(counts) if counts else 0,
                        "average_contributing_markets": round(sum(counts) / len(counts), 3) if counts else 0,
                    }
                )
            else:
                latest = last or {}
                base.update(
                    {
                        "market": latest.get("market"),
                        "market_id": None,
                        "state": latest.get("state"),
                        "source": "world_bank",
                        "is_model_estimate": True,
                        "price_value_type": "world_bank_close_estimate",
                        "base_observation_price": _as_float(latest.get("base_observation_price")),
                        "geography_level": latest.get("geography_level"),
                        "geography_name": latest.get("geography_name"),
                        "latest_estimate_price": self._price(last) if last else None,
                        "latest_low_estimate": latest.get("low_estimate"),
                        "latest_high_estimate": latest.get("high_estimate"),
                        "latest_trust_score": latest.get("trust_score"),
                        "latest_is_direct_observation": bool(latest.get("source_data_is_direct_observation")),
                        "source_data_is_direct_observation": bool(latest.get("source_data_is_direct_observation")),
                        "data_coverage": _as_float(latest.get("data_coverage")),
                        "data_coverage_recent": _as_float(latest.get("data_coverage_recent")),
                        "index_confidence_score": _as_float(latest.get("index_confidence_score")),
                        "last_survey_point": latest.get("last_survey_point"),
                    }
                )
            base["models"] = self.model_availability(dataset, series_id)
            result.append(base)
        return result

    @staticmethod
    def _price(row: dict[str, Any]) -> float | None:
        return row.get("price") if "price" in row else row.get("mean_price")

    def _load_metrics(self, dataset: str, series_ids: set[str] | None = None) -> list[dict[str, Any]]:
        if dataset == "aggregate":
            return self._load_common_aggregate_metrics()
        path = REPORT_DIR / "model_comparison.csv" if dataset == "market" else REPORT_DIR / "world_bank" / "model_comparison.csv"
        if dataset not in {"market", "world_bank"}:
            return []
        rows = _read_csv(path)
        output = []
        for row in rows:
            if row.get("split") != "test":
                continue
            if series_ids is not None and row.get("series_id") not in series_ids:
                continue
            model = _standard_model_name(row.get("model", ""))
            output.append(
                {
                    "series_id": row.get("series_id"),
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "split": row.get("split", "test"),
                    "status": row.get("status", ""),
                    "rmse": _public_metric(row.get("rmse", row.get("test_rmse"))),
                    "mae": _public_metric(row.get("mae", row.get("test_mae"))),
                    "mape": _public_metric(row.get("mape", row.get("test_mape"))),
                    "r2": _public_metric(row.get("r2", row.get("test_r2"))),
                    "transformation": row.get("transformation", "none") or "none",
                    "n_train": _as_int(row.get("n_train")),
                    "n_validation": _as_int(row.get("n_validation")),
                    "n_test": _as_int(row.get("n_test")),
                    "test_months": row.get("test_months", ""),
                    "notes": _public_note(row.get("notes")),
                }
            )
        return output

    def _load_common_aggregate_metrics(self) -> list[dict[str, Any]]:
        """Load the authoritative identical-test-date aggregate audit."""

        comparison = _read_csv(COMMON_AUDIT_DIR / "common_test_date_model_comparison.csv")
        eligibility_rows = _read_csv(COMMON_AUDIT_DIR / "model_date_eligibility.csv")
        transformations = {
            (row.get("series_id"), row.get("model")): row.get("transformation") or "none"
            for row in eligibility_rows
        }
        output: list[dict[str, Any]] = []
        for row in comparison:
            model = _standard_model_name(row.get("model", ""))
            valid = _as_bool(row.get("methodologically_valid"))
            status = "evaluated" if valid else (row.get("status") or "unavailable")
            test_dates = row.get("test_dates", "")
            dates = _parse_month_list(test_dates)
            output.append(
                {
                    "series_id": row.get("series_id"),
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "split": "common_test",
                    "status": status,
                    "rmse": _public_metric(row.get("rmse")),
                    "mae": _public_metric(row.get("mae")),
                    "mape": _public_metric(row.get("mape")),
                    "r2": _public_metric(row.get("r2")),
                    "bias": _public_metric(row.get("bias")),
                    "transformation": transformations.get((row.get("series_id"), row.get("model")), "none"),
                    "n_train": None,
                    "n_validation": None,
                    "n_test": _as_int(row.get("common_test_observations")),
                    "test_months": test_dates,
                    "first_test_date": row.get("first_test_date") or (dates[0] if dates else None),
                    "last_test_date": row.get("last_test_date") or (dates[-1] if dates else None),
                    "evaluation_validity": valid,
                    "test_date_alignment": "common" if valid else "not_comparable",
                    "comparison_source": "common_test_date_audit",
                    "notes": _public_note(
                        "Exact common test dates; original tabular and LSTM training cutoffs are retained and may differ. "
                        + (row.get("reason") or "")
                    ),
                }
            )
        return output

    def model_availability(self, dataset: str, series_id: str) -> list[dict[str, Any]]:
        self._ensure_dataset_enabled(dataset)
        # _build_series already iterates the authoritative rows; repeating a
        # 300k-row scan for every World Bank catalogue series would make app
        # startup quadratic.
        if dataset != "world_bank" and not any(row["series_id"] == series_id for row in self._rows(dataset)):
            return []
        if self.market_only:
            return [
                {
                    "model": "naive_last_observation",
                    "label": MODEL_LABELS["naive_last_observation"],
                    "available": True,
                    "status": "available",
                    "notes": "Previous calendar-month price baseline; lightweight deployment path.",
                }
            ]
        if dataset == "world_bank":
            record = self._world_bank_champions.get("records", {}).get(series_id) if self._world_bank_champions.get("available") else None
            selected = str(record.get("selected_model")) if record else None
            output: list[dict[str, Any]] = []
            for model in MODEL_ORDER:
                is_champion = bool(record and model == selected)
                if is_champion:
                    output.append({
                        "model": model,
                        "label": MODEL_LABELS[model],
                        "available": True,
                        "status": "available",
                        "production": True,
                        "research_only": False,
                        "transformation": record.get("selected_transformation", "none"),
                        "notes": "Registered World Bank production champion; close-estimate forecast.",
                    })
                else:
                    output.append({
                        "model": model,
                        "label": MODEL_LABELS[model],
                        "available": False,
                        "status": "unavailable",
                        "production": False,
                        "research_only": bool(record),
                        "notes": (
                            "Research challenger; not the registered production forecast."
                            if record else self._world_bank_champions.get("reason", "Production registry unavailable.")
                        ),
                    })
            return output
        records = {row["model"]: row for row in self._metrics_for(dataset, series_id)}
        available: list[dict[str, Any]] = []
        for model in MODEL_ORDER:
            if model == "naive_last_observation":
                available.append(
                    {
                        "model": model,
                        "label": MODEL_LABELS[model],
                        "available": True,
                        "status": "available",
                        "notes": "Previous calendar-month price baseline.",
                    }
                )
                continue
            if model == "lstm":
                if dataset == "aggregate":
                    metric = records.get(model)
                    lstm_python = _lstm_interpreter()
                    artifact_available = (
                        lstm_python is not None
                        and lstm_python.exists()
                        and
                        _lstm_model_path(series_id).exists()
                        and _lstm_preprocessing_path(series_id).exists()
                    )
                    metric_available = bool(
                        metric
                        and metric.get("status") == "evaluated"
                        and metric.get("evaluation_validity") is True
                    )
                    available_flag = artifact_available and metric_available
                    available.append(
                        {
                            "model": model,
                            "label": MODEL_LABELS[model],
                            "available": available_flag,
                            "status": "available" if available_flag else "unavailable",
                            "notes": "Saved TensorFlow/Keras LSTM artifact and common-date metrics available."
                            if available_flag
                            else "Saved LSTM artifact, forecast output, or common-date metric is unavailable.",
                        }
                    )
                    continue
                available.append(
                    {
                        "model": model,
                        "label": MODEL_LABELS[model],
                        "available": False,
                        "status": "unavailable",
                        "notes": "No validated LSTM artifact is available for this dataset.",
                    }
                )
                continue
            path = _model_path(dataset, series_id, model)
            metric = records.get(model)
            path_available = path.exists()
            # World Bank has many normalized catalogue series but only the
            # explicitly selected training panel has artifacts.  Avoid loading
            # every serialized estimator during app startup; forecast() still
            # performs the guarded deserialization before inference.
            artifact_loadable = (
                path_available
                if dataset == "world_bank"
                else path_available and self._artifact_is_loadable(dataset, series_id, model)
            )
            metric_available = metric is not None and metric.get("status") in {"trained", "evaluated"}
            available.append(
                {
                    "model": model,
                    "label": MODEL_LABELS[model],
                    "available": bool(path_available and artifact_loadable and metric_available),
                    "status": "available" if path_available and artifact_loadable and metric_available else "unavailable",
                    "notes": "Saved model artifact available."
                    if path_available and artifact_loadable and metric_available
                    else (
                        "Not trained for market-level data."
                        if dataset == "market" and model == "xgboost"
                        else (
                            "Saved artifact is incompatible with the active Python environment."
                            if path_available and not artifact_loadable
                            else "No usable saved artifact for this series."
                        )
                    ),
                }
            )
        return available

    def _artifact_is_loadable(self, dataset: str, series_id: str, model: str) -> bool:
        key = (dataset, series_id, model)
        if key not in self._artifact_compatibility:
            try:
                _load_artifact(_model_path(dataset, series_id, model))
                self._artifact_compatibility[key] = True
            except DashboardError:
                self._artifact_compatibility[key] = False
        return self._artifact_compatibility[key]

    def _metrics_for(self, dataset: str, series_id: str | None = None) -> list[dict[str, Any]]:
        if dataset == "market":
            rows = self._market_metrics
        elif dataset == "aggregate":
            rows = self._aggregate_metrics
        else:
            rows = self._world_bank_metrics
        if series_id is None:
            return rows
        return [row for row in rows if row["series_id"] == series_id]

    def overview(self) -> dict[str, Any]:
        market_model_counts = Counter()
        aggregate_model_counts = Counter()
        world_bank_model_counts = Counter()
        for series in self.market_series:
            for item in series["models"]:
                if item["available"]:
                    market_model_counts[item["model"]] += 1
        if self.world_bank_only:
            for series in self.world_bank_series:
                for item in series["models"]:
                    if item["available"]:
                        world_bank_model_counts[item["model"]] += 1
        elif not self.market_only:
            for series in self.aggregate_series:
                for item in series["models"]:
                    if item["available"]:
                        aggregate_model_counts[item["model"]] += 1
            for series in self.world_bank_series:
                for item in series["models"]:
                    if item["available"]:
                        world_bank_model_counts[item["model"]] += 1
        if self.world_bank_only:
            world_bank = self._world_bank_overview()
            dataset_facts = {
                "source": "World Bank Nigeria Real Time Food Prices; c_ close estimates",
                "historical_observations": world_bank.get("historical_observations", 0),
                "commodity_count": world_bank.get("commodity_count", 0),
                "market_count": world_bank.get("market_count"),
                "state_count": world_bank.get("state_count"),
                "date_start": world_bank.get("date_start"),
                "date_end": world_bank.get("date_end"),
                "earliest_observation_date": world_bank.get("earliest_observation_date"),
                "latest_observation_date": world_bank.get("latest_observation_date"),
                "frequency": "monthly",
                "currency": "NGN",
                "statistics_source": "filtered_world_bank_champion_rows",
                "fallback_used": False,
                "coverage_scope": "registered_champion_subset",
                "source_data_is_direct_observation": False,
                "note": "Statistics describe registered World Bank champion series only, not the full source catalogue. Prices are close estimates.",
            }
            registry_status = {
                "available": self._world_bank_champions.get("available", False),
                "reason": self._world_bank_champions.get("reason"),
                "generated_at": self._world_bank_champions.get("generated_at"),
                "champion_count": self._world_bank_champions.get("champion_count", 0),
            }
            sources = {
                "world_bank": {**world_bank, "champion_registry": registry_status},
                "wfp": {"available": False, "reason": "WFP datasets are disabled by AGRIPRICE_DEPLOY_SCOPE=world_bank."},
                "aggregate": {"available": False, "reason": "WFP aggregate data is disabled by AGRIPRICE_DEPLOY_SCOPE=world_bank."},
            }
            model_status = {
                "market": {},
                "aggregate_final_validation": {},
                "world_bank": dict(world_bank_model_counts),
                "lstm": {"aggregate_available_series": 0, "market_available_series": 0, "inference_environment": "Not applicable to the World Bank champion panel."},
            }
            return {
                "deployment_scope": WORLD_BANK_SCOPE,
                "available_datasets": [WORLD_BANK_SCOPE],
                "dataset": dataset_facts,
                "freshness": {
                    "reference_date": self.reference_date.isoformat(),
                    "thresholds_months": dict(FRESHNESS_THRESHOLDS_MONTHS),
                    "meaning": "Freshness describes the age of the latest available close estimate, not the date of a direct survey.",
                },
                "experiments": {
                    "market_level_series": 0,
                    "aggregate_series": 0,
                    "world_bank_champion_series": len(self.world_bank_series),
                    "full_world_bank_catalogue_loaded": False,
                },
                "model_status": model_status,
                "sources": sources,
            }

        dataset_facts = {
            **self.source_summary,
            "frequency": self.source_summary.get("frequency", "monthly"),
            "currency": self.source_summary.get("currency", "NGN"),
            "source": (
                "WFP Nigeria Food Prices dataset; values loaded from data/raw/train.parquet"
                if self.source_summary.get("statistics_source") == "dynamic_raw_parquet"
                else "WFP Nigeria Food Prices dataset; values reported in the verified dataset inspection report"
            ),
        }
        if self.market_only:
            sources = {
                "wfp": {
                    "label": "WFP Nigeria Food Prices (approved market scope)",
                    "statistics": dataset_facts,
                    "series_count": len(self.market_series),
                },
                "aggregate": {
                    "available": False,
                    "reason": "Disabled by AGRIPRICE_DEPLOY_SCOPE=approved_market.",
                },
                "world_bank": {
                    "available": False,
                    "reason": "Disabled by AGRIPRICE_DEPLOY_SCOPE=approved_market.",
                },
            }
            experiments = {
                "market_level_series": len(self.market_series),
                "aggregate_series": 0,
                "market_level_history_file": "approved_monthly_reindexed.csv",
                "aggregate_history_file": None,
            }
            model_status = {
                "market": dict(market_model_counts),
                "aggregate_final_validation": {},
                "world_bank": {},
                "lstm": {
                    "aggregate_available_series": 0,
                    "market_available_series": market_model_counts.get("lstm", 0),
                    "inference_environment": "LSTM is not loaded during market-only startup",
                },
            }
        else:
            sources = {
                "wfp": {
                    "label": "WFP Nigeria Food Prices",
                    "statistics": dataset_facts,
                    "series_count": len(self.market_series) + len(self.aggregate_series),
                },
                "world_bank": {
                    **self._world_bank_overview(),
                    "champion_registry": {
                        "available": self._world_bank_champions.get("available", False),
                        "reason": self._world_bank_champions.get("reason"),
                        "generated_at": self._world_bank_champions.get("generated_at"),
                        "champion_count": self._world_bank_champions.get("champion_count", 0),
                    },
                },
            }
            experiments = {
                "market_level_series": len(self.market_series),
                "aggregate_series": len(self.aggregate_series),
                "market_level_history_file": "approved_monthly_reindexed.csv",
                "aggregate_history_file": "aggregate_monthly_reindexed.csv",
            }
            model_status = {
                "market": dict(market_model_counts),
                "aggregate_final_validation": dict(aggregate_model_counts),
                "world_bank": dict(world_bank_model_counts),
                "lstm": {
                    "aggregate_available_series": aggregate_model_counts.get("lstm", 0),
                    "market_available_series": market_model_counts.get("lstm", 0),
                    "inference_environment": "Windows .venv-lstm or Linux active Python (TensorFlow 2.21.0)",
                },
            }
        return {
            "deployment_scope": self.deploy_scope or "local_full",
            "available_datasets": ["market"] if self.market_only else list(VALID_DATASETS),
            "dataset": dataset_facts,
            "freshness": {
                "reference_date": self.reference_date.isoformat(),
                "thresholds_months": dict(FRESHNESS_THRESHOLDS_MONTHS),
                "meaning": "Freshness describes the age of the latest recorded observation only; it does not indicate price accuracy.",
            },
            "experiments": experiments,
            "model_status": model_status,
            "sources": sources,
        }

    def _world_bank_overview(self) -> dict[str, Any]:
        if not self.world_bank_rows:
            return {
                "available": False,
                "source": "World Bank Nigeria Real Time Food Prices",
                "reason": self._world_bank_data_error
                or self._world_bank_champions.get("reason")
                or "Normalized World Bank data has not been prepared.",
                "series_count": 0,
                "champion_series_count": self._world_bank_champions.get("champion_count", 0),
                "is_model_estimate": True,
                "source_data_is_direct_observation": False,
            }
        rows = self.world_bank_rows
        dates = []
        for row in rows:
            value = row.get("date")
            if not value:
                continue
            try:
                dates.append(date.fromisoformat(str(value)[:10]))
            except ValueError:
                continue
        market_rows = [row for row in rows if row.get("geography_level") == "market"]
        states = {row.get("state") for row in market_rows if row.get("state")}
        markets = {row.get("market") for row in market_rows if row.get("market")}
        survey_months = []
        for row in rows:
            value = row.get("last_survey_point")
            if not value:
                continue
            for date_format in ("%Y-%m-%d", "%Y-%m", "%b %Y", "%B %Y"):
                try:
                    parsed = datetime.strptime(str(value).strip(), date_format)
                except ValueError:
                    continue
                survey_months.append(parsed.strftime("%Y-%m"))
                break
        result = {
            "available": True,
            "source": "World Bank Nigeria Real Time Food Prices; c_ close estimates",
            "historical_observations": len(rows),
            "commodity_count": len({row.get("commodity_ticker") for row in rows}),
            "market_count": len(markets),
            "state_count": len(states),
            "geopolitical_zone_count": len({row.get("geography_name") for row in rows if row.get("geography_level") == "geopolitical_zone"}),
            "date_start": min(dates).strftime("%Y-%m") if dates else None,
            "date_end": max(dates).strftime("%Y-%m") if dates else None,
            "earliest_observation_date": min(dates).isoformat() if dates else None,
            "latest_observation_date": max(dates).isoformat() if dates else None,
            "frequency": "monthly",
            "currency": "NGN",
            "estimate_type": "close_estimate",
            "is_model_estimate": True,
            "series_count": len(self.world_bank_series),
            "trained_series_count": len({row["series_id"] for row in self._world_bank_metrics}),
            "survey_observation_period_end": max(survey_months, default=None),
            "data_coverage": _as_float(rows[0].get("data_coverage")) if rows else None,
            "data_coverage_recent": _as_float(rows[0].get("data_coverage_recent")) if rows else None,
            "index_confidence_score": _as_float(rows[0].get("index_confidence_score")) if rows else None,
            "note": "World Bank values are modelled/imputed close estimates; the latest estimated month is not treated as proof of a direct market survey.",
        }
        if self.world_bank_only:
            result.update({
                "coverage_scope": "registered_champion_subset",
                "full_source_catalogue_loaded": False,
                "champion_series_count": len(self.world_bank_series),
                "geography_levels": sorted({str(row.get("geography_level")) for row in rows if row.get("geography_level")}),
                "geography_names_count": len({str(row.get("geography_name")) for row in rows if row.get("geography_name")}),
                "market_count": len(markets),
                "state_count": len(states),
                "source_data_is_direct_observation": False,
                "note": "Statistics cover only the registered champion subset (National Average and Market Average series), not the full World Bank catalogue. Prices are c_ close estimates; the latest estimated month is not presented as a direct survey date.",
            })
            # Coverage scores belong to individual source rows and are not
            # reduced to an arbitrary single value for this multi-series panel.
            result.pop("data_coverage", None)
            result.pop("data_coverage_recent", None)
            result.pop("index_confidence_score", None)
        return result

    def series(self, dataset: str | None = None) -> dict[str, Any]:
        if dataset is None:
            if self.market_only:
                return {"market": self.market_series}
            if self.world_bank_only:
                return {"world_bank": self.world_bank_series}
            return {"market": self.market_series, "aggregate": self.aggregate_series, "world_bank": self.world_bank_series}
        self._rows(dataset)  # validates dataset
        return {
            dataset: self.market_series
            if dataset == "market"
            else self.aggregate_series
            if dataset == "aggregate"
            else self.world_bank_series
        }

    def _require_series(self, dataset: str, series_id: str) -> dict[str, Any]:
        if not series_id:
            raise DashboardError("missing_series", "series_id is required.", 400)
        series = self._series_map(dataset).get(series_id)
        if series is None:
            raise DashboardError("unknown_series", "The requested series is not available.", 404)
        return series

    def history(
        self,
        dataset: str,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        metadata = self._require_series(dataset, series_id)
        start_value, start_month = _parse_date_filter(start_date, "start_date")
        end_value, end_month = _parse_date_filter(end_date, "end_date")
        if start_month and end_month and month_index(start_month) > month_index(end_month):
            raise DashboardError(
                "invalid_date_range",
                "start_date must not be later than end_date.",
                400,
            )
        rows = self._rows_for_series(dataset, series_id)
        if start_month:
            rows = [row for row in rows if month_index(row["month"]) >= month_index(start_month)]
        if end_month:
            rows = [row for row in rows if month_index(row["month"]) <= month_index(end_month)]
        output = []
        for row in rows:
            item = {
                "month": row["month"],
                "date": row["date"],
                "price": self._price(row),
                "observed": bool(row["observed"]),
            }
            if dataset == "market":
                item["priceflag"] = row.get("priceflag")
                item["state"] = row.get("state")
            elif dataset == "aggregate":
                item["n_markets"] = row.get("n_markets", 0)
                item["n_records"] = row.get("n_records", 0)
            else:
                item.update(
                    {
                        "source": "world_bank",
                        "is_model_estimate": True,
                        "price_value_type": "world_bank_close_estimate",
                        "base_observation_price": _as_float(row.get("base_observation_price")),
                        "geography_level": row.get("geography_level"),
                        "geography_name": row.get("geography_name"),
                        "state": row.get("state"),
                        "market": row.get("market"),
                        "low_estimate": row.get("low_estimate"),
                        "high_estimate": row.get("high_estimate"),
                        "trust_score": row.get("trust_score"),
                        "index_confidence_score": _as_float(row.get("index_confidence_score")),
                        "last_survey_point": row.get("last_survey_point"),
                        "source_data_is_direct_observation": bool(row.get("source_data_is_direct_observation")),
                    }
                )
            output.append(item)
        return {
            "dataset": dataset,
            "series": metadata,
            "history": output,
            "requested_range": {
                "start_date": start_value.isoformat() if start_value else None,
                "end_date": end_value.isoformat() if end_value else None,
                "start_month": start_month,
                "end_month": end_month,
            },
            "returned_calendar_months": len(output),
            "returned_observed_months": sum(1 for row in output if row["observed"] and row["price"] is not None),
            "latest_observation": {
                "price": metadata.get("latest_actual_price"),
                "date": metadata.get("latest_observation_date"),
                "month": metadata.get("latest_observation_month"),
                "freshness_status": metadata.get("freshness_status"),
                "observation_age_months": metadata.get("observation_age_months"),
            },
        }

    def performance(self, dataset: str, series_id: str | None = None) -> dict[str, Any]:
        self._rows(dataset)
        if series_id is not None:
            self._require_series(dataset, series_id)
        records = self._metrics_for(dataset, series_id)
        if dataset == "aggregate":
            selection_protocol = (
                "Aggregate records come from the common-test-date audit. Each model uses identical target dates within a series; "
                "the original tabular and LSTM training cutoffs can differ because the experiments were preserved separately. "
                "Transformations were selected on validation data only."
            )
        elif dataset == "world_bank":
            selection_protocol = (
                "World Bank close-estimate models use calendar-ordered 70/15/15 splits within each selected commodity/location. "
                "Transformations are chosen on validation RMSE only, the test months are untouched, and stacking uses chronological out-of-fold base predictions. "
                "The naive baseline uses the same test dates."
            )
        else:
            selection_protocol = (
                "Final aggregate configurations were selected on validation data and evaluated once on untouched test periods. "
                "Market records are the existing chronological test metrics."
            )
        if dataset == "world_bank":
            champion = self._world_bank_champions.get("records", {}).get(series_id) if series_id else None
            for record in records:
                selected = bool(champion and record.get("model") == champion.get("selected_model"))
                record["production_champion"] = selected
                record["research_only"] = not selected
                if selected:
                    record["notes"] = (record.get("notes") or "") + " Registered production champion."
        return {
            "dataset": dataset,
            "series_id": series_id,
            "selection_protocol": selection_protocol,
            "records": records,
        }

    def _validate_forecast_request(self, dataset: str, series_id: str, model: str, horizon: int) -> dict[str, Any]:
        series = self._require_series(dataset, series_id)
        if horizon not in VALID_HORIZONS:
            raise DashboardError("invalid_horizon", "horizon must be 1, 3, or 6 months.", 400)
        allowed = {item["model"]: item for item in self.model_availability(dataset, series_id)}
        if model not in MODEL_ORDER:
            raise DashboardError("invalid_model", "The requested model name is not supported.", 400)
        if model not in allowed or not allowed[model]["available"]:
            raise DashboardError(
                "model_unavailable",
                allowed.get(model, {}).get(
                    "notes",
                    "This model is not available in the approved-market deployment scope.",
                ),
                422,
            )
        return series

    def _lstm_forecast(self, series_id: str, horizon: int) -> dict[str, Any]:
        """Run isolated TensorFlow inference and return validated forecast rows."""

        # A six-step recursive run contains the exact prefixes for the one- and
        # three-month requests, so one isolated TensorFlow process is enough per
        # series while preserving the requested response horizon.
        cache_key = (series_id, 6)
        if cache_key in self._lstm_forecast_cache:
            cached = self._lstm_forecast_cache[cache_key]
            return {**cached, "forecast": cached["forecast"][:horizon]}
        lstm_python = _lstm_interpreter()
        if lstm_python is None or not lstm_python.exists():
            raise DashboardError(
                "model_unavailable",
                "LSTM inference is unavailable because no configured TensorFlow interpreter exists.",
                503,
            )
        if not _lstm_model_path(series_id).exists() or not _lstm_preprocessing_path(series_id).exists():
            raise DashboardError(
                "model_unavailable",
                "The saved LSTM artifact is unavailable for this aggregate series.",
                422,
            )
        try:
            completed = subprocess.run(
                [
                    str(lstm_python),
                    "-m",
                    "src.lstm_dashboard_inference",
                    "--series-id",
                    series_id,
                    "--horizon",
                    "6",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DashboardError(
                "inference_timeout",
                "LSTM inference took too long to complete.",
                503,
            ) from exc
        except OSError as exc:
            raise DashboardError(
                "model_unavailable",
                "The isolated LSTM inference environment could not be started.",
                503,
            ) from exc
        try:
            payload = json.loads(completed.stdout.strip())
        except (TypeError, json.JSONDecodeError) as exc:
            raise DashboardError(
                "inference_failed",
                "The isolated LSTM inference process returned an invalid response.",
                503,
            ) from exc
        if completed.returncode != 0 or not payload.get("ok"):
            raise DashboardError(
                str(payload.get("code") or "inference_failed"),
                str(payload.get("message") or "The LSTM forecast could not be generated."),
                422,
            )
        forecast_rows = payload.get("forecast")
        if not isinstance(forecast_rows, list) or len(forecast_rows) != 6:
            raise DashboardError(
                "inference_failed",
                "The LSTM inference response did not contain the requested forecast horizon.",
                503,
            )
        previous_month = ""
        validated_rows = []
        for expected_step, row in enumerate(forecast_rows, start=1):
            month = str(row.get("forecast_month") or "")
            price = _public_metric(row.get("forecast_price"))
            if int(row.get("horizon_months", 0)) != expected_step or not month or price is None:
                raise DashboardError(
                    "inference_failed",
                    "The LSTM inference response contained invalid forecast values.",
                    503,
                )
            if previous_month and month_index(month) <= month_index(previous_month):
                raise DashboardError(
                    "inference_failed",
                    "The LSTM forecast dates are not chronological.",
                    503,
                )
            previous_month = month
            validated_rows.append(
                {
                    "horizon_months": expected_step,
                    "forecast_month": month,
                    "forecast_price": price,
                    "status": "recursive_forecast",
                }
            )
        result = {
            "origin_month": str(payload.get("origin_month") or ""),
            "transformation": str(payload.get("transformation") or ""),
            "forecast": validated_rows,
            "method": str(payload.get("method") or "recursive; predicted prices fed back; no actual future prices"),
        }
        self._lstm_forecast_cache[cache_key] = result
        return {**result, "forecast": result["forecast"][:horizon]}

    def forecast(self, dataset: str, series_id: str, model: str, horizon: int) -> dict[str, Any]:
        series = self._validate_forecast_request(dataset, series_id, model, horizon)
        champion_record = None
        if dataset == "world_bank":
            champion_record = self._world_bank_champions.get("records", {}).get(series_id)
            if not champion_record or champion_record.get("selected_model") != model:
                raise DashboardError(
                    "model_unavailable",
                    "Only the registered World Bank production champion can be forecast.",
                    422,
                )
        rows = self._rows_for_series(dataset, series_id)
        observed = [row for row in rows if row["observed"] and self._price(row) is not None]
        if not observed:
            raise DashboardError(
                "missing_history",
                "No observed historical prices are available for this series.",
                422,
            )
        last_month = max((row["month"] for row in observed), key=month_index)

        if model == "naive_last_observation":
            last_price = float(self._price(max(observed, key=lambda row: month_index(row["month"]))))
            forecast_rows = [
                {
                    "horizon_months": step,
                    "forecast_month": self._month_add(last_month, step),
                    "forecast_price": last_price,
                    "status": "recursive_forecast",
                }
                for step in range(1, horizon + 1)
            ]
            transformation = "none"
            recursive_method = "naive persistence from the last observed calendar month"
        elif dataset == "aggregate" and model == "lstm":
            inference = self._lstm_forecast(series_id, horizon)
            if inference["origin_month"] and inference["origin_month"] != last_month:
                raise DashboardError(
                    "inference_failed",
                    "The LSTM forecast origin does not match the latest observed month.",
                    503,
                )
            transformation = inference["transformation"]
            recursive_method = (
                "src.lstm_dashboard_inference using the saved TensorFlow/Keras model; "
                "predicted prices are fed back recursively and actual future prices are not used"
            )
            forecast_rows = inference["forecast"]
        elif dataset == "market":
            path = _model_path(dataset, series_id, model)
            predictor, metadata = _load_artifact(path)
            from .forecasting import recursive_forecast

            forecast_rows = recursive_forecast(
                predictor,
                rows,
                series_id,
                model,
                horizons=tuple(range(1, horizon + 1)),
            )
            if not any(row["status"] == "recursive_forecast" for row in forecast_rows):
                raise DashboardError(
                    "missing_calendar_history",
                    "The selected market series does not have the required recent calendar history for recursive forecasting.",
                    422,
                )
            transformation = "original"
            recursive_method = "src.forecasting.recursive_forecast; calendar lags and rolling features; future prices are predicted recursively"
            forecast_rows = [
                {
                    "horizon_months": row["horizon_months"],
                    "forecast_month": row["forecast_month"],
                    "forecast_price": row["forecast_price"],
                    "status": row["status"],
                }
                for row in forecast_rows
            ]
        elif dataset == "world_bank":
            artifact = champion_record.get("artifact_path") if champion_record else None
            if not artifact:
                raise DashboardError("model_unavailable", "The registered World Bank champion does not have a usable artifact.", 422)
            path = (ROOT / str(artifact)).resolve()
            if ROOT not in path.parents:
                raise DashboardError("model_unavailable", "The registered World Bank artifact path is invalid.", 503)
            predictor, metadata = _load_artifact(path)
            _validate_world_bank_champion_artifact(
                metadata,
                series_id,
                champion_record,
            )
            from .world_bank_training import _recursive_forecast as world_bank_recursive_forecast

            transformation = str(champion_record["selected_transformation"])
            forecast_rows = world_bank_recursive_forecast(
                predictor,
                rows,
                transformation,
                horizons=tuple(range(1, horizon + 1)),
            )
            if not any(row["status"] == "recursive_forecast" for row in forecast_rows):
                raise DashboardError(
                    "missing_calendar_history",
                    "The selected World Bank series does not have the required calendar history for recursive forecasting.",
                    422,
                )
            recursive_method = "src.world_bank_training recursive close-estimate adapter; predicted prices are fed back and actual future prices are not used"
            forecast_rows = [
                {
                    "horizon_months": row["horizon_months"],
                    "forecast_month": row["forecast_month"],
                    "forecast_price": row["forecast_price"],
                    "status": row["status"],
                }
                for row in forecast_rows
            ]
        else:
            path = _model_path(dataset, series_id, model)
            predictor, metadata = _load_artifact(path)
            from .final_validation import _recursive_price_forecast

            observed_months = [row["month"] for row in observed]
            if not _month_contiguous(observed_months, 3):
                raise DashboardError(
                    "missing_calendar_history",
                    "The latest three observed months are not calendar-contiguous; this model requires complete recent calendar history for recursive features.",
                    422,
                )
            transformation = str(metadata.get("transformation", "original"))
            history = [float(self._price(row)) for row in sorted(observed, key=lambda row: month_index(row["month"]))]
            predictions = _recursive_price_forecast(
                predictor,
                transformation,
                history,
                last_month,
                horizon,
            )
            forecast_rows = [
                {
                    "horizon_months": step,
                    "forecast_month": self._month_add(last_month, step),
                    "forecast_price": float(value),
                    "status": "recursive_forecast",
                }
                for step, value in enumerate(predictions, start=1)
            ]
            recursive_method = "src.final_validation recursive transformed-target adapter; actual future prices are not used"

        forecast_metadata = {
            "dataset": dataset,
            "series_id": series_id,
            "commodity": series.get("commodity"),
            "market": series.get("market"),
            "state": series.get("state"),
            "unit": series.get("unit"),
            "price_type": series.get("price_type"),
            "model": model,
            "transformation": transformation,
            "recursive_method": recursive_method,
            "horizon_months": horizon,
            "origin_month": last_month,
            "forecast_origin_month": last_month,
            "latest_observation_date": series.get("latest_observation_date"),
            "latest_actual_price": series.get("latest_actual_price"),
            "latest_estimate_price": series.get("latest_estimate_price"),
            "latest_low_estimate": series.get("latest_low_estimate"),
            "latest_high_estimate": series.get("latest_high_estimate"),
            "latest_trust_score": series.get("latest_trust_score"),
            "base_observation_price": series.get("base_observation_price"),
            "source_data_is_direct_observation": series.get("source_data_is_direct_observation"),
            "is_model_estimate": series.get("is_model_estimate", False),
            "price_value_type": series.get("price_value_type"),
            "freshness_status": series.get("freshness_status"),
            "observation_age_months": series.get("observation_age_months"),
        }
        if champion_record is not None:
            forecast_metadata["champion_selection"] = {
                "selected_model": champion_record.get("selected_model"),
                "selected_transformation": champion_record.get("selected_transformation"),
                "validation_metrics": champion_record.get("validation_metrics"),
                "validation_period": champion_record.get("validation_period"),
                "selection_criterion": champion_record.get("selection_criterion"),
                "experiment_version": champion_record.get("experiment_version"),
                "selection_timestamp": champion_record.get("selection_timestamp"),
            }
            forecast_metadata["source_data_is_direct_observation"] = False
            forecast_metadata["base_observation_price"] = series.get("base_observation_price")
        freshness_warning = None
        if series.get("freshness_status") in {"aging", "stale"}:
            freshness_warning = (
                f"Warning: This forecast is based on historical data ending "
                f"{series.get('latest_observation_date') or last_month}. "
                "It may not reflect current market conditions."
            )
        return {
            "dataset": dataset,
            "series": series,
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "transformation": transformation,
            "recursive_method": recursive_method,
            "horizon_months": horizon,
            "origin_month": last_month,
            "forecast_origin_month": last_month,
            "latest_observation_date": series.get("latest_observation_date"),
            "latest_actual_price": series.get("latest_actual_price"),
            "latest_estimate_price": series.get("latest_estimate_price"),
            "latest_low_estimate": series.get("latest_low_estimate"),
            "latest_high_estimate": series.get("latest_high_estimate"),
            "latest_trust_score": series.get("latest_trust_score"),
            "base_observation_price": series.get("base_observation_price"),
            "source_data_is_direct_observation": series.get("source_data_is_direct_observation"),
            "is_model_estimate": series.get("is_model_estimate", False),
            "price_value_type": series.get("price_value_type"),
            "freshness_status": series.get("freshness_status"),
            "observation_age_months": series.get("observation_age_months"),
            "champion_selection": forecast_metadata.get("champion_selection"),
            "freshness_explanation": series.get("freshness_explanation"),
            "freshness_warning": freshness_warning,
            "metadata": forecast_metadata,
            "historical": [
                {
                    "month": row["month"],
                    "date": row.get("date"),
                    "price": self._price(row),
                    "observed": bool(row["observed"]),
                }
                for row in rows
            ],
            "forecast": forecast_rows,
            "confidence_intervals": None,
            "confidence_interval_note": "Confidence intervals have not been implemented or validated for these sparse series.",
        }

    @staticmethod
    def _month_add(month: str, offset: int) -> str:
        return month_from_index(month_index(month) + offset)


def create_dashboard_service() -> DashboardService:
    return DashboardService()
