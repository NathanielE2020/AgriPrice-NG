"""Generate a full per-series WFP/NBS coverage and forecast-attempt audit.

Run with ``python -m src.commodity_coverage_audit``. The audit reads the
normalized source tables, performs the same real forecast request as the Flask
API, and writes only new audit reports. Runtime model caches go to the OS temp
directory and are not research model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .pooled_forecasting import assess_pooled_group, compatibility_key
from .wfp_nbs_dashboard_service import DashboardError, DashboardService, ROOT, SUPPORTED_DATASETS


CSV_OUTPUT = ROOT / "reports" / "final_commodity_coverage_audit.csv"
TXT_OUTPUT = ROOT / "reports" / "final_commodity_coverage_audit.txt"


def _valid_month(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m")
        return len(value) == 7
    except (TypeError, ValueError):
        return False


def _month_index(value: str) -> int:
    year, month = (int(part) for part in value.split("-"))
    return year * 12 + month - 1


def _coverage(history: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [
        row for row in history
        if _valid_month(str(row.get("month") or ""))
        and row.get("price") is not None
    ]
    months = sorted({str(row["month"]) for row in valid_rows}, key=_month_index)
    if not months:
        return {
            "reported_observations": 0, "first_month": "", "latest_month": "",
            "first_observation_date": "", "latest_observation_date": "",
            "calendar_month_span": 0, "missing_calendar_months": 0,
            "longest_missing_month_run": 0,
        }
    first, latest = months[0], months[-1]
    span = _month_index(latest) - _month_index(first) + 1
    longest = 0
    run = 0
    observed = set(months)
    for ordinal in range(_month_index(first), _month_index(latest) + 1):
        month = f"{ordinal // 12:04d}-{ordinal % 12 + 1:02d}"
        if month in observed:
            longest = max(longest, run)
            run = 0
        else:
            run += 1
    longest = max(longest, run)
    by_month = {str(row["month"]): row for row in valid_rows}
    first_row, last_row = by_month[first], by_month[latest]
    first_date = first_row.get("first_observation_date") or first
    latest_date = last_row.get("last_observation_date") or last_row.get("source_report_date") or latest
    return {
        "reported_observations": len(months),
        "first_month": first,
        "latest_month": latest,
        "first_observation_date": first_date,
        "latest_observation_date": latest_date,
        "calendar_month_span": span,
        "missing_calendar_months": span - len(months),
        "longest_missing_month_run": longest,
    }


def _place(record: dict[str, Any], source: str) -> str:
    if source == "WFP":
        return str(record.get("market") or "")
    return str(record.get("geography_name") or "")


def _record_fields(record: dict[str, Any], source: str) -> dict[str, Any]:
    coverage = _coverage(record.get("history", []))
    return {
        "source": source,
        "series_id": record.get("series_id", ""),
        "commodity": record.get("commodity", ""),
        "market_or_geography": _place(record, source),
        "geography_level": record.get("geography_level", "market" if source == "WFP" else ""),
        "state": record.get("state", ""),
        "unit": record.get("unit", ""),
        "price_type": record.get("price_type", ""),
        "price_flag": record.get("price_flag", ""),
        "methodology_regime": record.get("methodology_regime", ""),
        **coverage,
    }


def _direct_eligibility(record: dict[str, Any], source: str) -> dict[str, Any]:
    from .ondemand_forecasting import assess_nbs_series, assess_wfp_series

    if source == "WFP" and record.get("price_flag") != "actual":
        return {"eligible": False, "code": "not_direct_actual", "reason": "WFP price flag is not exactly actual; retained for history but not modeled as direct market observations."}
    return assess_wfp_series(record) if source == "WFP" else assess_nbs_series(record)


def run_coverage_audit(as_of: date | None = None) -> dict[str, Any]:
    as_of = as_of or date.today()
    cache = Path(tempfile.gettempdir()) / "agriprice-ng" / "coverage-audit-runtime-models"
    service = DashboardService(reference_date=as_of, forecast_cache_dir=cache)
    from app import create_app

    client = create_app(service).test_client()
    pool_groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for source_key in SUPPORTED_DATASETS:
        source_label = source_key.upper()
        for record in service._series[source_key]:
            if source_key == "wfp" and record.get("price_flag") != "actual":
                continue
            pool_groups[(source_key, compatibility_key(record, source_key))].append(record)

    pooled_results: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    print(f"Assessing {len(pool_groups)} source-compatible groups before API attempts...")
    for index, ((source_key, key), records) in enumerate(pool_groups.items(), start=1):
        result = assess_pooled_group(records[0], records, source_key, cache, as_of=as_of)
        pooled_results[(source_key, key)] = {
            field: result.get(field)
            for field in (
                "eligible", "code", "reason", "selected_model", "validation_scores",
                "series_count", "group_count", "training_rows", "validation_rows",
                "validation_group_count", "validation_cutoff_month",
            )
        }
        if index % 20 == 0 or index == len(pool_groups):
            print(f"  pooled groups assessed: {index}/{len(pool_groups)}")

    from .ondemand_forecasting import ForecastingError

    attempts: list[dict[str, Any]] = []
    total_series = sum(len(service._series[source]) for source in SUPPORTED_DATASETS)
    completed_series = 0
    print(f"Submitting one-month forecast attempts for {total_series} exact series...")
    for source_key in SUPPORTED_DATASETS:
        source_label = source_key.upper()
        for record in service._series[source_key]:
            row = _record_fields(record, source_label)
            direct = _direct_eligibility(record, source_label)
            key = compatibility_key(record, source_key)
            pool = pooled_results.get((source_key, key)) if (source_key != "wfp" or record.get("price_flag") == "actual") else None
            row.update({
                "individual_series_model_eligible": bool(direct.get("eligible")),
                "individual_series_eligibility_reason": direct.get("reason", ""),
                "individual_usable_supervised_rows": direct.get("usable_training_rows", 0),
                "pooled_global_model_eligible": bool(pool and pool.get("eligible")),
                "pooled_global_model": pool.get("selected_model") if pool else "",
                "pooled_validation_rmse": (pool.get("validation_scores", {}).get(pool.get("selected_model"), {}) or {}).get("rmse") if pool and pool.get("eligible") else "",
                "pooled_validation_observations": pool.get("validation_rows", 0) if pool else 0,
                "pooled_eligibility_reason": pool.get("reason", "not applicable: non-actual WFP price flag") if pool else "not applicable: non-actual WFP price flag" if source_key == "wfp" else "no compatible pooled group",
                "forecast_attempt_status": "not_attempted_no_reported_price" if row["reported_observations"] == 0 else "pending",
                "selected_model_or_baseline": "",
                "selected_strategy": "",
                "validation_rmse": "",
                "validation_mae": "",
                "validation_mape": "",
                "validation_r2": "",
                "validation_observations": "",
                "selected_series_validation_rmse": "",
                "selected_series_validation_mae": "",
                "selected_series_validation_mape": "",
                "selected_series_validation_r2": "",
                "selected_series_validation_observations": "",
                "pooled_training_series_count": "",
                "pooled_training_rows": "",
                "pooled_contributing_series": "",
                "forecast_available": False,
                "forecast_failure_code": "",
                "forecast_failure_reason": "",
            })
            if row["reported_observations"]:
                try:
                    response = client.get("/api/forecast", query_string={
                        "dataset": source_key, "series_id": record["series_id"],
                        "model": "auto", "mode": "on_demand", "horizon": "1",
                    })
                    payload = response.get_json(silent=True) or {}
                    if response.status_code != 200:
                        error = payload.get("error") or {}
                        row.update({
                            "forecast_attempt_status": "unavailable" if response.status_code < 500 else "error",
                            "forecast_failure_code": error.get("code", "http_error"),
                            "forecast_failure_reason": error.get("message", f"HTTP {response.status_code}"),
                        })
                        attempts.append(row)
                        completed_series += 1
                        if completed_series % 100 == 0 or completed_series == total_series:
                            print(f"  exact series attempted: {completed_series}/{total_series}")
                        continue
                    result = payload
                    metadata = result.get("metadata", {})
                    metrics = metadata.get("validation_metrics") or {}
                    selected_metrics = metadata.get("selected_series_validation_metrics") or (
                        metrics if metadata.get("strategy_level") == "individual_series" else {}
                    )
                    row.update({
                        "forecast_attempt_status": "generated",
                        "selected_model_or_baseline": result.get("model", ""),
                        "selected_strategy": metadata.get("strategy_level", "individual_series"),
                        "validation_rmse": metrics.get("rmse", ""),
                        "validation_mae": metrics.get("mae", ""),
                        "validation_mape": metrics.get("mape", ""),
                        "validation_r2": metrics.get("r2", ""),
                        "validation_observations": metrics.get("n", ""),
                        "selected_series_validation_rmse": selected_metrics.get("rmse", ""),
                        "selected_series_validation_mae": selected_metrics.get("mae", ""),
                        "selected_series_validation_mape": selected_metrics.get("mape", ""),
                        "selected_series_validation_r2": selected_metrics.get("r2", ""),
                        "selected_series_validation_observations": selected_metrics.get("n", ""),
                        "pooled_training_series_count": metadata.get("training_contributing_series_count", metadata.get("training_series_count", "")),
                        "pooled_training_rows": metadata.get("training_observation_rows", metadata.get("final_model_fit_rows", "")),
                        "pooled_contributing_series": json.dumps(metadata.get("other_series_contributors", []), ensure_ascii=False, sort_keys=True),
                        "forecast_available": bool(result.get("forecast")),
                        "forecast_failure_reason": "",
                    })
                except DashboardError as exc:
                    row.update({
                        "forecast_attempt_status": "unavailable",
                        "forecast_failure_code": exc.code,
                        "forecast_failure_reason": exc.message,
                    })
                except ForecastingError as exc:
                    row.update({
                        "forecast_attempt_status": "unavailable",
                        "forecast_failure_code": exc.code,
                        "forecast_failure_reason": exc.message,
                    })
                except Exception as exc:  # noqa: BLE001 - retained as an audit row, never swallowed silently.
                    row.update({
                        "forecast_attempt_status": "error",
                        "forecast_failure_code": type(exc).__name__,
                        "forecast_failure_reason": str(exc),
                    })
            attempts.append(row)
            completed_series += 1
            if completed_series % 100 == 0 or completed_series == total_series:
                print(f"  exact series attempted: {completed_series}/{total_series}")

    if CSV_OUTPUT.exists() or TXT_OUTPUT.exists():
        raise FileExistsError("Coverage audit outputs already exist; choose a new versioned filename rather than overwriting research reports.")
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(attempts[0]) if attempts else []
    with CSV_OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(attempts)

    lines = [
        "AgriPrice-NG full WFP/NBS series coverage and forecast-attempt audit",
        f"As-of date: {as_of.isoformat()}",
        "",
        "Method",
        "------",
        "Every normalized dashboard series was included. Each series with at least one reported value was submitted through the same service forecast method used by the Flask API for a one-month on-demand forecast.",
        "WFP pooling was limited to pure priceflag=actual and the exact commodity/unit/price-type/currency; aggregate and mixed-price-flag histories were audited but rejected for forecasting.",
        "NBS pooling was limited to the exact original item label, methodology regime, geography level, unit, price type and currency. National records were not pooled with zones, and no NBS records were pooled with WFP.",
        "Pooled strategies use group-held-out plus chronological validation. Predictions are recursively future-dated; no target prices were interpolated or forward-filled.",
        "",
        "Totals by source",
        "----------------",
    ]
    for source in ("WFP", "NBS"):
        rows = [row for row in attempts if row["source"] == source]
        status = Counter(row["forecast_attempt_status"] for row in rows)
        models = Counter(row["selected_model_or_baseline"] for row in rows if row["selected_model_or_baseline"])
        lines.append(
            f"{source}: {len(rows)} exact series; {len({row['commodity'] for row in rows})} source labels; "
            f"generated={status['generated']}; unavailable={status['unavailable']}; errors={status['error']}; "
            f"no reported price={status['not_attempted_no_reported_price']}."
        )
        lines.append("  Selected models/baselines: " + (", ".join(f"{name}={count}" for name, count in sorted(models.items())) or "none"))
    for flag in sorted({row["price_flag"] for row in attempts if row["source"] == "WFP"}):
        flagged = [row for row in attempts if row["source"] == "WFP" and row["price_flag"] == flag]
        latest = max((row["latest_observation_date"] for row in flagged if row["latest_observation_date"]), default="none")
        generated = sum(row["forecast_attempt_status"] == "generated" for row in flagged)
        lines.append(
            f"WFP priceflag={flag}: {len(flagged)} exact series; latest reported observation={latest}; "
            f"future forecast attempts generated={generated}."
        )
    lines.extend(["", "Totals by source commodity/item", "-------------------------------"])
    for source, commodity in sorted({(row["source"], row["commodity"]) for row in attempts}):
        rows = [row for row in attempts if row["source"] == source and row["commodity"] == commodity]
        generated = sum(row["forecast_attempt_status"] == "generated" for row in rows)
        eligible_individual = sum(bool(row["individual_series_model_eligible"]) for row in rows)
        eligible_pooled = sum(bool(row["pooled_global_model_eligible"]) for row in rows)
        lines.append(f"{source} | {commodity}: series={len(rows)}, individual eligible={eligible_individual}, pooled eligible={eligible_pooled}, forecast generated={generated}.")
    lines.extend([
        "",
        "Interpretation and limits",
        "--------------------------",
        "A generated forecast indicates that a defined validation procedure selected a model or baseline; it does not mean that the estimate is reliable or current. Check each row's latest actual/report date, missing-month run, validation sample count, metrics, model strategy and freshness in the CSV.",
        "A selected naive baseline is explicitly identified and may produce a flat estimate if it wins validation. Negative R-squared indicates the selected predictions were worse than predicting the validation mean under the usual R-squared definition; it does not mean a negative price.",
        "Pooled validation metrics summarize held-out compatible markets/geographies. The selected series receives its own metrics where it contributes validation rows; very short series may have none and inherit only the documented peer-pool validation evidence.",
        "Historical analysis remains independent of forecast eligibility and continues to show reported observations with missing months left missing.",
        "",
        f"Detailed CSV: {CSV_OUTPUT.relative_to(ROOT)}",
        f"CSV rows written: {len(attempts)}",
    ])
    TXT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(attempts), "csv": CSV_OUTPUT, "summary": TXT_OUTPUT, "as_of": as_of}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all exact WFP and NBS series and make real one-month forecast attempts.")
    parser.add_argument("--as-of", help="Freeze forecast date as YYYY-MM-DD; defaults to the local system date.")
    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    result = run_coverage_audit(as_of)
    print(f"Wrote {result['rows']} exact-series rows to {result['csv']} and {result['summary']}.")


if __name__ == "__main__":
    main()

