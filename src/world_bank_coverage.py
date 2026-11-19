"""Retired coverage analysis for the historical World Bank release."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .world_bank_ingestion import NORMALIZED_FILE

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "world_bank"


def write_coverage_reports() -> tuple[pd.DataFrame, str]:
    raise RuntimeError(
        "This coverage pipeline is retired. The active data audit covers WFP Nigeria and NBS."
    )
    frame = pd.read_csv(NORMALIZED_FILE)
    keys = ["commodity_ticker", "commodity", "unit", "price_type", "geography_level", "geography_name"]
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        months = pd.PeriodIndex(group["month"], freq="M")
        expected = pd.period_range(months.min(), months.max(), freq="M")
        present = set(months)
        missing = [str(month) for month in expected if month not in present]
        monthly_locations = group.groupby("month")["geo_id"].nunique()
        rows.append({
            **dict(zip(keys, key)),
            "observations": len(group),
            "first_month": str(months.min()),
            "last_month": str(months.max()),
            "unique_months": len(present),
            "missing_month_count": len(missing),
            "missing_months": ";".join(missing),
            "calendar_coverage_pct": round(100 * len(present) / len(expected), 3),
            "locations_per_month_min": int(monthly_locations.min()),
            "locations_per_month_max": int(monthly_locations.max()),
            "locations_per_month_average": round(float(monthly_locations.mean()), 3),
            "is_complete_calendar_series": not missing,
        })
    coverage = pd.DataFrame(rows)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(REPORT_ROOT / "coverage.csv", index=False)
    levels = frame.groupby("geography_level")["geography_name"].nunique().to_dict()
    market_labels = sorted(frame.loc[frame["geography_level"] == "market", "geography_name"].dropna().unique())
    state_labels = sorted(frame.loc[frame["geography_level"] == "market", "state"].dropna().unique())
    summary = "\n".join(
        [
            "AgriPrice-NG World Bank coverage summary",
            "==========================================",
            "",
            f"Normalized rows: {len(frame):,}",
            f"Date range: {frame['date'].min()} to {frame['date'].max()}",
            f"Commodity tickers: {frame['commodity_ticker'].nunique()}",
            f"Normalized commodity/location series: {len(coverage):,}",
            f"Geography labels by level: {levels}",
            f"Actual market labels ({len(market_labels)}): {', '.join(market_labels)}",
            f"State labels attached to markets ({len(state_labels)}): {', '.join(state_labels)}",
            "The accompanying World Bank details metadata reports 73 markets modelled/covered; this normalized row classification identifies 67 actual market labels plus aggregate/geopolitical records.",
            "Aggregate labels (National Average, Market Average and Geopolitical Zone) are not counted as actual markets or states.",
            "The normalized values use c_ close estimates. Low/high estimates and trust scores are retained.",
            "Missing c_ rows are omitted; no interpolation or forward fill is applied.",
            "The first trained panel uses six commodities at National Average and Market Average (12 series).",
        ]
    ) + "\n"
    (REPORT_ROOT / "coverage_summary.txt").write_text(summary, encoding="utf-8")
    return coverage, summary


if __name__ == "__main__":
    coverage, _ = write_coverage_reports()
    print(f"World Bank coverage rows: {len(coverage):,}")
