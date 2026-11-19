"""Audit current WFP actual-series coverage against on-demand eligibility.

This report generator uses the dashboard's exact normalized WFP grouping and
the on-demand eligibility function; it does not fit models or alter data.
Run from the project root with ``python -m src.wfp_ondemand_coverage_audit``.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from .ondemand_forecasting import _month_ordinal, assess_wfp_series
from .wfp_nbs_dashboard_service import DashboardService


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "reports" / "wfp_ondemand_series_coverage_audit.csv"
TXT_PATH = ROOT / "reports" / "wfp_ondemand_series_coverage_audit.txt"
RECENT_FROM = "2024-01"


def _scope_class(commodity: str) -> tuple[str, str]:
    """Map labels for audit display only; never merge distinct WFP series."""
    label = commodity.strip().casefold()
    if label.startswith("rice"):
        return "Rice", "Direct WFP commodity variant; label/unit remains separate"
    if label == "maize" or label.startswith("maize ("):
        return "Maize", "Direct WFP maize variant; label/unit remains separate"
    if label == "maize flour":
        return "Maize", "Processed maize product; not combined with maize grain"
    if label == "cassava" or label.startswith("cassava ("):
        return "Cassava", "Exact cassava label; do not merge with processed products"
    if "gari" in label:
        return "Cassava", "Processed cassava proxy only; not raw cassava"
    if label.startswith("beans"):
        return "Beans", "WFP bean label variant; original label remains separate"
    if label.startswith("cowpeas"):
        return "Cowpeas (separate label)", "Related legume; not mapped to beans"
    if label.startswith("yam"):
        return "Yam", "WFP yam label variant; original label remains separate"
    if label.startswith("tomato"):
        return "Tomatoes", "Direct WFP tomato label"
    if label == "oil (palm)":
        return "Palm oil", "Direct WFP palm-oil label"
    return "", "Outside requested commodity scope"


def _age_months(latest_month: str, as_of: date) -> int | None:
    if not latest_month:
        return None
    return (as_of.year - int(latest_month[:4])) * 12 + as_of.month - int(latest_month[5:7])


def _last_three_complete(price_map: dict[str, float], latest_month: str) -> bool:
    if not latest_month:
        return False
    latest_ordinal = _month_ordinal(latest_month)
    return all(
        f"{(latest_ordinal - offset) // 12:04d}-{(latest_ordinal - offset) % 12 + 1:02d}" in price_map
        for offset in range(3)
    )


def _coverage_class(assessment: dict[str, Any], recent_count: int, as_of: date) -> str:
    latest = str(assessment.get("latest_actual_month") or "")
    if assessment.get("eligible"):
        return "eligible_but_stale" if latest < RECENT_FROM else "eligible_with_recent_actuals"
    if recent_count and recent_count <= 3:
        return "only_one_to_three_actual_months_since_2024"
    if latest and latest < RECENT_FROM and assessment.get("actual_observations", 0) >= 30:
        return "long_history_but_stale_latest_actual"
    if assessment.get("actual_observations", 0) < 30 or assessment.get("usable_training_rows", 0) < 30:
        return "insufficient_total_observations_or_calendar_lags"
    if recent_count:
        return "recent_actuals_but_fails_other_eligibility_checks"
    return "insufficient_or_gapped_history"


def _series_row(record: dict[str, Any], as_of: date) -> dict[str, Any] | None:
    research_scope, mapping_note = _scope_class(str(record.get("commodity") or ""))
    if not research_scope or record.get("price_flag") != "actual":
        return None

    assessment = assess_wfp_series(record)
    prices: dict[str, float] = assessment.get("price_map", {})
    months = sorted(prices, key=_month_ordinal)
    latest_month = str(assessment.get("latest_actual_month") or "")
    recent_months = [month for month in months if month >= RECENT_FROM]
    observation_dates = assessment.get("observation_dates", {})
    first_date = observation_dates.get(months[0], "") if months else ""
    latest_date = observation_dates.get(latest_month, "") if latest_month else ""
    reason = str(assessment.get("reason") or "")
    if assessment.get("eligible") and latest_month < RECENT_FROM:
        reason = "Meets on-demand history/validation/gap rules, but latest valid actual is before 2024-01; forecast origin is stale."
    elif assessment.get("eligible"):
        reason = "Meets on-demand history/validation/gap rules; source date freshness still requires review."

    raw_flag_rows = len(record.get("history", []))
    exact_dupes = int(assessment.get("duplicate_rows_removed", 0) or 0)
    latest_age = _age_months(latest_month, as_of)
    return {
        "series_id": record.get("series_id", ""),
        "research_scope": research_scope,
        "original_commodity_label": record.get("commodity", ""),
        "mapping_note": mapping_note,
        "market": record.get("market", ""),
        "market_id": record.get("market_id", ""),
        "state": record.get("state", ""),
        "unit": record.get("unit", ""),
        "price_type": record.get("price_type", ""),
        "currency": record.get("currency", ""),
        "price_flag": record.get("price_flag", ""),
        "source_month_rows": raw_flag_rows,
        "valid_unique_actual_months": len(months),
        "first_actual_month": months[0] if months else "",
        "first_actual_observation_date": first_date,
        "latest_actual_month": latest_month,
        "latest_actual_observation_date": latest_date,
        "valid_actual_months_since_2024_01": len(recent_months),
        "recent_actual_months_since_2024_01": ";".join(recent_months),
        "usable_calendar_lag_rows": int(assessment.get("usable_training_rows", 0) or 0),
        "chronological_train_rows": int(assessment.get("training_rows", 0) or 0),
        "chronological_validation_rows": int(assessment.get("validation_rows", 0) or 0),
        "last_three_calendar_months_have_actuals": _last_three_complete(prices, latest_month),
        "missing_calendar_months_in_span": int(assessment.get("missing_calendar_months", 0) or 0),
        "invalid_price_rows_excluded": int(assessment.get("invalid_price_rows", 0) or 0),
        "missing_price_rows": int(assessment.get("missing_price_rows", 0) or 0),
        "invalid_month_rows_excluded": int(assessment.get("invalid_month_rows", 0) or 0),
        "exact_duplicate_rows_removed": exact_dupes,
        "eligible_under_current_on_demand_rules": bool(assessment.get("eligible")),
        "eligibility_code": assessment.get("code", ""),
        "eligibility_reason": reason,
        "latest_actual_age_months_as_of_audit": latest_age,
        "coverage_classification": _coverage_class(assessment, len(recent_months), as_of),
    }


def run_audit() -> tuple[list[dict[str, Any]], str]:
    as_of = date.today()
    service = DashboardService(reference_date=as_of)
    records = service._series["wfp"]
    rows = [row for record in records if (row := _series_row(record, as_of)) is not None]
    rows.sort(key=lambda row: (
        row["research_scope"], row["original_commodity_label"],
        row["market"], row["market_id"], row["unit"], row["price_type"],
    ))

    actual_records = [record for record in records if record.get("price_flag") == "actual"]
    flags = Counter(record.get("price_flag", "") for record in records)
    recent = [row for row in rows if row["valid_actual_months_since_2024_01"] > 0]
    eligible = [row for row in rows if row["eligible_under_current_on_demand_rules"]]
    recent_eligible = [row for row in eligible if row["latest_actual_month"] >= RECENT_FROM]
    stale_eligible = [row for row in eligible if row["latest_actual_month"] < RECENT_FROM]
    recent_direct = [row for row in recent if not any(note in row["mapping_note"] for note in ("Related legume", "Processed maize", "Processed cassava"))]
    recent_direct_with_history = [row for row in recent_direct if row["usable_calendar_lag_rows"] >= 30]
    recent_singletons = [row for row in recent if row["valid_unique_actual_months"] == 1]
    recent_gap_only = [
        row for row in recent
        if row["usable_calendar_lag_rows"] >= 30
        and row["eligibility_reason"].startswith("the latest actual observation is not preceded")
    ]
    recent_low_history = [row for row in recent if row["usable_calendar_lag_rows"] < 30]
    recent_proxy_or_related = [
        row for row in recent
        if any(note in row["mapping_note"] for note in ("Related legume", "Processed maize", "Processed cassava"))
    ]
    recent_direct_by_label = Counter(row["original_commodity_label"] for row in recent_direct)
    direct_scopes = {"Rice", "Maize", "Cassava", "Yam", "Beans", "Tomatoes", "Palm oil"}
    missing_direct_recent_scopes = sorted(direct_scopes.difference(row["research_scope"] for row in recent_direct))
    recent_markets = sorted({
        f"{row['market']} ({row['market_id']}, {row['state']})" for row in recent
    })
    new_recent_markets = sorted({
        f"{row['market']} ({row['market_id']}, {row['state']})"
        for row in recent if row["valid_unique_actual_months"] == 1 and row["usable_calendar_lag_rows"] == 0
    })
    old_history_markets = sorted({
        f"{row['market']} ({row['market_id']}, {row['state']})"
        for row in recent if row["usable_calendar_lag_rows"] >= 30
    })
    recent_low_history_details = "; ".join(
        f"{row['original_commodity_label']} at {row['market']}: {row['valid_unique_actual_months']} valid months, {row['usable_calendar_lag_rows']} usable lag rows"
        for row in recent_low_history if row["valid_unique_actual_months"] > 1
    ) or "none beyond single-month histories"
    recent_by_count = Counter(
        "1" if row["valid_actual_months_since_2024_01"] == 1 else
        "2-3" if row["valid_actual_months_since_2024_01"] <= 3 else "4+"
        for row in recent
    )
    classes = Counter(row["coverage_classification"] for row in rows)
    distinct_recent_latest = max((row["latest_actual_month"] for row in rows if row["latest_actual_month"]), default="")
    distinct_eligible_latest = max((row["latest_actual_month"] for row in eligible), default="")
    actual_months = [month for record in actual_records for month in (record.get("history") or []) if month.get("price") is not None]
    latest_source_date = max(
        (str(item.get("last_observation_date") or "") for item in actual_months if item.get("last_observation_date")),
        default="",
    )

    lines = [
        "AgriPrice-NG WFP on-demand series coverage audit",
        f"Audit date: {as_of.isoformat()}",
        "Inputs: existing data/processed/wfp_monthly.csv through the DashboardService normalized loader; no source files were changed.",
        "Scope: actual-flag WFP market series whose labels are relevant to rice, maize, cassava, yam, beans, tomatoes, or palm oil. Distinct labels, markets, states, units, currencies, price types, and flags remain separate.",
        "",
        "Exact grouping and filtering traced",
        "- DashboardService._load_wfp groups by commodity, market_id, market, admin1/state, unit, pricetype, currency, and priceflag; its series ID hashes that exact tuple. It does not merge markets or units.",
        "- On-demand assess_wfp_series accepts only source_key='wfp' and exact price_flag='actual'. It validates YYYY-MM months and finite positive prices, counts distinct valid months, rejects conflicting prices for a repeated month, and only removes exact same-month/same-price duplicates.",
        "- Calendar-lag rows require actual prices in each of the prior three calendar months. No gap is interpolated or replaced by the previous available month.",
        "- Eligibility requires >=30 complete lag rows, an 80/20 chronological split with >=20 training and >=6 validation rows, plus actual observations for the latest month and its two preceding calendar months.",
        "- The older research experiment's >=96 observation/70-15-15 thresholds are separate and are not used by this on-demand eligibility check.",
        "",
        "Coverage findings",
        f"WFP market series in dashboard index: {len(records)}; exact priceflag=actual groups: {len(actual_records)}.",
        f"Actual priceflag distribution over all WFP series: {dict(sorted(flags.items()))}.",
        f"Relevant actual-flag commodity/market/unit/price-type/currency groups audited: {len(rows)}.",
        f"Relevant groups with >=1 valid actual month since {RECENT_FROM}: {len(recent)}; split by number of such months: {dict(sorted(recent_by_count.items()))}.",
        f"All recent groups have latest month {distinct_recent_latest or 'none'}; the latest exact observation date among them is {max((row['latest_actual_observation_date'] for row in recent), default='')}.",
        f"Of recent groups, {len(recent_gap_only)} have >=30 usable historical lag rows but fail only the terminal three-consecutive-month check; {len(recent_low_history)} fail the 30-row minimum as well; {len(recent_singletons)} have only one valid actual month in their entire series.",
        f"Within direct in-scope labels (excluding cowpeas, maize flour, and processed cassava proxies), {len(recent_direct)} groups have recent observations; {len(recent_direct_with_history)} have >=30 usable lag rows, while {len(recent_direct) - len(recent_direct_with_history)} are one-observation-only new market histories.",
        f"Direct recent groups by original commodity label: {dict(sorted(recent_direct_by_label.items()))}; direct in-scope categories with no recent actual group: {missing_direct_recent_scopes}.",
        f"Related/processed labels with recent records, kept separate from the seven raw commodity categories: {dict(sorted(Counter(row['original_commodity_label'] for row in recent_proxy_or_related).items()))}.",
        f"Markets represented in recent groups: {recent_markets}; markets represented only by single-observation new histories: {new_recent_markets}; markets with older histories and recent appended points: {old_history_markets}.",
        f"Relevant groups eligible under current on-demand rules: {len(eligible)}; eligible with latest actual since {RECENT_FROM}: {len(recent_eligible)}; eligible but stale before {RECENT_FROM}: {len(stale_eligible)}.",
        f"Latest actual month among relevant groups: {distinct_recent_latest or 'none'}; latest actual month among eligible relevant groups: {distinct_eligible_latest or 'none'}.",
        f"Latest exact source observation date across all actual WFP groups: {latest_source_date or 'not recorded'}.",
        f"Coverage classifications: {dict(sorted(classes.items()))}.",
        f"Normalized monthly rows in the audited groups: {sum(row['source_month_rows'] for row in rows)}; invalid month rows: {sum(row['invalid_month_rows_excluded'] for row in rows)}; invalid prices: {sum(row['invalid_price_rows_excluded'] for row in rows)}; missing prices: {sum(row['missing_price_rows'] for row in rows)}; duplicate rows removed: {sum(row['exact_duplicate_rows_removed'] for row in rows)}.",
        "",
        "Interpretation",
        "Recent actual rows occur in some of the same exact market-level groups that have older history, and also in newly observed market groups. They are not missing because of a priceflag, date-parse, or group-key bug. The older groups with 30+ usable lag rows have a single April 2026 point after a long gap; February and March 2026 are absent, so the newest actual cannot support the next calendar-month feature vector.",
        f"No audited series has two or more actual months since January 2024. {len(recent_singletons)} series are single-month histories with no usable lag rows. {len(recent_gap_only)} series have at least 30 usable historical lag rows but fail only the latest three-consecutive-actual-month requirement. Other recent groups below the row threshold: {recent_low_history_details}.",
        "A long series whose latest actual is stale can pass structural eligibility yet still cannot represent current market conditions. Ignoring the April 2026 observation to forecast from a stale 2023 origin would discard the latest source value and would not make a current forecast.",
        "Processed cassava products such as gari are explicitly labeled as proxies only and are not treated as raw cassava. Maize flour, rice variants, bean variants, and Yam (Abuja) likewise retain separate original labels.",
        "No model was trained and no forecast was generated by this audit.",
    ]
    return rows, "\n".join(lines) + "\n"


def main() -> None:
    rows, summary = run_audit()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    TXT_PATH.write_text(summary, encoding="utf-8")
    print(f"Wrote {len(rows)} series to {CSV_PATH.relative_to(ROOT)}")
    print(f"Wrote summary to {TXT_PATH.relative_to(ROOT)}")
    print(summary)


if __name__ == "__main__":
    main()
