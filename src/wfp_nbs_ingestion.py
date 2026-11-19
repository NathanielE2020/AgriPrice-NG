"""Read-only-source ingestion and normalization for the active WFP/NBS inputs.

Source files under ``data/raw/wfp`` and ``data/raw/nbs`` are never written by
this module. It writes normalized monthly tables and audit reports beneath
``data/processed`` and ``reports`` only.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
RAW_WFP_DIR = ROOT / "data" / "raw" / "wfp"
RAW_NBS_DIR = ROOT / "data" / "raw" / "nbs"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
WFP_MONTHLY_PATH = PROCESSED_DIR / "wfp_monthly.csv"
NBS_MONTHLY_PATH = PROCESSED_DIR / "nbs_monthly.csv"
COVERAGE_PATH = REPORT_DIR / "wfp_nbs_series_coverage.csv"
NBS_COVERAGE_PATH = REPORT_DIR / "nbs_series_coverage.csv"
AUDIT_PATH = REPORT_DIR / "source_migration_audit.txt"

APPROVED_WFP_SERIES = (
    ("Sorghum (white)", "Dawanau", "100 KG", "Wholesale"),
    ("Maize (white)", "Dawanau", "100 KG", "Wholesale"),
    ("Millet", "Dawanau", "100 KG", "Wholesale"),
    ("Maize", "Illela (CBM)", "KG", "Wholesale"),
    ("Sorghum", "Mai Adoua (CBM)", "KG", "Wholesale"),
)

WFP_REQUIRED_COLUMNS = {
    "date", "admin1", "admin2", "market", "market_id", "category",
    "commodity", "commodity_id", "unit", "priceflag", "pricetype",
    "currency", "price", "usdprice",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files() -> list[Path]:
    return sorted(
        [p for p in RAW_WFP_DIR.rglob("*") if p.is_file()]
        + [p for p in RAW_NBS_DIR.rglob("*") if p.is_file()]
    )


def _read_wfp() -> tuple[pd.DataFrame, dict[str, Any]]:
    prices_path = RAW_WFP_DIR / "wfp_food_prices_nga.csv"
    markets_path = RAW_WFP_DIR / "wfp_markets_nga.csv"
    if not prices_path.is_file() or not markets_path.is_file():
        raise FileNotFoundError("The two required WFP CSV files were not found in data/raw/wfp.")

    rows = pd.read_csv(prices_path, low_memory=False)
    missing_columns = sorted(WFP_REQUIRED_COLUMNS.difference(rows.columns))
    if missing_columns:
        raise ValueError(f"WFP CSV is missing required columns: {missing_columns}")
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    invalid_dates = int(rows["date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"WFP contains {invalid_dates} rows with invalid dates.")
    rows["price"] = pd.to_numeric(rows["price"], errors="coerce")
    rows["usdprice"] = pd.to_numeric(rows["usdprice"], errors="coerce")
    rows["observation_month"] = rows["date"].dt.to_period("M").astype(str)
    rows["source"] = "WFP"
    rows["source_file"] = prices_path.name

    markets = pd.read_csv(markets_path, low_memory=False)
    if not {"market_id", "market"}.issubset(markets.columns):
        raise ValueError("WFP market metadata must contain market_id and market columns.")
    if markets["market_id"].duplicated().any():
        raise ValueError("WFP market metadata contains duplicate market_id values.")

    # The price CSV already carries locality fields. Record the metadata match
    # rate without replacing or silently correcting the source fields.
    metadata_ids = set(markets["market_id"].dropna())
    matched_ids = set(rows["market_id"].dropna()) & metadata_ids
    exact_key = [
        "date", "market_id", "commodity_id", "unit", "pricetype", "priceflag"
    ]
    summary = {
        "path": prices_path,
        "rows": len(rows),
        "raw_duplicate_rows": int(rows.duplicated().sum()),
        "duplicate_observation_keys": int(rows.duplicated(exact_key).sum()),
        "missing_values": int(rows.isna().sum().sum()),
        "invalid_dates": invalid_dates,
        "date_min": rows["date"].min(),
        "date_max": rows["date"].max(),
        "months": int(rows["observation_month"].nunique()),
        "commodities": int(rows["commodity"].nunique()),
        "markets": int(rows["market"].nunique()),
        "states": int(rows["admin1"].nunique()),
        "price_flags": rows["priceflag"].value_counts(dropna=False).to_dict(),
        "price_types": rows["pricetype"].value_counts(dropna=False).to_dict(),
        "units": int(rows["unit"].nunique()),
        "currencies": rows["currency"].value_counts(dropna=False).to_dict(),
        "market_metadata_rows": len(markets),
        "market_metadata_matched_ids": len(matched_ids),
        "market_metadata_unmatched_ids": len(set(rows["market_id"].dropna()) - metadata_ids),
    }

    # A reporting-month mean is needed for chart/time-series access. Do not
    # merge WFP's actual, aggregate, and mixed flags: priceflag is part of the
    # grouping key. Preserve exact input observation date bounds and count.
    group_columns = [
        "source", "source_file", "admin1", "admin2", "market", "market_id",
        "latitude", "longitude", "category", "commodity", "commodity_id",
        "unit", "priceflag", "pricetype", "currency", "observation_month",
    ]
    monthly = (
        rows.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            price=("price", "mean"),
            usdprice=("usdprice", "mean"),
            source_observation_count=("date", "size"),
            first_observation_date=("date", "min"),
            last_observation_date=("date", "max"),
        )
    )
    monthly["first_observation_date"] = monthly["first_observation_date"].dt.strftime("%Y-%m-%d")
    monthly["last_observation_date"] = monthly["last_observation_date"].dt.strftime("%Y-%m-%d")
    summary["monthly_rows"] = len(monthly)
    summary["monthly_groups_with_multiple_source_rows"] = int(
        (monthly["source_observation_count"] > 1).sum()
    )
    return monthly, summary


def _period_from_sheet(sheet_title: str) -> tuple[str, str]:
    match = re.fullmatch(r"Selected Food ([A-Za-z]+) (\d{4})", sheet_title.strip())
    if not match:
        raise ValueError(f"Unexpected NBS monthly sheet title: {sheet_title!r}")
    month_name, year_text = match.groups()
    month_number = list(calendar.month_name).index(month_name.title()) if month_name.title() in calendar.month_name else 0
    if not month_number:
        raise ValueError(f"Unknown month in NBS sheet title: {sheet_title!r}")
    year = int(year_text)
    return f"{year:04d}-{month_number:02d}", f"{month_name.title()}-{year % 100:02d}"


def _read_nbs() -> tuple[pd.DataFrame, dict[str, Any]]:
    workbooks = sorted(RAW_NBS_DIR.rglob("*.xlsx"))
    pdf_files = sorted(RAW_NBS_DIR.rglob("*.pdf"))
    if not workbooks:
        raise FileNotFoundError("No NBS monthly Excel workbooks were found in data/raw/nbs.")
    if not pdf_files:
        raise FileNotFoundError("No NBS monthly PDF reports were found in data/raw/nbs.")

    output: list[dict[str, Any]] = []
    workbook_summary: list[dict[str, Any]] = []
    seen_periods: set[str] = set()
    for path in workbooks:
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            national = next(
                (sheet for sheet in workbook.worksheets if sheet.title.startswith("Selected Food ")),
                None,
            )
            if national is None:
                raise ValueError(f"No Selected Food monthly sheet was found in {path.name}.")
            period, current_period_label = _period_from_sheet(national.title)
            if period in seen_periods:
                raise ValueError(f"More than one NBS workbook represents {period}.")
            seen_periods.add(period)
            national_rows = list(national.iter_rows(values_only=True))
            headers = [str(value).strip() if value is not None else "" for value in national_rows[0]]
            if "Item Label" not in headers:
                raise ValueError(f"NBS workbook {path.name} has no Item Label column.")
            current_header = f"Average of {current_period_label}"
            if current_header not in headers:
                raise ValueError(
                    f"NBS workbook {path.name} does not contain current-period column {current_header!r}."
                )
            item_index = headers.index("Item Label")
            current_index = headers.index(current_header)
            highest_index = headers.index("Highest") if "Highest" in headers else None
            lowest_index = headers.index("Lowest") if "Lowest" in headers else None
            national_items: set[str] = set()
            missing_national_prices = 0
            for row in national_rows[1:]:
                item = row[item_index] if item_index < len(row) else None
                if item is None or not str(item).strip():
                    continue
                item_label = str(item).strip()
                if item_label in national_items:
                    raise ValueError(f"Duplicate NBS item {item_label!r} for {period}.")
                national_items.add(item_label)
                price = row[current_index] if current_index < len(row) else None
                if price is None:
                    missing_national_prices += 1
                output.append({
                    "source": "NBS",
                    "source_file": path.name,
                    "source_sheet": national.title,
                    "observation_month": period,
                    "geography_level": "national",
                    "geography_name": "Nigeria",
                    "item_label": item_label,
                    "unit": "",
                    "price_measure": current_header,
                    "price_type": "reported_monthly_average",
                    "currency": "",
                    "price": price,
                    "highest_state_and_price_raw": (
                        row[highest_index] if highest_index is not None and highest_index < len(row) else ""
                    ),
                    "lowest_state_and_price_raw": (
                        row[lowest_index] if lowest_index is not None and lowest_index < len(row) else ""
                    ),
                })

            zone_sheet = next((sheet for sheet in workbook.worksheets if sheet.title == "Zone All item"), None)
            zone_count = 0
            missing_zone_prices = 0
            if zone_sheet is not None:
                zone_rows = list(zone_sheet.iter_rows(values_only=True))
                zone_headers = [str(value).strip() if value is not None else "" for value in zone_rows[0]]
                if not zone_headers or zone_headers[0] != "Item Label":
                    raise ValueError(f"NBS zone sheet in {path.name} has an unexpected first column.")
                zone_names = [value for value in zone_headers[1:] if value]
                zone_items: set[str] = set()
                for row in zone_rows[1:]:
                    item = row[0] if row else None
                    if item is None or not str(item).strip():
                        continue
                    item_label = str(item).strip()
                    if item_label in zone_items:
                        raise ValueError(f"Duplicate NBS zone item {item_label!r} for {period}.")
                    zone_items.add(item_label)
                    for idx, zone in enumerate(zone_names, start=1):
                        price = row[idx] if idx < len(row) else None
                        if price is None:
                            missing_zone_prices += 1
                        output.append({
                            "source": "NBS",
                            "source_file": path.name,
                            "source_sheet": zone_sheet.title,
                            "observation_month": period,
                            "geography_level": "zone",
                            "geography_name": zone,
                            "item_label": item_label,
                            "unit": "",
                            "price_measure": "zone_average",
                            "price_type": "reported_monthly_average",
                            "currency": "",
                            "price": price,
                            "highest_state_and_price_raw": "",
                            "lowest_state_and_price_raw": "",
                        })
                        zone_count += 1
            workbook_summary.append({
                "path": path,
                "period": period,
                "national_items": len(national_items),
                "zone_price_cells": zone_count,
                "missing_national_prices": missing_national_prices,
                "missing_zone_prices": missing_zone_prices,
                "zone_count": len(zone_names) if zone_sheet is not None else 0,
            })
        finally:
            workbook.close()

    # Validate PDFs without deriving price rows from figures/tables. The Excel
    # workbooks remain the only structured source used for normalized values.
    pdf_summary = []
    for path in pdf_files:
        reader = PdfReader(str(path), strict=True)
        if not reader.pages:
            raise ValueError(f"NBS PDF has no readable pages: {path.name}")
        extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        extracted_chars = len(extracted_text)
        if extracted_chars == 0:
            raise ValueError(f"NBS PDF text could not be extracted: {path.name}")
        pdf_summary.append({
            "path": path,
            "pages": len(reader.pages),
            "text_chars": extracted_chars,
            "naira_symbol_present": chr(0x20A6) in extracted_text,
        })

    normalized = pd.DataFrame(output)
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    # The workbook has no currency column. Its paired official PDFs provide
    # direct currency evidence, so annotate only when every report confirms it.
    if pdf_summary and all(record["naira_symbol_present"] for record in pdf_summary):
        normalized["currency"] = "NGN"
    duplicate_key = ["observation_month", "geography_level", "geography_name", "item_label"]
    duplicates = int(normalized.duplicated(duplicate_key).sum())
    if duplicates:
        raise ValueError(f"Normalized NBS data contains {duplicates} duplicate item/geography/month rows.")
    summary = {
        "rows": len(normalized),
        "national_rows": int((normalized["geography_level"] == "national").sum()),
        "zone_rows": int((normalized["geography_level"] == "zone").sum()),
        "items": int(normalized["item_label"].nunique()),
        "geographies": int(normalized["geography_name"].nunique()),
        "periods": sorted(normalized["observation_month"].unique()),
        "missing_prices": int(normalized["price"].isna().sum()),
        "duplicate_rows": int(normalized.duplicated().sum()),
        "duplicate_series_month_keys": duplicates,
        "workbooks": workbook_summary,
        "pdfs": pdf_summary,
    }
    return normalized, summary


def _approved_wfp_coverage(monthly: pd.DataFrame) -> pd.DataFrame:
    coverage: list[dict[str, Any]] = []
    for commodity, market, unit, price_type in APPROVED_WFP_SERIES:
        group = monthly.loc[
            (monthly["commodity"] == commodity)
            & (monthly["market"] == market)
            & (monthly["unit"] == unit)
            & (monthly["pricetype"] == price_type)
        ].copy()
        flags = Counter(group["priceflag"].fillna("").astype(str))
        direct = group.loc[group["priceflag"] == "actual"]
        months = sorted(pd.Period(value, freq="M") for value in direct["observation_month"].dropna().unique())
        if months:
            calendar_months = pd.period_range(months[0], months[-1], freq="M")
            missing = calendar_months.difference(months)
            gaps: list[int] = []
            current = 0
            for month in calendar_months:
                if month in missing:
                    current += 1
                    gaps.append(current)
                else:
                    current = 0
            first = str(months[0])
            last = str(months[-1])
            coverage_pct = 100 * len(months) / len(calendar_months)
            missing_count = len(missing)
            max_gap = max(gaps, default=0)
        else:
            first = last = ""
            coverage_pct = 0.0
            missing_count = max_gap = 0
        coverage.append({
            "commodity": commodity,
            "market": market,
            "unit": unit,
            "price_type": price_type,
            "actual_monthly_observations": len(months),
            "first_observation_month": first,
            "last_observation_month": last,
            "missing_calendar_months": missing_count,
            "calendar_coverage_percent": round(coverage_pct, 2),
            "maximum_consecutive_missing_months": max_gap,
            "other_priceflag_rows_kept_separate": sum(count for flag, count in flags.items() if flag != "actual"),
            "has_at_least_96_observed_months": len(months) >= 96,
            "continuous_monthly_coverage": missing_count == 0,
            "ready_for_forecasting": False,
            "note": "Pure 'actual' flags only; long gaps remain. Aggregate and mixed flags are retained separately, never merged.",
        })
    return pd.DataFrame(coverage)


def _nbs_coverage(monthly: pd.DataFrame) -> pd.DataFrame:
    coverage: list[dict[str, Any]] = []
    group_fields = ["item_label", "geography_level", "geography_name"]
    for (item, geography_level, geography_name), group in monthly.groupby(group_fields, dropna=False):
        observed = sorted(
            pd.Period(value, freq="M")
            for value in group["observation_month"].dropna().unique()
        )
        if observed:
            calendar_months = pd.period_range(observed[0], observed[-1], freq="M")
            missing = calendar_months.difference(observed)
            first, last = str(observed[0]), str(observed[-1])
            coverage_pct = 100 * len(observed) / len(calendar_months)
        else:
            missing = []
            first = last = ""
            coverage_pct = 0.0
        coverage.append({
            "item_label": item,
            "geography_level": geography_level,
            "geography_name": geography_name,
            "unit": "not stated in workbook",
            "price_type": "reported_monthly_average",
            "observed_months": len(observed),
            "first_observation_month": first,
            "last_observation_month": last,
            "missing_calendar_months": len(missing),
            "calendar_coverage_percent": round(coverage_pct, 2),
            "missing_prices": int(group["price"].isna().sum()),
            "ready_for_forecasting": False,
            "note": "Only five source months; no standardized unit or market-level observations.",
        })
    return pd.DataFrame(coverage)


def _render_audit(
    wfp: dict[str, Any],
    nbs: dict[str, Any],
    source_files: list[Path],
    coverage: pd.DataFrame,
) -> str:
    lines = [
        "AgriPrice-NG WFP + NBS Source Migration Audit",
        "Generated from the source files currently present; no price values are fabricated.",
        "Original input files were read only. Normalized outputs are separate under data/processed/.",
        "",
        "SOURCE FILE INVENTORY",
    ]
    for path in source_files:
        rel = path.relative_to(ROOT).as_posix()
        lines.append(
            f"- {rel} | {path.stat().st_size:,} bytes | {path.suffix.lower()} | SHA-256 {_sha256(path)}"
        )
    lines.extend([
        "",
        "WFP FOOD-PRICE CSV",
        f"- Source observations: {wfp['rows']:,}; normalized monthly rows: {wfp['monthly_rows']:,}.",
        f"- Actual source date range: {wfp['date_min'].date()} through {wfp['date_max'].date()}; observed months: {wfp['months']}.",
        f"- Coverage: {wfp['commodities']} commodities, {wfp['markets']} observed markets, {wfp['states']} states, {wfp['units']} source units.",
        f"- Price flags (kept separate): {wfp['price_flags']}.",
        f"- Price types: {wfp['price_types']}; currencies: {wfp['currencies']}.",
        f"- Missing cells: {wfp['missing_values']}; full duplicate rows: {wfp['raw_duplicate_rows']}; duplicate observation keys: {wfp['duplicate_observation_keys']}.",
        f"- Market metadata: {wfp['market_metadata_rows']} records; {wfp['market_metadata_matched_ids']} observed IDs match; {wfp['market_metadata_unmatched_ids']} observed IDs did not match.",
        f"- Monthly normalization averages multiple rows only inside the same commodity, market, unit, price type, currency, and exact priceflag; source date bounds and row counts are retained. No absent month is filled.",
        "",
        "APPROVED WFP MARKET SERIES (pure priceflag='actual')",
    ])
    for row in coverage.to_dict(orient="records"):
        lines.append(
            f"- {row['commodity']} | {row['market']} | {row['unit']} | {row['price_type']}: "
            f"{row['actual_monthly_observations']} observed months, {row['first_observation_month']}–{row['last_observation_month']}, "
            f"{row['missing_calendar_months']} missing months, {row['calendar_coverage_percent']:.2f}% coverage, "
            f"longest gap {row['maximum_consecutive_missing_months']} months."
        )
    lines.extend([
        "- These data support the approved series as separate historical series, but the series are irregular and some stop years before the latest source date. No missing prices are interpolated.",
        "",
        "NBS MONTHLY WORKBOOKS AND PDF REPORTS",
        f"- Structured workbook values: {nbs['rows']:,} item/geography/month rows ({nbs['national_rows']:,} national and {nbs['zone_rows']:,} zonal).",
        f"- Workbook observation periods: {', '.join(nbs['periods'])}; latest actual reporting period: {max(nbs['periods']) if nbs['periods'] else 'none'}.",
        f"- Coverage: {nbs['items']} source item labels and {nbs['geographies']} geography labels; missing prices: {nbs['missing_prices']}; duplicate rows: {nbs['duplicate_rows']}.",
        "- Each workbook's current-period 'Average of <month-year>' column is used; prior-month and prior-year comparator columns are not treated as new observations. The national sheet and six-zone sheet remain separate.",
        "- No unit is asserted: the workbooks have no standardized unit column, so the full source item label is retained. The paired official PDFs were checked for the Naira symbol; NGN is recorded only when each PDF confirms it.",
        "- PDFs were opened and all pages yielded extractable text. They are preserved as supporting reports, not parsed as extra price rows, avoiding double counting.",
        "- NBS covers only five monthly periods and does not support reliable multi-step forecasting at this stage. It must remain separate from WFP because geography and measurement/unit definitions differ.",
        "",
        "TRAINING STATUS",
        "- No models were trained. Existing model/evaluation outputs from retired datasets must not be shown as results for the new source scope.",
    ])
    return "\n".join(lines) + "\n"


def prepare_wfp_nbs_data() -> dict[str, Any]:
    """Validate inputs and write normalized data and coverage/audit reports."""
    source_files = _source_files()
    if not source_files:
        raise FileNotFoundError("No WFP or NBS source files were found in the expected raw directories.")
    wfp, wfp_summary = _read_wfp()
    nbs, nbs_summary = _read_nbs()
    coverage = _approved_wfp_coverage(wfp)
    nbs_coverage = _nbs_coverage(nbs)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    wfp.to_csv(WFP_MONTHLY_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    nbs.to_csv(NBS_MONTHLY_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    coverage.to_csv(COVERAGE_PATH, index=False)
    nbs_coverage.to_csv(NBS_COVERAGE_PATH, index=False)
    AUDIT_PATH.write_text(
        _render_audit(wfp_summary, nbs_summary, source_files, coverage), encoding="utf-8"
    )
    return {
        "wfp": wfp,
        "nbs": nbs,
        "wfp_summary": wfp_summary,
        "nbs_summary": nbs_summary,
        "coverage": coverage,
        "nbs_coverage": nbs_coverage,
        "outputs": [WFP_MONTHLY_PATH, NBS_MONTHLY_PATH, COVERAGE_PATH, NBS_COVERAGE_PATH, AUDIT_PATH],
    }


if __name__ == "__main__":
    result = prepare_wfp_nbs_data()
    print(
        "WFP/NBS source preparation completed without training: "
        f"WFP monthly rows={len(result['wfp']):,}; NBS rows={len(result['nbs']):,}; "
        f"approved WFP series={len(result['coverage'])}."
    )
