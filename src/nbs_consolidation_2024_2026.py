"""Consolidate the supplied 2024-2026 NBS monthly workbooks for audit use.

The workbook/PDF inputs are read-only. Only the normalized NBS CSV and the
three explicitly named audit outputs are written. The script intentionally
does not train models or join NBS and WFP observations.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
NBS_RAW = ROOT / "data" / "raw" / "nbs"
WFP_RAW = ROOT / "data" / "raw" / "wfp"
NBS_OUTPUT = ROOT / "data" / "processed" / "nbs_monthly.csv"
NBS_COVERAGE = ROOT / "reports" / "nbs_2024_2026_coverage.csv"
NBS_AUDIT = ROOT / "reports" / "nbs_2024_2026_consolidation_audit.txt"
READINESS = ROOT / "reports" / "wfp_nbs_final_data_readiness.txt"
WFP_MONTHLY = ROOT / "data" / "processed" / "wfp_monthly.csv"
YEARS = ("2024", "2025", "2026")
ASSESSMENT_MONTH = pd.Period("2026-11", freq="M")
PREVIOUSLY_APPROVED_WFP_SERIES = (
    ("Sorghum (white)", "Dawanau", "100 KG", "Wholesale"),
    ("Maize (white)", "Dawanau", "100 KG", "Wholesale"),
    ("Millet", "Dawanau", "100 KG", "Wholesale"),
    ("Maize", "Illela (CBM)", "KG", "Wholesale"),
    ("Sorghum", "Mai Adoua (CBM)", "KG", "Wholesale"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_hashes() -> dict[str, str]:
    files = [
        path
        for year in YEARS
        for path in (NBS_RAW / year).rglob("*")
        if path.is_file()
    ]
    files += [path for path in WFP_RAW.rglob("*") if path.is_file()]
    if WFP_MONTHLY.is_file():
        files.append(WFP_MONTHLY)
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in sorted(files)}


def _month_number(folder: str) -> int:
    normalized = folder.casefold().replace("appril", "april").replace("febuary", "february")
    for number in range(1, 13):
        name = calendar.month_name[number].casefold()
        if normalized == name or normalized.startswith(name[:3]):
            return number
    raise ValueError(f"Unrecognized NBS month folder: {folder!r}")


def _header_period(value: Any) -> pd.Period | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*Average\s+of\s+([A-Za-z]+)[- ](\d{2,4})\s*", value, re.I)
    if not match:
        return None
    month_text, year_text = match.groups()
    month = next(
        (i for i in range(1, 13) if calendar.month_name[i].casefold().startswith(month_text.casefold()[:3])),
        0,
    )
    if not month:
        return None
    year = int(year_text)
    if year < 100:
        year += 2000
    return pd.Period(year=year, month=month, freq="M")


def _report_metadata(folder: Path) -> dict[str, str]:
    pdfs = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.casefold() == ".pdf")
    if len(pdfs) > 1:
        raise ValueError(f"Expected no more than one paired PDF in {folder}, found {len(pdfs)}.")
    if not pdfs:
        return {
            "source_pdf": "",
            "source_report_date": "",
            "source_publication_month": "",
            "currency_evidence": "No paired PDF; NGN follows NBS Nigeria food-price series context, not workbook currency metadata.",
            "pdf_pages": "",
            "pdf_text_chars": "",
        }

    pdf = pdfs[0]
    reader = PdfReader(str(pdf), strict=False)
    if not reader.pages:
        raise ValueError(f"Paired NBS PDF contains no pages: {pdf}")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError(f"Paired NBS PDF has no extractable text: {pdf}")
    report_match = re.search(r"Report\s*Date\s*:\s*([A-Za-z]+\s+\d{4})", text, re.I)
    raw_report_date = report_match.group(1).strip() if report_match else ""
    publication_month = ""
    if raw_report_date:
        date_match = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", raw_report_date)
        if date_match:
            month_name, year = date_match.groups()
            month_number = next(
                (i for i in range(1, 13) if calendar.month_name[i].casefold().startswith(month_name.casefold()[:3])),
                0,
            )
            if month_number:
                publication_month = f"{year}-{month_number:02d}"
    # NBS PDFs spell out prices using N before numeric values; a day is not
    # shown in Report Date, so preserve month precision only.
    naira_notation = bool(re.search(r"(?:₦|\bN)\s*[\d,]+(?:\.\d{1,2})?", text))
    return {
        "source_pdf": pdf.name,
        "source_report_date": raw_report_date,
        "source_publication_month": publication_month,
        "currency_evidence": (
            "Paired NBS PDF contains Naira price notation."
            if naira_notation
            else "Paired PDF present; currency notation not machine-detected."
        ),
        "pdf_pages": str(len(reader.pages)),
        "pdf_text_chars": str(len(text)),
    }


def _norm_label(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _map_item(label: str) -> tuple[str, str, str]:
    key = _norm_label(label)
    direct: dict[str, tuple[str, str, str]] = {
        "beans brown sold loose": ("Beans", "direct_variant", "Brown beans; the original item label is retained."),
        "beans brown": ("Beans", "direct_variant", "Brown beans; the original item label is retained."),
        "beans white": ("Beans", "direct_variant", "White beans; retained as a separate product variant."),
        "beans white black eye sold loose": ("Beans", "related_variant_requires_review", "Black-eye bean/pulse variant; kept separate from brown and white bean items."),
        "broken rice ofada": ("Rice (local)", "related_variant", "Ofada/broken rice; local type is named, but its definition is distinct."),
        "rice local sold loose": ("Rice (local)", "direct_variant", "Explicit local rice label; source definition retained."),
        "local rice broken": ("Rice (local)", "direct_variant", "Explicit local rice label; source definition retained."),
        "rice local short grained": ("Rice (local)", "direct_variant", "Explicit local rice label; source definition retained."),
        "rice imported high quality sold loose": ("Rice (imported)", "direct_variant", "Explicit imported rice label; source definition retained."),
        "rice long grained imported": ("Rice (imported)", "direct_variant", "Explicit imported rice label; source definition retained."),
        "rice agric sold loose": ("Rice (local)", "related_variant_requires_review", "Agric rice label does not explicitly state origin; do not merge without definition review."),
        "rice medium grained": ("Rice (origin unspecified)", "ambiguous_do_not_merge", "Grain size is stated but origin is not; not assigned to local or imported rice."),
        "maize grain white sold loose": ("Maize", "direct_variant", "White maize grain; retained as a separate colour variant."),
        "maize corn grains white sold loose": ("Maize", "direct_variant", "White maize grain; retained as a separate colour variant."),
        "maize grain yellow sold loose": ("Maize", "direct_variant", "Yellow maize grain; retained as a separate colour variant."),
        "yam tuber": ("Yam", "direct_variant", "Yam tuber; original wording/case retained."),
        "tomato": ("Tomatoes", "direct_variant", "Tomato item; original label retained."),
        "tomatoes fresh": ("Tomatoes", "direct_variant", "Fresh tomatoes; original label retained."),
        "palm oil 1 bottle specify bottle": ("Palm oil", "related_variant_requires_review", "Bottle volume is unspecified; not comparable to a stated 75cl unit."),
        "palm oil 75cl": ("Palm oil", "direct_variant", "Palm oil with a 75cl quantity explicitly stated in the label."),
        "garri white sold loose": ("Cassava-derived product (processed gari)", "related_not_equivalent", "Processed gari is not raw cassava."),
        "garri yellow sold loose": ("Cassava-derived product (processed gari)", "related_not_equivalent", "Processed gari is not raw cassava."),
        "garri white": ("Cassava-derived product (processed gari)", "related_not_equivalent", "Processed gari is not raw cassava."),
        "garri yellow": ("Cassava-derived product (processed gari)", "related_not_equivalent", "Processed gari is not raw cassava."),
    }
    if key in direct:
        return direct[key]
    return ("", "out_of_scope", "No defensible mapping to the seven research commodities; source label retained.")


def _unit_from_label(label: str) -> tuple[str, str]:
    match = re.search(r"\b(\d+(?:[.,]\d+)?\s*(?:kg|g|cl|ml|litres?|liters?|l))\b", label, re.I)
    if match:
        return re.sub(r"\s+", "", match.group(1)).replace(",", "."), "explicit_item_label"
    if re.search(r"\bcrate\s+of\s+30\s+pieces\b", label, re.I):
        return "30 pieces (crate)", "explicit_item_label"
    if re.search(r"\b1\s*bottle\b", label, re.I):
        return "1 bottle (volume unspecified)", "explicit_item_label_volume_unspecified"
    return "", "not_explicit_in_item_label"


def _number_or_blank(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return str(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return ""


def _workbook_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    year = int(path.parents[1].name)
    month = _month_number(path.parent.name)
    folder_period = pd.Period(year=year, month=month, freq="M")
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        national = next(
            (sheet for sheet in workbook.worksheets if sheet.title.casefold().startswith("selected food")),
            None,
        )
        if national is None:
            raise ValueError(f"No national Selected Food sheet in {path}.")
        national_rows = list(national.iter_rows(values_only=True))
        if not national_rows:
            raise ValueError(f"Empty national sheet in {path}.")
        headers = [str(value).strip() if value is not None else "" for value in national_rows[0]]
        if not headers or not re.fullmatch(r"(?:Items?|Row)\s+Labels?", headers[0], re.I):
            raise ValueError(f"Unexpected item-label header in {path}: {headers[:1]!r}.")
        current_columns = [
            index for index, header in enumerate(headers)
            if _header_period(header) == folder_period
        ]
        if len(current_columns) != 1:
            raise ValueError(
                f"Expected exactly one current-month average header for {folder_period} in {path.name}; "
                f"found {[headers[i] for i in current_columns]!r}."
            )
        current_index = current_columns[0]
        current_header = headers[current_index]
        item_labels = [str(row[0]).strip() for row in national_rows[1:] if row and row[0] is not None and str(row[0]).strip()]
        if len(item_labels) != len(set(item_labels)):
            raise ValueError(f"Repeated item label within the national sheet in {path.name}.")
        highest_idx = headers.index("Highest") if "Highest" in headers else None
        lowest_idx = headers.index("Lowest") if "Lowest" in headers else None
        sheet_period = re.search(r"Selected Food\s+([A-Za-z]+)\.?\s+(\d{4})", national.title, re.I)
        title_period = ""
        if sheet_period:
            month_name, title_year = sheet_period.groups()
            title_month = next((i for i in range(1, 13) if calendar.month_name[i].casefold().startswith(month_name.casefold()[:3])), 0)
            if title_month:
                title_period = str(pd.Period(year=int(title_year), month=title_month, freq="M"))

        metadata = _report_metadata(path.parent)
        relative_path = path.relative_to(ROOT).as_posix()
        period_text = str(folder_period)
        regime = "pre_2025_cpi_rebase" if folder_period < pd.Period("2025-01", freq="M") else "post_2025_cpi_rebase"
        records: list[dict[str, Any]] = []

        def append_row(
            item: str,
            price_value: Any,
            sheet_name: str,
            geography_level: str,
            geography_name: str,
            price_measure: str,
            highest: Any = "",
            lowest: Any = "",
        ) -> None:
            mapping, mapping_status, rationale = _map_item(item)
            unit, unit_basis = _unit_from_label(item)
            price = _number_or_blank(price_value)
            records.append({
                "source": "NBS",
                "source_file": path.name,
                "source_relative_path": relative_path,
                "source_pdf": metadata["source_pdf"],
                "source_sheet": sheet_name,
                "reporting_year": year,
                "reporting_month": month,
                "observation_month": period_text,
                "source_report_date": metadata["source_report_date"],
                "source_publication_month": metadata["source_publication_month"],
                "source_publication_date": "",
                "methodology_regime": regime,
                "geography_level": geography_level,
                "geography_name": geography_name,
                "item_label": item,
                "original_item_name": item,
                "research_commodity_mapping": mapping,
                "mapping_status": mapping_status,
                "mapping_rationale": rationale,
                "unit": unit,
                "unit_basis": unit_basis,
                "price_measure": price_measure,
                "price_type": "reported_monthly_average",
                "currency": "NGN",
                "currency_evidence": metadata["currency_evidence"],
                "price": price,
                "highest_state_and_price_raw": highest if highest is not None else "",
                "lowest_state_and_price_raw": lowest if lowest is not None else "",
            })

        for row in national_rows[1:]:
            if not row or row[0] is None or not str(row[0]).strip():
                continue
            item = str(row[0]).strip()
            append_row(
                item,
                row[current_index] if current_index < len(row) else None,
                national.title,
                "national",
                "Nigeria",
                current_header,
                row[highest_idx] if highest_idx is not None and highest_idx < len(row) else "",
                row[lowest_idx] if lowest_idx is not None and lowest_idx < len(row) else "",
            )

        zone = next((sheet for sheet in workbook.worksheets if sheet.title.casefold() == "zone all item"), None)
        zone_items: list[str] = []
        zone_names: list[str] = []
        if zone is not None:
            zone_rows = list(zone.iter_rows(values_only=True))
            zone_headers = [str(value).strip() if value is not None else "" for value in zone_rows[0]]
            if not zone_headers or not re.fullmatch(r"(?:Items?|Row)\s+Labels?", zone_headers[0], re.I):
                raise ValueError(f"Unexpected first zone column in {path.name}: {zone_headers[:1]!r}.")
            zone_names = [value for value in zone_headers[1:] if value]
            if len(zone_names) != 6:
                raise ValueError(f"Expected six reported NBS zones in {path.name}, found {zone_names!r}.")
            for row in zone_rows[1:]:
                if not row or row[0] is None or not str(row[0]).strip():
                    continue
                item = str(row[0]).strip()
                zone_items.append(item)
                for index, zone_name in enumerate(zone_names, start=1):
                    append_row(
                        item,
                        row[index] if index < len(row) else None,
                        zone.title,
                        "zone",
                        zone_name,
                        "zone_average",
                    )
            if len(zone_items) != len(set(zone_items)):
                raise ValueError(f"Repeated item label within the zone sheet in {path.name}.")

        return records, {
            "path": path,
            "period": period_text,
            "workbook_sheet_title_period": title_period,
            "national_sheet": national.title,
            "national_item_count": len(item_labels),
            "zone_item_count": len(zone_items),
            "zone_names": zone_names,
            "records": len(records),
            "missing_price_cells": sum(not row["price"] for row in records),
            "source_publication_month": metadata["source_publication_month"],
            "source_report_date": metadata["source_report_date"],
            "source_pdf": metadata["source_pdf"],
            "currency_evidence": metadata["currency_evidence"],
            "pdf_pages": metadata["pdf_pages"],
            "pdf_text_chars": metadata["pdf_text_chars"],
        }
    finally:
        workbook.close()


def _deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    """Remove only repeated same-price records from the exact same source sheet."""
    seen: dict[tuple[str, ...], str] = {}
    result: list[dict[str, Any]] = []
    removed = 0
    same_key_different_price = 0
    key_columns = (
        "source_relative_path", "source_sheet", "observation_month", "geography_level",
        "geography_name", "item_label", "unit", "price_type", "price_measure",
    )
    for row in records:
        key = tuple(str(row.get(name, "")) for name in key_columns)
        if key in seen and seen[key] == str(row.get("price", "")):
            removed += 1
            continue
        if key in seen and seen[key] != str(row.get("price", "")):
            # Different reported values are not duplicates; retain both.
            same_key_different_price += 1
        else:
            seen[key] = str(row.get("price", ""))
        result.append(row)
    return result, removed, same_key_different_price


def _longest_missing_run(periods: list[pd.Period]) -> int:
    if not periods:
        return 0
    full = pd.period_range(min(periods), max(periods), freq="M")
    observed = set(periods)
    longest = current = 0
    for period in full:
        if period in observed:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _coverage(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = (
        "item_label", "research_commodity_mapping", "mapping_status", "unit",
        "price_type", "geography_level", "geography_name",
    )
    for row in records:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        periods = sorted({pd.Period(row["observation_month"], freq="M") for row in group})
        full = pd.period_range(periods[0], periods[-1], freq="M")
        missing = [str(value) for value in full if value not in set(periods)]
        publication_months = sorted({row["source_publication_month"] for row in group if row["source_publication_month"]})
        regimes = sorted({row["methodology_regime"] for row in group})
        output.append({
            "original_item_label": key[0],
            "research_commodity_mapping": key[1],
            "mapping_status": key[2],
            "unit": key[3],
            "price_type": key[4],
            "geography_level": key[5],
            "geography_name": key[6],
            "observation_count": len(group),
            "first_observation_month": str(periods[0]),
            "last_observation_month": str(periods[-1]),
            "unique_observed_months": len(periods),
            "missing_calendar_months": len(missing),
            "missing_month_list": ";".join(missing),
            "calendar_coverage_percent": round(100 * len(periods) / len(full), 2),
            "longest_consecutive_missing_months": _longest_missing_run(periods),
            "missing_price_count": sum(not row["price"] for row in group),
            "currency_values": ";".join(sorted({row["currency"] for row in group if row["currency"]})),
            "methodology_regimes": ";".join(regimes),
            "crosses_2025_cpi_rebase": len(regimes) > 1,
            "source_workbook_count": len({row["source_relative_path"] for row in group}),
            "source_pdf_count": len({row["source_pdf"] for row in group if row["source_pdf"]}),
            "source_publication_months": ";".join(publication_months),
            "ready_for_reliable_ml_forecasting": False,
            "assessment_note": "NBS national/zone monthly averages; preserve exact item/geography/unit; no interpolation; 2025 CPI rebasing requires definition review.",
        })
    return output


def _wfp_mapping(commodity: str) -> tuple[str, str]:
    name = commodity.casefold()
    if "rice" in name:
        if any(word in name for word in ("flour", "meal")):
            return "Rice-derived product (processed)", "related_not_equivalent"
        if "import" in name:
            return "Rice (imported)", "direct_variant"
        if "local" in name:
            return "Rice (local)", "direct_variant"
        return "Rice (origin unspecified)", "ambiguous_do_not_merge"
    if "maize" in name and any(word in name for word in ("flour", "meal", "bran")):
        return "Maize-derived product (flour)", "related_not_equivalent"
    if "maize" in name:
        return "Maize", "direct_variant"
    if "yam" in name:
        return "Yam", "direct_variant"
    if "tomato" in name:
        return "Tomatoes", "direct_variant"
    if "palm" in name and "oil" in name:
        return "Palm oil", "direct_variant"
    if "beans" in name:
        return "Beans", "direct_variant"
    if "cowpea" in name:
        return "Beans/pulse related", "related_not_equivalent"
    if "cassava" in name:
        if any(word in name for word in ("gari", "flour", "meal")):
            return "Cassava-derived product (processed)", "related_not_equivalent"
        return "Cassava", "direct_candidate"
    if "gari" in name:
        return "Cassava-derived product (processed)", "related_not_equivalent"
    return "", "out_of_scope"


def _read_wfp_readiness() -> dict[str, Any]:
    if not WFP_MONTHLY.is_file():
        return {"available": False, "reason": "Normalized WFP monthly CSV is missing."}
    frame = pd.read_csv(WFP_MONTHLY, low_memory=False)
    needed = {
        "commodity", "admin1", "market", "market_id", "unit", "priceflag", "pricetype",
        "currency", "observation_month", "last_observation_date", "source_observation_count",
    }
    absent = sorted(needed.difference(frame.columns))
    if absent:
        return {"available": False, "reason": f"WFP normalized schema missing: {absent}"}
    frame["month_period"] = pd.PeriodIndex(frame["observation_month"].astype(str), freq="M")
    frame["priceflag"] = frame["priceflag"].fillna("").astype(str)
    flag_counts = frame["priceflag"].value_counts(dropna=False).to_dict()
    actual = frame.loc[frame["priceflag"].eq("actual")].copy()
    if actual.empty:
        return {"available": True, "frame": frame, "actual": actual, "flag_counts": flag_counts, "groups": [], "scope_counts": {}}
    mapped = actual["commodity"].fillna("").map(_wfp_mapping)
    actual["research_commodity_mapping"] = [entry[0] for entry in mapped]
    actual["mapping_status"] = [entry[1] for entry in mapped]
    scope = actual.loc[actual["research_commodity_mapping"].ne("")].copy()
    start_24 = ASSESSMENT_MONTH - 23
    start_36 = ASSESSMENT_MONTH - 35
    group_columns = [
        "research_commodity_mapping", "mapping_status", "commodity", "admin1", "market", "market_id",
        "unit", "pricetype", "currency",
    ]
    groups: list[dict[str, Any]] = []
    if not scope.empty:
        for keys, group in scope.groupby(group_columns, dropna=False):
            periods = sorted(group["month_period"].dropna().unique())
            first, last = periods[0], periods[-1]
            window_24 = sum(start_24 <= period <= ASSESSMENT_MONTH for period in periods)
            window_36 = sum(start_36 <= period <= ASSESSMENT_MONTH for period in periods)
            calendar_range = pd.period_range(first, last, freq="M")
            missing = len(calendar_range) - len(periods)
            row = dict(zip(group_columns, keys))
            row.update({
                "monthly_observations": len(periods),
                "first_observation_month": str(first),
                "latest_observation_month": str(last),
                "latest_actual_observation_date": str(group["last_observation_date"].fillna("").max()),
                "missing_calendar_months": missing,
                "calendar_coverage_percent": round(100 * len(periods) / len(calendar_range), 2),
                "longest_missing_month_run": _longest_missing_run(periods),
                "observations_recent_24_months": window_24,
                "observations_recent_36_months": window_36,
                "observations_in_2026": sum(period.year == 2026 for period in periods),
                "has_observation_in_2026": any(period.year == 2026 for period in periods),
                "meets_96_months": len(periods) >= 96,
                "meets_80pct_coverage": len(periods) / len(calendar_range) >= 0.8,
                "meets_gap_le_3": _longest_missing_run(periods) <= 3,
            })
            groups.append(row)
    groups.sort(key=lambda row: (row["latest_observation_month"], row["monthly_observations"]), reverse=True)
    scope_counts = (
        actual.assign(mapped=[entry[0] for entry in actual["commodity"].fillna("").map(_wfp_mapping)])
        .loc[lambda value: value["mapped"].ne("")]
        .groupby("mapped", dropna=False)
        .agg(monthly_rows=("observation_month", "size"), source_commodities=("commodity", "nunique"), markets=("market", "nunique"))
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "available": True,
        "frame": frame,
        "actual": actual,
        "flag_counts": flag_counts,
        "groups": groups,
        "scope_counts": scope_counts,
        "actual_latest_date": actual["last_observation_date"].dropna().astype(str).max(),
        "all_latest_date": frame["last_observation_date"].dropna().astype(str).max(),
        "actual_rows": len(actual),
        "all_rows": len(frame),
        "actual_month_count": actual["month_period"].nunique(),
        "actual_commodity_count": actual["commodity"].nunique(),
        "actual_market_count": actual["market"].nunique(),
        "actual_state_count": actual["admin1"].nunique(),
        "recent24_start": str(start_24),
        "recent36_start": str(start_36),
    }


def _previously_approved_wfp_audit(frame: pd.DataFrame) -> list[dict[str, Any]]:
    actual = frame.loc[frame["priceflag"].fillna("").astype(str).eq("actual")].copy()
    actual["month_period"] = pd.PeriodIndex(actual["observation_month"].astype(str), freq="M")
    start_24 = ASSESSMENT_MONTH - 23
    start_36 = ASSESSMENT_MONTH - 35
    results: list[dict[str, Any]] = []
    for commodity, market, unit, price_type in PREVIOUSLY_APPROVED_WFP_SERIES:
        group = actual.loc[
            actual["commodity"].eq(commodity)
            & actual["market"].eq(market)
            & actual["unit"].eq(unit)
            & actual["pricetype"].eq(price_type)
        ]
        periods = sorted(group["month_period"].dropna().unique())
        if not periods:
            results.append({"commodity": commodity, "market": market, "unit": unit, "price_type": price_type, "monthly_observations": 0})
            continue
        full = pd.period_range(periods[0], periods[-1], freq="M")
        results.append({
            "commodity": commodity,
            "market": market,
            "unit": unit,
            "price_type": price_type,
            "monthly_observations": len(periods),
            "first_observation_month": str(periods[0]),
            "latest_observation_month": str(periods[-1]),
            "latest_actual_observation_date": str(group["last_observation_date"].fillna("").astype(str).max()),
            "missing_calendar_months": len(full) - len(periods),
            "calendar_coverage_percent": round(100 * len(periods) / len(full), 2),
            "longest_missing_month_run": _longest_missing_run(periods),
            "observations_recent_24_months": sum(start_24 <= period <= ASSESSMENT_MONTH for period in periods),
            "observations_recent_36_months": sum(start_36 <= period <= ASSESSMENT_MONTH for period in periods),
            "observations_in_2026": sum(period.year == 2026 for period in periods),
        })
    return results


def _fmt_group(row: dict[str, Any]) -> str:
    return (
        f"{row['research_commodity_mapping']} | {row['commodity']} | {row['market']} / {row['admin1']} | "
        f"{row['unit']} | {row['pricetype']} | n={row['monthly_observations']} | "
        f"{row['first_observation_month']} to {row['latest_observation_month']} | "
        f"coverage={row['calendar_coverage_percent']:.1f}% | missing={row['missing_calendar_months']} | "
        f"longest_gap={row['longest_missing_month_run']} | recent24={row['observations_recent_24_months']} | "
        f"recent36={row['observations_recent_36_months']} | 2026={row['observations_in_2026']}"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def consolidate() -> dict[str, Any]:
    before_hashes = _protected_hashes()
    source_files = sorted(
        path
        for year in YEARS
        for path in (NBS_RAW / year).rglob("*")
        if path.is_file()
    )
    workbooks = [path for path in source_files if path.suffix.casefold() in {".xlsx", ".xlsm"}]
    pdfs = [path for path in source_files if path.suffix.casefold() == ".pdf"]
    other_files = [path for path in source_files if path.suffix.casefold() not in {".xlsx", ".xlsm", ".pdf"}]
    if not workbooks:
        raise FileNotFoundError("No NBS Excel workbooks found under data/raw/nbs/2024, 2025, 2026.")

    all_records: list[dict[str, Any]] = []
    workbook_summaries: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in workbooks:
        records, summary = _workbook_records(path)
        all_records.extend(records)
        workbook_summaries.append(summary)
    all_records, exact_duplicates_removed, key_conflicts_kept = _deduplicate(all_records)
    workbook_summaries.sort(key=lambda item: item["period"])
    all_records.sort(key=lambda row: (
        row["observation_month"],
        0 if row["geography_level"] == "national" else 1,
        row["geography_name"],
        row["item_label"],
        row["unit"],
    ))

    periods = sorted({pd.Period(row["observation_month"], freq="M") for row in all_records})
    full_periods = pd.period_range(periods[0], periods[-1], freq="M") if periods else []
    missing_source_months = [str(period) for period in full_periods if period not in set(periods)]
    row_key = (
        "observation_month", "geography_level", "geography_name", "item_label", "unit", "price_type",
    )
    series_month_keys: Counter[tuple[str, ...]] = Counter(
        tuple(str(row.get(key, "")) for key in row_key) for row in all_records
    )
    duplicate_series_month_keys = sum(value - 1 for value in series_month_keys.values() if value > 1)
    coverage = _coverage(all_records)
    coverage_observation_distribution = Counter(int(row["unique_observed_months"]) for row in coverage)

    NBS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(NBS_OUTPUT, all_records)
    _write_csv(NBS_COVERAGE, coverage)

    label_periods: dict[str, set[str]] = defaultdict(set)
    label_map: dict[str, tuple[str, str, str]] = {}
    for row in all_records:
        label_periods[row["item_label"]].add(row["observation_month"])
        label_map[row["item_label"]] = (
            row["research_commodity_mapping"], row["mapping_status"], row["mapping_rationale"]
        )
    year_labels: dict[int, set[str]] = defaultdict(set)
    for row in all_records:
        year_labels[int(row["reporting_year"])].add(row["item_label"])
    pre_labels = year_labels.get(2024, set())
    post_labels = set().union(*(year_labels.get(y, set()) for y in (2025, 2026)))
    removed_labels = sorted(pre_labels - post_labels)
    new_labels = sorted(post_labels - pre_labels)
    year_label_changes: dict[str, dict[str, list[str]]] = {}
    for earlier, later in ((2024, 2025), (2025, 2026)):
        old_set = year_labels.get(earlier, set())
        new_set = year_labels.get(later, set())
        year_label_changes[f"{earlier}_to_{later}"] = {
            "removed": sorted(old_set - new_set),
            "introduced": sorted(new_set - old_set),
        }
    title_mismatches = [
        summary for summary in workbook_summaries
        if summary["workbook_sheet_title_period"] and summary["workbook_sheet_title_period"] != summary["period"]
    ]
    no_pdf_months = sorted({summary["period"] for summary in workbook_summaries if not summary["source_pdf"]})
    observed_geographies = sorted({(row["geography_level"], row["geography_name"]) for row in all_records})
    unique_items = sorted(label_periods)
    mapped = sorted({row["research_commodity_mapping"] for row in all_records if row["research_commodity_mapping"]})
    mapping_labels: dict[str, list[str]] = defaultdict(list)
    for label, (mapping, status, _) in label_map.items():
        if mapping:
            mapping_labels[mapping].append(f"{label} [{status}]")

    wfp = _read_wfp_readiness()
    previously_approved_wfp = (
        _previously_approved_wfp_audit(wfp["frame"]) if wfp.get("available") else []
    )
    report_lines = [
        "AgriPrice-NG NBS 2024-2026 Consolidation Audit",
        "Assessment date: 2026-11-19 (Africa/Lagos).",
        "Purpose: source consolidation and data-readiness audit only; no model training was performed.",
        "Raw NBS files under data/raw/nbs/2024, 2025, and 2026 were opened read-only. WFP inputs were read only.",
        "The output CSV uses only each workbook's current reporting-month Average column, not prior-month/prior-year comparison columns.",
        "",
        "FILE INVENTORY",
        f"- Files discovered recursively: {len(source_files)} total; {len(workbooks)} Excel workbooks; {len(pdfs)} PDFs; {len(other_files)} other files.",
        f"- Workbooks parsed successfully: {len(workbook_summaries)}; parse errors: {len(errors)}.",
    ]
    for path in source_files:
        report_lines.append(
            f"  {path.relative_to(ROOT).as_posix()} | {path.stat().st_size:,} bytes | {path.suffix.lower()} | SHA-256 {_sha256(path)}"
        )
    report_lines += [
        "",
        "OBSERVATION EXTRACTION AND DEDUPLICATION",
        f"- Monthly records extracted before exact duplicate removal: {sum(summary['records'] for summary in workbook_summaries):,}.",
        f"- Monthly records in consolidated CSV after exact duplicate removal: {len(all_records):,}.",
        f"- Exact duplicates removed (same workbook, sheet, period, item, geography, measurement and price): {exact_duplicates_removed}.",
        f"- Same-source series/month keys with differing prices retained rather than collapsed: {key_conflicts_kept}.",
        f"- Duplicate item/geography/month/unit observation keys after consolidation: {duplicate_series_month_keys}.",
        f"- Missing price cells retained blank (not imputed): {sum(not row['price'] for row in all_records):,}.",
        f"- Observation reporting-month span: {periods[0] if periods else 'none'} to {periods[-1] if periods else 'none'}.",
        f"- Distinct source reporting months present: {len(periods)}; missing months between the first and last source report: {', '.join(missing_source_months) if missing_source_months else 'none'}.",
        f"- Unique original item labels across all years: {len(unique_items)}.",
        f"- Exact item/unit/geography series observation-count distribution: {dict(sorted(coverage_observation_distribution.items()))} (some labels occur only in selected months due to source-label variation).",
        f"- Geography rows: {len(observed_geographies)} combinations; levels={sorted({level for level, _ in observed_geographies})}; names={sorted({name for _, name in observed_geographies})}.",
        "- Spreadsheet geography is Nigeria (national) plus six geopolitical zones; no state-level or market-level price rows are in these sheets.",
        "- The paired PDFs describe price collection across states/LGAs, but those detailed observations are not the structured data in the supplied workbooks.",
        "",
        "SOURCE WORKBOOK PERIODS AND SUPPORTING PDF DATES",
    ]
    for summary in workbook_summaries:
        report_lines.append(
            f"- {summary['period']}: {summary['path'].relative_to(ROOT).as_posix()} | "
            f"{summary['national_item_count']} national item rows, {summary['zone_item_count']} zone item labels, "
            f"{summary['records']} item/geography rows | PDF={summary['source_pdf'] or 'not present'} | "
            f"report date={summary['source_report_date'] or 'not available'} | "
            f"publication month={summary['source_publication_month'] or 'not available'} | "
            f"sheet title period={summary['workbook_sheet_title_period'] or 'unparsed'}"
        )
    report_lines += [
        f"- No paired PDF was present for observation months: {', '.join(no_pdf_months) if no_pdf_months else 'none'}.",
        "- Report Date is published at month/year precision only. source_report_date preserves that text; source_publication_date is blank because no report day is supplied.",
        "- The workbook sheet title is not authoritative when it conflicts with the current-period Average header. The header plus year/month folder was used to select the observation period.",
        f"- Workbook title/current-column mismatches: {len(title_mismatches)}.",
    ]
    for summary in title_mismatches:
        report_lines.append(
            f"  {summary['path'].relative_to(ROOT).as_posix()}: sheet title indicates {summary['workbook_sheet_title_period']}, "
            f"but current Average header and folder identify {summary['period']}."
        )
    report_lines += [
        "",
        "UNITS, CURRENCY, AND MEASUREMENT TYPE",
        "- The workbooks have no dedicated currency or standardized unit column. Currency is NGN, corroborated by Naira notation in paired official PDFs; the two months without PDFs are transparently marked as source-context evidence in currency_evidence.",
        f"- Item labels with a directly stated quantity unit: {sum(bool(row['unit']) for row in all_records):,} of {len(all_records):,} records. Other unit values remain blank; no unit has been inferred from price magnitude.",
        "- Explicit label units are retained as written/parsed (for example 75cl, 2kg, 450g, a 30-piece crate, or a bottle with unspecified volume). Labels that say only 'sold loose' are not assigned a unit by this consolidation.",
        "- National and zonal observations are monthly averages. National price_measure preserves the workbook's exact Average header; zone rows are marked zone_average. No retail/wholesale distinction is supplied in these sheets, so price_type remains reported_monthly_average.",
        "",
        "ITEM DEFINITION, GEOGRAPHY, AND METHODOLOGY CHANGES",
        f"- Distinct original item labels by workbook year: 2024={len(year_labels.get(2024, set()))}, 2025={len(year_labels.get(2025, set()))}, 2026={len(year_labels.get(2026, set()))}.",
        f"- Labels in 2024 vs the combined 2025/2026 set: 2024={len(pre_labels)}, post-rebase={len(post_labels)}, exact strings shared={len(pre_labels & post_labels)}.",
        f"- Labels no longer present after 2024: {', '.join(removed_labels) if removed_labels else 'none'}.",
        f"- Labels introduced/renamed in 2025/2026: {', '.join(new_labels) if new_labels else 'none'}.",
        f"- Exact label changes 2024 to 2025: removed={', '.join(year_label_changes['2024_to_2025']['removed']) or 'none'}; introduced={', '.join(year_label_changes['2024_to_2025']['introduced']) or 'none'}.",
        f"- Exact label changes 2025 to 2026: removed={', '.join(year_label_changes['2025_to_2026']['removed']) or 'none'}; introduced={', '.join(year_label_changes['2025_to_2026']['introduced']) or 'none'}.",
        "- The April 2025 official PDF states that NBS rebased the CPI, updated price/weight reference periods and basket composition, and changed some item specifications; it explicitly warns that some old/new item prices may not be comparable while others remain comparable.",
        "- The report methodology sections describe collection by NBS field staff across states/LGAs and report state averages and a national average. The supplied reports do not show a clear change to that fieldwork description, but the CPI reference periods, weights, basket and some item definitions did change.",
        "- For defensible modeling, January 2025 is marked as a methodology regime boundary. The normalized table does not silently splice item labels or average across that boundary. Where a label is unchanged, comparison still requires a documented item-definition check.",
        "- Geography headings are Nigeria plus North Central, North East, North West, South East, South South, and South West throughout the workbooks. Geographical resolution remains national/zone, not WFP market-level.",
        "- Unit/definition examples in the research scope: the December 2024 PDF narrative describes palm oil as 1 litre, while its workbook label says one bottle without volume; the consolidated unit field preserves the workbook's explicit-but-unsized bottle wording and does not silently convert it. The post-rebase label states 75cl. Rice labels are revised/reclassified; local/imported rice are kept separate; garri remains processed cassava product, not raw cassava.",
        "",
        "EXPLICIT MAPPING TO THE RESEARCH COMMODITIES",
    ]
    for commodity in ("Rice (local)", "Rice (imported)", "Rice (origin unspecified)", "Maize", "Cassava-derived product (processed gari)", "Yam", "Beans", "Tomatoes", "Palm oil"):
        labels = mapping_labels.get(commodity, [])
        report_lines.append(f"- {commodity}: {'; '.join(sorted(labels)) if labels else 'no defensible NBS item label found.'}")
    report_lines += [
        "- Raw cassava: no defensible raw-cassava item label was found. Garri/Gari white and yellow are retained as processed cassava-derived products and are not mapped to raw cassava.",
        "- Rice Medium Grained has no stated local/imported origin and is explicitly left ambiguous. Rice agric sold loose is a candidate local variant but remains flagged for definition review.",
        "- Mapped categories are metadata only; no labels with different product definitions or units are aggregated in nbs_monthly.csv.",
        "",
        "OUTPUTS",
        f"- Consolidated table: {NBS_OUTPUT.relative_to(ROOT).as_posix()} ({len(all_records):,} rows after exact same-source duplicate removal).",
        f"- Coverage report: {NBS_COVERAGE.relative_to(ROOT).as_posix()} ({len(coverage):,} item/unit/geography series rows).",
        "- Original downloaded XLSX and PDF files were not written to or renamed by this process.",
    ]
    if wfp.get("available"):
        direct_wfp_groups = [
            row for row in wfp["groups"]
            if row["mapping_status"] not in {"related_not_equivalent", "ambiguous_do_not_merge"}
        ]
        historical_candidates = sorted(
            (row for row in direct_wfp_groups if row["meets_96_months"]),
            key=lambda row: (row["monthly_observations"], row["calendar_coverage_percent"]),
            reverse=True,
        )
        report_lines += [
            "",
            "WFP CHECK (actual observations only; separate from NBS)",
            f"- Normalized WFP table rows: {wfp['all_rows']:,}; flag row counts: {wfp['flag_counts']}.",
            f"- Pure actual rows: {wfp['actual_rows']:,}; actual coverage: {wfp['actual_month_count']} months; "
            f"{wfp['actual_commodity_count']} commodity labels, {wfp['actual_market_count']} markets, {wfp['actual_state_count']} state labels.",
            f"- Latest pure actual observation date: {wfp['actual_latest_date']}; latest date across all flags: {wfp['all_latest_date']} (not necessarily an actual market observation).",
            f"- For research-scope mapped market series, recent windows are {wfp['recent24_start']} to {ASSESSMENT_MONTH} (24 months) and {wfp['recent36_start']} to {ASSESSMENT_MONTH} (36 months).",
            f"- Mapped actual monthly rows by research grouping: {wfp['scope_counts']}",
            f"- Across direct/defensible variants, the maximum actual-month count in those windows is {max((row['observations_recent_24_months'] for row in direct_wfp_groups), default=0)} (24 months) and {max((row['observations_recent_36_months'] for row in direct_wfp_groups), default=0)} (36 months); {sum(row['observations_in_2026'] > 0 for row in direct_wfp_groups)} series have one 2026 observation.",
            "- No raw-cassava commodity label was found in WFP actual rows; Cassava meal (gari, yellow) and Gari (white) remain processed products, not raw cassava.",
        ]
        report_lines.append("- Previously approved five WFP combinations, audited for historical context only (Sorghum/Millet are outside the current seven-commodity research scope):")
        for row in previously_approved_wfp:
            report_lines.append(
                f"  {row['commodity']} | {row['market']} | {row['unit']} | {row['price_type']} | "
                f"n={row.get('monthly_observations', 0)} | {row.get('first_observation_month', 'none')} to {row.get('latest_observation_month', 'none')} | "
                f"coverage={row.get('calendar_coverage_percent', 0):.2f}% | "
                f"missing={row.get('missing_calendar_months', 0)} | longest_gap={row.get('longest_missing_month_run', 0)} | "
                f"recent24={row.get('observations_recent_24_months', 0)} | recent36={row.get('observations_recent_36_months', 0)} | "
                f"2026={row.get('observations_in_2026', 0)}"
            )
        for row in wfp["groups"][:12]:
            report_lines.append(f"  {_fmt_group(row)}")
        sufficient = [row for row in direct_wfp_groups if row["meets_96_months"] and row["meets_80pct_coverage"] and row["meets_gap_le_3"]]
        report_lines.append(
            f"- Direct/defensible-variant WFP market series meeting all three historical criteria (>=96 months, >=80% coverage, longest gap <=3 months): {len(sufficient)}."
        )
        recent = [row for row in direct_wfp_groups if row["latest_observation_month"] >= "2026-09"]
        report_lines.append(f"- Mapped market series with an actual observation in Sep-Nov 2026: {len(recent)}.")
        report_lines.append("- Direct WFP actual market series with >=96 observations (historically longer, but not necessarily continuous/current):")
        for row in historical_candidates:
            report_lines.append(f"  {_fmt_group(row)}")
        if not historical_candidates:
            report_lines.append("  none")
    else:
        report_lines += ["", "WFP CHECK", f"- Not completed: {wfp.get('reason', 'unknown WFP load issue')}"]
    report_lines += [
        "",
        "INTERPRETATION",
        "- The NBS workbook reporting-month calendar is continuous from November 2024 through May 2026, but item definitions cross an official 2025 CPI rebasing boundary and only 17 post-rebase monthly observations exist per consistently reported item/geography.",
        "- As of 2026-11-19, the latest NBS observation is May 2026; no June-November 2026 workbook was present in the inspected folders. The data do not describe the current November 2026 price level.",
        "- NBS national/zone records can support historical description and, with explicit caveats, exploratory short-horizon baseline experiments. They are too short for reliable model selection, multi-step ML claims, or a 12-month LSTM sequence experiment.",
        "- Do not join NBS national/zone aggregates to WFP market-level prices. No model training or interpolation was performed.",
    ]
    NBS_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    NBS_AUDIT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    readiness_lines = [
        "AgriPrice-NG WFP + NBS Final Data Readiness",
        "Assessment date: 2026-11-19 (Africa/Lagos). No training was performed.",
        "",
        "DECISION",
        "No series in the supplied WFP/NBS data is genuinely ready to support a reliable model-based forecast of November 2026 market conditions.",
        "NBS is the more recent of the sources, but its latest actual reporting month is May 2026 (publication month June 2026) and the post-rebase series have only 17 months. WFP pure actual market observations stop earlier and are irregular. Forecasts must not be described as current observed prices.",
        "",
        "NBS READINESS",
        f"- Files: {len(workbooks)} valid workbooks and {len(pdfs)} paired reports; reporting months {periods[0]} through {periods[-1]} ({len(periods)} consecutive months).",
        f"- Latest NBS actual reporting month: {max(periods) if periods else 'none'}; latest supplied report publication month: {max((row['source_publication_month'] for row in all_records if row['source_publication_month']), default='not available')}.",
        f"- Exact item/unit/geography series counts by observed months: {dict(sorted(coverage_observation_distribution.items()))}; one- and 16-month labels are not silently treated as the 17-month stable post-rebase item series.",
        "- January 2025 CPI basket rebasing is a material methodology/definition boundary; retain pre- and post-rebase regimes separately unless item specifications are independently confirmed comparable.",
        "- For 2025-01 through 2026-05, most stable labels have at most 17 consecutive observations, which is inadequate for a robust chronological train/validation/test evaluation of five ML methods. A 12-month LSTM window would yield only about five supervised targets before reserving test data.",
        "- NBS geography is national and six geopolitical zones, not market-level; unit definitions are absent for most workbook item labels; no raw cassava item is available (gari is processed).",
        "- Suitable now for descriptive analysis and transparent baseline-only exploratory comparisons, not for production/academic claims of reliable multi-step ML forecasting.",
    ]
    direct_nbs_statuses = {"direct_variant", "related_variant", "related_variant_requires_review"}
    post_rebase_nbs: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in coverage:
        if row["mapping_status"] in direct_nbs_statuses and row["first_observation_month"] >= "2025-01":
            post_rebase_nbs[(row["research_commodity_mapping"], row["original_item_label"], row["unit"])].append(row)
    readiness_lines += [
        "- Post-rebase NBS candidates (each geography remains a separate national/zone series; counts shown across those separate geographies):"
    ]
    for (mapping, item, unit), rows in sorted(post_rebase_nbs.items()):
        counts = [int(row["unique_observed_months"]) for row in rows]
        geographies = sorted({f"{row['geography_level']}:{row['geography_name']}" for row in rows})
        gaps = sum(int(row["missing_calendar_months"]) for row in rows)
        readiness_lines.append(
            f"  {mapping} | {item} | unit={unit or 'not stated'} | geographies={len(geographies)} "
            f"({'; '.join(geographies)}) | monthly points={min(counts)}-{max(counts)} | "
            f"{min(row['first_observation_month'] for row in rows)} to {max(row['last_observation_month'] for row in rows)} | "
            f"within-series missing months={gaps}"
        )
    if not post_rebase_nbs:
        readiness_lines.append("  none")
    readiness_lines += ["", "WFP ACTUAL-OBSERVATION READINESS"]
    if wfp.get("available"):
        direct_wfp_groups = [
            row for row in wfp["groups"]
            if row["mapping_status"] not in {"related_not_equivalent", "ambiguous_do_not_merge"}
        ]
        historical_candidates = sorted(
            (row for row in direct_wfp_groups if row["meets_96_months"]),
            key=lambda row: (row["monthly_observations"], row["calendar_coverage_percent"]),
            reverse=True,
        )
        readiness_lines += [
            f"- Pure actual rows: {wfp['actual_rows']:,} of {wfp['all_rows']:,} normalized monthly rows; priceflag categories remain separate: {wfp['flag_counts']}.",
            f"- Latest pure actual observation: {wfp['actual_latest_date']}; latest date in the file across all flags: {wfp['all_latest_date']}. Aggregate/estimated flag dates are not direct observed prices.",
            f"- Research-mapped direct plus separately flagged related/ambiguous actual market series: {len(wfp['groups'])}; direct/defensible variants: {len(direct_wfp_groups)}.",
            f"- Among direct/defensible variants, {sum(row['meets_96_months'] for row in direct_wfp_groups)} have >=96 observations, {sum(row['meets_80pct_coverage'] for row in direct_wfp_groups)} have >=80% calendar coverage, and {sum(row['meets_gap_le_3'] for row in direct_wfp_groups)} have no gap longer than 3 months.",
            f"- Direct/defensible variants meeting all three historical criteria: {sum(row['meets_96_months'] and row['meets_80pct_coverage'] and row['meets_gap_le_3'] for row in direct_wfp_groups)}. Direct/defensible variants with actual prices in Sep-Nov 2026: {sum(row['latest_observation_month'] >= '2026-09' for row in direct_wfp_groups)}.",
            f"- Latest actual records by research grouping: {wfp['scope_counts']}.",
            f"- Maximum actual observations per direct/defensible series in the trailing 24/36 months: {max((row['observations_recent_24_months'] for row in direct_wfp_groups), default=0)}/{max((row['observations_recent_36_months'] for row in direct_wfp_groups), default=0)}; {sum(row['observations_in_2026'] > 0 for row in direct_wfp_groups)} series have one observation in 2026.",
            "- WFP contains no raw-cassava commodity label in actual observations; gari/cassava-meal labels are treated as processed products.",
        ]
        readiness_lines.append("- Previously approved five WFP combinations (historical context only; not current candidates):")
        for row in previously_approved_wfp:
            readiness_lines.append(
                f"  {row['commodity']} | {row['market']} | {row['unit']} | {row['price_type']} | "
                f"n={row.get('monthly_observations', 0)} | {row.get('first_observation_month', 'none')} to {row.get('latest_observation_month', 'none')} | "
                f"coverage={row.get('calendar_coverage_percent', 0):.2f}% | "
                f"missing={row.get('missing_calendar_months', 0)} | longest_gap={row.get('longest_missing_month_run', 0)} | "
                f"recent24={row.get('observations_recent_24_months', 0)} | recent36={row.get('observations_recent_36_months', 0)} | "
                f"2026={row.get('observations_in_2026', 0)}"
            )
        readiness_lines.append("- Direct WFP series with >=96 monthly observations (historically longer; listed separately from related products):")
        for row in historical_candidates:
            readiness_lines.append(f"  {_fmt_group(row)}")
        if not historical_candidates:
            readiness_lines.append("  none")
        recent_candidates = [row for row in wfp["groups"] if row["observations_recent_24_months"] > 0]
        if recent_candidates:
            readiness_lines += ["- Most recent WFP actual candidate lines:"]
            for row in recent_candidates[:10]:
                readiness_lines.append(f"  {_fmt_group(row)}")
        else:
            readiness_lines.append("- No mapped WFP actual market series had observations in the trailing 24-month window.")
    else:
        readiness_lines.append(f"- WFP audit could not be completed: {wfp.get('reason', 'unknown error')}.")
    readiness_lines += [
        "",
        "DEFENSIBLE NEXT USE",
        "1. Preserve NBS item labels, units and national/zone geographies as separate series; explicitly flag the 2025 CPI basket break.",
        "2. Re-run source freshness checks when newer NBS releases or direct WFP market observations are supplied. Require actual reported values, not aggregate/imputed rows, for a claim about current prices.",
        "3. If proceeding with an academic forecasting experiment before longer data are available, limit it to post-rebase NBS series, use a naive baseline and walk-forward/rolling-origin evaluation, and label all results exploratory because the sample is short.",
        "4. Do not train on WFP aggregate/mixed flags as if they were direct market reports and do not combine NBS national/zone averages with WFP market series.",
        "",
        "TRAINING STATUS: No models trained; no missing prices/months filled; no raw inputs modified.",
    ]
    READINESS.parent.mkdir(parents=True, exist_ok=True)
    READINESS.write_text("\n".join(readiness_lines) + "\n", encoding="utf-8")

    after_hashes = _protected_hashes()
    if before_hashes != after_hashes:
        changed = sorted(set(before_hashes) | set(after_hashes))
        changed = [path for path in changed if before_hashes.get(path) != after_hashes.get(path)]
        raise RuntimeError(f"Protected source/WFP hashes changed unexpectedly: {changed}")
    with NBS_AUDIT.open("a", encoding="utf-8") as handle:
        handle.write("\nPROTECTION CHECK\n")
        handle.write(f"- Protected raw NBS files, raw WFP files and processed WFP table hashes unchanged: {len(before_hashes)} files verified.\n")
        handle.write("- data/raw/ files were not modified. The processed NBS CSV was intentionally regenerated for this consolidation.\n")

    return {
        "discovered_files": len(source_files),
        "workbooks": len(workbooks),
        "pdfs": len(pdfs),
        "rows_extracted": sum(summary["records"] for summary in workbook_summaries),
        "rows_unique": len(all_records),
        "exact_duplicates_removed": exact_duplicates_removed,
        "first_month": str(periods[0]) if periods else "",
        "last_month": str(periods[-1]) if periods else "",
        "unique_item_labels": len(unique_items),
        "coverage_rows": len(coverage),
        "source_hashes_verified": len(before_hashes),
        "outputs": [NBS_OUTPUT, NBS_COVERAGE, NBS_AUDIT, READINESS],
    }


if __name__ == "__main__":
    result = consolidate()
    for key, value in result.items():
        if key == "outputs":
            print("outputs:")
            for path in value:
                print(f"  {path.relative_to(ROOT).as_posix()}")
        else:
            print(f"{key}: {value}")
