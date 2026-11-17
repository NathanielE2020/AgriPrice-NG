"""Data loading and approved-series filtering for AgriPrice-NG."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .parquet_compat import read_parquet_rows, to_date


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PARQUET_PATH = RAW_DIR / "train.parquet"
MARKET_METADATA_PATH = RAW_DIR / "wfp_markets_nga.csv"

APPROVED_SERIES = (
    ("Sorghum (white)", "Dawanau", "100 KG", "Wholesale"),
    ("Maize (white)", "Dawanau", "100 KG", "Wholesale"),
    ("Millet", "Dawanau", "100 KG", "Wholesale"),
    ("Maize", "Illela (CBM)", "KG", "Wholesale"),
    ("Sorghum", "Mai Adoua (CBM)", "KG", "Wholesale"),
)


def series_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row["commodity"]),
        str(row["market"]),
        str(row["unit"]),
        str(row["pricetype"]),
    )


def load_all_rows() -> list[dict]:
    rows = read_parquet_rows(PARQUET_PATH)
    for row in rows:
        row["date"] = to_date(row["date"])
    return rows


def load_approved_rows() -> tuple[list[dict], dict]:
    rows = load_all_rows()
    approved = set(APPROVED_SERIES)
    filtered = [row for row in rows if series_key(row) in approved]
    counts = Counter(series_key(row) for row in filtered)
    missing_series = [series for series in APPROVED_SERIES if counts[series] == 0]
    if missing_series:
        raise ValueError(f"Approved series missing from dataset: {missing_series}")

    # The inspection found no duplicate observation keys. Keep this assertion
    # so a later dataset revision cannot silently overwrite an observation.
    seen: set[tuple] = set()
    duplicates: list[tuple] = []
    for row in filtered:
        key = (*series_key(row), row["date"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate approved-series observations found: {duplicates[:5]}")
    return filtered, dict(counts)


def load_market_metadata() -> list[dict]:
    if not MARKET_METADATA_PATH.exists():
        return []
    with MARKET_METADATA_PATH.open(newline="", encoding="utf-8-sig") as file_obj:
        rows = list(csv.reader(file_obj))
    if len(rows) <= 2:
        return []
    header = rows[0]
    return [dict(zip(header, row)) for row in rows[2:] if row]
