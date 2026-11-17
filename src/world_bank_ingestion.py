"""World Bank Nigeria food-price ingestion and normalization.

The World Bank release is intentionally kept separate from the WFP pipeline.
This module reads the downloaded CSV files, validates the wide estimate table,
and writes a normalized, source-labelled monthly table under
``data/processed/world_bank``.  It never edits anything below ``data/raw`` and
never fills missing prices.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORLD_BANK_ROOT = ROOT / "data" / "raw" / "external" / "world_bank_nigeria_2026"
EXTRACTED_ROOT = WORLD_BANK_ROOT / "extracted"
PROCESSED_ROOT = ROOT / "data" / "processed" / "world_bank"
RAW_FILE_NAME = "NGA_RTFP_mkt_2007_2026-11-02.csv"
DETAIL_FILE_NAME = "NGA_RTP_details_mkt_2007_2026-11-02.csv"
TICKER_FILE_NAME = "NGA_RTP_ticker_info_mkt_2007_2026-11-02.csv"
NORMALIZED_FILE = PROCESSED_ROOT / "world_bank_monthly.csv"
METADATA_FILE = PROCESSED_ROOT / "source_metadata.json"

REQUIRED_COLUMNS = {
    "ISO3",
    "country",
    "adm1_name",
    "adm2_name",
    "mkt_name",
    "geo_id",
    "price_date",
    "currency",
    "start_dense_data",
    "last_survey_point",
}


class WorldBankDataError(ValueError):
    """Raised when the downloaded World Bank release is invalid."""


def source_paths(root: Path = EXTRACTED_ROOT) -> dict[str, Path]:
    return {
        "prices": root / RAW_FILE_NAME,
        "details": root / DETAIL_FILE_NAME,
        "tickers": root / TICKER_FILE_NAME,
    }


def _read_tickers(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise WorldBankDataError(f"Ticker metadata is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ticker", "full_name", "units"}
    if not rows or not required.issubset(rows[0]):
        raise WorldBankDataError("Ticker metadata does not contain ticker, full_name and units columns")
    return {str(row["ticker"]): row for row in rows}


def _read_details(path: Path) -> dict[str, str]:
    if not path.exists():
        raise WorldBankDataError(f"Details metadata is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def _geography(row: pd.Series) -> tuple[str, str | None]:
    """Classify records without counting aggregate labels as states/markets."""

    adm1 = str(row["adm1_name"])
    market = str(row["mkt_name"])
    if adm1 == "National Average":
        return "national", "National Average"
    if adm1 == "Market Average" or market == "Market Average":
        return "market_average", "Market Average"
    if adm1 == "Geopolitical Zone":
        return "geopolitical_zone", str(row["adm2_name"])
    # For a market-level series the market label, not the state label, is the
    # location key.  The state is retained separately below.
    return "market", market


def load_world_bank_raw(
    root: Path = EXTRACTED_ROOT,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]], dict[str, str]]:
    """Load and validate the source release without mutating it."""

    paths = source_paths(root)
    for key, path in paths.items():
        if not path.exists():
            raise WorldBankDataError(f"World Bank {key} file is missing: {path}")
    frame = pd.read_csv(paths["prices"], low_memory=False)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise WorldBankDataError(f"World Bank price file is missing required columns: {missing}")
    frame["price_date"] = pd.to_datetime(frame["price_date"], format="%Y-%m-%d", errors="coerce")
    if frame["price_date"].isna().any():
        raise WorldBankDataError("World Bank price file contains invalid price_date values")
    tickers = _read_tickers(paths["tickers"])
    details = _read_details(paths["details"])
    estimate_columns = [column for column in frame.columns if column.startswith("c_")]
    for column in estimate_columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].dropna() < 0).any():
            raise WorldBankDataError(f"Negative values found in {column}")
    duplicates = frame.duplicated(["price_date", "geo_id"])
    if duplicates.any():
        raise WorldBankDataError("Duplicate World Bank location/month records were found")
    return frame, tickers, details


def normalize_world_bank(
    frame: pd.DataFrame,
    tickers: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Convert wide ``c_``/``l_``/``h_`` fields to one row per series/month."""

    rows: list[dict[str, Any]] = []
    source_records = frame.to_dict("records")
    for ticker, metadata in tickers.items():
        close_column = f"c_{ticker}"
        if close_column not in frame.columns:
            continue
        for source in source_records:
            # _geography only needs the location fields; using a small Series
            # here keeps the validation helper readable without the very slow
            # row-wise pandas iterrows path.
            location = pd.Series(source)
            level, state_or_zone = _geography(location)
            close = source[close_column]
            if pd.isna(close):
                # Missing target prices remain missing; there is no forward fill.
                continue
            base_observation = source.get(ticker)
            rows.append(
                {
                    "source": "world_bank",
                    "source_release": "NGA_RTFP_mkt_2007_2026-11-02",
                    "date": source["price_date"].strftime("%Y-%m-%d") if hasattr(source["price_date"], "strftime") else str(source["price_date"]),
                    "month": source["price_date"].strftime("%Y-%m") if hasattr(source["price_date"], "strftime") else str(source["price_date"])[:7],
                    "commodity_ticker": ticker,
                    "commodity": metadata.get("full_name") or ticker,
                    "unit": metadata.get("units") or "",
                    "price_type": "close_estimate",
                    "price": float(close),
                    "low_estimate": float(source[f"l_{ticker}"]) if pd.notna(source.get(f"l_{ticker}")) else None,
                    "high_estimate": float(source[f"h_{ticker}"]) if pd.notna(source.get(f"h_{ticker}")) else None,
                    "trust_score": float(source[f"trust_{ticker}"]) if pd.notna(source.get(f"trust_{ticker}")) else None,
                    "base_observation_price": float(base_observation) if pd.notna(base_observation) else None,
                    "is_model_estimate": True,
                    "geography_level": level,
                    "geography_name": state_or_zone,
                    "state": str(source["adm1_name"]) if level == "market" else None,
                    "admin1_label": str(source["adm1_name"]),
                    "admin2": str(source["adm2_name"]),
                    "market": str(source["mkt_name"]),
                    "geo_id": str(source["geo_id"]),
                    "latitude": float(source["lat"]) if pd.notna(source.get("lat")) else None,
                    "longitude": float(source["lon"]) if pd.notna(source.get("lon")) else None,
                    "currency": str(source["currency"]),
                    "data_coverage": float(source["data_coverage"]),
                    "data_coverage_recent": float(source["data_coverage_recent"]),
                    "index_confidence_score": float(source["index_confidence_score"]),
                    "spatially_interpolated": bool(source["spatially_interpolated"]),
                    "last_survey_point": str(source["last_survey_point"]),
                    "source_data_is_direct_observation": bool(pd.notna(base_observation)),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise WorldBankDataError("No World Bank close-estimate rows were available")
    result = result.sort_values(["commodity_ticker", "geography_level", "geography_name", "date"]).reset_index(drop=True)
    duplicate_key = ["commodity_ticker", "geo_id", "month", "unit", "price_type"]
    if result.duplicated(duplicate_key).any():
        raise WorldBankDataError("Normalized World Bank data contains duplicate series/month rows")
    return result


def prepare_world_bank_data(
    root: Path = EXTRACTED_ROOT,
    output_path: Path = NORMALIZED_FILE,
) -> dict[str, Any]:
    frame, tickers, details = load_world_bank_raw(root)
    normalized = normalize_world_bank(frame, tickers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    metadata = {
        "source": "World Bank Nigeria Real Time Food Prices (NGA_RTFP_mkt_2007_2026-11-02)",
        "raw_file": str(source_paths(root)["prices"].relative_to(ROOT)),
        "raw_rows": int(len(frame)),
        "normalized_rows": int(len(normalized)),
        "date_start": normalized["date"].min(),
        "date_end": normalized["date"].max(),
        "commodities": sorted(normalized["commodity_ticker"].unique().tolist()),
        "missing_close_by_ticker": {
            ticker: int(frame[f"c_{ticker}"].isna().sum())
            for ticker in tickers
            if f"c_{ticker}" in frame.columns
        },
        "geography_levels": normalized["geography_level"].value_counts().to_dict(),
        "state_labels": sorted(normalized.loc[normalized["geography_level"] == "market", "state"].dropna().unique().tolist()),
        "market_labels": sorted(normalized.loc[normalized["geography_level"] == "market", "market"].dropna().unique().tolist()),
        "details": details,
        "price_definition": "c_ close estimate; World Bank modelled/imputed estimate, not necessarily a direct survey observation",
        "missing_target_policy": "Rows with missing close estimates are omitted from the normalized table; no interpolation or forward fill is applied.",
    }
    import json

    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"raw": frame, "normalized": normalized, "tickers": tickers, "details": details, "metadata": metadata}


if __name__ == "__main__":
    result = prepare_world_bank_data()
    print(f"World Bank raw rows: {len(result['raw']):,}")
    print(f"Normalized rows: {len(result['normalized']):,}")
    print(f"Date range: {result['metadata']['date_start']} to {result['metadata']['date_end']}")
