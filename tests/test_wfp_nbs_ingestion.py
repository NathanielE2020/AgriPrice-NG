from __future__ import annotations

import csv
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_new_original_wfp_and_nbs_files_exist_in_the_canonical_locations():
    protected_sources = [
        ROOT / "data/raw/wfp/wfp_food_prices_nga.csv",
        ROOT / "data/raw/wfp/wfp_markets_nga.csv",
        ROOT / "data/raw/nbs/2024",
        ROOT / "data/raw/nbs/2025",
        ROOT / "data/raw/nbs/2026",
    ]
    if not all(path.exists() for path in protected_sources):
        pytest.skip("Original source data is intentionally excluded from the release checkout.")
    assert (ROOT / "data/raw/wfp/wfp_food_prices_nga.csv").is_file()
    assert (ROOT / "data/raw/wfp/wfp_markets_nga.csv").is_file()
    assert len(list((ROOT / "data/raw/nbs/2024").rglob("*.xlsx"))) == 2
    assert len(list((ROOT / "data/raw/nbs/2025").rglob("*.xlsx"))) == 12
    assert len(list((ROOT / "data/raw/nbs/2026").rglob("*.xlsx"))) == 5
    assert len(list((ROOT / "data/raw/nbs/2024").rglob("*.pdf"))) == 2
    assert len(list((ROOT / "data/raw/nbs/2025").rglob("*.pdf"))) == 10
    assert len(list((ROOT / "data/raw/nbs/2026").rglob("*.pdf"))) == 5


def test_nbs_normalization_has_only_reported_months_and_preserves_unknown_units():
    path = ROOT / "data/processed/nbs_monthly.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5_600
    assert {row["observation_month"] for row in rows} == (
        {"2024-11", "2024-12"}
        | {f"2025-{month:02d}" for month in range(1, 13)}
        | {f"2026-{month:02d}" for month in range(1, 6)}
    )
    assert {row["currency"] for row in rows} == {"NGN"}
    assert any(row["unit"] for row in rows)
    assert any(not row["unit"] for row in rows)
    assert {row["geography_level"] for row in rows} == {"national", "zone"}
    assert all(row["price"] != "" for row in rows)
    assert all(row["source_relative_path"].startswith("data/raw/nbs/") for row in rows)
    assert any(row["source_report_date"] for row in rows)
    assert any(row["mapping_status"] == "related_not_equivalent" for row in rows)
    assert not any(
        row["item_label"].casefold().startswith("garri") and row["research_commodity_mapping"] == "Cassava"
        for row in rows
    )


def test_source_coverage_reports_flag_irregular_series_and_short_nbs_panel():
    with (ROOT / "reports/wfp_nbs_series_coverage.csv").open(encoding="utf-8-sig", newline="") as handle:
        wfp = list(csv.DictReader(handle))
    assert len(wfp) == 5
    assert all(row["ready_for_forecasting"] == "False" for row in wfp)
    assert all(int(row["missing_calendar_months"]) > 0 for row in wfp)

    with (ROOT / "reports/nbs_2024_2026_coverage.csv").open(encoding="utf-8-sig", newline="") as handle:
        nbs = list(csv.DictReader(handle))
    assert len(nbs) == 589
    assert {int(row["observation_count"]) for row in nbs}.issubset({1, 2, 16, 17, 19})
    assert all(row["ready_for_reliable_ml_forecasting"] == "False" for row in nbs)
    assert all(int(row["missing_price_count"]) == 0 for row in nbs)
