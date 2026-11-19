from __future__ import annotations

import math
from datetime import date

import pytest

import src.ondemand_forecasting as forecasting
from src.ondemand_forecasting import ForecastingError, assess_wfp_series, generate_wfp_forecast


def _monthly_record(count: int = 54, gap: str | None = None) -> dict:
    year, month = 2019, 1
    history = []
    for index in range(count):
        key = f"{year:04d}-{month:02d}"
        if key != gap:
            price = 100.0 + 1.4 * index + 8.0 * math.sin(2.0 * math.pi * month / 12.0)
            history.append({
                "month": key,
                "price": price,
                "last_observation_date": f"{key}-15",
                "price_flag": "actual",
            })
        month += 1
        if month == 13:
            month = 1
            year += 1
    return {
        "series_id": "wfp-test-maize-market-actual",
        "source": "WFP",
        "source_key": "wfp",
        "commodity": "Maize",
        "market": "Test market",
        "market_id": "999",
        "state": "Test state",
        "unit": "KG",
        "price_type": "Retail",
        "currency": "NGN",
        "price_flag": "actual",
        "history": history,
    }


def test_on_demand_forecast_uses_chronological_validation_and_future_dates(tmp_path):
    record = _monthly_record()
    outcome = generate_wfp_forecast(
        record,
        record["series_id"],
        6,
        cache_dir=tmp_path,
        as_of=date(2023, 6, 1),
    )

    origin = record["history"][-1]["month"]
    assert outcome["metadata"]["forecast_origin_month"] == origin
    assert len(outcome["forecast"]) == 6
    assert [row["step"] for row in outcome["forecast"]] == list(range(1, 7))
    assert outcome["forecast"][0]["month"] > origin
    assert outcome["forecast"][-1]["month"] > origin
    assert outcome["metadata"]["validation_months"]
    assert outcome["metadata"]["selection_basis"].startswith("lowest chronological validation RMSE")
    comparison = outcome["metadata"]["model_comparison_validation"]
    assert {row["n"] for row in comparison} == {outcome["metadata"]["validation_rows"]}
    selected = outcome["metadata"]["validation_metrics"]
    assert selected == {
        key: value for key, value in next(row for row in comparison if row["model"] == outcome["model"]).items()
        if key != "model"
    }
    assert outcome["metadata"]["runtime_cache_status"] == "saved"
    assert outcome["metadata"]["confidence_intervals"] is False


def test_runtime_cache_reuses_model_without_retraining(tmp_path, monkeypatch):
    record = _monthly_record()
    first = generate_wfp_forecast(record, record["series_id"], 3, cache_dir=tmp_path, as_of=date(2023, 6, 1))

    def fail_if_called():
        raise AssertionError("validation estimators were retrained despite a matching runtime artifact")

    monkeypatch.setattr(forecasting, "_make_estimators", fail_if_called)
    second = generate_wfp_forecast(record, record["series_id"], 6, cache_dir=tmp_path, as_of=date(2023, 6, 1))
    assert second["model"] == first["model"]
    assert second["metadata"]["runtime_cache_status"] == "hit"
    assert second["metadata"]["model_reuse"] == "runtime_cache_hit"
    assert len(second["forecast"]) == 6


def test_insufficient_history_reports_actual_usable_counts():
    record = _monthly_record(count=20)
    assessment = assess_wfp_series(record)
    assert not assessment["eligible"]
    assert assessment["usable_training_rows"] < 30
    with pytest.raises(ForecastingError, match="complete calendar-lag training rows"):
        generate_wfp_forecast(record, record["series_id"], 1)


def test_latest_calendar_gap_is_not_filled_or_treated_as_previous_month():
    record = _monthly_record(count=54, gap="2023-05")
    assessment = assess_wfp_series(record)
    assert not assessment["eligible"]
    assert "2023-05" in assessment["reason"]
    assert assessment["missing_calendar_months"] == 1


def test_aggregate_or_mixed_wfp_price_flags_are_rejected():
    record = _monthly_record()
    record["price_flag"] = "actual,aggregate"
    assessment = assess_wfp_series(record)
    assert not assessment["eligible"]
    assert assessment["code"] == "not_direct_actual"
    assert "pure priceflag=actual" in assessment["reason"]


def test_conflicting_duplicate_month_is_not_silently_aggregated():
    record = _monthly_record()
    duplicate = dict(record["history"][-1])
    duplicate["price"] += 10
    record["history"].append(duplicate)
    assessment = assess_wfp_series(record)
    assert not assessment["eligible"]
    assert assessment["code"] == "conflicting_monthly_observations"


def test_invalid_prices_are_excluded_without_interpolation():
    record = _monthly_record()
    record["history"][4]["price"] = 0
    assessment = assess_wfp_series(record)
    assert assessment["invalid_price_rows"] == 1
    assert assessment["missing_calendar_months"] == 1
