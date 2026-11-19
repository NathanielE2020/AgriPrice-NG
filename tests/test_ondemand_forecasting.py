from __future__ import annotations

import math
from datetime import date

import pytest

import src.ondemand_forecasting as forecasting
from src.ondemand_forecasting import ForecastingError, assess_nbs_series, assess_wfp_series, generate_nbs_forecast, generate_wfp_forecast


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
        as_of=date(2023, 6, 19),
    )

    origin = record["history"][-1]["month"]
    assert outcome["metadata"]["forecast_origin_month"] == origin
    assert len(outcome["forecast"]) == 6
    assert [row["step"] for row in outcome["forecast"]] == list(range(1, 7))
    assert outcome["forecast"][0]["month"] > origin
    assert outcome["forecast"][-1]["month"] > origin
    assert outcome["metadata"]["validation_months"]
    assert outcome["metadata"]["training_rows"] + outcome["metadata"]["validation_rows"] == outcome["metadata"]["usable_training_rows"]
    assert outcome["metadata"]["training_last_month"] < outcome["metadata"]["validation_first_month"]
    assert outcome["metadata"]["validation_last_month"] == outcome["metadata"]["final_model_fit_last_month"]
    assert outcome["metadata"]["validation_design"].startswith("chronological 80/20 split")
    assert outcome["metadata"]["final_model_fit_rows"] == outcome["metadata"]["usable_training_rows"]
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
    first = generate_wfp_forecast(record, record["series_id"], 3, cache_dir=tmp_path, as_of=date(2023, 6, 19))

    def fail_if_called():
        raise AssertionError("validation estimators were retrained despite a matching runtime artifact")

    monkeypatch.setattr(forecasting, "_make_estimators", fail_if_called)
    second = generate_wfp_forecast(record, record["series_id"], 6, cache_dir=tmp_path, as_of=date(2023, 6, 19))
    assert second["model"] == first["model"]
    assert second["metadata"]["runtime_cache_status"] == "hit"
    assert second["metadata"]["model_reuse"] == "runtime_cache_hit"
    assert len(second["forecast"]) == 6


def test_insufficient_history_reports_actual_usable_counts():
    record = _monthly_record(count=10)
    assessment = assess_wfp_series(record)
    assert not assessment["eligible"]
    assert assessment["usable_training_rows"] < 30
    with pytest.raises(ForecastingError, match="gap-aware supervised rows"):
        generate_wfp_forecast(record, record["series_id"], 1, as_of=date(2019, 10, 19))


def test_latest_calendar_gap_is_not_filled_or_treated_as_previous_month():
    record = _monthly_record(count=54, gap="2023-05")
    assessment = assess_wfp_series(record)
    assert assessment["eligible"]
    assert assessment["missing_calendar_months"] == 1
    assert "2023-05" not in assessment["price_map"]
    june_row = next(row for row in assessment["feature_rows"] if row["month"] == "2023-06")
    assert june_row["prior_observation_months"] == ["2023-02", "2023-03", "2023-04"]
    assert june_row["features"]["months_to_target"] == 2
    assert june_row["features"]["gap_lag_1_to_lag_2"] == 1
    assert "observation_lag_1" in forecasting.FEATURE_COLUMNS
    assert "lag_1" not in forecasting.FEATURE_COLUMNS


def test_recursive_forecast_keeps_historical_gap_missing_in_returned_history(tmp_path):
    record = _monthly_record(count=54, gap="2023-05")
    outcome = generate_wfp_forecast(record, record["series_id"], 3, cache_dir=tmp_path, as_of=date(2023, 6, 19))
    assert [row["month"] for row in outcome["forecast"]] == ["2023-07", "2023-08", "2023-09"]
    assert all(row["temporal_status"] == "future_as_of_generation" for row in outcome["forecast"])
    assert all(row["month"] != "2023-05" for row in outcome["historical"])
    assert max(row["month"] for row in outcome["historical"]) == "2023-06"
    assert outcome["metadata"]["missing_calendar_months"] == 1


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


def test_stale_origin_is_rejected_before_a_long_recursive_bridge(tmp_path):
    record = _monthly_record()
    with pytest.raises(ForecastingError) as caught:
        generate_wfp_forecast(record, record["series_id"], 6, cache_dir=tmp_path, as_of=date(2026, 11, 19))
    assert caught.value.code == "stale_source_data"
    assert "41 calendar month(s) old" in caught.value.message
    assert "cannot be forecast as current market conditions" in caught.value.message


def test_recent_origin_uses_only_a_short_bounded_calendar_bridge(tmp_path):
    record = _monthly_record()
    outcome = generate_wfp_forecast(
        record, record["series_id"], 2, cache_dir=tmp_path,
        as_of=date(2023, 8, 19),
    )
    assert outcome["metadata"]["source_age_months"] == 2
    assert outcome["metadata"]["forecast_bridge_months"] == 2
    assert outcome["metadata"]["freshness_limit_months"] == 3
    assert outcome["metadata"]["max_forecast_bridge_months"] == 3
    assert [row["month"] for row in outcome["forecast"]] == ["2023-09", "2023-10"]


def test_observation_dated_after_generation_day_is_rejected_before_training(monkeypatch):
    record = _monthly_record()
    record["history"][-1]["last_observation_date"] = "2026-11-20"
    monkeypatch.setattr(forecasting, "_make_estimators", lambda: (_ for _ in ()).throw(AssertionError("future data must be rejected before fitting")))
    with pytest.raises(ForecastingError) as caught:
        generate_wfp_forecast(record, record["series_id"], 1, as_of=date(2026, 11, 19))
    assert caught.value.code == "actual_data_after_generation_date"


def test_all_integer_horizons_one_through_six_are_supported(tmp_path):
    record = _monthly_record()
    for horizon in range(1, 7):
        outcome = generate_wfp_forecast(record, record["series_id"], horizon, cache_dir=tmp_path, as_of=date(2023, 6, 19))
        assert len(outcome["forecast"]) == horizon
        assert outcome["forecast"][-1]["month"] == f"2023-{6 + horizon:02d}" if horizon < 7 else False


def test_nbs_assessment_and_forecast_keep_source_identity_separate(tmp_path):
    record = _monthly_record()
    record.update({
        "source": "NBS", "source_key": "nbs", "price_flag": None,
        "market": None, "geography_level": "national", "geography_name": "Nigeria",
    })
    assessment = assess_nbs_series(record)
    assert assessment["eligible"]
    wrong_source = assess_nbs_series(_monthly_record())
    assert not wrong_source["eligible"]
    assert wrong_source["code"] == "source_not_supported"
    outcome = generate_nbs_forecast(record, record["series_id"], 1, cache_dir=tmp_path, as_of=date(2023, 6, 19))
    assert outcome["metadata"]["source"] == "NBS"
    assert outcome["metadata"]["latest_actual_month_in_source"] is None
    assert outcome["metadata"]["source_observation_count"] == 54
    assert outcome["series"]["source_key"] == "nbs"


def test_pooled_fallback_uses_compatible_peers_and_cuts_future_peer_observations(tmp_path):
    from src.ondemand_forecasting import generate_wfp_forecast
    from src.pooled_forecasting import compatibility_key

    def make_market(series_id: str, market: str, count: int) -> dict:
        history = []
        year, month = 2017, 1
        for index in range(count):
            value = 150 + index * 1.7 + (len(market) % 3) * 6
            key = f"{year:04d}-{month:02d}"
            history.append({"month": key, "price": value, "last_observation_date": f"{key}-15", "price_flag": "actual"})
            month += 1
            if month == 13:
                month = 1
                year += 1
        return {
            "series_id": series_id, "source": "WFP", "source_key": "wfp",
            "commodity": "Maize", "market": market, "market_id": series_id,
            "state": "Test", "unit": "KG", "price_type": "Retail",
            "currency": "NGN", "price_flag": "actual", "history": history,
        }

    selected = make_market("short-market", "Selected market", 25)
    peers = [selected] + [make_market(f"peer-{index}", f"Peer {index}", 72) for index in range(5)]
    incompatible = dict(peers[1], commodity="Rice (local)")
    assert compatibility_key(selected, "wfp") != compatibility_key(incompatible, "wfp")

    result = generate_wfp_forecast(
        selected, selected["series_id"], 1, cache_dir=tmp_path,
        as_of=date(2019, 1, 19), pool_records=peers,
    )
    assert result["metadata"]["strategy_level"] == "pooled_global"
    assert result["metadata"]["training_last_month"] == "2019-01"
    assert result["metadata"]["validation_training_target_last_month"] < result["metadata"]["validation_cutoff_month"]
    assert all(item["commodity"] == "Maize" for item in result["metadata"]["other_series_contributors"])
    assert all(row["temporal_status"] == "future_as_of_generation" for row in result["forecast"])


def test_short_exact_history_can_use_a_rolling_origin_baseline_only(tmp_path):
    record = _monthly_record(count=17)
    result = generate_wfp_forecast(
        record, record["series_id"], 2, cache_dir=tmp_path,
        as_of=date(2020, 6, 19), pool_records=[record],
    )
    metadata = result["metadata"]
    assert metadata["strategy_level"] == "individual_series_short_history_baseline"
    assert metadata["model_type"] == "validated baseline"
    assert metadata["validation_rows"] == 6
    assert len(metadata["validation_predictions"]) == metadata["validation_rows"]
    assert {item["month"] for item in metadata["validation_predictions"]} == set(metadata["validation_months"])
    assert metadata["selection_basis"].startswith("lowest RMSE among simple baselines")
    assert result["forecast"]
