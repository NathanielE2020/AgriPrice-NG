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
    result = generate_wfp_forecast(record, record["series_id"], 1, as_of=date(2019, 10, 19))
    assert result["metadata"]["strategy_level"] == "exact_series_fallback"
    assert result["metadata"]["validation_metrics"]["n"] == 0
    assert len(result["forecast"]) == 1


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


def test_one_positive_observation_gets_a_carry_forward_forecast(tmp_path):
    record = _monthly_record(count=1)
    result = generate_wfp_forecast(record, record["series_id"], 3, cache_dir=tmp_path, as_of=date(2026, 11, 19))
    assert result["model"] == "last_price_carry_forward"
    assert [row["price"] for row in result["forecast"]] == [record["history"][0]["price"]] * 3
    assert result["metadata"]["confidence_level"] == "very low"
    assert result["metadata"]["validation_metrics"]["n"] == 0


def test_one_positive_nbs_observation_gets_a_nonzero_carry_forward_forecast(tmp_path):
    record = _monthly_record(count=1)
    record.update({
        "series_id": "nbs-test-yam-national-post-cpi",
        "source": "NBS",
        "source_key": "nbs",
        "price_flag": None,
        "market": "",
        "market_id": "",
        "state": "",
        "geography_level": "national",
        "geography_name": "Nigeria",
        "commodity": "Yam Tuber",
        "unit": "Not stated in workbook",
        "verified_unit": "1kg",
        "methodology_regime": "post_2025_cpi_rebase",
    })
    last_price = record["history"][0]["price"]
    result = generate_nbs_forecast(
        record, record["series_id"], 3, cache_dir=tmp_path, as_of=date(2019, 2, 19),
    )
    assert result["metadata"]["source"] == "NBS"
    assert result["model"] == "last_price_carry_forward"
    assert [row["price"] for row in result["forecast"]] == [last_price] * 3
    assert all(math.isfinite(row["price"]) and row["price"] > 0 for row in result["forecast"])


def test_april_actual_origin_displays_december_future_forecast_with_one_observation(tmp_path):
    record = _monthly_record(count=1)
    record["history"][0].update({"month": "2026-04", "last_observation_date": "2026-04-15"})
    result = generate_wfp_forecast(record, record["series_id"], 6, cache_dir=tmp_path, as_of=date(2026, 11, 19))
    assert [row["month"] for row in result["forecast"]] == [
        "2026-12", "2027-01", "2027-02", "2027-03", "2027-04", "2027-05",
    ]
    assert [row["price"] for row in result["forecast"]] == [record["history"][0]["price"]] * 6
    assert result["historical"] == record["history"]
    assert result["metadata"]["forecast_origin_month"] == "2026-04"
    assert result["metadata"]["forecast_bridge_months"] == 7


@pytest.mark.parametrize(
    ("prices", "should_rise"),
    [([100.0, 110.0, 121.0, 133.1], True), ([133.1, 121.0, 110.0, 100.0], False)],
)
def test_robust_movement_tracks_supported_rising_and_falling_directions(prices, should_rise, tmp_path):
    record = _monthly_record(count=len(prices))
    months = ["2026-01", "2026-02", "2026-03", "2026-04"]
    record["history"] = [
        {"month": month, "price": price, "last_observation_date": f"{month}-15"}
        for month, price in zip(months, prices)
    ]
    outcome = generate_wfp_forecast(record, record["series_id"], 2, cache_dir=tmp_path, as_of=date(2026, 11, 19))
    first = outcome["forecast"][0]["price"]
    assert (first > prices[-1]) is should_rise
    assert all(math.isfinite(row["price"]) and row["price"] > 0 for row in outcome["forecast"])


def test_irregular_observation_gap_is_month_normalized_and_shrunk(tmp_path):
    record = _monthly_record(count=2)
    record["history"] = [
        {"month": "2026-01", "price": 100.0, "last_observation_date": "2026-01-15"},
        {"month": "2026-04", "price": 133.1, "last_observation_date": "2026-04-15"},
    ]
    movement = forecasting._robust_monthly_movement({"2026-01": 100.0, "2026-04": 133.1})
    expected_log_change_per_month = math.log(133.1 / 100.0) / 3
    assert movement["change_count"] == 1
    assert movement["monthly_log_rate"] == pytest.approx(expected_log_change_per_month * 0.5)
    outcome = generate_wfp_forecast(record, record["series_id"], 3, cache_dir=tmp_path, as_of=date(2026, 11, 19))
    expected_first = 133.1 * math.exp(movement["monthly_log_rate"] * 8)
    expected_second = expected_first * math.exp(movement["monthly_log_rate"])
    assert outcome["forecast"][0]["price"] == pytest.approx(expected_first)
    assert outcome["forecast"][1]["price"] == pytest.approx(expected_second)
    assert outcome["forecast"][0]["percentage_change_from_origin"] == pytest.approx((expected_first / 133.1 - 1) * 100)
    assert [row["month"] for row in outcome["forecast"]] == ["2026-12", "2027-01", "2027-02"]
    assert outcome["metadata"]["forecast_origin_month"] == "2026-04"
    assert outcome["metadata"]["forecast_bridge_months"] == 7
    assert outcome["metadata"]["forecast_start_month"] == "2026-12"
    assert outcome["historical"][-1]["month"] == "2026-04"
    assert outcome["metadata"]["latest_observation_date_in_source"] == "2026-04-15"
    assert "not actual observations" in outcome["warning"]


def test_horizons_use_calendar_months_and_cross_year_from_observation_origin(tmp_path):
    record = _monthly_record(count=1)
    record["history"][0]["month"] = "2026-11"
    record["history"][0]["last_observation_date"] = "2026-11-15"
    expected = ["2026-12", "2027-01", "2027-02", "2027-03", "2027-04", "2027-05"]
    for horizon in range(1, 7):
        outcome = generate_wfp_forecast(record, record["series_id"], horizon, cache_dir=tmp_path, as_of=date(2026, 11, 19))
        assert [row["month"] for row in outcome["forecast"]] == expected[:horizon]
        assert [row["step"] for row in outcome["forecast"]] == list(range(1, horizon + 1))


def test_current_month_is_bridged_and_not_displayed_as_a_forecast(tmp_path):
    record = _monthly_record(count=1)
    record["history"][0]["month"] = "2026-10"
    record["history"][0]["last_observation_date"] = "2026-10-15"
    result = generate_wfp_forecast(record, record["series_id"], 1, cache_dir=tmp_path, as_of=date(2026, 11, 19))
    assert result["forecast"][0]["month"] == "2026-12"
    assert result["forecast"][0]["temporal_status"] == "future_as_of_generation"
    assert result["metadata"]["forecast_classification"] == "future_as_of_generation"
    assert result["metadata"]["forecast_bridge_months"] == 1
    assert result["metadata"]["is_current_market_forecast"] is True


def test_future_observation_is_never_used_to_move_forecast_origin():
    record = _monthly_record(count=2)
    record["history"][-1].update({"month": "2026-12", "last_observation_date": "2026-12-15"})
    with pytest.raises(ForecastingError) as caught:
        generate_wfp_forecast(record, record["series_id"], 1, as_of=date(2026, 11, 19))
    assert caught.value.code == "actual_data_after_generation_date"


def test_recursive_forecast_keeps_historical_gap_missing_in_returned_history(tmp_path):
    record = _monthly_record(count=54, gap="2023-05")
    outcome = generate_wfp_forecast(record, record["series_id"], 3, cache_dir=tmp_path, as_of=date(2023, 6, 19))
    assert [row["month"] for row in outcome["forecast"]] == ["2023-07", "2023-08", "2023-09"]
    assert all(row["temporal_status"] == "future_as_of_generation" for row in outcome["forecast"])
    assert all(row["month"] != "2023-05" for row in outcome["historical"])
    assert max(row["month"] for row in outcome["historical"]) == "2023-06"
    assert outcome["metadata"]["missing_calendar_months"] == 1


def test_aggregate_or_mixed_wfp_price_flags_are_kept_separate_and_forecast_as_their_own_series(tmp_path):
    record = _monthly_record()
    record["price_flag"] = "actual,aggregate"
    assessment = assess_wfp_series(record)
    assert assessment["eligible"]
    result = generate_wfp_forecast(record, record["series_id"], 2, cache_dir=tmp_path, as_of=date(2023, 6, 19))
    assert result["metadata"]["price_flag"] == "actual,aggregate"
    assert result["metadata"]["latest_actual_month_in_source"] is None
    assert "not direct actuals" in result["metadata"]["historical_price_label"]


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


def test_stale_origin_gets_future_window_with_stale_warning(tmp_path):
    record = _monthly_record()
    result = generate_wfp_forecast(record, record["series_id"], 6, cache_dir=tmp_path, as_of=date(2026, 11, 19))
    assert [row["month"] for row in result["forecast"]] == ["2026-12", "2027-01", "2027-02", "2027-03", "2027-04", "2027-05"]
    assert result["metadata"]["source_age_months"] == 41
    assert result["metadata"]["forecast_classification"] == "future_as_of_generation"
    assert result["metadata"]["forecast_bridge_months"] == 41
    assert result["metadata"]["is_current_market_forecast"] is False
    assert "stale" in result["warning"].lower()


def test_recent_origin_bridges_to_months_after_generation_month(tmp_path):
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
        "commodity": "Rice local sold loose", "methodology_regime": "post_2025_cpi_rebase",
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
    assert outcome["metadata"]["commodity"] == "Rice local sold loose"
    assert outcome["metadata"]["geography_name"] == "Nigeria"
    assert outcome["metadata"]["methodology_regime"] == "post_2025_cpi_rebase"


def test_world_bank_close_estimate_is_not_accepted_as_wfp_actual_data():
    record = _monthly_record(count=3)
    record.update({"source": "world_bank", "source_key": "world_bank", "price_type": "close_estimate", "is_model_estimate": True})
    assessment = assess_wfp_series(record)
    assert not assessment["eligible"]
    assert assessment["code"] == "source_not_supported"


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
        as_of=date(2022, 6, 19), pool_records=peers,
    )
    assert result["metadata"]["strategy_level"] == "pooled_global"
    assert result["metadata"]["training_last_month"] == "2019-01"
    assert result["metadata"]["validation_training_target_last_month"] < result["metadata"]["validation_cutoff_month"]
    assert all(item["commodity"] == "Maize" for item in result["metadata"]["other_series_contributors"])
    assert all(row["temporal_status"] == "future_as_of_generation" for row in result["forecast"])
    assert result["forecast"][0]["month"] == "2022-07"
    assert result["metadata"]["forecast_bridge_months"] == 41


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
