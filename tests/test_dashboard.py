from __future__ import annotations

from pathlib import Path

import pytest

from app import create_app
from src.wfp_nbs_dashboard_service import DashboardService
from src.ondemand_forecasting import ForecastingError, assess_wfp_series


@pytest.fixture(scope="module")
def service():
    return DashboardService()


@pytest.fixture(scope="module")
def client(service):
    return create_app(service).test_client()


def test_dashboard_page_and_overview_use_only_migrated_sources(client):
    page = client.get("/")
    assert page.status_code == 200
    assert b"WFP Nigeria" in page.data
    assert b"NBS monthly summaries" in page.data
    assert b"World Bank" not in page.data

    overview = client.get("/api/overview").get_json()
    assert overview["available_datasets"] == ["wfp", "nbs"]
    assert overview["default_dataset"] == "wfp"
    assert set(overview["sources"]) == {"wfp", "nbs"}
    assert overview["model_status"]["available_model_count"] == 29
    assert overview["model_status"]["trained_on_current_sources"] is True
    assert overview["model_status"]["historical_forecast_series_count"] == 7
    assert overview["model_status"]["current_market_forecast_series_count"] == 0


def test_latest_value_dom_matches_render_latest_and_local_assets_load(client):
    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    for element_id in (
        "analysisLatestPrice",
        "analysisLatestContext",
        "analysisLatestDate",
        "analysisLatestAge",
        "analysisFreshness",
        "analysisFreshnessExplanation",
        "forecastLatestPrice",
        "forecastLatestContext",
        "forecastLatestDate",
        "forecastLatestAge",
        "forecastFreshness",
    ):
        assert f'id="{element_id}"' in html

    for asset_path in ("/static/js/dashboard.js", "/static/css/dashboard.css"):
        assert client.get(asset_path).status_code == 200
    javascript = client.get("/static/js/dashboard.js").get_data(as_text=True)
    assert 'mode: "on_demand"' in javascript
    assert "state.forecastRequestId" in javascript


def test_overview_facts_match_prepared_source_data(client):
    overview = client.get("/api/overview").get_json()
    wfp = overview["datasets"]["wfp"]
    nbs = overview["datasets"]["nbs"]
    assert wfp["historical_observations"] == 91_216
    assert wfp["commodity_count"] == 43
    assert wfp["market_count"] == 68
    assert wfp["state_count"] == 14
    assert wfp["earliest_observation_date"] == "2002-01-15"
    assert wfp["latest_observation_date"] == "2026-11-15"
    assert wfp["latest_actual_observation_date"] == "2026-04-15"
    assert set(wfp["price_flags"]) == {"actual", "aggregate", "actual,aggregate"}
    assert nbs["historical_observations"] == 5_600
    assert nbs["date_start"] == "2024-11"
    assert nbs["date_end"] == "2026-05"
    assert nbs["commodity_count"] == 86
    assert nbs["market_count"] is None
    assert nbs["state_count"] is None
    assert nbs["geography_count"] == 7


def test_series_catalogue_keeps_source_identities_separate(client):
    payload = client.get("/api/series").get_json()
    assert set(payload) == {"wfp", "nbs"}
    assert len(payload["wfp"]) == 3_432
    assert len(payload["nbs"]) == 589
    assert all(item["source"] == "WFP" for item in payload["wfp"])
    assert all(item["source"] == "NBS" for item in payload["nbs"])
    assert all(item["price_flag"] for item in payload["wfp"])
    assert all(item["geography_level"] in {"national", "zone"} for item in payload["nbs"])


def test_approved_wfp_market_series_are_present_as_actual_series(client):
    series = client.get("/api/series?dataset=wfp").get_json()
    approved = {
        ("Sorghum (white)", "Dawanau", "100 KG", "Wholesale"),
        ("Maize (white)", "Dawanau", "100 KG", "Wholesale"),
        ("Millet", "Dawanau", "100 KG", "Wholesale"),
        ("Maize", "Illela (CBM)", "KG", "Wholesale"),
        ("Sorghum", "Mai Adoua (CBM)", "KG", "Wholesale"),
    }
    found = {
        (item["commodity"], item["market"], item["unit"], item["price_type"])
        for item in series if item["price_flag"] == "actual"
    }
    assert approved.issubset(found)


def test_forecast_catalogue_only_exposes_exact_pure_actual_wfp_ids(client, service):
    catalogue = client.get("/api/series?dataset=wfp").get_json()
    actual = [item for item in catalogue if item["price_flag"] == "actual"]
    non_actual = [item for item in catalogue if item["price_flag"] != "actual"]
    assert actual and non_actual
    assert all(item["forecast_request_available"] for item in actual)
    assert not any(item["forecast_request_available"] for item in non_actual)

    javascript = client.get("/static/js/dashboard.js").get_data(as_text=True)
    assert "const forecastItems = items.filter((item) => isPureActualForecastSeries(item, dataset));" in javascript
    assert "state.forecastSelected[state.dataset] = event.target.value;" in javascript
    assert "if (!isPureActualForecastSeries(selectedSeries, dataset))" in javascript
    assert "series_id: seriesId" in javascript

    # IDs are formed from the same exact grouping dimensions as the loaded
    # records, with priceflag retained as a separate key.
    for item in actual[:5]:
        record = service._series_index[("wfp", item["series_id"])]
        assert record["price_flag"] == "actual"
        assert record["series_id"] == item["series_id"]


def test_forecast_api_routes_multiple_commodity_market_ids_without_substitution(client, service, monkeypatch):
    requested = (
        ("Beans (red)", "Bullunkutu"),
        ("Rice (local)", "Baga Road"),
        ("Maize", "Illela (CBM)"),
    )
    records = []
    for commodity, market in requested:
        record = next(
            item for item in service._series["wfp"]
            if item["commodity"] == commodity
            and item["market"] == market
            and item["price_flag"] == "actual"
            and assess_wfp_series(item)["eligible"]
        )
        records.append(record)
    assert len({item["commodity"] for item in records}) == len(requested)
    assert len({item["market"] for item in records}) == len(requested)

    routed = []

    def fake_generate(record, series_id, horizon, **_kwargs):
        assessment = assess_wfp_series(record)
        if not assessment["eligible"]:
            raise ForecastingError(assessment["code"], assessment["reason"])
        routed.append((record["series_id"], record["commodity"], record["market"], record["price_flag"]))
        return {"metadata": {"series_id": series_id, "horizon": horizon}, "forecast": []}

    from src import ondemand_forecasting

    monkeypatch.setattr(ondemand_forecasting, "generate_wfp_forecast", fake_generate)
    for record in records:
        response = client.get("/api/forecast", query_string={
            "dataset": "wfp",
            "series_id": record["series_id"],
            "model": "auto",
            "mode": "on_demand",
            "horizon": "1",
        })
        assert response.status_code == 200, response.get_json()
        payload = response.get_json()
        assert payload["series"]["series_id"] == record["series_id"]
        assert payload["series"]["commodity"] == record["commodity"]
        assert payload["series"]["market"] == record["market"]
        assert payload["series"]["price_flag"] == "actual"
    assert routed == [
        (record["series_id"], record["commodity"], record["market"], "actual")
        for record in records
    ]


def test_duplicate_commodity_market_flags_remain_separate_and_ineligible_flags_are_explained(
    client, service, monkeypatch
):
    grouping_fields = ("commodity", "market_id", "market", "state", "unit", "price_type", "currency")
    groups = {}
    for record in service._series["wfp"]:
        key = tuple(record[field] for field in grouping_fields)
        groups.setdefault(key, {})[record["price_flag"]] = record

    variants = next(
        group for group in groups.values()
        if {"actual", "aggregate", "actual,aggregate"}.issubset(group)
        and assess_wfp_series(group["actual"])["eligible"]
    )
    assert len({variants[flag]["series_id"] for flag in ("actual", "aggregate", "actual,aggregate")}) == 3

    aggregate_only = next(
        record for record in service._series["wfp"]
        if record["price_flag"] == "aggregate"
        and "actual" not in groups[tuple(record[field] for field in grouping_fields)]
        and "actual,aggregate" not in groups[tuple(record[field] for field in grouping_fields)]
    )

    def exact_series_probe(record, series_id, horizon, **_kwargs):
        assessment = assess_wfp_series(record)
        if not assessment["eligible"]:
            raise ForecastingError(assessment["code"], assessment["reason"])
        return {"metadata": {"series_id": series_id, "horizon": horizon}, "forecast": []}

    from src import ondemand_forecasting

    monkeypatch.setattr(ondemand_forecasting, "generate_wfp_forecast", exact_series_probe)
    actual_response = client.get("/api/forecast", query_string={
        "dataset": "wfp", "series_id": variants["actual"]["series_id"],
        "model": "auto", "mode": "on_demand", "horizon": "1",
    })
    assert actual_response.status_code == 200
    assert actual_response.get_json()["series"]["series_id"] == variants["actual"]["series_id"]

    for flagged_record in (variants["aggregate"], variants["actual,aggregate"], aggregate_only):
        response = client.get("/api/forecast", query_string={
            "dataset": "wfp", "series_id": flagged_record["series_id"],
            "model": "auto", "mode": "on_demand", "horizon": "1",
        })
        assert response.status_code == 422
        error = response.get_json()["error"]
        assert error["code"] == "not_direct_actual"
        assert "aggregate or mixed-price-flag" in error["message"]


def test_history_keeps_real_months_and_does_not_fill_gaps(client):
    series = client.get("/api/series?dataset=wfp").get_json()
    item = next(row for row in series if row["commodity"] == "Sorghum (white)" and row["market"] == "Dawanau" and row["price_flag"] == "actual")
    result = client.get("/api/history", query_string={"dataset": "wfp", "series_id": item["series_id"]})
    assert result.status_code == 200
    payload = result.get_json()
    assert payload["gaps_filled"] is False
    history = payload["history"]
    assert len(history) == 199
    months = {row["month"] for row in history}
    assert "2019-05" not in months
    assert history[0]["month"] == "2003-08"
    assert history[-1]["month"] == "2021-12"


def test_nbs_history_is_a_distinct_reported_average_series(client):
    series = client.get("/api/series?dataset=nbs").get_json()
    item = next(row for row in series if row["commodity"] == "Yam Tuber" and row["geography_level"] == "national")
    result = client.get("/api/history", query_string={"dataset": "nbs", "series_id": item["series_id"]})
    assert result.status_code == 200
    payload = result.get_json()
    assert payload["series"]["source"] == "NBS"
    assert payload["series"]["unit"] == "Not stated in workbook"
    assert len(payload["history"]) == 17
    assert [row["month"] for row in payload["history"]] == [
        f"{year}-{month:02d}" for year, months in ((2025, range(1, 13)), (2026, range(1, 6))) for month in months
    ]


def test_history_date_filters_and_invalid_requests(client):
    item = next(
        row for row in client.get("/api/series?dataset=nbs").get_json()
        if row["first_observed_month"] <= "2026-02" and row["last_observed_month"] >= "2026-04"
    )
    filtered = client.get("/api/history", query_string={"dataset": "nbs", "series_id": item["series_id"], "start_date": "2026-02", "end_date": "2026-04"})
    assert filtered.status_code == 200
    assert [row["month"] for row in filtered.get_json()["history"]] == ["2026-02", "2026-03", "2026-04"]
    assert client.get("/api/history", query_string={"dataset": "nbs", "series_id": item["series_id"], "start_date": "2026-13"}).status_code == 400
    assert client.get("/api/history", query_string={"dataset": "nbs", "series_id": item["series_id"], "start_date": "2026-04", "end_date": "2026-02"}).status_code == 400
    assert client.get("/api/history", query_string={"dataset": "nbs", "series_id": "missing"}).status_code == 404


def test_ineligible_series_has_no_metrics_and_forecast_is_unavailable(client):
    item = client.get("/api/series?dataset=wfp").get_json()[0]
    performance = client.get("/api/performance", query_string={"dataset": "wfp", "series_id": item["series_id"]})
    assert performance.status_code == 200
    body = performance.get_json()
    assert body["status"] in {"not_trained", "not_eligible"}
    assert body["records"] == []
    assert body["legacy_results_included"] is False

    for horizon in ("1", "3", "6"):
        response = client.get("/api/forecast", query_string={"dataset": "wfp", "series_id": item["series_id"], "model": "random_forest", "horizon": horizon})
        assert response.status_code == 422
        assert response.get_json()["error"]["code"] == "forecast_unavailable"
    assert client.get("/api/forecast", query_string={"dataset": "wfp", "series_id": item["series_id"], "model": "random_forest", "horizon": "2"}).status_code == 400


def test_old_or_unknown_datasets_are_not_available(client):
    for dataset in ("world_bank", "market", "aggregate", "unknown"):
        response = client.get("/api/series", query_string={"dataset": dataset})
        assert response.status_code == 404
        assert response.get_json()["error"]["code"] == "dataset_unavailable"


def test_processing_outputs_exist_and_sources_are_not_used_as_model_artifacts():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "data/processed/wfp_monthly.csv",
        "data/processed/nbs_monthly.csv",
        "reports/wfp_nbs_series_coverage.csv",
        "reports/nbs_series_coverage.csv",
        "reports/source_migration_audit.txt",
    ):
        assert (root / relative).is_file()
    entrypoint = (root / "train.py").read_text(encoding="utf-8").lower()
    assert "world_bank" not in entrypoint
    assert "train_wfp_nbs_models" in entrypoint


def test_saved_forecasts_performance_and_unavailable_combinations(client):
    series = client.get("/api/series?dataset=wfp").get_json()
    trained = next(item for item in series if item.get("forecast_available"))
    performance = client.get("/api/performance", query_string={"dataset": "wfp", "series_id": trained["series_id"]})
    assert performance.status_code == 200
    body = performance.get_json()
    assert body["status"] == "evaluated"
    assert body["summary"]["selection_basis"].startswith("validation RMSE")
    assert body["summary"]["test_date_count"] > 0
    assert all(row["test_dates"] for row in body["records"])
    assert any(row["champion_selected_on_validation"] for row in body["records"])

    model = trained["champion_model"]
    for horizon in (1, 3, 6):
        response = client.get("/api/forecast", query_string={
            "dataset": "wfp", "series_id": trained["series_id"],
            "model": model, "horizon": str(horizon),
        })
        assert response.status_code == 200
        result = response.get_json()
        assert len(result["forecast"]) == horizon
        assert result["metadata"]["forecast_classification"] == "historical_only"
        assert result["metadata"]["confidence_intervals"] is False
        assert result["warning"].startswith("Historical-only")

    unsupported_model = client.get("/api/forecast", query_string={
        "dataset": "wfp", "series_id": trained["series_id"],
        "model": "random_forest" if model != "random_forest" else "xgboost", "horizon": "1",
    })
    assert unsupported_model.status_code == 422
    assert unsupported_model.get_json()["error"]["code"] == "model_unavailable"


def test_on_demand_forecasts_a_registered_wfp_series_from_latest_actual(tmp_path):
    from datetime import date

    service = DashboardService(reference_date=date(2026, 11, 19), forecast_cache_dir=tmp_path)
    registered = next(iter(service._training_series))
    record = service._series_index[("wfp", registered)]
    client = create_app(service).test_client()
    for horizon in (1, 3, 6):
        result = client.get("/api/forecast", query_string={
            "dataset": "wfp", "series_id": registered, "model": "auto",
            "mode": "on_demand", "horizon": str(horizon),
        })
        assert result.status_code == 200, result.get_json()
        payload = result.get_json()
        assert payload["series"]["series_id"] == registered
        assert payload["series"]["commodity"] == record["commodity"]
        assert payload["series"]["market"] == record["market"]
        origin = payload["metadata"]["forecast_origin_month"]
        assert payload["metadata"]["latest_actual_month_in_source"] == origin
        assert len(payload["forecast"]) == horizon
        assert [row["month"] for row in payload["forecast"]] == [
            f"{int(origin[:4]) + (int(origin[5:]) + step - 1) // 12:04d}-{(int(origin[5:]) + step - 1) % 12 + 1:02d}"
            for step in range(1, horizon + 1)
        ]
        assert payload["metadata"]["validation_metrics"]["n"] >= 6
        assert payload["metadata"]["selection_basis"].startswith("lowest chronological validation RMSE")
        assert "not a live or guaranteed market price" in payload["warning"]


def test_on_demand_forecasts_a_qualified_series_without_saved_forecasts(tmp_path):
    from datetime import date

    service = DashboardService(reference_date=date(2026, 11, 19), forecast_cache_dir=tmp_path)
    candidate = next(
        record for record in service._series["wfp"]
        if record["price_flag"] == "actual"
        and record["series_id"] not in service._training_series
        and assess_wfp_series(record)["eligible"]
    )
    assert not candidate.get("forecast_available")
    response = create_app(service).test_client().get("/api/forecast", query_string={
        "dataset": "wfp", "series_id": candidate["series_id"],
        "mode": "on_demand", "horizon": "1",
    })
    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    assert payload["series"]["series_id"] == candidate["series_id"]
    assert payload["metadata"]["actual_observation_count"] >= 30
    assert payload["forecast"][0]["month"] > payload["metadata"]["forecast_origin_month"]


def test_on_demand_api_excludes_aggregate_flags_and_keeps_nbs_separate(client):
    wfp_aggregate = next(item for item in client.get("/api/series?dataset=wfp").get_json() if item["price_flag"] != "actual")
    response = client.get("/api/forecast", query_string={
        "dataset": "wfp", "series_id": wfp_aggregate["series_id"],
        "model": "auto", "mode": "on_demand", "horizon": "1",
    })
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "not_direct_actual"

    short_actual = next(
        item for item in client.get("/api/series?dataset=wfp").get_json()
        if item["price_flag"] == "actual" and item["observed_months"] < 10
    )
    response = client.get("/api/forecast", query_string={
        "dataset": "wfp", "series_id": short_actual["series_id"],
        "model": "auto", "mode": "on_demand", "horizon": "1",
    })
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "insufficient_or_gapped_history"
    assert "complete calendar-lag training rows" in response.get_json()["error"]["message"]

    nbs_series = client.get("/api/series?dataset=nbs").get_json()[0]
    response = client.get("/api/forecast", query_string={
        "dataset": "nbs", "series_id": nbs_series["series_id"],
        "model": "auto", "mode": "on_demand", "horizon": "1",
    })
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "nbs_forecast_unavailable"


def test_training_metrics_recompute_from_saved_predictions():
    import numpy as np
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    metrics = pd.read_csv(root / "reports/training/model_comparison.csv")
    predictions = pd.read_csv(root / "reports/training/test_predictions.csv")
    trained = metrics[metrics.status == "trained_and_evaluated"]
    for row in trained.itertuples():
        values = predictions[
            (predictions.series_id == row.series_id)
            & (predictions.model == row.model)
            & (predictions.partition == "test")
        ]
        assert len(values) == row.test_n
        actual = values.actual_price.to_numpy(dtype=float)
        predicted = values.predicted_price.to_numpy(dtype=float)
        errors = predicted - actual
        assert np.isclose(np.sqrt(np.mean(errors**2)), row.test_rmse)
        assert np.isclose(np.mean(np.abs(errors)), row.test_mae)
        assert np.isclose(np.mean(errors), row.test_bias)
        nonzero = actual != 0
        if nonzero.any():
            assert np.isclose(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100, row.test_mape)
    for _, group in trained.groupby("series_id"):
        date_sets = {tuple(row.test_dates.split(";")) for row in group.itertuples()}
        assert len(date_sets) == 1


def test_calendar_features_and_lstm_sequences_do_not_bridge_missing_months():
    import pandas as pd

    from src.model_training_pipeline import build_lstm_sequences, build_monthly_features

    prices = {
        pd.Period("2020-01", freq="M"): 10.0,
        pd.Period("2020-02", freq="M"): 11.0,
        pd.Period("2020-04", freq="M"): 13.0,
        pd.Period("2020-05", freq="M"): 14.0,
        pd.Period("2020-06", freq="M"): 15.0,
        pd.Period("2020-07", freq="M"): 16.0,
    }
    features = build_monthly_features(prices)
    assert list(map(str, features.index)) == ["2020-07"]
    assert features.loc[pd.Period("2020-07", freq="M"), "lag_1"] == 15.0
    assert features.loc[pd.Period("2020-07", freq="M"), "lag_3"] == 13.0
    sequences, _, targets = build_lstm_sequences(prices, window=3)
    assert len(sequences) == 1
    assert str(targets[0]) == "2020-07"


def test_nbs_is_documented_but_not_trained():
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    eligibility = pd.read_csv(root / "reports/training/series_eligibility.csv")
    nbs = eligibility[eligibility.source == "NBS"]
    assert not nbs.empty
    assert not nbs.tree_model_eligible.any()
    assert set(nbs[nbs.methodology_regime == "post_2025_cpi_rebase"].actual_monthly_observations.unique()) <= {16, 17}


def test_model_registry_artifacts_and_forecast_horizons_are_consistent():
    import json
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "reports/training/model_registry.json").read_text(encoding="utf-8"))
    assert registry["source_scope"] == ["WFP", "NBS"]
    assert registry["summary"]["historical_wfp_series_trained"] == 7
    assert registry["summary"]["model_series_fits"] == 29
    assert registry["summary"]["current_market_forecasts"] == 0
    forecasts = pd.read_csv(root / "outputs/training/forecasts.csv")
    for series_id, record in registry["series"].items():
        assert record["forecast_classification"] == "historical_only"
        assert all((root / path).is_file() for path in record["artifact_paths"].values())
        assert forecasts[forecasts.series_id == series_id].requested_horizon.astype(int).value_counts().to_dict() == {1: 1, 3: 3, 6: 6}
        trained = [name for name in record["available_models"] if name != "naive_last_observation"]
        assert {"random_forest", "xgboost", "gradient_boosting", "stacking"}.issubset(trained)
