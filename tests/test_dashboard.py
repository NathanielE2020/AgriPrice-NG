from __future__ import annotations

from pathlib import Path

import pytest

from app import create_app
from src.wfp_nbs_dashboard_service import DashboardService


@pytest.fixture(scope="module")
def client():
    return create_app(DashboardService()).test_client()


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
