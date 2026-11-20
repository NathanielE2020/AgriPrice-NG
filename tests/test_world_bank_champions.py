"""Retirement guard: old champion artifacts must never reach active APIs."""

from app import create_app
from src.wfp_nbs_dashboard_service import DashboardService


def test_active_overview_reports_only_registered_wfp_nbs_historical_experiments():
    overview = create_app(DashboardService()).test_client().get("/api/overview").get_json()
    status = overview["model_status"]
    assert status["trained_on_current_sources"] is True
    assert status["available_model_count"] == 29
    assert status["trained_series_count"] == 7
    assert status["historical_forecast_series_count"] == 7
    assert status["current_market_forecast_series_count"] == 0


def test_performance_api_never_substitutes_historical_champion_metrics():
    client = create_app(DashboardService()).test_client()
    series = client.get("/api/series?dataset=nbs").get_json()[0]
    payload = client.get("/api/performance", query_string={"dataset": "nbs", "series_id": series["series_id"]}).get_json()
    assert payload["legacy_results_included"] is False
    assert payload["status"] in {"validation_evaluated", "not_eligible", "not_trained", "overall_evaluated", "overall_coverage"}
    if payload["status"] == "validation_evaluated":
        assert payload["summary"]["evaluation_period"] == "validation"
        assert all(row["n"] > 0 for row in payload["records"] if row["status"] == "validation_evaluated")
    elif payload["status"] in {"overall_evaluated", "overall_coverage"}:
        assert payload["overall_evaluation"] is True
        assert payload["summary"]["evaluation_period"] == "validation"
        assert payload["predictions_by_date"] == []
        assert all(row["rmse"] is None and row["mae"] is None and row["bias"] is None for row in payload["records"])
    else:
        assert payload["records"] == []
