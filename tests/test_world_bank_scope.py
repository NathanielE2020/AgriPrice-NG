"""Scope checks for the WFP/NBS-only application and deployment configuration."""

from pathlib import Path

from app import create_app
from src.wfp_nbs_dashboard_service import DashboardService


def test_active_scope_defaults_to_wfp_and_lists_nbs_separately():
    service = DashboardService()
    assert service.default_dataset == "wfp"
    assert service.deploy_scope == "wfp_nbs"
    payload = create_app(service).test_client().get("/api/overview").get_json()
    assert payload["available_datasets"] == ["wfp", "nbs"]


def test_active_routes_reject_retired_and_old_wfp_names():
    client = create_app(DashboardService()).test_client()
    for dataset in ("world_bank", "market", "aggregate"):
        assert client.get("/api/series", query_string={"dataset": dataset}).status_code == 404


def test_deployment_and_dashboard_do_not_select_retired_data_sources():
    root = Path(__file__).resolve().parents[1]
    app_source = (root / "app.py").read_text(encoding="utf-8").lower()
    train_source = (root / "train.py").read_text(encoding="utf-8").lower()
    service_source = (root / "src" / "wfp_nbs_dashboard_service.py").read_text(encoding="utf-8").lower()
    page_source = (root / "templates" / "dashboard.html").read_text(encoding="utf-8").lower()
    render_source = (root / "render.yaml").read_text(encoding="utf-8").lower()
    assert "wfp_nbs_dashboard_service" in app_source
    assert "world_bank" not in train_source
    assert "world_bank" not in service_source
    assert "world bank" not in page_source
    assert "wfp_nbs" in render_source


def test_runtime_reports_only_registered_series_and_rejects_unavailable_forecasts():
    client = create_app(DashboardService()).test_client()
    series = client.get("/api/series?dataset=wfp").get_json()[0]
    assert series["models"] == []
    performance = client.get("/api/performance", query_string={"dataset": "wfp", "series_id": series["series_id"]}).get_json()
    assert performance["records"] == []
    forecast = client.get("/api/forecast", query_string={"dataset": "wfp", "series_id": series["series_id"], "model": "naive", "horizon": 1})
    assert forecast.status_code == 422
    assert forecast.get_json()["error"]["code"] == "forecast_unavailable"
