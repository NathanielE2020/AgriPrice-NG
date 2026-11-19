"""Retirement guards kept to ensure the former production source stays disabled."""

import pytest

from app import create_app
from src.wfp_nbs_dashboard_service import DashboardService


def test_retired_dataset_is_rejected_and_new_sources_are_only_catalogue():
    client = create_app(DashboardService()).test_client()
    response = client.get("/api/series?dataset=world_bank")
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "dataset_unavailable"
    catalogue = client.get("/api/series").get_json()
    assert set(catalogue) == {"wfp", "nbs"}


def test_active_service_does_not_report_a_legacy_champion_registry():
    client = create_app(DashboardService()).test_client()
    overview = client.get("/api/overview").get_json()
    assert set(overview["sources"]) == {"wfp", "nbs"}
    assert "world_bank" not in overview["sources"]
    assert "champion_registry" not in overview["model_status"]


def test_legacy_world_bank_entrypoints_are_retired():
    from src.world_bank_coverage import write_coverage_reports
    from src.world_bank_ingestion import prepare_world_bank_data
    from src.world_bank_training import _legacy_train_world_bank, train_world_bank

    for entrypoint in (
        prepare_world_bank_data,
        write_coverage_reports,
        train_world_bank,
        _legacy_train_world_bank,
    ):
        with pytest.raises(RuntimeError, match="retired"):
            entrypoint()
