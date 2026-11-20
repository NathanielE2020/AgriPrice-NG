"""Retirement guards kept to ensure the former production source stays disabled."""

import pytest
import pandas as pd

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


def test_world_bank_normalization_keeps_close_estimates_explicitly_estimated():
    from src.world_bank_ingestion import normalize_world_bank

    row = {
        "price_date": pd.Timestamp("2026-11-01"), "adm1_name": "Lagos",
        "adm2_name": "Lagos Island", "mkt_name": "Test Market", "geo_id": "NG001",
        "currency": "NGN", "c_RICE": 640.0, "l_RICE": 600.0, "h_RICE": 680.0,
        "trust_RICE": 0.9, "RICE": 630.0, "lat": 6.45, "lon": 3.39,
        "data_coverage": 0.8, "data_coverage_recent": 0.7,
        "index_confidence_score": 0.9, "spatially_interpolated": False,
        "last_survey_point": "2026-10-15",
    }
    normalized = normalize_world_bank(
        pd.DataFrame([row]), {"RICE": {"full_name": "Rice", "units": "KG"}},
    )
    estimate = normalized.iloc[0]
    assert estimate["source"] == "world_bank"
    assert estimate["price_type"] == "close_estimate"
    assert bool(estimate["is_model_estimate"]) is True
    assert bool(estimate["source_data_is_direct_observation"]) is True
