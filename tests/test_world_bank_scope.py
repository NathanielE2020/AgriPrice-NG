"""Focused tests for the memory-limited World Bank deployment scope."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.dashboard_service import (
    WORLD_BANK_COLUMNS,
    DashboardError,
    DashboardService,
)


def _champion_ids() -> tuple[str, ...]:
    return tuple(f"Commodity {index} | National Average | Kg | close_estimate" for index in range(12))


def _registry(ids: tuple[str, ...] | None = None) -> dict:
    series_ids = ids or _champion_ids()
    records = {
        series_id: {
            "series_id": series_id,
            "selected_model": "naive_last_observation",
            "selected_transformation": "none",
            "artifact_path": None,
            "experiment_version": "test-version",
        }
        for series_id in series_ids
    }
    return {
        "available": True,
        "reason": "fixture registry",
        "records": records,
        "generated_at": "2026-11-18T00:00:00Z",
        "champion_count": len(records),
        "experiment_version": "test-version",
    }


def _rows_for(ids: tuple[str, ...]) -> list[dict]:
    rows = []
    for index, series_id in enumerate(ids):
        commodity, geography, unit, price_type = series_id.split(" | ")
        rows.append({
            "source": "world_bank",
            "date": "2026-10-15",
            "month": "2026-10",
            "commodity_ticker": f"commodity_{index}",
            "commodity": commodity,
            "unit": unit,
            "price_type": price_type,
            "price": 100.0 + index,
            "low_estimate": 95.0,
            "high_estimate": 105.0,
            "trust_score": 0.8,
            "base_observation_price": 98.0,
            "is_model_estimate": True,
            "geography_level": "national",
            "geography_name": geography,
            "state": None,
            "market": None,
            "data_coverage": 0.9,
            "data_coverage_recent": 0.9,
            "index_confidence_score": 0.8,
            "last_survey_point": "2026-10",
            "source_data_is_direct_observation": False,
            "series_id": series_id,
            "observed": True,
        })
    return rows


class WorldBankScopeTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"AGRIPRICE_DEPLOY_SCOPE": "world_bank"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _service_with_fixtures(self, ids: tuple[str, ...] | None = None, registry=None):
        ids = ids or _champion_ids()
        registry = registry or _registry(ids)
        events: list[str] = []

        def load_registry(self, require_expected_count=False):
            events.append("registry")
            if not registry["available"]:
                return registry
            if require_expected_count and len(registry["records"]) != 12:
                return {"available": False, "reason": "fixture registry count mismatch", "records": {}, "generated_at": None, "champion_count": 0}
            return registry

        def load_rows(champion_series_ids=None):
            events.append("rows")
            self_ids = tuple(champion_series_ids or ())
            return _rows_for(self_ids)

        def load_metrics(self, dataset, series_ids=None):
            events.append(f"metrics:{dataset}")
            if dataset == "world_bank":
                return [{"series_id": series_id, "model": "naive_last_observation", "status": "evaluated", "rmse": 1.0, "mae": 1.0, "mape": 1.0, "r2": 0.0} for series_id in (series_ids or ())]
            return []

        def unexpected_loader(*args, **kwargs):
            raise AssertionError("disabled dataset loader was called")

        with (
            patch.object(DashboardService, "_load_world_bank_champion_registry", new=load_registry),
            patch.object(DashboardService, "_load_world_bank_rows", new=staticmethod(load_rows)),
            patch.object(DashboardService, "_load_market_rows", new=staticmethod(unexpected_loader)),
            patch.object(DashboardService, "_load_aggregate_rows", new=staticmethod(unexpected_loader)),
            patch.object(DashboardService, "_load_source_summary", new=staticmethod(unexpected_loader)),
            patch.object(DashboardService, "_load_metrics", new=load_metrics),
        ):
            service = DashboardService()
        return service, events

    def test_world_bank_scope_load_order_default_and_rows(self):
        service, events = self._service_with_fixtures()
        self.assertEqual(service.default_dataset, "world_bank")
        self.assertEqual(service.deploy_scope, "world_bank")
        self.assertEqual(service.market_rows, [])
        self.assertEqual(service.aggregate_rows, [])
        self.assertEqual(len(service.world_bank_series), 12)
        self.assertEqual(len(service.world_bank_rows), 12)
        self.assertLess(events.index("registry"), events.index("rows"))
        self.assertEqual(events.count("metrics:world_bank"), 1)
        self.assertNotIn("metrics:market", events)
        self.assertNotIn("metrics:aggregate", events)

    def test_world_bank_access_control_and_series_catalogue(self):
        service, _ = self._service_with_fixtures()
        self.assertEqual(set(service.series()), {"world_bank"})
        self.assertEqual(len(service.series("world_bank")["world_bank"]), 12)
        for dataset in ("market", "aggregate"):
            with self.subTest(dataset=dataset), self.assertRaises(DashboardError) as context:
                service.series(dataset)
            self.assertEqual(context.exception.code, "dataset_unavailable")
            self.assertEqual(context.exception.status_code, 404)

    def test_world_bank_overview_uses_world_bank_subset_facts(self):
        service, _ = self._service_with_fixtures()
        overview = service.overview()
        self.assertEqual(overview["deployment_scope"], "world_bank")
        self.assertEqual(overview["available_datasets"], ["world_bank"])
        self.assertEqual(overview["dataset"]["source"], "World Bank Nigeria Real Time Food Prices; c_ close estimates")
        self.assertEqual(overview["dataset"]["historical_observations"], 12)
        self.assertEqual(overview["dataset"]["coverage_scope"], "registered_champion_subset")
        self.assertFalse(overview["dataset"]["source_data_is_direct_observation"])
        self.assertFalse(overview["sources"]["wfp"]["available"])
        self.assertEqual(overview["sources"]["world_bank"]["series_count"], 12)
        self.assertFalse(overview["sources"]["world_bank"]["full_source_catalogue_loaded"])
        self.assertEqual(overview["model_status"]["world_bank"], {"naive_last_observation": 12})
        self.assertEqual(overview["sources"]["world_bank"]["survey_observation_period_end"], "2026-10")

    def test_unregistered_series_cannot_be_exposed(self):
        service, _ = self._service_with_fixtures()
        with self.assertRaises(DashboardError) as context:
            service.history("world_bank", "Unregistered | National Average | Kg | close_estimate")
        self.assertEqual(context.exception.code, "unknown_series")
        self.assertEqual(context.exception.status_code, 404)

    def test_invalid_registry_fails_closed_without_reading_csv(self):
        invalid = {"available": False, "reason": "bad registry", "records": {}, "generated_at": None, "champion_count": 0}
        service, events = self._service_with_fixtures(ids=(), registry=invalid)
        self.assertEqual(service.world_bank_rows, [])
        self.assertEqual(service.world_bank_series, [])
        self.assertNotIn("rows", events)
        self.assertNotIn("metrics:world_bank", events)
        self.assertFalse(service.overview()["sources"]["world_bank"]["available"])
        self.assertIn("bad registry", service.overview()["sources"]["world_bank"]["reason"])

    def test_chunk_loader_filters_exact_registered_series_and_columns(self):
        selected_ids = _champion_ids()[:2]
        rows = []
        for index, series_id in enumerate((*selected_ids, "Not Registered | National Average | Kg | close_estimate")):
            commodity, geography, unit, price_type = series_id.split(" | ")
            row = {column: None for column in WORLD_BANK_COLUMNS}
            row.update({
                "source": "world_bank", "date": "2026-10-15", "month": "2026-10",
                "commodity_ticker": f"c{index}", "commodity": commodity, "unit": unit,
                "price_type": price_type, "price": 100 + index, "low_estimate": 90,
                "high_estimate": 110, "trust_score": 0.8, "base_observation_price": 98,
                "is_model_estimate": True, "geography_level": "national", "geography_name": geography,
                "state": None, "market": None, "data_coverage": 0.9, "data_coverage_recent": 0.9,
                "index_confidence_score": 0.8, "last_survey_point": "2026-10",
                "source_data_is_direct_observation": False,
            })
            rows.append(row)
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "world_bank.csv"
            pd.DataFrame(rows, columns=WORLD_BANK_COLUMNS).to_csv(fixture, index=False)
            DashboardService._load_world_bank_rows.cache_clear()
            with patch("src.world_bank_ingestion.NORMALIZED_FILE", fixture):
                loaded = DashboardService._load_world_bank_rows(tuple(selected_ids))
                empty = DashboardService._load_world_bank_rows(())
            DashboardService._load_world_bank_rows.cache_clear()
        self.assertEqual({row["series_id"] for row in loaded}, set(selected_ids))
        self.assertEqual(len(loaded), 2)
        self.assertTrue(all(row["is_model_estimate"] for row in loaded))
        self.assertTrue(all(row["source_data_is_direct_observation"] is False for row in loaded))
        self.assertTrue(all("latitude" not in row and "longitude" not in row for row in loaded))
        self.assertEqual(empty, [])

    def test_world_bank_series_api_rejects_wfp_datasets(self):
        service, _ = self._service_with_fixtures()
        from app import create_app

        api = create_app(service).test_client()
        page = api.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'data-deploy-scope="world_bank"', page.data)
        self.assertIn(b"normalized World Bank Nigeria Real Time Food Prices data", page.data)
        self.assertIn(b"test metrics are retained for audit", page.data)
        self.assertIn("World Bank Nigeria Food Prices · close-estimate production view".encode(), page.data)
        self.assertNotIn(b"<strong>LSTM</strong>", page.data)
        self.assertEqual(api.get("/api/series").get_json().keys(), {"world_bank"})
        for dataset in ("market", "aggregate"):
            response = api.get("/api/series", query_string={"dataset": dataset})
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.get_json()["error"]["code"], "dataset_unavailable")
        series_id = _champion_ids()[0]
        performance = api.get(
            "/api/performance",
            query_string={"dataset": "world_bank", "series_id": series_id},
        )
        self.assertEqual(performance.status_code, 200)
        record = performance.get_json()["records"][0]
        self.assertTrue(record["production_champion"])
        self.assertFalse(record["research_only"])

    def test_local_dashboard_keeps_full_research_copy(self):
        from types import SimpleNamespace

        from app import create_app

        page = create_app(SimpleNamespace(deploy_scope="")).test_client().get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"WFP Nigeria Food Prices", page.data)
        self.assertIn(b"<strong>LSTM</strong>", page.data)
        self.assertNotIn(b"close-estimate production view", page.data)


if __name__ == "__main__":
    unittest.main()
