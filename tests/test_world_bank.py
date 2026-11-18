"""World Bank ingestion and dashboard source-separation tests."""

from __future__ import annotations

import unittest

import pandas as pd

from app import app
from src.dashboard_service import DashboardService
from src.world_bank_ingestion import EXTRACTED_ROOT, NORMALIZED_FILE, load_world_bank_raw


class WorldBankIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_raw_release_and_ticker_metadata(self):
        frame, tickers, details = load_world_bank_raw(EXTRACTED_ROOT)
        self.assertEqual(frame.shape, (17686, 137))
        self.assertEqual(frame["price_date"].min().strftime("%Y-%m"), "2007-01")
        self.assertEqual(frame["price_date"].max().strftime("%Y-%m"), "2026-11")
        self.assertEqual(frame.groupby("price_date").size().nunique(), 1)
        self.assertEqual(int(frame.groupby("price_date").size().iloc[0]), 74)
        self.assertEqual(tickers["rice"]["units"], "2.8 KG")
        self.assertEqual(details["number_of_markets_covered"], "73")

    def test_normalized_table_preserves_source_estimate_metadata(self):
        self.assertTrue(NORMALIZED_FILE.exists())
        frame = pd.read_csv(NORMALIZED_FILE)
        self.assertTrue((frame["source"] == "world_bank").all())
        self.assertTrue((frame["price_type"] == "close_estimate").all())
        self.assertTrue((frame["is_model_estimate"] == True).all())  # noqa: E712
        self.assertTrue(frame["trust_score"].notna().all())
        self.assertTrue(frame["low_estimate"].notna().all())
        self.assertTrue(frame["high_estimate"].notna().all())
        self.assertEqual(frame.duplicated(["commodity_ticker", "geo_id", "month", "unit", "price_type"]).sum(), 0)
        self.assertEqual(set(frame["geography_level"].unique()), {"market", "national", "market_average", "geopolitical_zone"})
        self.assertEqual(frame.loc[frame["geography_level"] == "market", "market"].nunique(), 67)

    def test_dashboard_exposes_world_bank_without_mixing_wfp(self):
        overview = self.client.get("/api/overview").get_json()
        self.assertEqual(overview["sources"]["world_bank"]["date_end"], "2026-11")
        self.assertEqual(overview["sources"]["world_bank"]["market_count"], 67)
        series = self.client.get("/api/series", query_string={"dataset": "world_bank"})
        self.assertEqual(series.status_code, 200)
        items = series.get_json()["world_bank"]
        national = next(item for item in items if item["series_id"] == "Beans (red) | National Average | 2.5 KG | close_estimate")
        self.assertTrue(national["is_model_estimate"])
        self.assertEqual(national["geography_level"], "national")
        self.assertIsNotNone(national["latest_low_estimate"])
        history = self.client.get("/api/history", query_string={"dataset": "world_bank", "series_id": national["series_id"]})
        self.assertEqual(history.status_code, 200)
        self.assertTrue(history.get_json()["history"][0]["is_model_estimate"])

    def test_world_bank_series_index_reuses_loaded_rows(self):
        service = DashboardService()
        self.assertEqual(
            sum(len(rows) for rows in service.world_bank_rows_by_series.values()),
            len(service.world_bank_rows),
        )
        series_id = "Beans (red) | National Average | 2.5 KG | close_estimate"
        indexed = service.world_bank_rows_by_series[series_id]
        self.assertIs(service._rows_for_series("world_bank", series_id)[0], indexed[0])
        self.assertTrue(any(row is indexed[0] for row in service.world_bank_rows))
        self.assertEqual(service._rows_for_series("world_bank", "not-real"), [])

    def test_world_bank_forecast_artifact_and_unavailable_market_are_explicit(self):
        series_id = "Beans (red) | National Average | 2.5 KG | close_estimate"
        service = DashboardService()
        champion = service._world_bank_champions["records"][series_id]["selected_model"]
        for horizon in (1, 3, 6):
            response = self.client.get("/api/forecast", query_string={"dataset": "world_bank", "series_id": series_id, "model": champion, "horizon": horizon})
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            payload = response.get_json()
            self.assertEqual(len(payload["forecast"]), horizon)
            self.assertTrue(payload["is_model_estimate"])
            self.assertEqual(payload["confidence_intervals"], None)
        challenger = "random_forest" if champion != "random_forest" else "gradient_boosting"
        non_champion = self.client.get("/api/forecast", query_string={"dataset": "world_bank", "series_id": series_id, "model": challenger, "horizon": 1})
        self.assertEqual(non_champion.status_code, 422)
        unsupported = self.client.get("/api/forecast", query_string={"dataset": "world_bank", "series_id": "Beans (red) | Aba | 2.5 KG | close_estimate", "model": champion, "horizon": 1})
        self.assertEqual(unsupported.status_code, 422)


if __name__ == "__main__":
    unittest.main()
