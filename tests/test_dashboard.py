"""Read-only Flask dashboard smoke and API contract tests."""

from __future__ import annotations

import unittest
from datetime import date
import os
from pathlib import Path
import sys
from unittest.mock import patch

from app import app
from src.dashboard_service import (
    DashboardError,
    DashboardService,
    ROOT,
    WINDOWS_LSTM_PYTHON,
    _lstm_interpreter,
    _load_artifact,
)


class DashboardApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_page_overview_and_series(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        overview = self.client.get("/api/overview")
        self.assertEqual(overview.status_code, 200)
        facts = overview.get_json()["dataset"]
        self.assertEqual(facts["historical_observations"], 54883)
        self.assertEqual(facts["commodity_count"], 42)
        self.assertFalse(facts["fallback_used"])
        self.assertEqual(facts["statistics_source"], "dynamic_raw_parquet")
        series = self.client.get("/api/series").get_json()
        self.assertEqual(len(series["market"]), 5)
        self.assertEqual(len(series["aggregate"]), 6)

    def test_polished_frontend_assets_and_sections(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)
        for section in ("overview-section", "analysis-section", "forecast-section", "performance-section", "about-section"):
            self.assertIn(section, html)
        self.assertIn("chart.umd.min.js", html)
        css = self.client.get("/static/css/dashboard.css")
        js = self.client.get("/static/js/dashboard.js")
        self.assertEqual(css.status_code, 200)
        self.assertEqual(js.status_code, 200)
        css.close()
        js.close()

    def test_history_and_performance(self):
        market_id = "Millet | Dawanau | 100 KG | Wholesale"
        aggregate_id = "Millet | 2.6 KG | Retail"
        history = self.client.get(
            "/api/history",
            query_string={"dataset": "market", "series_id": market_id},
        )
        self.assertEqual(history.status_code, 200)
        self.assertTrue(any(not row["observed"] for row in history.get_json()["history"]))

        performance = self.client.get(
            "/api/performance",
            query_string={"dataset": "aggregate", "series_id": aggregate_id},
        )
        self.assertEqual(performance.status_code, 200)
        records = performance.get_json()["records"]
        self.assertEqual(
            {row["model"] for row in records},
            {"naive_last_observation", "random_forest", "gradient_boosting", "xgboost", "stacking", "lstm"},
        )
        self.assertTrue(all(row["status"] == "evaluated" for row in records))
        self.assertTrue(all(row["test_date_alignment"] == "common" for row in records))
        self.assertTrue(all(row["n_test"] > 0 for row in records))
        self.assertTrue(all(row["evaluation_validity"] is True for row in records))
        self.assertTrue(all(row["first_test_date"] and row["last_test_date"] for row in records))
        self.assertTrue(all("bias" in row for row in records))
        self.assertIn("common-test-date audit", performance.get_json()["selection_protocol"])

    def test_latest_observation_metadata_and_deterministic_freshness(self):
        service = DashboardService(reference_date=date(2026, 11, 15))
        aggregate = next(item for item in service.aggregate_series if item["series_id"] == "Millet | 2.6 KG | Retail")
        self.assertEqual(aggregate["latest_observation_date"], "2025-11-15")
        self.assertAlmostEqual(aggregate["latest_actual_price"], 990.5569230769231)
        self.assertEqual(aggregate["observation_age_months"], 12)
        self.assertEqual(aggregate["freshness_status"], "aging")
        market = next(item for item in service.market_series if item["series_id"] == "Millet | Dawanau | 100 KG | Wholesale")
        self.assertEqual(market["latest_observation_date"], "2022-11-15")
        self.assertEqual(market["freshness_status"], "stale")

    def test_overview_statistics_are_loaded_dynamically_from_parquet_rows(self):
        fake_rows = [
            {"date": date(2024, 2, 15), "commodity": "Test A", "market": "Market 1", "admin1": "State 1", "currency": "NGN"},
            {"date": date(2025, 3, 15), "commodity": "Test B", "market": "Market 2", "admin1": "State 2", "currency": "NGN"},
            {"date": date(2025, 3, 15), "commodity": "Test A", "market": "Market 1", "admin1": "State 1", "currency": "NGN"},
        ]
        with patch("src.dashboard_service.read_parquet_rows", return_value=fake_rows):
            service = DashboardService(reference_date=date(2025, 4, 1))
        facts = service.overview()["dataset"]
        self.assertEqual(facts["historical_observations"], 3)
        self.assertEqual(facts["commodity_count"], 2)
        self.assertEqual(facts["market_count"], 2)
        self.assertEqual(facts["state_count"], 2)
        self.assertEqual(facts["earliest_observation_date"], "2024-02-15")
        self.assertEqual(facts["latest_observation_date"], "2025-03-15")
        self.assertEqual(facts["statistics_source"], "dynamic_raw_parquet")
        self.assertFalse(facts["fallback_used"])

    def test_overview_identifies_inspection_report_fallback(self):
        with patch("src.dashboard_service.read_parquet_rows", side_effect=RuntimeError("reader unavailable")):
            service = DashboardService(reference_date=date(2026, 11, 15))
        facts = service.overview()["dataset"]
        self.assertTrue(facts["fallback_used"])
        self.assertEqual(facts["statistics_source"], "verified_inspection_report_fallback")
        self.assertIn("could not be loaded", facts["fallback_reason"])

    def test_history_date_filtering_and_invalid_ranges(self):
        series_id = "Millet | 2.6 KG | Retail"
        filtered = self.client.get(
            "/api/history",
            query_string={"dataset": "aggregate", "series_id": series_id, "start_date": "2014-05", "end_date": "2015-12"},
        )
        self.assertEqual(filtered.status_code, 200)
        payload = filtered.get_json()
        self.assertEqual(payload["requested_range"]["start_month"], "2014-05")
        self.assertEqual(payload["requested_range"]["end_month"], "2015-12")
        self.assertTrue(all("2014-05" <= row["month"] <= "2015-12" for row in payload["history"]))
        self.assertTrue(any(not row["observed"] and row["price"] is None for row in payload["history"]))

        reversed_range = self.client.get(
            "/api/history",
            query_string={"dataset": "aggregate", "series_id": series_id, "start_date": "2022-01", "end_date": "2021-12"},
        )
        self.assertEqual(reversed_range.status_code, 400)
        self.assertEqual(reversed_range.get_json()["error"]["code"], "invalid_date_range")

        malformed = self.client.get(
            "/api/history",
            query_string={"dataset": "aggregate", "series_id": series_id, "start_date": "not-a-date"},
        )
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(malformed.get_json()["error"]["code"], "invalid_date")

    def test_stale_forecast_exposes_real_origin_and_warning(self):
        series_id = "Rice (imported) | 50 KG | Wholesale"
        response = self.client.get(
            "/api/forecast",
            query_string={"dataset": "aggregate", "series_id": series_id, "model": "lstm", "horizon": 3},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["latest_observation_date"], "2023-01-15")
        self.assertEqual(payload["origin_month"], "2023-01")
        self.assertEqual(payload["freshness_status"], "stale")
        self.assertIn("2023-01-15", payload["freshness_warning"])
        self.assertEqual(
            [row["horizon_months"] for row in payload["forecast"]],
            [1, 2, 3],
        )
        self.assertTrue(all(row["forecast_month"] > payload["origin_month"] for row in payload["forecast"]))

    def test_model_availability_keeps_lstm_aggregate_only(self):
        aggregate_id = "Millet | 2.6 KG | Retail"
        aggregate = self.client.get(
            "/api/series",
            query_string={"dataset": "aggregate"},
        ).get_json()["aggregate"]
        aggregate_series = next(item for item in aggregate if item["series_id"] == aggregate_id)
        aggregate_models = {item["model"]: item for item in aggregate_series["models"]}
        self.assertTrue(aggregate_models["lstm"]["available"])

        market_id = "Millet | Dawanau | 100 KG | Wholesale"
        market = self.client.get(
            "/api/series",
            query_string={"dataset": "market"},
        ).get_json()["market"]
        market_series = next(item for item in market if item["series_id"] == market_id)
        market_models = {item["model"]: item for item in market_series["models"]}
        self.assertFalse(market_models["lstm"]["available"])

    def test_lstm_interpreter_selection_is_portable(self):
        with patch.dict(os.environ, {"LSTM_PYTHON": "custom/python"}, clear=False):
            self.assertEqual(_lstm_interpreter(), (ROOT / "custom/python").resolve())

        with patch.dict(os.environ, {}, clear=True):
            selected = _lstm_interpreter()
        if os.name == "nt":
            expected = WINDOWS_LSTM_PYTHON if WINDOWS_LSTM_PYTHON.exists() else None
        else:
            expected = Path(sys.executable)
        self.assertEqual(selected, expected)

    def test_supported_forecasts_and_validation_errors(self):
        series_id = "Millet | 2.6 KG | Retail"
        for horizon in (1, 3, 6):
            response = self.client.get(
                "/api/forecast",
                query_string={
                    "dataset": "aggregate",
                    "series_id": series_id,
                    "model": "xgboost",
                    "horizon": horizon,
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(len(payload["forecast"]), horizon)
            self.assertIsNone(payload["confidence_intervals"])

        unsupported = self.client.get(
            "/api/forecast",
            query_string={
                "dataset": "market",
                "series_id": "Millet | Dawanau | 100 KG | Wholesale",
                "model": "xgboost",
                "horizon": 1,
            },
        )
        self.assertEqual(unsupported.status_code, 422)

        invalid_horizon = self.client.get(
            "/api/forecast",
            query_string={
                "dataset": "aggregate",
                "series_id": series_id,
                "model": "xgboost",
                "horizon": 2,
            },
        )
        self.assertEqual(invalid_horizon.status_code, 400)

        unknown_series = self.client.get(
            "/api/forecast",
            query_string={
                "dataset": "aggregate",
                "series_id": "not-real",
                "model": "xgboost",
                "horizon": 1,
            },
        )
        self.assertEqual(unknown_series.status_code, 404)

        missing_calendar_history = self.client.get(
            "/api/forecast",
            query_string={
                "dataset": "market",
                "series_id": "Maize (white) | Dawanau | 100 KG | Wholesale",
                "model": "random_forest",
                "horizon": 6,
            },
        )
        self.assertEqual(missing_calendar_history.status_code, 422)

    def test_lstm_forecasts_all_approved_aggregate_series_and_horizons(self):
        aggregate_series = self.client.get(
            "/api/series",
            query_string={"dataset": "aggregate"},
        ).get_json()["aggregate"]
        for item in aggregate_series:
            for horizon in (1, 3, 6):
                response = self.client.get(
                    "/api/forecast",
                    query_string={
                        "dataset": "aggregate",
                        "series_id": item["series_id"],
                        "model": "lstm",
                        "horizon": horizon,
                    },
                )
                self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
                payload = response.get_json()
                forecast = payload["forecast"]
                self.assertEqual(len(forecast), horizon)
                self.assertIsNone(payload["confidence_intervals"])
                self.assertEqual(payload["transformation"], "price_change" if item["series_id"] != "Rice (imported) | 50 KG | Wholesale" else "original")
                months = [row["forecast_month"] for row in forecast]
                self.assertEqual(months, sorted(months))
                self.assertEqual([row["horizon_months"] for row in forecast], list(range(1, horizon + 1)))
                self.assertTrue(all(isinstance(row["forecast_price"], (int, float)) for row in forecast))
                self.assertNotIn("actual_price", forecast[0])

    def test_missing_artifact_is_reported_without_traceback(self):
        with self.assertRaises(DashboardError) as context:
            _load_artifact(
                Path("models/final_validation/aggregate/not-real/xgboost.joblib")
            )
        self.assertEqual(context.exception.code, "model_unavailable")
        self.assertEqual(context.exception.status_code, 422)

    def test_missing_lstm_artifact_is_reported_safely(self):
        service = DashboardService()
        series_id = "Millet | 2.6 KG | Retail"
        with patch("src.dashboard_service._lstm_model_path", return_value=Path("models/not-real/lstm.keras")):
            with self.assertRaises(DashboardError) as context:
                service.forecast("aggregate", series_id, "lstm", 1)
        self.assertEqual(context.exception.code, "model_unavailable")
        self.assertEqual(context.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
