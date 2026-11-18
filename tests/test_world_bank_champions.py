"""Leakage and registry invariants for the World Bank champion experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.dashboard_service import (
    DashboardError,
    DashboardService,
    _validate_world_bank_champion_artifact,
)
from src.world_bank_training import select_champion, select_model_transform


class WorldBankChampionTests(unittest.TestCase):
    def test_naive_can_win_and_test_rmse_is_ignored(self):
        winner = select_champion([
            {"model": "naive_last_observation", "transformation": "none", "validation_rmse": 1.0, "test_metrics": {"rmse": 999}, "status": "evaluated"},
            {"model": "xgboost", "transformation": "original", "validation_rmse": 2.0, "test_metrics": {"rmse": 0}, "status": "evaluated"},
        ])
        self.assertEqual(winner["model"], "naive_last_observation")

    def test_ml_can_win_and_ties_use_simplicity(self):
        winner = select_champion([
            {"model": "random_forest", "transformation": "original", "validation_rmse": 1.0, "status": "evaluated"},
            {"model": "gradient_boosting", "transformation": "original", "validation_rmse": 1.0, "status": "evaluated"},
        ])
        self.assertEqual(winner["model"], "gradient_boosting")
        transform = select_model_transform([
            {"transformation": "percentage_change", "validation_rmse": 1.0, "status": "evaluated"},
            {"transformation": "original", "validation_rmse": 1.0, "status": "evaluated"},
        ])
        self.assertEqual(transform["transformation"], "original")

    def test_missing_registry_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing.json"
            service = DashboardService.__new__(DashboardService)
            service.market_only = False
            with patch("src.dashboard_service.WORLD_BANK_CHAMPION_REGISTRY", missing):
                result = service._load_world_bank_champion_registry()
        self.assertFalse(result["available"])
        self.assertEqual(result["champion_count"], 0)

    def test_world_bank_artifact_metadata_must_match_registry(self):
        record = {
            "selected_model": "stacking",
            "selected_transformation": "percentage_change",
            "experiment_version": "world-bank-champion-v1",
        }
        metadata = {
            "source": "world_bank",
            "series_id": "Rice | National Average | 2.8 KG | close_estimate",
            "selected_model": "stacking",
            "transformation": "percentage_change",
            "production_champion": True,
            "experiment_version": "world-bank-champion-v1",
        }
        series_id = metadata["series_id"]

        # Correct metadata must pass.
        _validate_world_bank_champion_artifact(metadata, series_id, record)

        # Every identity field must be independently enforced.
        mismatches = {
            "source": "other_source",
            "series_id": "Wrong series",
            "selected_model": "random_forest",
            "transformation": "original",
            "production_champion": False,
            "experiment_version": "wrong-version",
        }
        for field, wrong_value in mismatches.items():
            with self.subTest(field=field):
                corrupted = {**metadata, field: wrong_value}
                with self.assertRaises(DashboardError) as context:
                    _validate_world_bank_champion_artifact(
                        corrupted, series_id, record
                    )
                self.assertEqual(context.exception.code, "model_unavailable")
                self.assertEqual(context.exception.status_code, 503)

    def test_world_bank_artifact_rejects_missing_metadata(self):
        record = {
            "selected_model": "stacking",
            "selected_transformation": "percentage_change",
            "experiment_version": "world-bank-champion-v1",
        }
        with self.assertRaises(DashboardError):
            _validate_world_bank_champion_artifact(
                {}, "Example series", record
            )

    def test_naive_registry_does_not_need_artifact(self):
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "series": [{
                "series_id": "Example | National Average | Kg | close_estimate",
                "selected_model": "naive_last_observation",
                "selected_transformation": "none",
                "artifact_path": None,
            }],
        }
        with tempfile.TemporaryDirectory() as folder:
            registry = Path(folder) / "champion_models.json"
            registry.write_text(json.dumps(payload), encoding="utf-8")
            service = DashboardService.__new__(DashboardService)
            service.market_only = False
            with patch("src.dashboard_service.WORLD_BANK_CHAMPION_REGISTRY", registry):
                result = service._load_world_bank_champion_registry()
        self.assertTrue(result["available"])
        self.assertEqual(result["records"][payload["series"][0]["series_id"]]["selected_model"], "naive_last_observation")


    def test_registry_rejects_duplicate_series_ids(self):
        series = {
            "series_id": "Example | National Average | Kg | close_estimate",
            "selected_model": "naive_last_observation",
            "selected_transformation": "none",
            "artifact_path": None,
        }
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "series": [series, dict(series)],
        }

        with tempfile.TemporaryDirectory() as folder:
            registry = Path(folder) / "champion_models.json"
            registry.write_text(json.dumps(payload), encoding="utf-8")
            service = DashboardService.__new__(DashboardService)
            service.market_only = False
            with patch("src.dashboard_service.WORLD_BANK_CHAMPION_REGISTRY", registry):
                result = service._load_world_bank_champion_registry()

        self.assertFalse(result["available"])
        self.assertIn("(ValueError)", result["reason"])

    def test_registry_rejects_future_timestamp(self):
        future = datetime.now(timezone.utc) + __import__("datetime").timedelta(minutes=10)
        payload = {
            "generated_at": future.isoformat(),
            "series": [{
                "series_id": "Example | National Average | Kg | close_estimate",
                "selected_model": "naive_last_observation",
                "selected_transformation": "none",
                "artifact_path": None,
            }],
        }

        with tempfile.TemporaryDirectory() as folder:
            registry = Path(folder) / "champion_models.json"
            registry.write_text(json.dumps(payload), encoding="utf-8")
            service = DashboardService.__new__(DashboardService)
            service.market_only = False
            with patch("src.dashboard_service.WORLD_BANK_CHAMPION_REGISTRY", registry):
                result = service._load_world_bank_champion_registry()

        self.assertFalse(result["available"])
        self.assertIn("timestamp is in the future", result["reason"])

    def test_registry_rejects_artifact_outside_champion_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_dir = root / "models"
            champion_dir = model_dir / "world_bank" / "champions"
            champion_dir.mkdir(parents=True)
            outside_artifact = root / "outside.joblib"
            outside_artifact.write_bytes(b"test")

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "series": [{
                    "series_id": "Example | National Average | Kg | close_estimate",
                    "selected_model": "xgboost",
                    "selected_transformation": "original",
                    "artifact_path": "outside.joblib",
                }],
            }
            registry = root / "champion_models.json"
            registry.write_text(json.dumps(payload), encoding="utf-8")

            service = DashboardService.__new__(DashboardService)
            service.market_only = False
            with (
                patch("src.dashboard_service.WORLD_BANK_CHAMPION_REGISTRY", registry),
                patch("src.dashboard_service.MODEL_DIR", model_dir),
                patch("src.dashboard_service.ROOT", root),
            ):
                result = service._load_world_bank_champion_registry()

        self.assertFalse(result["available"])
        self.assertIn("(ValueError)", result["reason"])

    def test_registry_accepts_artifact_inside_champion_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_dir = root / "models"
            champion_dir = model_dir / "world_bank" / "champions"
            champion_dir.mkdir(parents=True)
            artifact = champion_dir / "champion.joblib"
            artifact.write_bytes(b"test")

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "series": [{
                    "series_id": "Example | National Average | Kg | close_estimate",
                    "selected_model": "xgboost",
                    "selected_transformation": "original",
                    "artifact_path": str(artifact.relative_to(root)).replace("\\\\", "/"),
                }],
            }
            registry = root / "champion_models.json"
            registry.write_text(json.dumps(payload), encoding="utf-8")

            service = DashboardService.__new__(DashboardService)
            service.market_only = False
            with (
                patch("src.dashboard_service.WORLD_BANK_CHAMPION_REGISTRY", registry),
                patch("src.dashboard_service.MODEL_DIR", model_dir),
                patch("src.dashboard_service.ROOT", root),
            ):
                result = service._load_world_bank_champion_registry()

        self.assertTrue(result["available"], result["reason"])
        self.assertEqual(result["champion_count"], 1)


if __name__ == "__main__":
    unittest.main()
