"""Isolated LSTM inference adapter used by the Flask dashboard.

The Flask application delegates this module to the configured TensorFlow
interpreter and loads the saved Keras model plus its JSON scalers.  On Windows
that is normally ``.venv-lstm``; on Linux it is normally the active project
environment.  It performs the same calendar-window,
target reconstruction, and recursive predicted-price feedback used by the
completed LSTM experiment.  It never reads actual prices after the forecast
origin.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .lstm_experiment_stage2 import (
    MODELS_DIR,
    WINDOW,
    _inverse_scale,
    _load_monthly_groups,
    _reconstruct,
    _scale,
)
from .parquet_compat import month_from_index, month_index


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:120]


def _error(code: str, message: str) -> dict:
    return {"ok": False, "code": code, "message": message}


def forecast_series(series_id: str, horizon: int) -> dict:
    if horizon not in (1, 3, 6):
        return _error("invalid_horizon", "horizon must be 1, 3, or 6 months.")

    model_dir = MODELS_DIR / _slug(series_id)
    model_path = model_dir / "lstm.keras"
    preprocessing_path = model_dir / "lstm_preprocessing.json"
    if not model_path.exists() or not preprocessing_path.exists():
        return _error("model_unavailable", "The saved LSTM artifact is unavailable for this series.")

    try:
        import tensorflow as tf

        groups = _load_monthly_groups()
        rows = groups.get(series_id)
        if not rows:
            return _error("unknown_series", "The requested aggregate series is not available to the LSTM experiment.")

        metadata = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        transformation = str(metadata.get("transformation") or metadata.get("selected_transformation") or "")
        if not transformation:
            return _error("model_unavailable", "The saved LSTM transformation metadata is missing.")
        input_scaler = metadata.get("input_scaler")
        target_scaler = metadata.get("target_scaler")
        if not isinstance(input_scaler, dict) or not isinstance(target_scaler, dict):
            return _error("model_unavailable", "The saved LSTM scaler metadata is invalid.")
        if not all(math.isfinite(float(input_scaler.get(key))) for key in ("mean", "scale")):
            return _error("model_unavailable", "The saved LSTM input scaler is invalid.")
        if not all(math.isfinite(float(target_scaler.get(key))) for key in ("mean", "scale")):
            return _error("model_unavailable", "The saved LSTM target scaler is invalid.")

        observed = {
            row["month"]: float(row["mean_price"])
            for row in rows
            if row.get("observed") and row.get("mean_price") is not None
        }
        if not observed:
            return _error("missing_history", "No observed historical prices are available for this series.")
        origin_month = max(observed, key=month_index)
        history_months = [
            month_from_index(month_index(origin_month) - offset)
            for offset in range(WINDOW - 1, -1, -1)
        ]
        if any(month not in observed for month in history_months):
            return _error(
                "missing_calendar_history",
                "The latest 12 calendar months are not all observed; LSTM recursive forecasting is unavailable.",
            )

        model = tf.keras.models.load_model(model_path, compile=False)
        values = [observed[month] for month in history_months]
        forecasts: list[dict] = []
        for step in range(1, horizon + 1):
            sequence = np.asarray(values[-WINDOW:], dtype=float).reshape(1, WINDOW, 1)
            scaled = _scale(sequence.reshape(-1, 1), input_scaler).reshape(1, WINDOW, 1)
            transformed_scaled = model.predict(scaled, verbose=0).reshape(-1)
            transformed = float(_inverse_scale(transformed_scaled, target_scaler)[0])
            price = float(_reconstruct(np.asarray([transformed]), np.asarray([values[-1]]), transformation)[0])
            if not math.isfinite(price):
                return _error("non_finite_prediction", "The saved LSTM produced a non-finite forecast.")
            price = max(price, 0.0)
            forecast_month = month_from_index(month_index(origin_month) + step)
            forecasts.append(
                {
                    "horizon_months": step,
                    "forecast_month": forecast_month,
                    "forecast_price": price,
                    "status": "recursive_forecast",
                }
            )
            values.append(price)

        return {
            "ok": True,
            "series_id": series_id,
            "transformation": transformation,
            "origin_month": origin_month,
            "forecast": forecasts,
            "method": "recursive; predicted prices fed back; no actual future prices",
        }
    except Exception:
        # The Flask process receives a safe message, never a local path or
        # TensorFlow stack trace.
        return _error("inference_failed", "The isolated LSTM inference process could not generate a forecast.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--horizon", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(forecast_series(args.series_id, args.horizon), separators=(",", ":")))


if __name__ == "__main__":
    main()
