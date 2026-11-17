"""Exploratory TensorFlow/Keras LSTM experiment for the approved series."""

from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np

from .evaluate_models import regression_metrics
from .plotting import write_actual_predicted_svg
from .parquet_compat import month_from_index, month_index


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = ROOT / "outputs" / "plots"
WINDOW = 12


def build_sequences(monthly_rows: list[dict], window: int = WINDOW):
    by_month = {row["month"]: row["price"] for row in monthly_rows}
    if not by_month:
        return [], []
    months = sorted(by_month, key=month_index)
    sequences = []
    targets = []
    target_months = []
    for month in months:
        current_index = month_index(month)
        history_months = [month_from_index(current_index - offset) for offset in range(window, 0, -1)]
        history = [by_month.get(history_month) for history_month in history_months]
        target = by_month.get(month)
        if target is None or any(value is None for value in history):
            continue
        sequences.append([float(value) for value in history])
        targets.append(float(target))
        target_months.append(month)
    return sequences, targets, target_months


def _split(values):
    n = len(values)
    if n == 0:
        return [], [], []
    train_end = max(1, int(n * 0.70))
    validation_end = max(train_end + 1, int(n * 0.85)) if n >= 3 else n
    validation_end = min(validation_end, n)
    return values[:train_end], values[train_end:validation_end], values[validation_end:]


def run_lstm_experiment(prepared: dict) -> list[dict]:
    # Keep the legacy market-level path importable in the lightweight
    # TensorFlow-only environment used by the isolated experiment.
    from sklearn.preprocessing import StandardScaler

    monthly_by_series: dict[str, list[dict]] = {}
    for row in prepared["monthly_rows"]:
        series_id = " | ".join(
            [row["commodity"], row["market"], row["unit"], row["price_type"]]
        )
        monthly_by_series.setdefault(series_id, []).append(row)

    try:
        import tensorflow as tf
    except ImportError as exc:
        tensorflow_note = f"TensorFlow/Keras unavailable: {exc}"
        tf = None

    report_rows: list[dict] = []
    for series_id, monthly_rows in sorted(monthly_by_series.items()):
        sequences, targets, target_months = build_sequences(monthly_rows)
        train_seq, validation_seq, test_seq = _split(list(range(len(sequences))))
        base = {
            "series_id": series_id,
            "sequence_window": WINDOW,
            "sequence_count": len(sequences),
            "train_sequences": len(train_seq),
            "validation_sequences": len(validation_seq),
            "test_sequences": len(test_seq),
            "rmse": None,
            "mae": None,
            "mape": None,
            "r2": None,
            "status": "not_trained",
            "notes": "",
        }
        if len(sequences) < 20:
            base["status"] = "insufficient_sequences"
            sequence_note = (
                "Fewer than 20 complete 12-month sequences; exploratory LSTM "
                "would not have a defensible chronological evaluation."
            )
            if tf is None:
                sequence_note += f" {tensorflow_note}"
            base["notes"] = sequence_note
            report_rows.append(base)
            continue
        if tf is None:
            base["status"] = "missing_tensorflow"
            base["notes"] = tensorflow_note
            report_rows.append(base)
            continue

        np.random.seed(42)
        tf.random.set_seed(42)
        x = np.asarray(sequences, dtype=float)
        y = np.asarray(targets, dtype=float)
        x_train = x[train_seq]
        y_train = y[train_seq]
        x_validation = x[validation_seq]
        y_validation = y[validation_seq]
        x_test = x[test_seq]
        y_test = y[test_seq]
        scaler = StandardScaler()
        scaler.fit(x_train.reshape(-1, 1))
        x_train_scaled = scaler.transform(x_train.reshape(-1, 1)).reshape(-1, WINDOW, 1)
        x_validation_scaled = scaler.transform(x_validation.reshape(-1, 1)).reshape(-1, WINDOW, 1)
        x_test_scaled = scaler.transform(x_test.reshape(-1, 1)).reshape(-1, WINDOW, 1)
        target_scaler = StandardScaler()
        target_scaler.fit(y_train.reshape(-1, 1))
        y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_validation_scaled = target_scaler.transform(y_validation.reshape(-1, 1)).ravel()

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(WINDOW, 1)),
                tf.keras.layers.LSTM(16),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        model.fit(
            x_train_scaled,
            y_train_scaled,
            validation_data=(x_validation_scaled, y_validation_scaled),
            epochs=100,
            batch_size=8,
            verbose=0,
            callbacks=[callback],
        )
        predicted_scaled = model.predict(x_test_scaled, verbose=0).ravel()
        predicted = target_scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()
        metrics = regression_metrics(y_test, predicted)
        slug = "_".join(part.replace(" ", "_") for part in series_id.split(" | "))
        model_dir = MODELS_DIR / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_dir / "lstm.keras")
        joblib.dump(
            {"input_scaler": scaler, "target_scaler": target_scaler},
            model_dir / "lstm_preprocessing.joblib",
        )
        write_actual_predicted_svg(
            PLOTS_DIR / f"{slug}__lstm.svg",
            f"{series_id} — exploratory LSTM",
            [target_months[index] for index in test_seq],
            y_test.tolist(),
            predicted.tolist(),
        )
        base.update(metrics)
        base["status"] = "trained_exploratory"
        base["notes"] = "Small 12-month LSTM with early stopping; not a reliable series-level result."
        report_rows.append(base)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "lstm_experiment.csv").open("w", newline="", encoding="utf-8") as file_obj:
        fields = list(report_rows[0].keys()) if report_rows else []
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)
    return report_rows


def run_isolated_lstm_experiment() -> dict:
    """Run the separate six-series TensorFlow experiment.

    The original ``run_lstm_experiment(prepared)`` function above remains
    available for the legacy market-level training command.  This adapter
    keeps the new aggregate LSTM outputs isolated from those legacy paths.
    """

    from .lstm_experiment_stage2 import run_isolated_lstm_experiment as runner

    return runner()


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    result = run_isolated_lstm_experiment()
    trained = sum(item.get("status") == "trained" for item in result["series"].values())
    print(f"LSTM experiment completed: {trained}/{len(result['series'])} series trained")
