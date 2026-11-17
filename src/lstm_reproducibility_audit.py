"""Create an auditable, defense-ready record from the completed LSTM run.

The audit is read-only with respect to the raw data and previously trained
models.  It reads the isolated LSTM artifacts and writes additional audit
documents below ``reports/lstm_experiment``.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from .aggregate_preprocessing import series_id
from .lstm_experiment_stage2 import (
    FINAL_COMPARISON_PATH,
    MONTHLY_PATH,
    REPORT_DIR,
    _build_sequences,
    _fit_final,
    _load_monthly_groups,
    _metrics,
    _split_indices,
    _slug,
)
from .parquet_compat import month_index


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "lstm_experiment"
OUTPUT_PREDICTIONS = ROOT / "outputs" / "lstm_experiment" / "test_predictions.csv"
CONFIG_PATH = REPORT_DIR / "experiment_config.json"
SEQUENCE_PATH = REPORT_DIR / "sequence_counts.csv"
SELECTION_PATH = REPORT_DIR / "target_selection.csv"


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float_or_blank(value):
    if value in (None, "", "None"):
        return ""
    return float(value)


def _coverage(rows: list[dict]) -> dict:
    observed = {row["month"] for row in rows if row["observed"] and row["mean_price"] is not None}
    first = rows[0]["month"]
    last = rows[-1]["month"]
    missing = [
        f"{(idx - 1) // 12:04d}-{((idx - 1) % 12) + 1:02d}"
        for idx in range(month_index(first), month_index(last) + 1)
        if f"{(idx - 1) // 12:04d}-{((idx - 1) % 12) + 1:02d}" not in observed
    ]
    return {
        "original_observations": len(observed),
        "first_observation_month": first,
        "last_observation_month": last,
        "missing_calendar_month_count": len(missing),
        "missing_calendar_months": ";".join(missing),
    }


def _model_details(model_path: Path) -> dict:
    """Inspect the saved Keras graph and optimizer without fitting it."""

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=True)
        layers = []
        for layer in model.layers:
            name = layer.__class__.__name__
            if name == "LSTM":
                layers.append(
                    f"LSTM(units={layer.units}, activation={layer.activation.__name__}, recurrent_activation={layer.recurrent_activation.__name__})"
                )
            elif name == "Dropout":
                layers.append(f"Dropout(rate={layer.rate})")
            elif name == "Dense":
                activation = getattr(layer.activation, "__name__", str(layer.activation))
                layers.append(f"Dense(units={layer.units}, activation={activation})")
            else:
                layers.append(name)
        optimizer = model.optimizer.__class__.__name__ if model.optimizer is not None else "not_saved"
        learning_rate = ""
        if model.optimizer is not None:
            learning_rate = float(model.optimizer.learning_rate.numpy())
        loss = getattr(model.loss, "__name__", str(model.loss))
        return {
            "architecture": "Input(shape=(12, 1)) -> " + " -> ".join(layers),
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "loss": loss,
            "input_shape": str(model.input_shape),
            "status": "loaded",
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        return {
            "architecture": "",
            "optimizer": "",
            "learning_rate": "",
            "loss": "",
            "input_shape": "",
            "status": f"load_failed: {exc}",
        }


def _prediction_audit() -> tuple[list[dict], dict[str, list[dict]]]:
    predictions = _read_csv(OUTPUT_PREDICTIONS)
    comparison = _read_csv(REPORT_DIR / "model_comparison.csv")
    reported = {
        row["series_id"]: row
        for row in comparison
        if row.get("model") == "lstm"
    }
    grouped: dict[str, list[dict]] = {}
    for row in predictions:
        grouped.setdefault(row["series_id"], []).append(row)
    audits: list[dict] = []
    for current_series, rows in sorted(grouped.items()):
        actual = np.asarray([float(row["actual_price"]) for row in rows], dtype=float)
        predicted = np.asarray([float(row["predicted_price"]) for row in rows], dtype=float)
        metrics = _metrics(actual, predicted)
        months = [row["month"] for row in rows]
        duplicate_months = len(months) != len(set(months))
        source = reported.get(current_series, {})
        differences = {}
        for key in ("rmse", "mae", "mape", "r2", "bias"):
            saved_key = f"test_{key}" if key != "bias" else "bias"
            saved = _float_or_blank(source.get(saved_key, ""))
            recalculated = metrics.get(key)
            differences[key] = "" if saved == "" or recalculated is None else abs(float(saved) - float(recalculated))
        matches = all(value == "" or float(value) <= 1e-9 for value in differences.values())
        audits.append(
            {
                "series_id": current_series,
                "saved_prediction_count": len(rows),
                "test_months": ";".join(months),
                "actual_values_finite": bool(np.isfinite(actual).all()),
                "predicted_values_finite": bool(np.isfinite(predicted).all()),
                "duplicate_test_months": duplicate_months,
                "reported_rmse": source.get("test_rmse", ""),
                "recalculated_rmse": metrics["rmse"],
                "rmse_abs_difference": differences["rmse"],
                "reported_mae": source.get("test_mae", ""),
                "recalculated_mae": metrics["mae"],
                "mae_abs_difference": differences["mae"],
                "reported_mape": source.get("test_mape", ""),
                "recalculated_mape": metrics["mape"],
                "mape_abs_difference": differences["mape"],
                "reported_r2": source.get("test_r2", ""),
                "recalculated_r2": metrics["r2"],
                "r2_abs_difference": differences["r2"],
                "reported_bias": source.get("bias", ""),
                "recalculated_bias": metrics["bias"],
                "bias_abs_difference": differences["bias"],
                "metrics_match_saved_report": matches,
            }
        )
    return audits, grouped


def _timed_final_refit_reproduction(groups: dict[str, list[dict]], metadata_by_series: dict[str, dict]) -> dict[str, float | str]:
    """Time a no-save replay of each final train+validation fit.

    The original run did not persist final-refit wall-clock duration.  This
    replay uses the saved transformation, split dates, epoch count and random
    seed, but its timing is explicitly labelled as a reproduction measurement,
    not silently substituted for the historical run's duration.
    """

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - environment-specific
        return {series: f"unavailable: {exc}" for series in groups}
    durations: dict[str, float | str] = {}
    for current_series in sorted(groups):
        metadata = metadata_by_series[current_series]
        x, y, _, lags = _build_sequences(groups[current_series])
        train_i, validation_i, _ = _split_indices(len(x))
        fit_i = np.concatenate([train_i, validation_i])
        try:
            replay = _fit_final(
                tf,
                x[fit_i],
                y[fit_i],
                lags[fit_i],
                x[fit_i[:1]],
                metadata["selected_transformation"],
                int(metadata["final_fit_epochs"]),
            )
            durations[current_series] = float(replay["duration_seconds"])
            del replay
        except Exception as exc:  # pragma: no cover - environment-specific
            durations[current_series] = f"failed: {exc}"
    return durations


def run_reproducibility_audit() -> dict:
    groups = _load_monthly_groups()
    sequence_rows = {row["series_id"]: row for row in _read_csv(SEQUENCE_PATH)}
    selection_rows = _read_csv(SELECTION_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    prediction_audits, grouped_predictions = _prediction_audit()
    prediction_map = {row["series_id"]: row for row in prediction_audits}
    metadata_by_series = {
        current_series: json.loads(
            (MODELS_DIR / _slug(current_series) / "lstm_preprocessing.json").read_text(encoding="utf-8")
        )
        for current_series in groups
    }
    timed_reproduction = _timed_final_refit_reproduction(groups, metadata_by_series)

    audit_rows: list[dict] = []
    partition_rows: list[dict] = []
    for current_series in sorted(groups):
        slug = _slug(current_series)
        metadata_path = MODELS_DIR / slug / "lstm_preprocessing.json"
        metadata = metadata_by_series[current_series]
        sequence = sequence_rows[current_series]
        selected = metadata["selected_transformation"]
        selected_training = next(
            row
            for row in selection_rows
            if row["series_id"] == current_series
            and row["transformation"] == selected
            and row["status"] == "evaluated"
        )
        history_path = REPORT_DIR / "history" / f"{slug}.csv"
        history = _read_csv(history_path)
        selected_history = [
            row
            for row in history
            if row["stage"] == "validation_candidate"
            and row["transformation"] == selected
        ]
        best_val_loss = min(float(row["val_loss"]) for row in selected_history if row["val_loss"] not in ("", None))
        details = _model_details(MODELS_DIR / slug / "lstm.keras")
        train_months = metadata["train_target_months"]
        validation_months = metadata["validation_target_months"]
        test_months = metadata["test_target_months"]
        for partition, months in (("train", train_months), ("validation", validation_months), ("test", test_months)):
            partition_rows.append(
                {
                    "series_id": current_series,
                    "partition": partition,
                    "sequence_count": len(months),
                    "first_target_month": months[0] if months else "",
                    "last_target_month": months[-1] if months else "",
                    "target_months": ";".join(months),
                }
            )
        pred_audit = prediction_map[current_series]
        audit_rows.append(
            {
                "series_id": current_series,
                **_coverage(groups[current_series]),
                "valid_sequence_count": sequence["complete_sequence_count"],
                "train_sequence_count": sequence["train_sequences"],
                "validation_sequence_count": sequence["validation_sequences"],
                "test_sequence_count": sequence["test_sequences"],
                "train_target_months": ";".join(train_months),
                "validation_target_months": ";".join(validation_months),
                "test_target_months": ";".join(test_months),
                "input_sequence_length_months": metadata["sequence_window_months"],
                "prediction_target": "selected transformed target reconstructed to monthly mean_price",
                "selected_transformation": selected,
                "input_scaling_method": "training-only StandardScaler equivalent: (x - mean) / population_std",
                "input_scaler_mean": metadata["input_scaler"]["mean"],
                "input_scaler_scale": metadata["input_scaler"]["scale"],
                "target_scaling_method": "training-only StandardScaler equivalent on transformed target",
                "target_scaler_mean": metadata["target_scaler"]["mean"],
                "target_scaler_scale": metadata["target_scaler"]["scale"],
                "architecture": details["architecture"],
                "optimizer": details["optimizer"],
                "learning_rate": details["learning_rate"],
                "loss": details["loss"],
                "batch_size": config["batch_size"],
                "max_epochs": config["max_epochs"],
                "early_stopping_patience": config["early_stopping_patience"],
                "validation_candidate_actual_epochs": selected_training["epochs_run"],
                "validation_candidate_best_epoch": selected_training["best_epoch"],
                "validation_candidate_best_loss": best_val_loss,
                "validation_candidate_early_stop_epoch": selected_training["epochs_run"],
                "final_refit_epochs": metadata["final_fit_epochs"],
                "final_refit_early_stopping": "not used; fixed to validation-selected best epoch",
                "validation_candidate_duration_seconds": selected_training["duration_seconds"],
                "final_refit_duration_seconds": "not persisted by original run",
                "timed_reproduction_final_refit_duration_seconds": timed_reproduction.get(current_series, ""),
                "saved_test_prediction_count": pred_audit["saved_prediction_count"],
                "evaluation_coverage_percent": 100.0 * pred_audit["saved_prediction_count"] / len(test_months),
                "metrics_match_saved_report": pred_audit["metrics_match_saved_report"],
                "model_reload_status": details["status"],
                "random_seed": metadata["random_seed"],
            }
        )

    audit_fields = list(audit_rows[0].keys())
    _write_csv(REPORT_DIR / "reproducibility_audit.csv", audit_rows, audit_fields)
    _write_csv(
        REPORT_DIR / "partition_dates.csv",
        partition_rows,
        ["series_id", "partition", "sequence_count", "first_target_month", "last_target_month", "target_months"],
    )
    prediction_fields = list(prediction_audits[0].keys()) if prediction_audits else []
    _write_csv(REPORT_DIR / "prediction_metric_audit.csv", prediction_audits, prediction_fields)

    methodology = _methodology_document(audit_rows, prediction_audits, config)
    (REPORT_DIR / "methodology_for_defense.md").write_text(methodology, encoding="utf-8")

    summary = {
        "series_count": len(audit_rows),
        "all_models_reloaded": all(row["model_reload_status"] == "loaded" for row in audit_rows),
        "all_prediction_metrics_match": all(row["metrics_match_saved_report"] for row in prediction_audits),
        "all_test_coverage_complete": all(float(row["evaluation_coverage_percent"]) == 100.0 for row in audit_rows),
        "final_refit_duration_recording_gap": "The original completed run did not persist final refit duration; validation candidate duration is preserved. The runner now records final_fit_duration_seconds for future runs.",
        "timed_reproduction_final_refits": True,
        "artifacts": [
            "reproducibility_audit.csv",
            "partition_dates.csv",
            "prediction_metric_audit.csv",
            "methodology_for_defense.md",
        ],
    }
    (REPORT_DIR / "reproducibility_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"audit": audit_rows, "predictions": prediction_audits, "summary": summary}


def _methodology_document(audit_rows: list[dict], prediction_audits: list[dict], config: dict) -> str:
    lines = [
        "# AgriPrice-NG LSTM Experiment - Methodology for Project Defense",
        "",
        "## What was tested",
        "",
        "The experiment predicts the next monthly aggregate food price for six approved commodity/unit/price-type series. Each series was trained independently; series were not pooled and retail, wholesale, and different units were not combined.",
        "",
        "The source is the prepared monthly aggregate file `data/processed/aggregate_monthly_reindexed.csv`. A monthly target is the existing unweighted `mean_price`. Missing calendar months remain missing. No target price was interpolated or forward-filled.",
        "",
        "## How the sequences were constructed",
        "",
        "The input length is 12 months. For a target month t, the input is the prices at t-12, t-11, ..., t-1, and the target is the price at t (or a selected movement transformation later reconstructed to price). A sequence is created only when every one of those calendar months is observed.",
        "",
        "This rule prevents a previous available observation from being incorrectly treated as the previous calendar month. It also means that a gap removes sequences whose input window crosses the gap.",
        "",
        "## Why overlapping windows are not independent observations",
        "",
        "With a 12-month window, the sequence for target month 2024-02 uses 2023-02 through 2024-01. The next sequence for 2024-03 uses 2023-03 through 2024-02. Eleven of the twelve input months overlap. Therefore 100 valid windows do not mean 100 independent historical experiments: they are correlated views of the same finite price history. The count is useful for describing the number of training examples, but it must not be described as the number of independent observations or equivalent to 100 separate years of evidence.",
        "",
        "## Chronological partitions",
        "",
        "Sequence target months were ordered by calendar date and split without shuffling: 70% training, 15% validation, and 15% test. The validation period was used to choose the target transformation and the best epoch. The test period was not used for transformation, architecture, epoch, or scaling decisions.",
        "",
        "Validation and test input windows may contain earlier observations from the training period because those prices were available at the historical forecast origin. No input contains an observation after its target month.",
        "",
        "Exact target months for every partition are in `partition_dates.csv`; the compact scalar audit is in `reproducibility_audit.csv`.",
        "",
        "## Scaling and target transformations",
        "",
        "Inputs were standardized using parameters fitted on training input values only. The target was standardized separately using training target values only. The saved means and scales for every series are recorded in `reproducibility_audit.csv` and the corresponding `models/lstm_experiment/*/lstm_preprocessing.json` file.",
        "",
        "Four candidates were considered on validation data: original price, log1p price, price change (current price minus previous calendar-month price), and percentage change. Validation predictions were reconstructed to price before calculating validation RMSE. The selected transformation was then refit on training plus validation data and evaluated once on the untouched test period.",
        "",
        "## LSTM architecture and training configuration",
        "",
        f"The primary configuration was a 12-by-1 input, LSTM({config['architecture']['lstm_units']}), Dropout({config['architecture']['dropout']}), and Dense(1) linear output. The optimizer was Adam with learning rate {config.get('learning_rate', 0.001)}; loss was mean squared error; batch size was {config['batch_size']}; maximum epochs was {config['max_epochs']}; early-stopping patience was {config['early_stopping_patience']}; shuffle was disabled; and random seed was {config['random_seed']}.",
        "",
        "The validation candidate used early stopping with restored best weights. The final refit used the validation-selected best epoch count on training plus validation data, because the test set must remain untouched. The original run persisted validation-candidate duration but did not persist final-refit duration; this is reported as an explicit provenance gap rather than estimated. A no-save replay of each final refit was timed separately and is labelled timed_reproduction_final_refit_duration_seconds; it is not claimed to be the original wall-clock duration. The runner has been updated to record final-refit duration on future executions.",
        "",
        "## Evaluation and coverage",
        "",
        "Metrics are RMSE, MAE, MAPE with zero actuals excluded from the MAPE denominator, R-squared, and bias (mean prediction minus actual). The saved test predictions are in `outputs/lstm_experiment/test_predictions.csv` and include actual price, LSTM prediction, naive previous-calendar-month value, month, and transformation.",
        "",
        "The metric audit recalculates metrics from those saved predictions. It checks finite values, duplicate test months, test coverage, and absolute differences from the reported LSTM rows. Results are in `prediction_metric_audit.csv`.",
        "",
        "## Reproducibility record",
        "",
        "| Series | Observations | Missing months | Valid sequences | Train/validation/test | Selected target | Test coverage |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for row in audit_rows:
        display_series = row["series_id"].replace("|", "\\|")
        lines.append(
            f"| {display_series} | {row['original_observations']} | {row['missing_calendar_month_count']} | {row['valid_sequence_count']} | {row['train_sequence_count']}/{row['validation_sequence_count']}/{row['test_sequence_count']} | {row['selected_transformation']} | {float(row['evaluation_coverage_percent']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Results and defensible interpretation",
            "",
            "The LSTM must be compared with the naive baseline on the exact same test dates. On that same-date sequence comparison, it improved RMSE for Oil (palm) and Yam, but not for Millet, Rice (imported) 2.8 KG, Rice (imported) 50 KG, or Rice (local). This is the verified result; no general claim that LSTM is the best model is justified.",
            "",
            "The existing Random Forest, Gradient Boosting, XGBoost, and Stacking metrics were copied from the final-validation report. Their tabular test dates can differ from the LSTM sequence test dates, so rows marked with different alignment must not be presented as a perfectly matched head-to-head test. The comparison file records the exact dates and overlap count.",
            "",
            "## Recursive forecasting",
            "",
            "For 1-, 3-, and 6-month forecasts, the model predicts one month, appends that prediction to the history, and uses the updated history to predict the next month. Actual future prices are never used. Forecast uncertainty accumulates with the horizon, especially for series with missing historical months. No confidence intervals were claimed because none were implemented and validated.",
            "",
            "## Suggested defense answers",
            "",
            "**Why not randomly split the observations?** Random splitting would allow later price regimes to influence training and would not represent deployment, where the future is unknown. Chronological partitions preserve the forecasting direction.",
            "",
            "**Why is the LSTM result exploratory?** Each series has only about 100-125 complete 12-month windows, the windows overlap heavily, and the test sets contain only 15-19 target months. These constraints limit statistical certainty even though the experiment is leakage-aware.",
            "",
            "**Why can the naive model win?** A price series with strong persistence can be difficult to improve upon with a small dataset. Beating a more complex model is not required; the correct conclusion is that the naive baseline is a strong benchmark for these series.",
            "",
            "**What should be reported in the thesis?** Report the exact partitions, missing months, scaling parameters, transformations, model configuration, saved predictions, metric audit, and the limitation that overlapping windows are correlated rather than independent.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    result = run_reproducibility_audit()
    print(json.dumps(result["summary"], indent=2))
