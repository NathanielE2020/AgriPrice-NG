# Common-test-date comparison decision

## Decision

The audit selected **Option A: common eligible prediction dates** wherever all six models had saved, leakage-safe predictions on the same target months. Metrics are calculated separately for each commodity/unit/price-type series; units and price types are never pooled into an unqualified score.

The existing tabular artifacts were not retrained. Their saved model bundles were loaded, their recorded validation-selected transformations were applied, and predictions were reproduced only on the tabular models' recorded final-validation test months. The LSTM predictions were loaded from `outputs/lstm_experiment/test_predictions.csv`. The naive baseline uses the actual previous calendar-month lag for each target month.

## Leakage and alignment checks

- A target month was included only when every compared model had a prediction and the actual price matched the aggregate monthly source.
- A tabular prediction was accepted only within its saved test period and when the fitted period ended before the target month.
- Exact train, validation, and test boundaries for the tabular feature split and the LSTM sequence split are recorded in `model_date_eligibility.csv` and the text report.
- Tabular transformations are marked validation-selected in the saved metadata and final-validation report; LSTM transformation selection is marked `validation` in its target-selection and reproducibility artifacts.
- Predictions were finite, dates were deduplicated, and all excluded dates are listed in `prediction_coverage.csv`.
- The audit did not use test RMSE to select a model or transformation.

## Important limitation

Same-date evaluation removes the largest comparison defect, but the historical experiments were not trained from one identical split: the tabular models and LSTM can have different training cutoffs because the LSTM requires a complete 12-month sequence. Therefore, the audit supports a defensible same-observation comparison for series with a non-empty all-model intersection, but a fully controlled six-model experiment would still retrain all models using one common chronological train/validation/test boundary.

## Reproducibility files

- `common_test_date_audit_report.txt`: date boundaries, decisions, and exclusions.
- `common_test_dates.csv`: one row per common target month with actual and model predictions.
- `common_test_date_model_comparison.csv`: per-series metrics on identical dates.
- `prediction_coverage.csv`: available, included, and excluded dates for every model.
- `model_date_eligibility.csv`: artifact, cutoff, transformation-selection, and eligibility checks.
