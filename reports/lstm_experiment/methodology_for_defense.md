# AgriPrice-NG LSTM Experiment - Methodology for Project Defense

## What was tested

The experiment predicts the next monthly aggregate food price for six approved commodity/unit/price-type series. Each series was trained independently; series were not pooled and retail, wholesale, and different units were not combined.

The source is the prepared monthly aggregate file `data/processed/aggregate_monthly_reindexed.csv`. A monthly target is the existing unweighted `mean_price`. Missing calendar months remain missing. No target price was interpolated or forward-filled.

## How the sequences were constructed

The input length is 12 months. For a target month t, the input is the prices at t-12, t-11, ..., t-1, and the target is the price at t (or a selected movement transformation later reconstructed to price). A sequence is created only when every one of those calendar months is observed.

This rule prevents a previous available observation from being incorrectly treated as the previous calendar month. It also means that a gap removes sequences whose input window crosses the gap.

## Why overlapping windows are not independent observations

With a 12-month window, the sequence for target month 2024-02 uses 2023-02 through 2024-01. The next sequence for 2024-03 uses 2023-03 through 2024-02. Eleven of the twelve input months overlap. Therefore 100 valid windows do not mean 100 independent historical experiments: they are correlated views of the same finite price history. The count is useful for describing the number of training examples, but it must not be described as the number of independent observations or equivalent to 100 separate years of evidence.

## Chronological partitions

Sequence target months were ordered by calendar date and split without shuffling: 70% training, 15% validation, and 15% test. The validation period was used to choose the target transformation and the best epoch. The test period was not used for transformation, architecture, epoch, or scaling decisions.

Validation and test input windows may contain earlier observations from the training period because those prices were available at the historical forecast origin. No input contains an observation after its target month.

Exact target months for every partition are in `partition_dates.csv`; the compact scalar audit is in `reproducibility_audit.csv`.

## Scaling and target transformations

Inputs were standardized using parameters fitted on training input values only. The target was standardized separately using training target values only. The saved means and scales for every series are recorded in `reproducibility_audit.csv` and the corresponding `models/lstm_experiment/*/lstm_preprocessing.json` file.

Four candidates were considered on validation data: original price, log1p price, price change (current price minus previous calendar-month price), and percentage change. Validation predictions were reconstructed to price before calculating validation RMSE. The selected transformation was then refit on training plus validation data and evaluated once on the untouched test period.

## LSTM architecture and training configuration

The primary configuration was a 12-by-1 input, LSTM(32), Dropout(0.2), and Dense(1) linear output. The optimizer was Adam with learning rate 0.001; loss was mean squared error; batch size was 8; maximum epochs was 150; early-stopping patience was 15; shuffle was disabled; and random seed was 42.

The validation candidate used early stopping with restored best weights. The final refit used the validation-selected best epoch count on training plus validation data, because the test set must remain untouched. The original run persisted validation-candidate duration but did not persist final-refit duration; this is reported as an explicit provenance gap rather than estimated. A no-save replay of each final refit was timed separately and is labelled timed_reproduction_final_refit_duration_seconds; it is not claimed to be the original wall-clock duration. The runner has been updated to record final-refit duration on future executions.

## Evaluation and coverage

Metrics are RMSE, MAE, MAPE with zero actuals excluded from the MAPE denominator, R-squared, and bias (mean prediction minus actual). The saved test predictions are in `outputs/lstm_experiment/test_predictions.csv` and include actual price, LSTM prediction, naive previous-calendar-month value, month, and transformation.

The metric audit recalculates metrics from those saved predictions. It checks finite values, duplicate test months, test coverage, and absolute differences from the reported LSTM rows. Results are in `prediction_metric_audit.csv`.

## Reproducibility record

| Series | Observations | Missing months | Valid sequences | Train/validation/test | Selected target | Test coverage |
|---|---:|---:|---:|---|---|---:|
| Millet \| 2.6 KG \| Retail | 127 | 12 | 100 | 70/15/15 | price_change | 100.0% |
| Oil (palm) \| L \| Retail | 134 | 5 | 112 | 78/17/17 | price_change | 100.0% |
| Rice (imported) \| 2.8 KG \| Retail | 133 | 6 | 104 | 72/16/16 | price_change | 100.0% |
| Rice (imported) \| 50 KG \| Wholesale | 127 | 2 | 103 | 72/15/16 | original | 100.0% |
| Rice (local) \| 2.7 KG \| Retail | 131 | 6 | 103 | 72/15/16 | price_change | 100.0% |
| Yam \| 2.5 KG \| Retail | 137 | 0 | 125 | 87/19/19 | price_change | 100.0% |

## Results and defensible interpretation

The LSTM must be compared with the naive baseline on the exact same test dates. On that same-date sequence comparison, it improved RMSE for Oil (palm) and Yam, but not for Millet, Rice (imported) 2.8 KG, Rice (imported) 50 KG, or Rice (local). This is the verified result; no general claim that LSTM is the best model is justified.

The existing Random Forest, Gradient Boosting, XGBoost, and Stacking metrics were copied from the final-validation report. Their tabular test dates can differ from the LSTM sequence test dates, so rows marked with different alignment must not be presented as a perfectly matched head-to-head test. The comparison file records the exact dates and overlap count.

## Recursive forecasting

For 1-, 3-, and 6-month forecasts, the model predicts one month, appends that prediction to the history, and uses the updated history to predict the next month. Actual future prices are never used. Forecast uncertainty accumulates with the horizon, especially for series with missing historical months. No confidence intervals were claimed because none were implemented and validated.

## Suggested defense answers

**Why not randomly split the observations?** Random splitting would allow later price regimes to influence training and would not represent deployment, where the future is unknown. Chronological partitions preserve the forecasting direction.

**Why is the LSTM result exploratory?** Each series has only about 100-125 complete 12-month windows, the windows overlap heavily, and the test sets contain only 15-19 target months. These constraints limit statistical certainty even though the experiment is leakage-aware.

**Why can the naive model win?** A price series with strong persistence can be difficult to improve upon with a small dataset. Beating a more complex model is not required; the correct conclusion is that the naive baseline is a strong benchmark for these series.

**What should be reported in the thesis?** Report the exact partitions, missing months, scaling parameters, transformations, model configuration, saved predictions, metric audit, and the limitation that overlapping windows are correlated rather than independent.
