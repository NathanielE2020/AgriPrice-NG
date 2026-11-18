# AgriPrice-NG

Machine-learning analysis and prediction of agricultural commodity prices in the Nigerian market.

## Phase 3 pipeline

Run from the project directory:

```text
python train.py
```

To prepare and audit the calendar series without fitting models:

```text
python train.py --prepare-only
```

To run the separate approved aggregate-series experiment without changing the market-level
experiment outputs:

```text
python train.py --aggregate-only
```

To run the read-only Phase 3.3 aggregate forecasting diagnosis:

```text
python train.py --diagnostics
```

To run the separate Phase 3.4 final validation experiment (use the project-local
`.venv` when it contains XGBoost/TensorFlow):

```text
python train.py --final-validation
```

To run the isolated TensorFlow LSTM recovery experiment, use the dedicated
`.venv-lstm` environment. This command trains only the six approved aggregate
series and does not overwrite the previous market-level or final-validation
artifacts:

```text
.venv-lstm\Scripts\python.exe -m src.lstm_experiment
```

The LSTM experiment uses complete 12-month calendar windows, chronological
70/15/15 sequence splits, training-only scaling, validation-selected target
transformations, and recursive 1-, 3-, and 6-month forecasts. Overlapping
windows are reported as training examples, not independent observations.

## Phase 4 read-only dashboard

Start the minimal Flask dashboard from the project-local environment:

```text
.venv\Scripts\Activate.ps1
python app.py
```

The dashboard is a read-only Flask and Chart.js application. Its responsive single-page
interface includes an overview, historical price analysis, recursive forecasting,
model-performance comparison, and project methodology sections. It consumes the
existing JSON endpoints for overview facts, approved market-level and aggregate series,
historical monthly prices, model metrics, and validated 1-, 3-, and 6-month forecasts.
Forecast requests never retrain models, fill missing target prices, or return confidence
intervals. LSTM is available only for the six approved aggregate series; market-level
LSTM remains unsupported.

The frontend assets are `templates/base.html`, `templates/dashboard.html`,
`static/css/dashboard.css`, and `static/js/dashboard.js`. Bootstrap provides the
responsive layout and Chart.js renders historical, actual-versus-forecast, RMSE, and
MAE charts. All displayed values are loaded from the Flask API; no sample or mock
prices are embedded in the browser code.

### Historical exploration and data freshness

`/api/history` accepts optional inclusive `start_date` and `end_date` parameters in
`YYYY-MM` or `YYYY-MM-DD` form. The dashboard uses these filters at monthly
resolution, preserves calendar gaps, and rejects malformed or reversed ranges with
an HTTP 400 response. Every series includes its own earliest/latest observation
dates, latest real recorded price, and freshness metadata. Freshness is calculated
relative to the service reference date using the configurable thresholds in
`src/dashboard_service.py`: fresh (0–3 complete months), recent (4–6), aging
(7–12), and stale (more than 12). These labels describe age only; they do not
indicate price accuracy.

Forecast responses expose the actual last observation date and price, forecast
origin month, age/freshness status, and a warning for aging or stale series. A stale
forecast is still generated from the saved model and its real historical input; it
is not moved to a later date and should not be interpreted as today's market price.

Dataset-wide overview facts are calculated dynamically from `data/raw/train.parquet`
through `pyarrow` in the Flask `.venv` (`pyarrow 22.0.0`, constrained in
`requirements.txt`). If a future environment genuinely cannot load the Parquet
file, the API explicitly marks `fallback_used: true` and identifies the verified
inspection-report fallback in `statistics_source`; it does not silently present
that fallback as dynamic data. Rerunning the deliberate preprocessing pipeline
allows future raw revisions to update the approved series; the dashboard never
retrains models automatically or treats newly added data as covered by existing
model artifacts.

Aggregate performance is loaded from
`reports/common_test_date_audit/common_test_date_model_comparison.csv`, so the
six model families and naive baseline are compared on identical target dates
within each series. The report also records that the preserved experiments use
different training cutoffs.

The Flask app runs in the active project environment. On Windows, LSTM
inference is delegated to `.venv-lstm` (TensorFlow 2.21.0) when that
environment exists. On Linux, the dashboard delegates to the active Python
interpreter (or the interpreter supplied through `LSTM_PYTHON`) through
`src/lstm_dashboard_inference.py`. In both cases the subprocess remains
isolated from the Flask process.
The adapter loads each saved `.keras` model and its JSON scalers, reconstructs
the selected target transformation, and recursively feeds predicted prices back
into the next 12-month input window. Actual future prices are never read. Start
the dashboard with:

```text
.venv\Scripts\Activate.ps1
python app.py
```

If the selected TensorFlow interpreter or a required LSTM artifact is
unavailable, the API returns a clear model-unavailable response. Forecast
confidence intervals are intentionally `null` because no validated interval
method exists.

### Linux/Render deployment preparation

The repository includes `render.yaml` and `.python-version` for a Python 3.12.5
Render web service. The Blueprint uses the Free plan with
`AGRIPRICE_DEPLOY_SCOPE=world_bank`. In that deployment scope the validated
champion registry is loaded first, then the service reads the normalized World
Bank CSV in bounded chunks and retains only rows belonging to its 12 registered
champion series. WFP market and aggregate datasets, the full World Bank panel,
and WFP source-summary scan are disabled. World Bank values are `c_` close
estimates (modelled/imputed), not necessarily direct survey observations. Local
development remains full-scope when the variable is absent; the separate
`approved_market` scope remains supported.

The production process is a single Gunicorn worker with one thread and no
application preload:

```text
python -m pip install -r requirements-deploy.txt
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120 app:app
```

The Render environment variables are `PYTHON_VERSION=3.12.5` and
`AGRIPRICE_DEPLOY_SCOPE=world_bank`. World Bank champion inference does not use
TensorFlow or LSTM. On Linux, leave `LSTM_PYTHON` unset to use the active
interpreter if the separate WFP aggregate LSTM functionality is used in another
scope. The World Bank deployment must include:

- `data/processed/world_bank/world_bank_monthly.csv`;
- `reports/world_bank/champion_models.json` and
  `reports/world_bank/model_comparison.csv`;
- all 12 artifacts under `models/world_bank/champions/`;
- `app.py`, `src/`, `templates/`, `static/`, `requirements-deploy.txt`,
  `.python-version`, and `render.yaml`.

`requirements-deploy.txt` pins Flask, Gunicorn, NumPy, Pandas, scikit-learn,
Joblib, and XGBoost to versions matching the environment that produced the
saved champion artifacts. TensorFlow and training-only dependencies are not
included. `requirements.txt` remains the complete local research/training
environment. The WFP raw files and processed tables, non-champion World Bank
rows, and unrelated experimental artifacts are not required by this
World-Bank-only service, but remain preserved for local reproducibility.

The remaining reports, raw World Bank release files, and forecast outputs are
retained for reproducibility and project review. The application does not
retrain or write model artifacts at runtime.

Final validation selects target transformations on the chronological validation period
only, refits the selected configuration on training plus validation rows, and evaluates
once on the untouched test period. It never overwrites earlier Phase 3/3.2/3.3 outputs.

The pipeline reads `data/raw/train.parquet` without modifying it. It reindexes each approved
commodity/market/unit/price-type series to calendar months, keeps missing target months missing,
and drops rows whose required calendar lags or rolling values are unavailable.

Approved Phase 3 series:

1. Sorghum (white) | Dawanau | 100 KG | Wholesale
2. Maize (white) | Dawanau | 100 KG | Wholesale
3. Millet | Dawanau | 100 KG | Wholesale
4. Maize | Illela (CBM) | KG | Wholesale
5. Sorghum | Mai Adoua (CBM) | KG | Wholesale

The tree models use chronological 70/15/15 train/validation/test splits. Stacking trains its
Ridge meta-model on chronological out-of-fold base-model predictions only. LSTM is exploratory
and requires at least 20 complete 12-month sequences for a defensible result.

## Important outputs

- `reports/model_performance_report.txt`: counts, evaluation results, dependency statuses, and limitations.
- `reports/model_comparison.csv`: baseline and model metrics.
- `reports/individual_series_metrics.csv`: same metrics with explicit series identifiers.
- `reports/lstm_experiment.csv`: complete 12-month sequence counts and LSTM status per series.
- `outputs/plots/`: actual-versus-predicted SVG charts.
- `outputs/forecasts/recursive_forecasts.csv`: 1-, 3-, and 6-month recursive forecast rows, including explicit unavailable statuses.
- `models/`: fitted tree and stacking models plus preprocessing metadata.
- `data/processed/`: reindexed monthly data, usable features, sample counts, and supplementary aggregates.

## Aggregate experiment outputs

The aggregate experiment uses six independent commodity/unit/price-type series. Its outputs are
isolated from the market-level run:

- `models/aggregate/`: aggregate Random Forest, Gradient Boosting, optional XGBoost, and stacking artifacts.
- `reports/aggregate/data_verification.csv`: raw-record aggregate calculation and market-composition checks.
- `reports/aggregate/training_sample_counts.csv`: aggregate calendar and usable-feature counts.
- `reports/aggregate/model_comparison.csv`: aggregate baseline and model metrics.
- `reports/aggregate/lstm_experiment.csv`: 12-month sequence counts and TensorFlow/Keras status.
- `reports/aggregate/xgboost_status.csv`: explicit XGBoost training status per aggregate series.
- `reports/aggregate/model_performance_report.txt`: aggregate experiment report and limitations.
- `outputs/aggregate/plots/`: aggregate actual-versus-predicted charts.
- `outputs/aggregate/forecasts/recursive_forecasts.csv`: 1-, 3-, and 6-month recursive forecasts.

Phase 3.3 diagnostic outputs are written under `reports/diagnostics/`. They include aligned
prediction-error analysis, original/log/change/percentage-change target comparisons, and a
linear-trend baseline comparison. This command does not overwrite saved models or prior reports.

Phase 3.4 outputs are isolated under `reports/final_validation/`,
`models/final_validation/aggregate/`, and `outputs/final_validation/`. The key files are
`final_model_comparison.csv`, `selection_audit.csv`, `aggregate_training_sample_counts.csv`,
`lstm_sequence_counts.csv`, `xgboost_training_report.txt`, `lstm_training_report.txt`,
`tensorflow_install_attempt.txt`, `final_validation_report.txt`, and `final_forecasts.csv`. Blank metrics mean that a model
was unavailable or failed; no missing result is fabricated. LSTM uses complete 12-month
calendar sequences and may therefore have different test months from the tabular models.

The isolated TensorFlow LSTM experiment writes to `models/lstm_experiment/`,
`reports/lstm_experiment/`, and `outputs/lstm_experiment/`. Its key files are
`sequence_counts.csv`, `target_selection.csv`, `model_comparison.csv`,
`test_predictions.csv`, `winner_by_series.csv`, `training_report.txt`, and
`forecasts/recursive_forecasts.csv`. LSTM results use a same-sequence-date naive
baseline; comparisons with final-validation tree metrics are marked when their
tabular test dates differ.

To audit a fair comparison on identical target months, use the project
environment, which contains the saved XGBoost-compatible dependencies:

```text
.venv\Scripts\python.exe -m src.common_test_date_audit
```

The read-only audit writes `reports/common_test_date_audit/`. It reloads the
saved tabular bundles, applies their recorded validation-selected transformations,
recalculates their saved test RMSE values, and intersects their legitimate test
dates with the saved LSTM test dates. It does not retrain or overwrite previous
experiments. Exact dates, actual values, predictions, coverage, eligibility, and
per-series RMSE/MAE/MAPE/R2 are recorded in the generated CSV files. Same-date
comparison does not remove the limitation that the historical tabular and LSTM
runs used different training cutoffs; a fully controlled study would retrain all
models from one common chronological split.

For research reproducibility and project-defense review, run the read-only audit
with the TensorFlow environment:

```text
.venv-lstm\Scripts\python.exe -m src.lstm_reproducibility_audit
```

It writes `reports/lstm_experiment/reproducibility_audit.csv` (one complete
record per series), `partition_dates.csv` (the exact train/validation/test target
months), `prediction_metric_audit.csv` (metrics recalculated from saved test
predictions), and `methodology_for_defense.md`. The audit reloads every saved
LSTM, verifies test coverage and metric equality, and explicitly reports that the
original run did not persist final-refit wall-clock duration; separately labelled
no-save replay timings are not presented as the historical run duration.

The aggregate target is the monthly unweighted `mean_price` from the existing aggregate audit,
with `n_markets` retained as metadata. Units and retail/wholesale prices are never combined.

If XGBoost or TensorFlow is not installed, the run remains reproducible and records the exact
missing dependency instead of fabricating metrics. Missing historical calendar prices are never
automatically interpolated.

## World Bank source experiment

The downloaded World Bank Nigeria release is preserved under
`data/raw/external/world_bank_nigeria_2026/` and is never merged into the WFP
files. Prepare and validate its normalized close-estimate table with:

```text
.venv\Scripts\python.exe train.py --world-bank-prepare-only
```

The isolated model experiment can then be rerun with:

```text
.venv\Scripts\python.exe train.py --world-bank-only
```

`src/world_bank_ingestion.py` maps the wide `c_`, `l_`, `h_`, and `trust_`
columns using the official ticker metadata. The normalized table is written to
`data/processed/world_bank/world_bank_monthly.csv` with source, unit, location,
geography level, survey-period metadata, low/high estimates, and trust scores.
Rows with missing close estimates are omitted; no interpolation or forward fill
is performed. The central `c_` values are World Bank modelled/imputed estimates,
not guaranteed direct market observations. `base_observation_price` and
`source_data_is_direct_observation` are retained so that this distinction is
visible.

The source has 67 actual market labels, 15 state labels (including the source's
distinct `Kano State (1)` label), five geopolitical-zone records, one National
Average, and one Market Average. Aggregate labels are not counted as markets or
states. The first trained panel contains six commodities (`beans`, imported
rice, millet, white maize, white sorghum, and yam) at National Average and
Market Average, giving 12 source-specific series. The other normalized market
and zone records remain available for later experiments rather than being
silently combined.

World Bank artifacts and reports are isolated under `models/world_bank/`,
`reports/world_bank/`, and `outputs/world_bank/`. Tree-model transformations are
selected on chronological validation data, stacking uses chronological
out-of-fold base predictions, and test dates are identical within each series.
Forecasts for 1, 3, and 6 months are recursive; actual future values are never
used and confidence intervals are returned as unavailable. The dashboard's
`World Bank estimates` source selector exposes the normalized history, estimate
ranges, trust scores, trained model availability, and explicit estimate
labelling without changing the existing WFP views.

The production champion–challenger registry is
`reports/world_bank/champion_models.json`. It records one validation-selected
champion per trained series, while the untouched test metrics and challenger
outputs remain available in the research reports. ML champions are refit on
training plus validation data and stored under
`models/world_bank/champions/`; a naive champion needs no serialized artifact.
The dashboard accepts only the registered champion for World Bank forecasting
and fails closed when the registry is missing, stale, malformed, or references
an unavailable artifact. The companion methodology reports are
`reports/world_bank/champion_training_report.txt` and
`reports/world_bank/champion_integration_report.txt`.

The World Bank panel currently marks LSTM unavailable rather than reusing a
WFP-trained sequence model. Existing TensorFlow LSTM artifacts remain
source-specific; a World Bank LSTM will only be enabled after a separate
isolated run has saved and verified its predictions and metrics.
