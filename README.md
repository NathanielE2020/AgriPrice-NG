# AgriPrice-NG

**Machine Learning Based Analysis and Prediction of Agricultural Commodities Prices in Nigerian Market**

AgriPrice-NG uses only the existing WFP Nigeria food-price data and NBS Selected Food Price Watch releases. The sources, product definitions, units, and geographies remain separate. The current fitted experiment is historical-only; it is not a live November 2026 price forecast.

## Active data

- WFP original CSVs are preserved locally under `data/raw/` and are excluded from the release. The active normalized input is `data/processed/wfp_monthly.csv`, which retains commodity, market, state, unit, price type, currency, monthly date, and price flag. Only pure `actual` observations are eligible for WFP training; aggregate and mixed flags are excluded.
- NBS original monthly workbooks and paired PDF reports remain locally under `data/raw/nbs/` and are excluded from the release. The active normalized input is `data/processed/nbs_monthly.csv`, with exact item labels and separate national/zone geography. Units are not inferred where the workbook does not state them. The January 2025 CPI rebase is retained as a methodology boundary.
- Neither NBS national/zone summaries nor WFP aggregate/mixed flags are combined with WFP direct market observations. Garri is not treated as raw cassava.

## Training and evaluation

Run the pipeline from the project root with the project environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.venv\Scripts\python.exe train.py
```

The command reads the two normalized CSVs and the approved WFP mapping audit. It does not rerun ingestion or write to raw or normalized datasets. A read-only input check is available with:

```powershell
.venv\Scripts\python.exe train.py --check-only
```

The experiment uses actual monthly WFP prices only, exact calendar lags 1–3, prior-only rolling 3-month statistics, month-of-year features, and a calendar trend. No missing target prices are interpolated or carried forward. A training row is excluded unless the target and each required prior calendar month are actual observations.

The fixed chronological 70/15/15 experiment compares the naive last-calendar-month baseline, Random Forest, XGBoost, Gradient Boosting, and leakage-safe stacking. Stacking's Ridge meta-model is trained on expanding-window out-of-fold base predictions from the training partition. The champion is selected by validation RMSE; test results are audit-only. Metrics include RMSE, MAE, zero-safe MAPE, R², and bias. Exact test dates and saved predictions are recorded.

LSTM uses a 12-month univariate sequence, a small `LSTM(16)` architecture, training-only MinMax scaling, inverse transformation, and early stopping. It is attempted only when train, validation, and test contain enough complete contiguous sequences. Adjacent windows overlap and are not independent observations.

### Current observed eligibility and results

The completed run trained four tree/stacking methods on seven historical WFP market series and one exploratory LSTM on Maize (white) at Dawanau. These are Maize (grain varieties) and imported rice series from Illela, Mai Adoua, Jibia, and Dawanau. The Rice (imported) and Maize market histories end in March 2017; Dawanau's full pure-actual history reaches December 2021, but its model window ends April 2019 because later observations are separated by a long gap. The Ibadan maize candidate has 77 usable lag rows and did not meet the 60-row training threshold after splitting. Other research commodities do not currently meet the predeclared historical training thresholds; NBS post-rebase series have only 17 monthly observations and remain descriptive-only.

Test periods are small (14–29 shared dates), irregular historical months; results are exploratory and unstable. The naive baseline is validation-selected champion on several series and remains hard to beat. Where another model wins validation, that does not guarantee lower test error. No supplied series supports a reliable forecast of current market conditions. All generated recursive outputs are labelled `historical_only`, with their source latest date and forecast origin recorded; do not describe them as current prices.

## Outputs

- `reports/model_training_evaluation_report.txt` — methodology, eligibility, results, and limitations.
- `reports/training/series_eligibility.csv` — WFP/NBS per-series counts, gaps, splits, and eligibility decisions.
- `reports/training/model_comparison.csv` — validation/test metrics, champion selection, exact test dates, and model status.
- `reports/training/test_predictions.csv` — saved actual and predicted values used for the reported metrics.
- `reports/training/model_registry.json` — selected champion, model availability, artifacts, and forecast freshness metadata.
- `reports/training/lstm_training_report.txt` and `lstm_experiment_summary.json` — LSTM architecture, sequence counts, scaling, and training record.
- `models/wfp_nbs/<run_id>/` — fitted artifacts for the experiment run.
- `outputs/training/forecasts.csv` — saved recursive 1-, 3-, and 6-month projections for validation-selected champions; every row is historical-only.
- `outputs/training/plots/` — actual-versus-predicted SVG charts.

## Dashboard

Start the existing Flask application:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`. Historical performance and saved champion forecasts are loaded from the registry and output tables; the dashboard does not load model libraries or retrain at startup. Unavailable series/models are reported as such, and forecast requests only return saved outputs for the validation-selected champion. Forecasts have no confidence intervals.

For Render, `render.yaml` selects Python 3.12.5, the `wfp_nbs` scope, one Gunicorn worker/thread, and `requirements.txt`. That file contains only Flask and Gunicorn. Install `requirements-training.txt` only in a research environment; it adds the data, tree-model, and TensorFlow training dependencies. The service reads the normalized WFP/NBS CSVs and registered reports/forecast CSV, not the original files under `data/raw/`. Forecasts remain historical-only. Render Free memory use is not guaranteed by local measurements.

## Project checks

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q app.py train.py src tests
node --check static/js/dashboard.js
```

Retired legacy modules may remain in the repository as historical source, but the active `app.py`, `train.py`, and WFP/NBS dashboard service do not import World Bank data or artifacts. Backups and original source data are retained.
