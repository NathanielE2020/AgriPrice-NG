# AgriPrice-NG

**Machine Learning Based Analysis and Prediction of Agricultural Commodities Prices in Nigerian Market**

AgriPrice-NG uses the existing WFP Nigeria food-price data and NBS Selected Food Price Watch releases. Sources, product definitions, units, and geographies remain separate. The dashboard exposes every normalized series for historical analysis and a forecast eligibility attempt. On-demand forecasting prefers an exact series model, then may use a same-source, same-definition pooled model only after group-held-out chronological validation. WFP aggregate and mixed-price-flag series are never treated as direct actual prices; NBS item labels, CPI-rebase regimes and national/zone levels remain distinct. Forecast requests return only future calendar months, and stale origins are explicitly labeled rather than presented as current market prices.

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

The completed research run trained four tree/stacking methods on seven WFP market series and one exploratory LSTM on Maize (white) at Dawanau. The research eligibility audit considered only mapped research candidates, required at least 96 actual months and 80 complete lag rows, then required at least 60 training, 10 validation, and 10 test rows. Thus the seven-series count describes that conservative comparison experiment, not every WFP market series that can be forecast on demand. NBS post-rebase series have only 17 monthly observations and remain descriptive-only.

Test periods are small (14–29 shared dates) and irregular; those historical research results are exploratory and unstable. The on-demand individual-series path retains its conservative threshold of 30 usable supervised rows, 20 training rows and 6 validation rows. For a sparse exact series, the added pooled fallback may borrow only within WFP's exact commodity/unit/price-type/currency or within an exact NBS item/methodology-regime/geography-level/unit definition. Pooled models predict log monthly price movement, preserve the selected series' price scale, and are evaluated with held-out markets/geographies and a chronological cutoff; global validation metrics and selected-series metrics (when available) are labeled separately. Naive last-observation and gap-adjusted trend baselines remain candidates, and the seasonal-naive baseline is included only when year-back prices are present on the same validation dates. No method is guaranteed to beat a baseline. Missing prices are never interpolated or forward-filled. As of November 2026, WFP's latest direct actual observations are in April 2026 and NBS reports extend through May 2026; forecasts from those cutoffs are stale-source estimates, not current-market prices. No confidence intervals are provided.

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

Open `http://127.0.0.1:5000`. Historical performance and saved research forecasts remain loaded from the registry and output tables. Forecast selectors are populated from every normalized WFP and NBS series that has reported prices; a selected non-actual WFP flag reaches the backend and receives a specific rejection rather than being silently omitted. The service does not train models at startup. For individual-eligible series it evaluates Random Forest, Gradient Boosting, last-observation, trend and, when fully supported on common dates, seasonal-naive candidates using chronological validation. If the exact series is too sparse, a compatible pooled strategy is considered and assessed through held-out geography plus chronological validation; insufficient or unsupported pools return an explanation. A selected naive baseline remains a baseline, not an advanced AI model. Runtime scikit-learn models are cached under the operating system's temporary directory; Render's filesystem is ephemeral, so artifacts may be rebuilt after restart. When the latest observation predates the current month, recursive estimates bridge the calendar gap internally; they are never represented as actual data or returned as history. API horizons are 1 through 6 future calendar months. Source age is shown prominently, and old data must not be interpreted as current prices. No confidence intervals are provided. Run `python -m src.commodity_coverage_audit --as-of YYYY-MM-DD` to make an actual one-month forecast attempt for every normalized series and write a row-level audit without modifying source data.

For Render, `render.yaml` selects Python 3.12.5, the `wfp_nbs` scope, one Gunicorn worker/thread, and `requirements.txt`. Production requirements include Flask, Gunicorn, NumPy, scikit-learn, and joblib for on-demand forecasting; TensorFlow and XGBoost are not required by this path. Install `requirements-training.txt` only in a research environment; it adds separate experiment dependencies. The service reads normalized WFP/NBS CSVs and registered reports, not original files under `data/raw/`. Local verification currently finds WFP actual observations through April 2026 and NBS reports through May 2026. As of November 2026 the eligible WFP forecasts are future-dated but use seven-month-old observations; they are explicitly not presented as current-market forecasts. Render Free memory use is not guaranteed by local measurements.

## Project checks

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q app.py train.py src tests
node --check static/js/dashboard.js
```

Retired legacy modules may remain in the repository as historical source, but the active `app.py`, `train.py`, and WFP/NBS dashboard service do not import World Bank data or artifacts. Backups and original source data are retained.
