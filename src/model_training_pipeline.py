"""Leakage-aware historical WFP/NBS model experiments.

The pipeline reads only the already-normalized WFP and NBS tables. It never
rewrites source or normalized data. WFP and NBS series remain separate; NBS
series split at the January 2025 CPI rebasing boundary and are not trained
because their post-rebase histories are too short for the required comparison.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
WFP_PATH = ROOT / "data" / "processed" / "wfp_monthly.csv"
NBS_PATH = ROOT / "data" / "processed" / "nbs_monthly.csv"
WFP_SELECTION_PATH = ROOT / "reports" / "wfp_nbs_series_availability.csv"
REPORT_DIR = ROOT / "reports" / "training"
MODEL_DIR = ROOT / "models" / "wfp_nbs"
OUTPUT_DIR = ROOT / "outputs" / "training"
PLOT_DIR = OUTPUT_DIR / "plots"

RESEARCH_COMMODITIES = (
    "Rice (local)", "Rice (imported)", "Maize", "Cassava", "Yam", "Beans",
    "Tomatoes", "Palm oil",
)
DIRECT_MAPPING_STATUSES = {"direct_match", "direct_variant"}
IN_SCOPE_MAPPING_STATUSES = DIRECT_MAPPING_STATUSES | {
    "related_variant", "related_variant_requires_review", "related_not_equivalent",
    "ambiguous_do_not_merge",
}
MIN_RAW_MONTHS = 96
MIN_TREE_TRAIN_ROWS = 60
MIN_VALIDATION_ROWS = 10
MIN_TEST_ROWS = 10
MIN_LSTM_TRAIN_SEQUENCES = 60
MIN_LSTM_VALIDATION_SEQUENCES = 10
MIN_LSTM_TEST_SEQUENCES = 10
LSTM_WINDOW = 12
FEATURE_COLUMNS = (
    "lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_std_3",
    "month_sin", "month_cos", "trend_month",
)
MODEL_NAMES = (
    "naive_last_observation", "random_forest", "xgboost",
    "gradient_boosting", "stacking", "lstm",
)


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _market_id(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return _text(value)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:120] or "series"


def _series_id(source: str, fields: tuple[str, ...]) -> str:
    label = "-".join(re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-") for part in fields)
    digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:12]
    return f"{source}-{label[:70]}-{digest}"


def _wfp_dashboard_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        _text(row.get("commodity")), _market_id(row.get("market_id")),
        _text(row.get("market")), _text(row.get("admin1")), _text(row.get("unit")),
        _text(row.get("pricetype")), _text(row.get("currency")), "actual",
    )


def _month_period(value: Any) -> pd.Period:
    return pd.Period(str(value), freq="M")


def _calendar_facts(months: Iterable[pd.Period]) -> dict[str, Any]:
    values = sorted(set(months))
    if not values:
        return {
            "first_month": "", "latest_month": "", "span_months": 0,
            "missing_months": 0, "missing_month_list": "", "coverage_percent": 0.0,
            "longest_consecutive_missing_months": 0, "longest_consecutive_observed_months": 0,
        }
    calendar = pd.period_range(values[0], values[-1], freq="M")
    have = set(values)
    missing = [month for month in calendar if month not in have]
    missing_runs: list[int] = []
    observed_runs: list[int] = []
    current_missing = current_observed = 0
    for month in calendar:
        if month in have:
            current_observed += 1
            if current_missing:
                missing_runs.append(current_missing)
                current_missing = 0
        else:
            current_missing += 1
            if current_observed:
                observed_runs.append(current_observed)
                current_observed = 0
    if current_missing:
        missing_runs.append(current_missing)
    if current_observed:
        observed_runs.append(current_observed)
    return {
        "first_month": str(values[0]), "latest_month": str(values[-1]),
        "span_months": len(calendar), "missing_months": len(missing),
        "missing_month_list": ";".join(map(str, missing)),
        "coverage_percent": 100.0 * len(values) / len(calendar),
        "longest_consecutive_missing_months": max(missing_runs, default=0),
        "longest_consecutive_observed_months": max(observed_runs, default=0),
    }


def _longest_observed_run(price_map: dict[pd.Period, float]) -> list[pd.Period]:
    months = sorted(price_map)
    runs: list[list[pd.Period]] = []
    for month in months:
        if not runs or month.ordinal != runs[-1][-1].ordinal + 1:
            runs.append([month])
        else:
            runs[-1].append(month)
    # Prefer the latest run when two runs have equal length.
    return max(runs, key=lambda run: (len(run), run[-1].ordinal), default=[])


def build_monthly_features(price_map: dict[pd.Period, float]) -> pd.DataFrame:
    """Build only calendar-correct features; gaps stay NaN and rows are excluded."""
    if not price_map:
        return pd.DataFrame(columns=["target", *FEATURE_COLUMNS])
    periods = pd.period_range(min(price_map), max(price_map), freq="M")
    prices = pd.Series([price_map.get(month, np.nan) for month in periods], index=periods, dtype=float)
    frame = pd.DataFrame(index=periods)
    frame["target"] = prices
    for lag in (1, 2, 3):
        frame[f"lag_{lag}"] = prices.shift(lag)
    prior = prices.shift(1)
    frame["rolling_mean_3"] = prior.rolling(3, min_periods=3).mean()
    frame["rolling_std_3"] = prior.rolling(3, min_periods=3).std(ddof=0)
    month_number = np.asarray([month.month for month in periods], dtype=float)
    frame["month_sin"] = np.sin(2.0 * np.pi * month_number / 12.0)
    frame["month_cos"] = np.cos(2.0 * np.pi * month_number / 12.0)
    frame["trend_month"] = np.arange(len(periods), dtype=float)
    return frame.dropna(subset=["target", *FEATURE_COLUMNS])


def build_lstm_sequences(price_map: dict[pd.Period, float], window: int = LSTM_WINDOW):
    """Return complete contiguous monthly windows and their target months."""
    months = sorted(price_map)
    sequences: list[list[float]] = []
    targets: list[float] = []
    target_months: list[pd.Period] = []
    for month in months:
        prior = [month - offset for offset in range(window, 0, -1)]
        if all(item in price_map for item in prior):
            sequences.append([float(price_map[item]) for item in prior])
            targets.append(float(price_map[month]))
            target_months.append(month)
    return np.asarray(sequences, dtype=float), np.asarray(targets, dtype=float), target_months


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 split on time-sorted usable target rows, never shuffled."""
    n = len(frame)
    train_end = int(math.floor(n * 0.70))
    validation_end = train_end + int(math.floor(n * 0.15))
    return frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:]


def _date_split_cutoffs(months: list[pd.Period]) -> tuple[pd.Period, pd.Period]:
    calendar = pd.period_range(min(months), max(months), freq="M")
    first_cut = max(1, int(math.floor(len(calendar) * 0.70)))
    second_cut = max(first_cut + 1, int(math.floor(len(calendar) * 0.85)))
    second_cut = min(second_cut, len(calendar) - 1)
    return calendar[first_cut - 1], calendar[second_cut - 1]


def _partition_dates(index: Iterable[pd.Period], first_cut: pd.Period, second_cut: pd.Period):
    months = list(index)
    train = [month for month in months if month <= first_cut]
    validation = [month for month in months if first_cut < month <= second_cut]
    test = [month for month in months if month > second_cut]
    return train, validation, test


def _wfp_candidates(wfp: pd.DataFrame) -> list[dict[str, Any]]:
    if not WFP_SELECTION_PATH.is_file():
        raise FileNotFoundError("The verified WFP series mapping audit is missing.")
    audit = pd.read_csv(WFP_SELECTION_PATH, keep_default_na=False)
    actual = wfp[(wfp["priceflag"] == "actual") & wfp["price"].notna()].copy()
    actual["observation_month"] = actual["observation_month"].astype(str)
    group_columns = ["commodity", "market_id", "market", "admin1", "unit", "pricetype", "currency", "priceflag"]
    groups: dict[tuple[str, ...], pd.DataFrame] = {}
    for raw_key, group in actual.groupby(group_columns, dropna=False, sort=False):
        normalized = tuple(_market_id(x) if i == 1 else _text(x) for i, x in enumerate(raw_key))
        groups[normalized] = group
    candidates = []
    for _, mapping in audit.iterrows():
        key = (
            _text(mapping.get("commodity")), _market_id(mapping.get("market_id")),
            _text(mapping.get("market")), _text(mapping.get("state")),
            _text(mapping.get("unit")), _text(mapping.get("price_type")),
            _text(mapping.get("currency")), _text(mapping.get("price_flag")),
        )
        group = groups.get(key, pd.DataFrame(columns=actual.columns)).copy()
        group = group.sort_values("observation_month")
        if not group.empty and group["observation_month"].duplicated().any():
            # The normalized table is expected to contain one monthly target per key.
            raise ValueError(f"Duplicate WFP monthly series key: {key}.")
        prices = {_month_period(row.observation_month): float(row.price) for row in group.itertuples()}
        candidates.append({
            "source": "WFP", "source_key": "wfp", "mapping": mapping.to_dict(),
            "key": key, "rows": group, "price_map": prices,
            "research_commodity": _text(mapping.get("research_commodity")),
            "mapping_status": _text(mapping.get("mapping_status")),
            "series_id": _series_id("wfp", key),
        })
    return candidates


def _nbs_candidates(nbs: pd.DataFrame) -> list[dict[str, Any]]:
    relevant = nbs[
        nbs["research_commodity_mapping"].notna()
        & nbs["mapping_status"].isin(IN_SCOPE_MAPPING_STATUSES)
    ].copy()
    columns = [
        "item_label", "geography_level", "geography_name", "unit", "price_type",
        "currency", "methodology_regime", "research_commodity_mapping", "mapping_status",
    ]
    candidates = []
    for key, group in relevant.groupby(columns, dropna=False, sort=True):
        item, level, geography, unit, price_type, currency, regime, research, mapping_status = key
        group = group.sort_values("observation_month")
        periods = [_month_period(value) for value in group["observation_month"]]
        prices = {
            _month_period(row.observation_month): float(row.price)
            for row in group.itertuples() if pd.notna(row.price)
        }
        candidates.append({
            "source": "NBS", "source_key": "nbs", "rows": group,
            "price_map": prices,
            "research_commodity": _text(research), "mapping_status": _text(mapping_status),
            "item_label": _text(item), "geography_level": _text(level),
            "geography_name": _text(geography), "unit": _text(unit),
            "price_type": _text(price_type), "currency": _text(currency),
            "methodology_regime": _text(regime),
            "series_id": _series_id("nbs", tuple(_text(x) for x in (item, level, geography, unit, price_type, currency))),
            "periods": periods,
        })
    return candidates


def _eligibility_row(candidate: dict[str, Any]) -> dict[str, Any]:
    price_map = candidate["price_map"]
    observed_months = sorted(price_map)
    calendar = _calendar_facts(observed_months)
    features = build_monthly_features(price_map)
    tree_train, tree_validation, tree_test = chronological_split(features)
    run = _longest_observed_run(price_map)
    sequences, _, sequence_months = build_lstm_sequences(price_map)
    lstm_train, lstm_validation, lstm_test = chronological_split(pd.DataFrame(index=sequence_months))
    direct = candidate["mapping_status"] in DIRECT_MAPPING_STATUSES
    reasons: list[str] = []
    if not direct:
        reasons.append("mapping is related or ambiguous; not a direct target match")
    if candidate["source_key"] == "nbs":
        reasons.append("NBS post-rebase series have fewer than 96 monthly observations; pre-rebase data are a separate methodology regime")
        if not candidate.get("unit"):
            reasons.append("unit is not explicitly stated; item labels remain separate")
    else:
        if len(observed_months) < MIN_RAW_MONTHS:
            reasons.append(f"fewer than {MIN_RAW_MONTHS} actual monthly observations")
        if len(features) < 80:
            reasons.append("fewer than 80 complete calendar-lag supervised rows")
        if len(tree_train) < MIN_TREE_TRAIN_ROWS:
            reasons.append(f"fewer than {MIN_TREE_TRAIN_ROWS} chronological training rows after calendar-lag exclusions")
        if len(tree_validation) < MIN_VALIDATION_ROWS:
            reasons.append(f"fewer than {MIN_VALIDATION_ROWS} validation rows")
        if len(tree_test) < MIN_TEST_ROWS:
            reasons.append(f"fewer than {MIN_TEST_ROWS} test rows")
    eligible_tree = (
        candidate["source_key"] == "wfp" and direct
        and len(observed_months) >= MIN_RAW_MONTHS and len(features) >= 80
        and len(tree_train) >= MIN_TREE_TRAIN_ROWS
        and len(tree_validation) >= MIN_VALIDATION_ROWS
        and len(tree_test) >= MIN_TEST_ROWS
    )
    if not eligible_tree and candidate["source_key"] == "wfp" and direct and not reasons:
        reasons.append("does not meet the minimum chronological sample counts")
    if candidate["source_key"] == "nbs":
        eligible_tree = False
    latest_period = max(observed_months) if observed_months else None
    as_of = pd.Period(date.today(), freq="M")
    recent24_start = as_of - 23
    recent36_start = as_of - 35
    recent24_count = sum(recent24_start <= month <= as_of for month in observed_months)
    recent36_count = sum(recent36_start <= month <= as_of for month in observed_months)
    if candidate["source_key"] == "wfp" and not candidate["rows"].empty:
        first_observation_date = _text(candidate["rows"].iloc[0].get("first_observation_date"))
        latest_observation_date = _text(candidate["rows"].iloc[-1].get("last_observation_date"))
    else:
        first_observation_date = calendar["first_month"]
        latest_observation_date = calendar["latest_month"]
    run_map = {month: price_map[month] for month in run}
    run_sequences, _, run_sequence_months = build_lstm_sequences(run_map)
    if run:
        run_calendar = list(pd.period_range(min(run), max(run), freq="M"))
        first_cut, second_cut = _date_split_cutoffs(run_calendar)
        run_seq_train, run_seq_validation, run_seq_test = _partition_dates(
            run_sequence_months, first_cut, second_cut,
        )
    else:
        run_seq_train, run_seq_validation, run_seq_test = [], [], []
    eligible_lstm = bool(
        eligible_tree and len(run) >= MIN_RAW_MONTHS
        and len(run_seq_train) >= MIN_LSTM_TRAIN_SEQUENCES
        and len(run_seq_validation) >= MIN_LSTM_VALIDATION_SEQUENCES
        and len(run_seq_test) >= MIN_LSTM_TEST_SEQUENCES
    )
    return {
        "series_id": candidate["series_id"], "source": candidate["source"],
        "research_commodity": candidate["research_commodity"],
        "mapping_status": candidate["mapping_status"],
        "commodity": candidate["key"][0] if candidate["source_key"] == "wfp" else candidate["item_label"],
        "item_label": candidate.get("item_label", ""),
        "market": candidate["key"][2] if candidate["source_key"] == "wfp" else "",
        "state_or_geography_level": candidate["key"][3] if candidate["source_key"] == "wfp" else candidate["geography_level"],
        "geography_name": candidate.get("geography_name", ""),
        "unit": candidate["key"][4] if candidate["source_key"] == "wfp" else candidate.get("unit", ""),
        "price_type": candidate["key"][5] if candidate["source_key"] == "wfp" else candidate.get("price_type", ""),
        "currency": candidate["key"][6] if candidate["source_key"] == "wfp" else candidate.get("currency", ""),
        "methodology_regime": candidate.get("methodology_regime", ""),
        "actual_monthly_observations": len(observed_months),
        "first_actual_month": calendar["first_month"], "latest_actual_month": calendar["latest_month"],
        "first_actual_observation_date": first_observation_date,
        "latest_actual_observation_date": latest_observation_date,
        "months_since_latest_actual": _month_age(date.today(), latest_period) if latest_period else None,
        "actual_observations_last_24_months": recent24_count,
        "actual_observations_last_36_months": recent36_count,
        "actual_observations_in_2026": sum(month.year == 2026 for month in observed_months),
        "has_actual_observation_in_2026": any(month.year == 2026 for month in observed_months),
        "calendar_months_in_span": calendar["span_months"],
        "missing_calendar_months": calendar["missing_months"],
        "missing_calendar_month_list": calendar["missing_month_list"],
        "calendar_coverage_percent": round(calendar["coverage_percent"], 4),
        "longest_consecutive_missing_months": calendar["longest_consecutive_missing_months"],
        "longest_consecutive_observed_months": calendar["longest_consecutive_observed_months"],
        "complete_lag_1_to_3_target_rows": len(features),
        "actual_months_excluded_for_missing_required_lags": len(set(observed_months) - set(features.index)),
        "actual_month_list_excluded_for_missing_required_lags": ";".join(map(str, sorted(set(observed_months) - set(features.index)))),
        "tree_train_rows_70_15_15": len(tree_train),
        "tree_validation_rows_70_15_15": len(tree_validation),
        "tree_test_rows_70_15_15": len(tree_test),
        "valid_lstm_sequences_window_12": len(sequences),
        "lstm_train_sequences_row_split": len(lstm_train),
        "lstm_validation_sequences_row_split": len(lstm_validation),
        "lstm_test_sequences_row_split": len(lstm_test),
        "lstm_longest_run_train_sequences": len(run_seq_train),
        "lstm_longest_run_validation_sequences": len(run_seq_validation),
        "lstm_longest_run_test_sequences": len(run_seq_test),
        "longest_contiguous_run_start": str(run[0]) if run else "",
        "longest_contiguous_run_end": str(run[-1]) if run else "",
        "longest_contiguous_run_months": len(run),
        "actual_months_outside_longest_contiguous_run": len(set(observed_months) - set(run)),
        "actual_month_list_outside_longest_contiguous_run": ";".join(map(str, sorted(set(observed_months) - set(run)))),
        "tree_model_eligible": eligible_tree,
        "lstm_model_eligible": eligible_lstm,
        "eligibility_status": "eligible_historical_wfp_experiment" if eligible_tree else "not_eligible",
        "eligibility_reason": "; ".join(reasons) if reasons else ("Eligible for historical WFP experiment only; not a current-market forecast series." if eligible_tree else "No qualifying training history."),
    }


def _estimators(seed: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=1.0,
            random_state=seed, n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.03, max_depth=2,
            min_samples_leaf=2, loss="huber", random_state=seed,
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=250, max_depth=2, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", random_state=seed, n_jobs=1,
            verbosity=0,
        )
    except Exception:
        # The caller records the unavailable status without claiming metrics.
        pass
    return models


def _fit_stacking(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> dict[str, Any]:
    """Fit a Ridge meta-model only from expanding-window OOF base predictions."""
    base_names = [name for name in ("random_forest", "xgboost", "gradient_boosting") if name in _estimators(seed)]
    n = len(X)
    warmup = max(30, int(math.floor(n * 0.50)))
    remaining = n - warmup
    if remaining < 20 or len(base_names) < 2:
        raise ValueError("insufficient train-only walk-forward predictions for a leakage-safe meta-model")
    fold_count = min(4, max(2, remaining // 5))
    boundaries = np.linspace(warmup, n, fold_count + 1, dtype=int)
    oof_rows: list[np.ndarray] = []
    oof_targets: list[float] = []
    for fold in range(fold_count):
        start, end = int(boundaries[fold]), int(boundaries[fold + 1])
        if end <= start:
            continue
        fold_models = _estimators(seed)
        fit_end = start
        for name in base_names:
            fold_models[name].fit(X.iloc[:fit_end], y.iloc[:fit_end])
        oof_rows.extend(
            np.column_stack([fold_models[name].predict(X.iloc[index:index + 1]) for name in base_names])[0]
            for index in range(start, end)
        )
        oof_targets.extend(float(value) for value in y.iloc[start:end])
    if len(oof_rows) < 20:
        raise ValueError("fewer than 20 leakage-safe walk-forward predictions were produced")
    meta = Ridge(alpha=1.0)
    meta.fit(np.asarray(oof_rows, dtype=float), np.asarray(oof_targets, dtype=float))
    base_models = _estimators(seed)
    for name in base_names:
        base_models[name].fit(X, y)
    return {
        "model_name": "stacking", "base_names": base_names,
        "base_models": {name: base_models[name] for name in base_names},
        "meta_model": meta, "oof_count": len(oof_rows),
        "oof_method": "expanding-window chronological predictions within training partition",
    }


def _predict_stacking(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    matrix = np.column_stack([bundle["base_models"][name].predict(X) for name in bundle["base_names"]])
    return np.asarray(bundle["meta_model"].predict(matrix), dtype=float)


def _lstm_sequences_for_months(price_map: dict[pd.Period, float], months: Iterable[pd.Period]):
    lookup = set(months)
    X, y, targets = build_lstm_sequences(price_map)
    keep = [i for i, month in enumerate(targets) if month in lookup]
    return X[keep], y[keep], [targets[i] for i in keep]


def _fit_lstm(
    price_map: dict[pd.Period, float], train_months: list[pd.Period],
    validation_months: list[pd.Period], seed: int = 42,
):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()
    X_train, y_train, train_dates = _lstm_sequences_for_months(price_map, train_months)
    X_val, y_val, val_dates = _lstm_sequences_for_months(price_map, validation_months)
    if len(X_train) < MIN_LSTM_TRAIN_SEQUENCES or len(X_val) < MIN_LSTM_VALIDATION_SEQUENCES:
        raise ValueError("insufficient chronological LSTM training or validation sequences")
    fit_values = [value for month, value in price_map.items() if month <= max(train_months)]
    scale_min = float(min(fit_values))
    scale_max = float(max(fit_values))
    scale_range = scale_max - scale_min or 1.0
    X_train_scaled = ((X_train - scale_min) / scale_range).reshape((-1, LSTM_WINDOW, 1))
    y_train_scaled = (y_train - scale_min) / scale_range
    X_val_scaled = ((X_val - scale_min) / scale_range).reshape((-1, LSTM_WINDOW, 1))
    y_val_scaled = (y_val - scale_min) / scale_range

    inputs = tf.keras.Input(shape=(LSTM_WINDOW, 1), name="monthly_price_window")
    hidden = tf.keras.layers.LSTM(16, name="lstm_16")(inputs)
    hidden = tf.keras.layers.Dropout(0.10, name="dropout_10pct")(hidden)
    outputs = tf.keras.layers.Dense(1, name="price_output")(hidden)
    model = tf.keras.Model(inputs, outputs, name="agriprice_small_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, min_delta=1e-5,
    )
    started = time.perf_counter()
    history = model.fit(
        X_train_scaled, y_train_scaled, validation_data=(X_val_scaled, y_val_scaled),
        epochs=200, batch_size=min(8, len(X_train_scaled)), shuffle=False,
        callbacks=[callback], verbose=0,
    )
    duration = time.perf_counter() - started
    actual_epochs = len(history.history.get("loss", []))
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    predictions = model.predict(X_val_scaled, verbose=0).reshape(-1) * scale_range + scale_min
    details = {
        "architecture": "Input(12,1) -> LSTM(16) -> Dropout(0.10) -> Dense(1)",
        "optimizer": "Adam", "learning_rate": 0.001, "loss": "mean_squared_error",
        "batch_size": min(8, len(X_train_scaled)), "maximum_epochs": 200,
        "actual_epochs": actual_epochs, "early_stopping_epoch": actual_epochs,
        "best_validation_loss_scaled_mse": float(min(history.history["val_loss"])),
        "best_epoch": best_epoch, "training_seconds": duration,
        "train_sequence_count": len(X_train), "validation_sequence_count": len(X_val),
        "train_first_target_month": str(min(train_dates)), "train_last_target_month": str(max(train_dates)),
        "train_target_months": ";".join(map(str, train_dates)),
        "validation_first_target_month": str(min(val_dates)), "validation_last_target_month": str(max(val_dates)),
        "validation_target_months": ";".join(map(str, val_dates)),
        "input_window_months": LSTM_WINDOW,
        "scaling": "MinMax scaling with min/max fitted only on observed prices dated on or before the training cutoff",
        "scaler_min": scale_min, "scaler_max": scale_max,
        "sequences_overlap": True,
        "overlap_note": "Adjacent windows share 11 of 12 input months. Window count is not an independent-observation count.",
    }
    return model, (scale_min, scale_range), details, dict(zip(val_dates, predictions))


def _lstm_predict(model, scale: tuple[float, float], price_map: dict[pd.Period, float], months: Iterable[pd.Period]):
    scale_min, scale_range = scale
    X, _, dates = _lstm_sequences_for_months(price_map, months)
    if not len(X):
        return {}
    X_scaled = ((X - scale_min) / scale_range).reshape((-1, LSTM_WINDOW, 1))
    y_scaled = model.predict(X_scaled, verbose=0).reshape(-1)
    return dict(zip(dates, (y_scaled * scale_range + scale_min).astype(float)))


def _metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float | int | None]:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if not len(y):
        return {"rmse": None, "mae": None, "mape": None, "r2": None, "bias": None, "n": 0, "mape_n": 0}
    nonzero = y != 0
    return {
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "mae": float(mean_absolute_error(y, p)),
        "mape": float(np.mean(np.abs((y[nonzero] - p[nonzero]) / y[nonzero])) * 100.0) if nonzero.any() else None,
        "r2": float(r2_score(y, p)) if len(y) >= 2 else None,
        "bias": float(np.mean(p - y)), "n": int(len(y)), "mape_n": int(nonzero.sum()),
    }


def _prediction_row(candidate: dict[str, Any], partition: str, model: str, month: pd.Period, actual: float, predicted: float) -> dict[str, Any]:
    meta = candidate["mapping"] if candidate["source_key"] == "wfp" else {}
    return {
        "series_id": candidate["series_id"], "source": candidate["source"],
        "research_commodity": candidate["research_commodity"],
        "commodity": _text(meta.get("commodity")) if meta else candidate.get("item_label", ""),
        "market": _text(meta.get("market")) if meta else "",
        "geography": _text(meta.get("state")) if meta else candidate.get("geography_name", ""),
        "unit": _text(meta.get("unit")) if meta else candidate.get("unit", ""),
        "price_type": _text(meta.get("price_type")) if meta else candidate.get("price_type", ""),
        "partition": partition, "model": model, "target_month": str(month),
        "actual_price": float(actual), "predicted_price": float(predicted),
        "error_predicted_minus_actual": float(predicted - actual),
    }


def _fit_one_series(candidate: dict[str, Any], eligibility: dict[str, Any], artifact_root: Path):
    price_map_all = candidate["price_map"]
    features_all = build_monthly_features(price_map_all)
    full_run = _longest_observed_run(price_map_all)
    full_run_map = {month: price_map_all[month] for month in full_run}
    full_sequences, _, full_sequence_months = build_lstm_sequences(price_map_all)

    # The only LSTM-capable series is evaluated on its longest uninterrupted
    # block so that tree models, stacking, baseline, and LSTM share one test set.
    use_lstm_window = (
        len(full_run) >= 96
        and len(full_sequences) >= 80
        and len(full_run_map) - LSTM_WINDOW >= 80
    )
    if use_lstm_window:
        modeling_map = full_run_map
        feature_frame = build_monthly_features(modeling_map)
        calendar = pd.period_range(min(modeling_map), max(modeling_map), freq="M")
        first_cut, second_cut = _date_split_cutoffs(list(calendar))
        train_dates, validation_dates, test_dates = _partition_dates(feature_frame.index, first_cut, second_cut)
        train_frame = feature_frame.loc[train_dates]
        validation_frame = feature_frame.loc[validation_dates]
        test_frame = feature_frame.loc[test_dates]
        sequence_X, sequence_y, sequence_months = build_lstm_sequences(modeling_map)
        seq_train_dates, seq_val_dates, seq_test_dates = _partition_dates(sequence_months, first_cut, second_cut)
        seq_train_X, seq_train_y, _ = _lstm_sequences_for_months(modeling_map, seq_train_dates)
        seq_val_X, _, _ = _lstm_sequences_for_months(modeling_map, seq_val_dates)
        seq_test_X, _, _ = _lstm_sequences_for_months(modeling_map, seq_test_dates)
        lstm_eligible = (
            len(seq_train_X) >= MIN_LSTM_TRAIN_SEQUENCES
            and len(seq_val_X) >= MIN_LSTM_VALIDATION_SEQUENCES
            and len(seq_test_X) >= MIN_LSTM_TEST_SEQUENCES
        )
        if not lstm_eligible:
            use_lstm_window = False
    else:
        lstm_eligible = False

    if not use_lstm_window:
        modeling_map = price_map_all
        feature_frame = features_all
        train_frame, validation_frame, test_frame = chronological_split(feature_frame)
        train_dates = list(train_frame.index)
        validation_dates = list(validation_frame.index)
        test_dates = list(test_frame.index)
        seq_train_dates = seq_val_dates = seq_test_dates = []
        lstm_eligible = False

    if len(train_frame) < MIN_TREE_TRAIN_ROWS or len(validation_frame) < MIN_VALIDATION_ROWS or len(test_frame) < MIN_TEST_ROWS:
        raise ValueError("eligibility changed after modeling-window construction; refusing to train a small sample")

    # If LSTM is available, use precisely the same validation/test target dates
    # for all model comparisons. The temporal boundaries above are shared.
    if lstm_eligible:
        seq_val_set, seq_test_set = set(seq_val_dates), set(seq_test_dates)
        common_val = sorted(set(validation_frame.index) & seq_val_set)
        common_test = sorted(set(test_frame.index) & seq_test_set)
        if len(common_val) >= MIN_LSTM_VALIDATION_SEQUENCES and len(common_test) >= MIN_LSTM_TEST_SEQUENCES:
            validation_eval_dates, test_eval_dates = common_val, common_test
        else:
            lstm_eligible = False
            validation_eval_dates, test_eval_dates = list(validation_frame.index), list(test_frame.index)
    else:
        validation_eval_dates, test_eval_dates = list(validation_frame.index), list(test_frame.index)

    X_train = train_frame.loc[:, FEATURE_COLUMNS]
    y_train = train_frame["target"]
    X_validation = validation_frame.loc[validation_eval_dates, FEATURE_COLUMNS]
    y_validation = validation_frame.loc[validation_eval_dates, "target"]
    X_test = test_frame.loc[test_eval_dates, FEATURE_COLUMNS]
    y_test = test_frame.loc[test_eval_dates, "target"]

    scores: dict[str, dict[str, Any]] = {}
    val_predictions: dict[str, dict[pd.Period, float]] = {}
    test_predictions: dict[str, dict[pd.Period, float]] = {}
    fitted_eval: dict[str, Any] = {}
    model_errors: dict[str, str] = {}
    duration_by_model: dict[str, float] = {}
    oof_count: int | None = None

    baseline_val = validation_frame.loc[validation_eval_dates, "lag_1"].astype(float).to_dict()
    baseline_test = test_frame.loc[test_eval_dates, "lag_1"].astype(float).to_dict()
    val_predictions["naive_last_observation"] = baseline_val
    test_predictions["naive_last_observation"] = baseline_test

    estimators = _estimators()
    for name, estimator in estimators.items():
        started = time.perf_counter()
        try:
            estimator.fit(X_train, y_train)
            val_predictions[name] = dict(zip(validation_eval_dates, estimator.predict(X_validation).astype(float)))
            test_predictions[name] = dict(zip(test_eval_dates, estimator.predict(X_test).astype(float)))
            fitted_eval[name] = estimator
            duration_by_model[name] = time.perf_counter() - started
        except Exception as exc:
            model_errors[name] = f"{type(exc).__name__}: {exc}"

    if "xgboost" not in estimators:
        model_errors["xgboost"] = "XGBoost could not be imported in the project environment."

    started = time.perf_counter()
    try:
        stacking = _fit_stacking(X_train, y_train)
        val_predictions["stacking"] = dict(zip(validation_eval_dates, _predict_stacking(stacking, X_validation)))
        test_predictions["stacking"] = dict(zip(test_eval_dates, _predict_stacking(stacking, X_test)))
        fitted_eval["stacking"] = stacking
        oof_count = int(stacking["oof_count"])
        duration_by_model["stacking"] = time.perf_counter() - started
    except Exception as exc:
        model_errors["stacking"] = f"{type(exc).__name__}: {exc}"

    lstm_details: dict[str, Any] | None = None
    eval_lstm = None
    eval_lstm_scale = None
    if lstm_eligible:
        try:
            eval_lstm, eval_lstm_scale, lstm_details, val_lstm = _fit_lstm(
                modeling_map, train_dates, validation_dates,
            )
            lstm_details["test_sequence_count"] = len(test_eval_dates)
            lstm_details["test_first_target_month"] = str(min(test_eval_dates))
            lstm_details["test_last_target_month"] = str(max(test_eval_dates))
            lstm_details["test_target_months"] = ";".join(map(str, test_eval_dates))
            lstm_details["validation_target_months"] = ";".join(map(str, validation_eval_dates))
            _, _, _, test_lstm = None, None, None, None
            # _fit_lstm returns validation predictions; evaluate test with the
            # same train-fitted scale and model, never fitting on test data.
            test_lstm = _lstm_predict(eval_lstm, eval_lstm_scale, modeling_map, test_eval_dates)
            val_predictions["lstm"] = {month: val_lstm[month] for month in validation_eval_dates if month in val_lstm}
            test_predictions["lstm"] = {month: test_lstm[month] for month in test_eval_dates if month in test_lstm}
            if set(test_predictions["lstm"]) != set(test_eval_dates) or set(val_predictions["lstm"]) != set(validation_eval_dates):
                raise ValueError("LSTM could not predict the shared validation/test dates")
            duration_by_model["lstm"] = float(lstm_details["training_seconds"])
        except Exception as exc:
            model_errors["lstm"] = f"{type(exc).__name__}: {exc}"
            lstm_eligible = False

    available_for_validation = [name for name in MODEL_NAMES if name in val_predictions and len(val_predictions[name]) == len(validation_eval_dates)]
    validation_metrics: dict[str, dict[str, Any]] = {}
    for name in available_for_validation:
        validation_metrics[name] = _metrics(
            y_validation.reindex(validation_eval_dates),
            [val_predictions[name][month] for month in validation_eval_dates],
        )
    if not validation_metrics:
        raise RuntimeError("No model produced predictions on the shared validation dates.")
    champion = min(validation_metrics, key=lambda name: validation_metrics[name]["rmse"])
    champion_validation_rmse = validation_metrics[champion]["rmse"]

    # Test predictions are generated for every fitted model exactly once and
    # are never used to choose the champion.
    metrics_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    test_models = [name for name in MODEL_NAMES if name in test_predictions and set(test_predictions[name]) == set(test_eval_dates)]
    for name in MODEL_NAMES:
        status = "trained_and_evaluated" if name in test_models else "unavailable"
        if name == "lstm" and status == "unavailable" and not model_errors.get(name):
            model_errors[name] = "Insufficient complete 12-month windows for a reliable chronological train/validation/test experiment."
        if name == "xgboost" and status == "unavailable" and not model_errors.get(name):
            model_errors[name] = "XGBoost model was unavailable."
        if name == "naive_last_observation" and status == "unavailable":
            continue
        vals = validation_metrics.get(name, {})
        mets = _metrics(y_test.reindex(test_eval_dates), [test_predictions[name][month] for month in test_eval_dates]) if name in test_models else {}
        metrics_records.append({
            "series_id": candidate["series_id"], "source": "WFP",
            "research_commodity": candidate["research_commodity"],
            "commodity": _text(candidate["mapping"].get("commodity")),
            "market": _text(candidate["mapping"].get("market")),
            "state": _text(candidate["mapping"].get("state")),
            "unit": _text(candidate["mapping"].get("unit")),
            "price_type": _text(candidate["mapping"].get("price_type")),
            "model": name, "status": status,
            "validation_rmse": vals.get("rmse"), "validation_mae": vals.get("mae"),
            "validation_mape": vals.get("mape"), "validation_r2": vals.get("r2"),
            "validation_bias": vals.get("bias"), "validation_n": vals.get("n"),
            "test_rmse": mets.get("rmse"), "test_mae": mets.get("mae"),
            "test_mape": mets.get("mape"), "test_r2": mets.get("r2"),
            "test_bias": mets.get("bias"), "test_n": mets.get("n"),
            "test_mape_n": mets.get("mape_n"),
            "test_first_month": str(min(test_eval_dates)) if test_eval_dates else "",
            "test_last_month": str(max(test_eval_dates)) if test_eval_dates else "",
            "test_dates": ";".join(map(str, test_eval_dates)),
            "train_dates": ";".join(map(str, train_dates)),
            "validation_dates": ";".join(map(str, validation_eval_dates)),
            "champion_selected_on_validation": name == champion,
            "champion_validation_rmse": champion_validation_rmse,
            "fit_seconds": duration_by_model.get(name), "stacking_oof_count": oof_count if name == "stacking" else None,
            "unavailable_reason": model_errors.get(name, ""),
            "evaluation_note": "Untouched chronological test; predictions use only source data through the prior calendar month.",
        })
        if name in test_models:
            for month in test_eval_dates:
                prediction_records.append(_prediction_row(
                    candidate, "test", name, month,
                    float(y_test.loc[month]), float(test_predictions[name][month]),
                ))
        elif name != "naive_last_observation":
            prediction_records.append(_prediction_row(
                candidate, "test", name, min(test_eval_dates),
                float(y_test.loc[min(test_eval_dates)]), float("nan"),
            ))

    # Refit final forecasting artifacts on admissible observed data through the
    # forecast origin; these refits are distinct from untouched test models.
    forecast_origin = max(modeling_map)
    forecast_fit_rows = build_monthly_features({m: v for m, v in modeling_map.items() if m <= forecast_origin})
    final_models: dict[str, Any] = {}
    artifact_dir = artifact_root / _slug(candidate["series_id"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}
    final_estimators = _estimators()
    for name, estimator in final_estimators.items():
        if name not in test_models:
            continue
        estimator.fit(forecast_fit_rows.loc[:, FEATURE_COLUMNS], forecast_fit_rows["target"])
        final_models[name] = estimator
        path = artifact_dir / f"{name}.joblib"
        import joblib

        joblib.dump({
            "model": estimator, "model_name": name, "series_id": candidate["series_id"],
            "feature_columns": list(FEATURE_COLUMNS), "target": "WFP monthly actual price in NGN per source unit",
            "training_window_start": str(min(modeling_map)), "training_window_end": str(forecast_origin),
            "training_fit_scope": "full eligible observed history through forecast origin; separate from test evaluation fit",
        }, path)
        artifact_paths[name] = str(path.relative_to(ROOT).as_posix())

    if "stacking" in test_models:
        full_stack = _fit_stacking(forecast_fit_rows.loc[:, FEATURE_COLUMNS], forecast_fit_rows["target"])
        final_models["stacking"] = full_stack
        path = artifact_dir / "stacking.joblib"
        import joblib

        joblib.dump({
            **full_stack, "series_id": candidate["series_id"],
            "feature_columns": list(FEATURE_COLUMNS), "target": "WFP monthly actual price in NGN per source unit",
            "training_window_start": str(min(modeling_map)), "training_window_end": str(forecast_origin),
            "training_fit_scope": "full eligible observed history; meta-model uses expanding-window OOF predictions",
        }, path)
        artifact_paths["stacking"] = str(path.relative_to(ROOT).as_posix())

    final_lstm = None
    final_lstm_scale = None
    final_lstm_metadata: dict[str, Any] | None = None
    if "lstm" in test_models:
        final_seq_X, final_seq_y, final_seq_months = build_lstm_sequences(modeling_map)
        final_cut = max(1, int(len(final_seq_months) * 0.80))
        final_train_months = final_seq_months[:final_cut]
        final_val_months = final_seq_months[final_cut:]
        final_lstm, final_lstm_scale, final_lstm_metadata, _ = _fit_lstm(
            modeling_map, final_train_months, final_val_months,
        )
        lstm_path = artifact_dir / "lstm.keras"
        final_lstm.save(lstm_path)
        artifact_paths["lstm"] = str(lstm_path.relative_to(ROOT).as_posix())
        preproc_path = artifact_dir / "lstm_preprocessing.json"
        _write_json(preproc_path, {
            "window_months": LSTM_WINDOW, "scale_min": final_lstm_scale[0],
            "scale_range": final_lstm_scale[1], "scaling": "training-only MinMax",
            "feature": "monthly price only", "metadata": final_lstm_metadata,
        })
        artifact_paths["lstm_preprocessing"] = str(preproc_path.relative_to(ROOT).as_posix())

    history_metadata = candidate["mapping"]
    output_record = {
        "series_id": candidate["series_id"], "source_key": "wfp",
        "research_commodity": candidate["research_commodity"],
        "commodity": _text(history_metadata.get("commodity")),
        "market": _text(history_metadata.get("market")),
        "market_id": _market_id(history_metadata.get("market_id")),
        "state": _text(history_metadata.get("state")),
        "unit": _text(history_metadata.get("unit")),
        "price_type": _text(history_metadata.get("price_type")),
        "currency": _text(history_metadata.get("currency")),
        "price_flag": "actual", "actual_monthly_observations": len(price_map_all),
        "latest_actual_month": str(max(price_map_all)),
        "latest_actual_date": _text(candidate["rows"].iloc[-1].get("last_observation_date")) if not candidate["rows"].empty else "",
        "modeling_window_start": str(min(modeling_map)), "modeling_window_end": str(max(modeling_map)),
        "forecast_origin_month": str(max(modeling_map)),
        "modeling_window_actual_months": len(modeling_map),
        "modeling_window_scope": "longest_contiguous_actual_run" if use_lstm_window else "calendar_lag_valid_rows_across_actual_segments",
        "latest_observation_age_months": _month_age(date.today(), max(price_map_all)),
        "forecast_classification": "historical_only",
        "historical_forecast_note": "Forecast origin is historical and does not represent current Nigerian market prices.",
        "champion": champion, "champion_selection_basis": "lowest validation RMSE on shared validation dates; test metrics were not used",
        "champion_validation_rmse": champion_validation_rmse,
        "validation_dates": ";".join(map(str, validation_eval_dates)),
        "test_dates": ";".join(map(str, test_eval_dates)),
        "available_models": sorted(test_models), "artifact_paths": artifact_paths,
        "lstm_experiment": lstm_details if lstm_details else {
            "status": "unavailable_insufficient_sequences",
            "valid_complete_12_month_windows_in_full_series": len(full_sequences),
            "reason": model_errors.get("lstm", "Fewer than the configured train, validation, or test sequence minimums."),
        },
        "stacking_oof_count": oof_count,
    }
    return metrics_records, prediction_records, output_record, final_models, (modeling_map, forecast_fit_rows, final_lstm, final_lstm_scale, final_lstm_metadata)


def _month_age(reference: date, month: pd.Period) -> int:
    return max(0, (reference.year - month.year) * 12 + reference.month - month.month)


def _recursive_tree_forecast(model: Any, model_name: str, price_map: dict[pd.Period, float], origin: pd.Period, horizon: int):
    history = dict(price_map)
    start = min(price_map)
    forecasts = []
    for step in range(1, horizon + 1):
        target_month = origin + step
        if model_name == "naive_last_observation":
            required = [target_month - 1]
            if any(month not in history for month in required):
                raise ValueError("last-calendar-month price is unavailable at recursive forecast origin")
            predicted = history[target_month - 1]
        else:
            feature_values = [history.get(target_month - lag) for lag in (1, 2, 3)]
            if any(value is None for value in feature_values):
                raise ValueError("three consecutive calendar-month price lags are required for recursive forecasting")
            prior = np.asarray(feature_values, dtype=float)
            month = target_month.month
            features = pd.DataFrame([{
                "lag_1": prior[0], "lag_2": prior[1], "lag_3": prior[2],
                "rolling_mean_3": float(prior.mean()),
                "rolling_std_3": float(prior.std(ddof=0)),
                "month_sin": float(np.sin(2 * np.pi * month / 12)),
                "month_cos": float(np.cos(2 * np.pi * month / 12)),
                "trend_month": float(target_month.ordinal - start.ordinal),
            }], columns=FEATURE_COLUMNS)
            if model_name == "stacking":
                predicted = float(_predict_stacking(model, features)[0])
            else:
                predicted = float(model.predict(features)[0])
        history[target_month] = predicted
        forecasts.append({
            "step": step, "forecast_month": str(target_month),
            "forecast_price": float(predicted),
            "quality_flag": "negative_model_output" if predicted < 0 else "model_estimate",
        })
    return forecasts


def _recursive_lstm_forecast(model: Any, scale: tuple[float, float], price_map: dict[pd.Period, float], origin: pd.Period, horizon: int):
    scale_min, scale_range = scale
    history = dict(price_map)
    forecasts = []
    for step in range(1, horizon + 1):
        target_month = origin + step
        inputs = [history.get(target_month - lag) for lag in range(LSTM_WINDOW, 0, -1)]
        if any(value is None for value in inputs):
            raise ValueError("12 consecutive actual or previously predicted calendar months are required")
        X = ((np.asarray(inputs, dtype=float) - scale_min) / scale_range).reshape((1, LSTM_WINDOW, 1))
        predicted_scaled = float(model.predict(X, verbose=0).reshape(-1)[0])
        predicted = predicted_scaled * scale_range + scale_min
        history[target_month] = predicted
        forecasts.append({
            "step": step, "forecast_month": str(target_month),
            "forecast_price": float(predicted),
            "quality_flag": "negative_model_output" if predicted < 0 else "model_estimate",
        })
    return forecasts


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Period,)):
        return str(value)
    if isinstance(value, (Path,)):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")


def _run_metadata(candidate: dict[str, Any], trained: dict[str, Any], final_bundle: tuple, generation_date: str):
    modeling_map, _, final_lstm, final_lstm_scale, _ = final_bundle
    origin = max(modeling_map)
    champion = trained["champion"]
    forecasts: list[dict[str, Any]] = []
    forecast_error = ""
    try:
        for horizon in (1, 3, 6):
            if champion == "lstm":
                rows = _recursive_lstm_forecast(final_lstm, final_lstm_scale, modeling_map, origin, horizon)
            else:
                predictor = None if champion == "naive_last_observation" else trained["final_models"][champion]
                rows = _recursive_tree_forecast(predictor, champion, modeling_map, origin, horizon)
            for row in rows:
                forecasts.append({
                    "series_id": candidate["series_id"], "source": "WFP",
                    "research_commodity": candidate["research_commodity"],
                    "commodity": trained["record"]["commodity"], "market": trained["record"]["market"],
                    "state": trained["record"]["state"], "unit": trained["record"]["unit"],
                    "price_type": trained["record"]["price_type"], "currency": trained["record"]["currency"],
                    "model": champion, "requested_horizon": horizon,
                    "forecast_step": row["step"], "forecast_origin_month": str(origin),
                    "forecast_month": row["forecast_month"], "forecast_price": row["forecast_price"],
                    "forecast_generation_date": generation_date,
                    "latest_actual_month_in_source": trained["record"]["latest_actual_month"],
                    "latest_actual_date_in_source": trained["record"]["latest_actual_date"],
                    "forecast_classification": "historical_only",
                    "forecast_method": "recursive; each predicted price is fed back as a future lag; no actual future prices are accessed",
                    "quality_flag": row["quality_flag"],
                })
    except Exception as exc:
        forecast_error = f"{type(exc).__name__}: {exc}"
    trained["record"]["forecast_available"] = bool(forecasts)
    trained["record"]["forecast_unavailable_reason"] = forecast_error
    trained["forecasts"] = forecasts
    return trained


def _eligibility_summary_text(rows: list[dict[str, Any]]) -> str:
    eligible_wfp = [r for r in rows if r["source"] == "WFP" and r["tree_model_eligible"]]
    nbs_rows = [r for r in rows if r["source"] == "NBS"]
    direct_counts: dict[str, int] = {}
    for row in rows:
        if row["source"] == "WFP" and row["mapping_status"] in DIRECT_MAPPING_STATUSES:
            direct_counts[row["research_commodity"]] = direct_counts.get(row["research_commodity"], 0) + 1
    lines = [
        "AgriPrice-NG training eligibility overview",
        "============================================",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "Only the existing normalized WFP/NBS data were read. No source, normalized, backup, or legacy data were modified.",
        "WFP training uses only priceflag=actual and exact commodity/market/state/unit/price-type/currency groups.",
        "NBS national and zone series are independent, exact item-label series; pre/post January 2025 CPI regimes are not bridged.",
        "",
        "Eligibility method",
        "- Historical WFP tree/stacking eligibility: at least 96 actual months, at least 80 complete rows with actual calendar lags 1-3, at least 60 chronological training rows, and at least 10 validation and 10 test rows.",
        "- Features are calculated on a full monthly calendar grid, with no target interpolation or forward-fill. A target row is usable only when the target and all required prior calendar months are actual observations.",
        "- Eligible old series are historical experiments, not current-market forecasts. Freshness is reported separately and does not make a model eligible for current conditions.",
        "- LSTM requires a complete uninterrupted 12-month input window and minimum counts of 60 train, 10 validation, and 10 test sequences. Overlapping windows share input observations and do not increase independent historical information.",
        "- Test predictions are generated after validation-based model selection; no test metric is used for selection.",
        "",
        f"WFP direct-mapped series meeting historical tree experiment threshold: {len(eligible_wfp)}",
    ]
    for row in eligible_wfp:
        lines.append(
            f"- {row['research_commodity']} | {row['commodity']} | {row['market']} | {row['unit']} | {row['price_type']} | actual={row['actual_monthly_observations']} | usable lag rows={row['complete_lag_1_to_3_target_rows']} | coverage={row['calendar_coverage_percent']:.2f}% | {row['first_actual_month']}..{row['latest_actual_month']}"
        )
    lines += [
        "",
        "NBS decision",
        f"- Relevant NBS item/geography/regime series documented: {len(nbs_rows)}.",
        "- No NBS models were trained: the post-rebase stable series contain at most 17 monthly points (2025-01 through 2026-05); pre-rebase 2024 observations are kept separate. This cannot support five-model train/validation/test comparisons or a defensible LSTM experiment.",
        "- NBS workbook units are not standardized for most labels. Exact item labels and geography are retained; no unit conversion or WFP/NBS merge is performed.",
        "",
        "Research scope coverage",
    ]
    for commodity in RESEARCH_COMMODITIES:
        if commodity == "Cassava":
            lines.append("- Cassava: no defensible raw-cassava WFP/NBS series was supplied; gari remains a processed cassava product and is excluded.")
            continue
        count = direct_counts.get(commodity, 0)
        if not count:
            lines.append(f"- {commodity}: no direct-mapped WFP market group in the selection audit; no cross-source substitution was attempted.")
    return "\n".join(lines) + "\n"


def train_wfp_nbs_models() -> dict[str, Any]:
    """Run eligibility, chronological evaluation, final fits, and historical forecasts."""
    for path in (WFP_PATH, NBS_PATH, WFP_SELECTION_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required prepared input is missing: {path.name}")
    wfp = pd.read_csv(WFP_PATH, low_memory=False)
    nbs = pd.read_csv(NBS_PATH, low_memory=False)
    required_wfp = {"observation_month", "commodity", "market", "market_id", "admin1", "unit", "pricetype", "currency", "priceflag", "price", "last_observation_date"}
    required_nbs = {"observation_month", "methodology_regime", "geography_level", "geography_name", "item_label", "research_commodity_mapping", "mapping_status", "unit", "price_type", "currency", "price"}
    if missing := sorted(required_wfp.difference(wfp.columns)):
        raise ValueError(f"Normalized WFP schema is missing: {missing}")
    if missing := sorted(required_nbs.difference(nbs.columns)):
        raise ValueError(f"Normalized NBS schema is missing: {missing}")
    wfp["price"] = pd.to_numeric(wfp["price"], errors="coerce")
    nbs["price"] = pd.to_numeric(nbs["price"], errors="coerce")
    wfp_candidates = _wfp_candidates(wfp)
    nbs_candidates = _nbs_candidates(nbs)
    candidates = wfp_candidates + nbs_candidates
    eligibility_rows = [_eligibility_row(candidate) for candidate in candidates]
    eligibility_by_id = {row["series_id"]: row for row in eligibility_rows}

    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    artifact_root = MODEL_DIR / run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    registry_series: dict[str, Any] = {}
    all_forecasts: list[dict[str, Any]] = []
    lstm_reports: list[dict[str, Any]] = []
    trained_count = 0
    model_status_counts: dict[str, int] = {}

    for candidate in wfp_candidates:
        eligibility = eligibility_by_id[candidate["series_id"]]
        if not eligibility["tree_model_eligible"]:
            continue
        metrics, predictions, record, final_models, final_bundle = _fit_one_series(candidate, eligibility, artifact_root)
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        record["forecast_generation_date"] = date.today().isoformat()
        trained = {
            "champion": record["champion"], "record": record,
            "final_models": final_models,
        }
        trained = _run_metadata(candidate, trained, final_bundle, date.today().isoformat())
        all_forecasts.extend(trained["forecasts"])
        registry_series[candidate["series_id"]] = record
        trained_count += sum(name != "naive_last_observation" for name in record["available_models"])
        for name in record["available_models"]:
            if name != "naive_last_observation":
                model_status_counts[name] = model_status_counts.get(name, 0) + 1
        if record.get("lstm_experiment", {}).get("architecture"):
            lstm_reports.append({
                "series_id": candidate["series_id"], "commodity": record["commodity"],
                "market": record["market"], "details": record["lstm_experiment"],
                "test_dates": record["test_dates"],
                "source_observation_count": eligibility["actual_monthly_observations"],
                "source_first_month": eligibility["first_actual_month"],
                "source_latest_month": eligibility["latest_actual_month"],
                "source_first_date": eligibility["first_actual_observation_date"],
                "source_latest_date": eligibility["latest_actual_observation_date"],
                "source_missing_month_count": eligibility["missing_calendar_months"],
                "source_missing_months": eligibility["missing_calendar_month_list"],
                "source_calendar_coverage_percent": eligibility["calendar_coverage_percent"],
                "source_longest_missing_run": eligibility["longest_consecutive_missing_months"],
                "modeling_window_start": record["modeling_window_start"],
                "modeling_window_end": record["modeling_window_end"],
                "modeling_window_observations": record["modeling_window_actual_months"],
            })

    generation = datetime.now(timezone.utc).isoformat(timespec="seconds")
    registry = {
        "project": "AgriPrice-NG", "experiment": "WFP/NBS historical forecasting experiment",
        "generated_at": generation, "run_id": run_id,
        "source_scope": ["WFP", "NBS"],
        "source_files": {
            "wfp_normalized": WFP_PATH.relative_to(ROOT).as_posix(),
            "nbs_normalized": NBS_PATH.relative_to(ROOT).as_posix(),
        },
        "series": registry_series,
        "summary": {
            "historical_wfp_series_trained": len(registry_series),
            "model_series_fits": trained_count,
            "model_series_fits_by_model": model_status_counts,
            "nbs_series_trained": 0,
            "current_market_forecasts": 0,
            "forecast_records": len(all_forecasts),
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(REPORT_DIR / "series_eligibility.csv", eligibility_rows)
    _write_csv(REPORT_DIR / "model_comparison.csv", all_metrics)
    _write_csv(REPORT_DIR / "test_predictions.csv", all_predictions)
    _write_csv(OUTPUT_DIR / "forecasts.csv", all_forecasts)
    _write_json(REPORT_DIR / "model_registry.json", registry)
    _write_json(REPORT_DIR / "training_configuration.json", {
        "generated_at": generation, "run_id": run_id,
        "inputs": [WFP_PATH.relative_to(ROOT).as_posix(), NBS_PATH.relative_to(ROOT).as_posix()],
        "target": "WFP monthly actual price (NGN per source unit); NBS targets separate and not trained",
        "wfp_price_flag": "actual only", "features": list(FEATURE_COLUMNS),
        "lag_months": [1, 2, 3], "rolling_window_months": 3,
        "split": "chronological 70/15/15; model-specific usable target rows; shared validation/test dates within each series",
        "eligibility_thresholds": {
            "minimum_actual_months": MIN_RAW_MONTHS,
            "minimum_complete_lag_rows": 80,
            "minimum_training_rows": MIN_TREE_TRAIN_ROWS,
            "minimum_validation_rows": MIN_VALIDATION_ROWS,
            "minimum_test_rows": MIN_TEST_ROWS,
            "lstm_window_months": LSTM_WINDOW,
            "lstm_min_train_validation_test_sequences": [MIN_LSTM_TRAIN_SEQUENCES, MIN_LSTM_VALIDATION_SEQUENCES, MIN_LSTM_TEST_SEQUENCES],
        },
        "models": {
            "random_forest": "300 trees; min_samples_leaf=2; seed=42",
            "xgboost": "250 trees; depth=2; learning_rate=0.03; subsample=0.9; colsample_bytree=0.9; seed=42",
            "gradient_boosting": "120 estimators; depth=2; learning_rate=0.03; Huber loss; seed=42",
            "stacking": "Random Forest + XGBoost + Gradient Boosting; Ridge meta-model from expanding-window training-only OOF predictions",
            "lstm": "Input(12,1) -> LSTM(16) -> Dropout(0.10) -> Dense(1); Adam(0.001); MSE; early stopping patience 15",
            "naive_last_observation": "actual previous calendar-month price; no substitute prior observation across gaps",
        },
        "notes": [
            "Only normalized WFP/NBS data were read; no raw data modified.",
            "No target interpolation, forward-fill, cross-source aggregation, or unit conversion was performed.",
            "All generated forecasts are historical-only and are not current November 2026 market prices.",
        ],
    })
    _write_json(REPORT_DIR / "lstm_experiment_summary.json", {
        "status": "trained" if lstm_reports else "unavailable_insufficient_sequences",
        "experiment_count": len(lstm_reports), "experiments": lstm_reports,
        "overlapping_window_note": "Overlapping sequences share input months; sequence count is not the count of independent observations.",
    })
    _write_text_report(registry, eligibility_rows, all_metrics, all_forecasts)
    _write_lstm_report(lstm_reports)
    _write_plots(all_metrics, all_predictions)
    return {
        "run_id": run_id, "eligible_wfp_series": len(registry_series),
        "trained_model_series_fits": trained_count,
        "model_status_counts": model_status_counts,
        "forecast_records": len(all_forecasts), "nbs_models_trained": 0,
        "registry": registry,
    }


def _write_text_report(registry, eligibility_rows, metrics, forecasts) -> None:
    eligible = [row for row in eligibility_rows if row["tree_model_eligible"]]
    lines = [
        "AgriPrice-NG Model Training and Evaluation Report",
        "===================================================",
        f"Generated: {registry['generated_at']}",
        f"Experiment run: {registry['run_id']}",
        "",
        "Decision and scope",
        "------------------",
        "The experiment reads only the existing normalized WFP and NBS datasets. WFP market histories and NBS national/zone summaries remain separate. No World Bank source, additional data, interpolation, forward-fill, price conversion, or synthetic observation was used.",
        "All model results and recursive outputs are historical experiments. No supplied series is current enough and continuous enough to support a November 2026 market-price claim.",
        "",
        "Eligibility and observations",
        "----------------------------",
        f"Relevant WFP/NBS series evaluated: {len(eligibility_rows)}; eligible historical WFP series trained: {len(eligible)}; NBS series trained: 0.",
    ]
    for row in eligible:
        model_record = registry["series"].get(row["series_id"], {})
        lines.append(
            f"- {row['research_commodity']} / {row['commodity']} / {row['market']} / {row['unit']} / {row['price_type']}: {row['actual_monthly_observations']} actual months ({row['first_actual_month']}..{row['latest_actual_month']}), coverage {row['calendar_coverage_percent']:.2f}%, missing {row['missing_calendar_months']}, longest missing run {row['longest_consecutive_missing_months']}; complete lag rows={row['complete_lag_1_to_3_target_rows']}; modeling window={model_record.get('modeling_window_start')}..{model_record.get('modeling_window_end')} ({model_record.get('modeling_window_scope')})."
        )
    lines += [
        "",
        "Feature and validation method",
        "------------------------------",
        "Each WFP target is the exact monthly-normalized actual price in its source unit and NGN. Price flags, commodity variants, market, state, unit, and retail/wholesale type remain separate.",
        "Monthly reindexing creates calendar positions only. The target is never imputed. Lag 1/2/3 and rolling 3-month mean/standard deviation require actual values in those exact prior calendar months; rows with missing required lags are excluded.",
        "Month sine/cosine and a calendar trend feature are added. Rolling features use only t-1 through t-3. Evaluation is chronological, not shuffled, using 70/15/15 splits over usable target rows. Where LSTM is eligible, the tree/stacking/naive metrics are restricted to the same validation and test target dates as LSTM.",
        "Model configurations are fixed and deliberately small. The champion is selected by validation RMSE. Test results are reported once and do not influence selection.",
        "The naive baseline uses only the immediately prior calendar month. It does not substitute the last nonmissing observation across a calendar gap.",
        "Stacking meta-training uses expanding-window out-of-fold base predictions generated only inside the training partition. No in-sample base prediction is used to train the meta-model.",
        "MAPE excludes zero actual values from its denominator and records the number of nonzero observations used. Bias is mean(prediction - actual).",
        "",
        "Results",
        "-------",
        f"Successful model-series fits (excluding naive baseline): {registry['summary']['model_series_fits']}; counts: {registry['summary']['model_series_fits_by_model']}.",
        "See reports/training/model_comparison.csv and reports/training/test_predictions.csv for per-model metrics and exact test dates.",
    ]
    for sid, series in registry["series"].items():
        rows = [row for row in metrics if row["series_id"] == sid]
        champion = series["champion"]
        champion_row = next((row for row in rows if row["model"] == champion), None)
        baseline_row = next((row for row in rows if row["model"] == "naive_last_observation"), None)
        if champion_row and baseline_row:
            beats = champion_row["test_rmse"] < baseline_row["test_rmse"]
            lines.append(
                f"- {series['research_commodity']} | {series['commodity']} | {series['market']}: validation champion={champion}; validation RMSE={series['champion_validation_rmse']:.4f}; untouched test RMSE={champion_row['test_rmse']:.4f}; naive test RMSE={baseline_row['test_rmse']:.4f}; champion lower test RMSE than naive={beats}; test n={champion_row['test_n']}; test dates={series['test_dates']}."
            )
    lines += [
        "",
        "Forecast interpretation and limitations",
        "-----------------------------------------",
        f"Recursive historical forecast rows produced: {len(forecasts)}.",
        "Forecast outputs are generated from an explicitly recorded historical origin and a validation-selected model refit on admissible observed data through that origin. Each predicted value is fed into later forecast features; actual prices after the origin are not read.",
        "Every forecast is labelled historical_only. These projections are not live/current prices and must not be used as a November 2026 outlook. Long gaps, structural change, inflation, and sparse market reporting limit external validity.",
        "If the latest source observation is outside the modeled continuous/lag-valid window, the output records the distinct latest actual date and forecast origin. No confidence intervals are provided.",
        "",
        "NBS limitations",
        "---------------",
        "NBS series are documented in the eligibility report but not trained. Stable post-rebase series have only 17 points, while pre-rebase observations are segregated by methodology regime. Item labels and national/zone geography remain exact; most units are not stated. Garri is not mapped to raw cassava.",
        "",
        "Artifact and output locations",
        "------------------------------",
        "Models: models/wfp_nbs/<run_id>/<series_id>/ (joblib artifacts; Keras LSTM only where eligible).",
        "Metrics and registry: reports/training/model_comparison.csv, series_eligibility.csv, model_registry.json, training_configuration.json.",
        "Forecasts and plots: outputs/training/forecasts.csv and outputs/training/plots/.",
    ]
    (ROOT / "reports" / "model_training_evaluation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_lstm_report(experiments: list[dict[str, Any]]) -> None:
    lines = [
        "AgriPrice-NG LSTM experiment record",
        "====================================",
        "A 12-month univariate sequence was used. The small network is Input(12,1) -> LSTM(16) -> Dropout(0.10) -> Dense(1), trained with Adam at 0.001, MSE, batch size up to 8, max 200 epochs, early-stopping patience 15, and restored best validation weights.",
        "Scaling min/max are fitted only on prices through the training cutoff and inverse transformation is applied before computing NGN price metrics.",
        "A complete sequence is admitted only where all 12 preceding calendar months and the target month have actual observations. No gap filling is used.",
        "Adjacent sequence windows overlap and share 11 of 12 input months; the number of windows is not the number of independent observations.",
        "",
    ]
    if not experiments:
        lines.append("No LSTM model was trained: no series met the configured chronological train/validation/test sequence minimums. This is an unavailable result, not a zero-error result.")
    for exp in experiments:
        lines.append(f"Series: {exp['commodity']} | {exp['market']} | {exp['series_id']}")
        lines.append("Source and window coverage:")
        lines.append(json.dumps({key: value for key, value in exp.items() if key != "details"}, indent=2, ensure_ascii=False, default=_json_default))
        lines.append(json.dumps(exp["details"], indent=2, ensure_ascii=False, default=_json_default))
        lines.append(f"Shared untouched test dates: {exp['test_dates']}")
        lines.append("")
    (REPORT_DIR / "lstm_training_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(metrics: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> None:
    from .plotting import write_actual_predicted_svg

    for sid in sorted({row["series_id"] for row in metrics}):
        rows = [row for row in predictions if row["series_id"] == sid and row["partition"] == "test"]
        baseline = {row["target_month"]: row["predicted_price"] for row in rows if row["model"] == "naive_last_observation"}
        champion_name = next((row["model"] for row in metrics if row["series_id"] == sid and row["champion_selected_on_validation"]), None)
        if not champion_name:
            continue
        champion_rows = {row["target_month"]: row["predicted_price"] for row in rows if row["model"] == champion_name and math.isfinite(row["predicted_price"])}
        actual_rows = {row["target_month"]: row["actual_price"] for row in rows if row["model"] == champion_name and math.isfinite(row["predicted_price"])}
        months = sorted(set(champion_rows) & set(actual_rows) & set(baseline))
        if not months:
            continue
        write_actual_predicted_svg(
            PLOT_DIR / f"{_slug(sid)}_test_comparison.svg",
            f"Actual vs {champion_name} and naive · {sid}", months,
            [actual_rows[m] for m in months], [champion_rows[m] for m in months],
            [baseline[m] for m in months],
        )
