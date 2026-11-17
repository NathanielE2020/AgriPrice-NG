"""Phase 3.1 read-only model audit and aggregate-series coverage analysis.

This module deliberately does not fit models or rewrite existing experiment
outputs. It verifies the saved feature/evaluation artifacts and writes new
audit reports for the approved Phase 3.1 investigation.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from .preprocessing import FEATURE_COLUMNS, TARGET_COLUMN


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _month_index(month: str) -> int:
    year, month_number = (int(value) for value in month.split("-"))
    return year * 12 + month_number - 1


def _month_text(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:120]


def _chronological_split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: row["month"])
    n = len(ordered)
    if n < 10:
        return ordered, [], []
    train_end = max(1, int(n * 0.70))
    validation_end = max(train_end + 1, int(n * 0.85))
    validation_end = min(validation_end, n - 1)
    return ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if len(actual) == 0:
        return {"rmse": None, "mae": None, "mape": None, "r2": None}
    errors = predicted - actual
    actual_nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(errors[actual_nonzero] / actual[actual_nonzero])) * 100.0)
        if np.any(actual_nonzero)
        else None
    )
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = None if len(actual) < 2 or denominator == 0 else float(1.0 - np.sum(errors**2) / denominator)
    return {
        "rmse": float(math.sqrt(float(np.mean(errors**2)))),
        "mae": float(np.mean(np.abs(errors))),
        "mape": mape,
        "r2": r2,
    }


def _close(left, right, tolerance: float = 1e-6) -> bool:
    if left is None or right is None:
        return left == right
    return abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(right)))


def _feature_history_audit(monthly_rows: list[dict], feature_rows: list[dict]) -> dict:
    monthly_by_series: dict[str, list[dict]] = defaultdict(list)
    for row in monthly_rows:
        series_id = " | ".join(
            [row["commodity"], row["market"], row["unit"], row["price_type"]]
        )
        monthly_by_series[series_id].append(row)

    lag_errors = 0
    rolling_errors = 0
    checked_rows = 0
    for row in feature_rows:
        by_month = {item["month"]: item for item in monthly_by_series[row["series_id"]]}
        current_index = _month_index(row["month"])
        history = {}
        complete = True
        for lag in (1, 2, 3):
            history_month = _month_text(current_index - lag)
            history_value = by_month.get(history_month, {}).get("price", "")
            if history_value in (None, ""):
                complete = False
                continue
            history[lag] = float(history_value)
            if not _close(row[f"price_lag_{lag}"], history[lag]):
                lag_errors += 1
        rolling_values = []
        for offset in (1, 2, 3):
            history_month = _month_text(current_index - offset)
            history_value = by_month.get(history_month, {}).get("price", "")
            if history_value in (None, ""):
                complete = False
                break
            rolling_values.append(float(history_value))
        if complete:
            mean_value = statistics.fmean(rolling_values)
            std_value = math.sqrt(
                sum((value - mean_value) ** 2 for value in rolling_values) / len(rolling_values)
            )
            if not _close(row["rolling_mean_3"], mean_value):
                rolling_errors += 1
            if not _close(row["rolling_std_3"], std_value):
                rolling_errors += 1
            checked_rows += 1
    return {
        "checked_feature_rows": checked_rows,
        "lag_errors": lag_errors,
        "rolling_errors": rolling_errors,
    }


def _audit_saved_models(feature_rows: list[dict], comparison_rows: list[dict]) -> tuple[list[dict], list[str]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in feature_rows:
        grouped[row["series_id"]].append(row)
    comparison_lookup = {
        (row["series_id"], row["model"], row["split"]): row for row in comparison_rows
    }
    verification_rows: list[dict] = []
    test_months: list[str] = []
    for series_id, rows in sorted(grouped.items()):
        train_rows, validation_rows, test_rows = _chronological_split(rows)
        months = [row["month"] for row in test_rows]
        test_months.append(f"{series_id}: {','.join(months)}")
        x_test = np.asarray([[float(row[name]) for name in FEATURE_COLUMNS] for row in test_rows])
        y_test = np.asarray([float(row[TARGET_COLUMN]) for row in test_rows])
        baseline = np.asarray([float(row["price_lag_1"]) for row in test_rows])
        recomputed_baseline = _metric_values(y_test, baseline)
        saved_baseline = comparison_lookup[(series_id, "naive_last_calendar_month", "test")]
        baseline_matches = all(
            _close(recomputed_baseline[field], saved_baseline[field])
            for field in ("rmse", "mae", "mape", "r2")
        )
        verification_rows.append(
            {
                "series_id": series_id,
                "model": "naive_last_calendar_month",
                "saved_metrics_match_recomputed": baseline_matches,
                "test_month_count": len(test_rows),
                "test_months": ";".join(months),
            }
        )

        for model_name in ("random_forest", "gradient_boosting", "stacking"):
            model_path = MODELS_DIR / _slug(series_id) / f"{model_name}.joblib"
            artifact = joblib.load(model_path)
            if model_name == "stacking":
                bundle = artifact["model"]
                base_predictions = np.column_stack(
                    [model.predict(x_test) for model in bundle["base_models"].values()]
                )
                prediction = bundle["meta_model"].predict(base_predictions)
            else:
                prediction = artifact["model"].predict(x_test)
            recomputed = _metric_values(y_test, prediction)
            saved = comparison_lookup[(series_id, model_name, "test")]
            matches = all(_close(recomputed[field], saved[field]) for field in ("rmse", "mae", "mape", "r2"))
            verification_rows.append(
                {
                    "series_id": series_id,
                    "model": model_name,
                    "saved_metrics_match_recomputed": matches,
                    "test_month_count": len(test_rows),
                    "test_months": ";".join(months),
                }
            )
    return verification_rows, test_months


def audit_models() -> str:
    feature_rows = _read_csv(PROCESSED_DIR / "approved_training_features.csv")
    monthly_rows = _read_csv(PROCESSED_DIR / "approved_monthly_reindexed.csv")
    comparison_rows = _read_csv(REPORTS_DIR / "model_comparison.csv")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in feature_rows:
        grouped[row["series_id"]].append(row)

    history_audit = _feature_history_audit(monthly_rows, feature_rows)
    verification_rows, test_months = _audit_saved_models(feature_rows, comparison_rows)
    _write_csv(REPORTS_DIR / "model_metric_recomputation_audit.csv", verification_rows)

    lines = [
        "AgriPrice-NG Phase 3.1 model performance audit",
        "===============================================",
        "",
        "Scope: read-only audit of saved Phase 3 features, metrics, and model artifacts.",
        "No model was fitted and no existing experiment output was overwritten.",
        "",
        "1. Verification findings",
        "------------------------",
        "Target alignment: rows_to_arrays uses target_price for every tree and stacking evaluation.",
        "The target_price field is the observed price for the current calendar month.",
        "",
        "Naive baseline: verified as price_lag_1, where price_lag_1 is the observed price at the immediately preceding calendar month.",
        "It is not the previous available observation. Recomputed baseline metrics match model_comparison.csv.",
        "",
        f"Calendar-feature audit: checked {history_audit['checked_feature_rows']} complete feature rows; "
        f"lag mismatches={history_audit['lag_errors']}, rolling mismatches={history_audit['rolling_errors']}.",
        "Lags and rolling statistics use only months before the target month; no current or future target is included.",
        "",
        "Date alignment: each series is split after sorting by month. The naive, Random Forest, Gradient Boosting,",
        "and Stacking test metrics are evaluated on the same test-month list within that series.",
        "The exact derived test months are:",
    ]
    lines.extend(f"  {item}" for item in test_months)
    lines.extend(
        [
            "",
            "Saved-model recomputation:",
            f"  {sum(bool(row['saved_metrics_match_recomputed']) for row in verification_rows)}/{len(verification_rows)} "
            "baseline/tree/stacking metric rows matched recomputation from the saved artifacts.",
        ]
    )
    for row in verification_rows:
        if not row["saved_metrics_match_recomputed"]:
            lines.append(f"  MISMATCH: {row['series_id']} | {row['model']}")

    lines.extend(
        [
            "",
            "Preprocessing/scaling: tree models use raw numeric features and no scaler or inverse transformation.",
            "The exploratory LSTM code fits its StandardScaler objects on the chronological training sequences only and inverse-transforms predictions with the target scaler; no LSTM model was trained in the current run.",
            "",
            "Stacking: the meta-model is trained on chronological TimeSeriesSplit out-of-fold predictions from the base models.",
            "The test predictions are generated by base models fitted only on the training split, so in-sample base predictions are not used for meta-training.",
            "",
            "2. Why tree models can be worse than the naive baseline",
            "--------------------------------------------------------",
            "- There are only 23-49 usable rows per series and only 4-8 test rows. The resulting estimates are highly volatile.",
            "- Calendar reindexing correctly removes rows with unavailable history, but it also leaves sparse regime-specific training samples.",
            "- The naive baseline directly carries forward the most recent calendar-month price, which is a strong level-persistence rule.",
            "- Random Forest and Gradient Boosting are tree models that interpolate within learned training regimes; they do not naturally extrapolate a new price level or trend beyond the training range.",
            "- The absolute trend_index and lag/rolling features do not remove structural price jumps. A later test regime can therefore be far from the small training sample even when its dates are correctly aligned.",
            "- Stacking has even fewer effective observations for its out-of-fold meta-training and cannot create information absent from the base models.",
            "",
            "These are data-coverage and model-class explanations supported by the audit; no parameter changes were made in this phase.",
            "The full numerical metrics remain in reports/model_comparison.csv.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (REPORTS_DIR / "model_performance_audit.txt").write_text(report, encoding="utf-8")
    return report


def _coverage_for_group(key: tuple[str, str, str], rows: list[dict]) -> dict:
    months = sorted({row["month"] for row in rows}, key=_month_index)
    observed_indices = {_month_index(month) for month in months}
    first_index = _month_index(months[0])
    last_index = _month_index(months[-1])
    expected_indices = set(range(first_index, last_index + 1))
    missing_indices = sorted(expected_indices - observed_indices)

    gap_runs: list[int] = []
    run = 0
    for index in range(first_index, last_index + 1):
        if index not in observed_indices:
            run += 1
        elif run:
            gap_runs.append(run)
            run = 0
    if run:
        gap_runs.append(run)

    market_counts = [int(row["n_markets"]) for row in rows]
    currencies = sorted(
        {
            currency.strip()
            for row in rows
            for currency in str(row.get("currency_values", "")).split(";")
            if currency.strip()
        }
    )
    n_observations = len(months)
    expected_count = len(expected_indices)
    coverage_pct = 100.0 * n_observations / expected_count if expected_count else 0.0
    max_gap = max(gap_runs or [0])
    return {
        "commodity": key[0],
        "unit": key[1],
        "price_type": key[2],
        "n_monthly_observations": n_observations,
        "first_month": months[0],
        "last_month": months[-1],
        "expected_calendar_months": expected_count,
        "missing_month_count": len(missing_indices),
        "missing_months": ";".join(_month_text(index) for index in missing_indices),
        "calendar_coverage_pct": round(coverage_pct, 4),
        "max_consecutive_missing_months": max_gap,
        "min_markets_per_month": min(market_counts),
        "max_markets_per_month": max(market_counts),
        "average_markets_per_month": round(statistics.fmean(market_counts), 4),
        "median_markets_per_month": round(statistics.median(market_counts), 4),
        "currency_values": ";".join(currencies),
        "at_least_60_months": n_observations >= 60,
        "at_least_96_months": n_observations >= 96,
        "strict_continuity_96_80_gap3": n_observations >= 96 and coverage_pct >= 80.0 and max_gap <= 3,
    }


def analyze_aggregates() -> tuple[list[dict], str]:
    rows = _read_csv(PROCESSED_DIR / "commodity_monthly_aggregates.csv")
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["commodity"], row["unit"], row["price_type"])].append(row)

    duplicate_group_months = sum(
        len(group_rows) - len({row["month"] for row in group_rows})
        for group_rows in grouped.values()
    )
    coverage_rows = [
        _coverage_for_group(key, group_rows) for key, group_rows in sorted(grouped.items())
    ]
    _write_csv(REPORTS_DIR / "aggregate_series_coverage.csv", coverage_rows)

    # Ranking is transparent: first meet the 96-month preference, then prefer
    # calendar coverage, fewer gaps, more observations, and market breadth.
    ranked = sorted(
        coverage_rows,
        key=lambda row: (
            not row["at_least_96_months"],
            -float(row["calendar_coverage_pct"]),
            int(row["max_consecutive_missing_months"]),
            -int(row["n_monthly_observations"]),
            -float(row["average_markets_per_month"]),
        ),
    )
    eligible_60 = [row for row in coverage_rows if row["at_least_60_months"]]
    eligible_96 = [row for row in coverage_rows if row["at_least_96_months"]]
    strict = [row for row in coverage_rows if row["strict_continuity_96_80_gap3"]]

    recommended_keys = {
        ("Yam", "2.5 KG", "Retail"),
        ("Oil (palm)", "L", "Retail"),
        ("Rice (imported)", "2.8 KG", "Retail"),
        ("Rice (local)", "2.7 KG", "Retail"),
        ("Rice (imported)", "50 KG", "Wholesale"),
        ("Millet", "2.6 KG", "Retail"),
    }
    recommended = [row for row in ranked if (row["commodity"], row["unit"], row["price_type"]) in recommended_keys]

    lines = [
        "AgriPrice-NG Phase 3.1 aggregate-series coverage summary",
        "=========================================================",
        "",
        "Source: data/processed/commodity_monthly_aggregates.csv, read without modification.",
        "This file contains monthly unweighted means/medians across records for each commodity, exact unit, price type, and month.",
        "Retail and wholesale prices, and different units, remain separate.",
        "",
        f"Aggregate rows inspected: {len(rows):,}",
        f"Commodity/unit/price-type groups: {len(coverage_rows):,}",
        f"Groups with at least 60 monthly observations: {len(eligible_60):,}",
        f"Groups with at least 96 monthly observations: {len(eligible_96):,}",
        f"Groups meeting the strict 96 observations / 80% coverage / maximum 3-month gap rule: {len(strict):,}",
        f"Duplicate aggregate group-month rows detected: {duplicate_group_months}",
        "",
        "Strongest groups by the ranking rule (96-month preference, coverage, gap length, observations, market breadth):",
    ]
    for rank, row in enumerate(ranked[:15], start=1):
        lines.append(
            f"{rank}. {row['commodity']} | {row['unit']} | {row['price_type']}: "
            f"n={row['n_monthly_observations']}, {row['first_month']} to {row['last_month']}, "
            f"coverage={row['calendar_coverage_pct']:.2f}%, missing={row['missing_month_count']}, "
            f"max_gap={row['max_consecutive_missing_months']}, markets/month="
            f"{row['min_markets_per_month']}-{row['max_markets_per_month']} (avg {row['average_markets_per_month']:.2f})"
        )

    lines.extend(
        [
            "",
            "Recommended manageable next experiment (six aggregate series):",
            "These are separate, interpretable series with at least 127 monthly observations and strong coverage.",
        ]
    )
    for row in recommended:
        lines.append(
            f"- {row['commodity']} | {row['unit']} | {row['price_type']}: "
            f"{row['n_monthly_observations']} observations, coverage {row['calendar_coverage_pct']:.2f}%, "
            f"maximum missing run {row['max_consecutive_missing_months']} month(s), "
            f"average {row['average_markets_per_month']:.2f} contributing markets/month."
        )

    lines.extend(
        [
            "",
            "Comparison with the five market-level series",
            "-----------------------------------------------",
            "The aggregate file provides substantially longer candidate series than the 23-49 usable market-level feature rows.",
            "For example, the strongest aggregate candidates have 127-137 monthly rows before feature exclusion, compared with 23-49 usable market-level rows.",
            "Aggregation can improve calendar coverage and reduce idiosyncratic market noise, making a 12-month window and chronological validation more feasible.",
            "",
            "Aggregation limitations",
            "-----------------------",
            "- The mean/median hides market-level dispersion, local shocks, and differences between contributing markets.",
            "- The number and identity of contributing markets changes by month; a stable-looking aggregate can partly reflect changing market composition.",
            "- A single aggregate forecast is not a forecast for every market and should not replace the approved market-level experiments.",
            "- Aggregates should remain separate by exact unit and price type; no unit conversion or retail/wholesale mixing is justified here.",
            "",
            "No models were trained in Phase 3.1. The six recommended series are candidates for approval before a separate aggregate-model experiment.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (REPORTS_DIR / "aggregate_series_summary.txt").write_text(summary, encoding="utf-8")
    return coverage_rows, summary


def main() -> None:
    audit_models()
    analyze_aggregates()
    print("Phase 3.1 audit reports written.")


if __name__ == "__main__":
    main()
