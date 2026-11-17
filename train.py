"""AgriPrice-NG training entry point."""

from __future__ import annotations

import argparse

from src.preprocessing import prepare_approved_data, split_feature_rows
from src.train_models import train_all


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare market-level monthly rows/features and print usable sample counts.",
    )
    mode.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Run the isolated approved aggregate-series experiment.",
    )
    mode.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run the read-only aggregate forecasting diagnosis and target experiments.",
    )
    mode.add_argument(
        "--final-validation",
        action="store_true",
        help="Run leakage-safe final validation on the approved aggregate series.",
    )
    mode.add_argument(
        "--world-bank-only",
        action="store_true",
        help="Prepare and train the isolated World Bank source experiment.",
    )
    mode.add_argument(
        "--world-bank-prepare-only",
        action="store_true",
        help="Normalize and validate the World Bank source without model training.",
    )
    args = parser.parse_args()
    if args.world_bank_prepare_only:
        from src.world_bank_ingestion import prepare_world_bank_data

        result = prepare_world_bank_data()
        print(
            "World Bank preparation completed; "
            f"raw rows={len(result['raw']):,}, normalized rows={len(result['normalized']):,}"
        )
        return
    if args.world_bank_only:
        from src.world_bank_training import train_world_bank

        print("Starting isolated World Bank experiment...")
        result = train_world_bank(rebuild_data=True)
        print(
            "World Bank training completed; "
            f"metric rows={len(result['metrics'])}, forecast rows={len(result['forecasts'])}"
        )
        return
    if args.aggregate_only:
        from src.aggregate_training import train_aggregate

        print("Starting aggregate-series experiment...")
        result = train_aggregate()
        print(f"Aggregate training completed; metric rows={len(result['metrics'])}")
        return
    if args.diagnostics:
        from src.diagnostics import run_diagnostics

        print("Starting aggregate forecasting diagnostics...")
        result = run_diagnostics()
        print(
            "Diagnostics completed; "
            f"error rows={len(result['errors'])}, "
            f"transformation rows={len(result['transformations'])}, "
            f"baseline rows={len(result['baselines'])}"
        )
        return
    if args.final_validation:
        from src.final_validation import run_final_validation

        print("Starting leakage-safe final validation...")
        result = run_final_validation()
        print(
            "Final validation completed; "
            f"comparison rows={len(result['comparison'])}, "
            f"forecast rows={len(result['forecasts'])}"
        )
        return

    prepared = prepare_approved_data()
    print(f"Approved source rows: {len(prepared['approved_rows']):,}")
    print(f"Reindexed calendar rows: {len(prepared['monthly_rows']):,}")
    print(f"Usable feature rows: {len(prepared['feature_rows']):,}")
    print("Per-series sample counts:")
    for diagnostic in prepared["diagnostics"]:
        print(
            f"  {diagnostic['series_id']}: observed={diagnostic['observed_rows']}, "
            f"calendar={diagnostic['calendar_rows']}, "
            f"usable={diagnostic['usable_feature_rows']}, "
            f"excluded={diagnostic['excluded_missing_history_rows']}"
        )
    splits = split_feature_rows(prepared["feature_rows"])
    print("Chronological split counts:")
    for series_id, split in splits.items():
        print(
            f"  {series_id}: train={len(split['train'])}, "
            f"validation={len(split['validation'])}, test={len(split['test'])}"
        )
    if not args.prepare_only:
        print("Starting model training...")
        result = train_all()
        print(f"Training completed; metric rows={len(result['metrics'])}")


if __name__ == "__main__":
    main()
