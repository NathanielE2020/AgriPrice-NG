"""Run the approved WFP/NBS historical model-training experiment."""

from __future__ import annotations

import argparse

from src.model_training_pipeline import train_wfp_nbs_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate historical WFP market series. NBS and WFP remain separate; "
            "the command reads existing normalized CSVs and never prepares or overwrites them."
        )
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Run a read-only import and input-schema check without training or writing outputs.",
    )
    args = parser.parse_args()
    if args.check_only:
        from src.model_training_pipeline import NBS_PATH, WFP_PATH, WFP_SELECTION_PATH

        missing = [path.name for path in (WFP_PATH, NBS_PATH, WFP_SELECTION_PATH) if not path.is_file()]
        if missing:
            parser.error(f"Required prepared inputs are missing: {', '.join(missing)}")
        print("Prepared WFP/NBS normalized inputs are present. No training was run.")
        return
    result = train_wfp_nbs_models()
    print(
        "Training complete: "
        f"{result['eligible_wfp_series']} historical WFP series; "
        f"{result['trained_model_series_fits']} trained model-series fits; "
        f"{result['forecast_records']} recursive historical forecast rows; "
        f"NBS models trained={result['nbs_models_trained']}."
    )
    print("See reports/model_training_evaluation_report.txt for data limits and results.")


if __name__ == "__main__":
    main()
