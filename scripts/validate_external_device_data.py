from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from griddx.external_validation import run_external_device_validation
from griddx.paths import OUTPUT_ROOT, data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate saved device models on a new device-day CSV.")
    parser.add_argument("--csv", type=Path, default=data_path("model_base_device_day_line_or_load_like_real.csv"))
    parser.add_argument("--model-dir", type=Path, default=OUTPUT_ROOT / "device_prediction")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "external_validation")
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=data_path("model_base_device_day_line_or_load_like_enriched.csv"),
        help="Optional source panel used only to inspect device/date overlap.",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="A real 0-3 outcome label. Omit it to run proxy-label consistency validation only.",
    )
    parser.add_argument(
        "--device-partition",
        choices=["test", "all"],
        default="test",
        help="test uses the saved group-split test devices; all includes training devices for compatibility checks.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Use <=0 to read all rows.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--torch-device", default="auto", help="auto, cpu, cuda, cuda:<id>, or mps.")
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA device index when CUDA is visible.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_external_device_validation(
        csv_path=args.csv,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        reference_csv=args.reference_csv,
        target_col=args.target_col,
        device_partition=args.device_partition,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        torch_device=args.torch_device,
        gpu_id=args.gpu_id,
    )
    print(f"\nValidation outputs: {args.output_dir}")
    if metrics.empty:
        print("No saved model artifacts were found.")
    else:
        columns = [
            column
            for column in [
                "model",
                "scoring_type",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "quadratic_weighted_kappa",
                "high_risk_recall_label_ge_2",
            ]
            if column in metrics
        ]
        print(metrics[columns].to_string(index=False))


if __name__ == "__main__":
    main()
