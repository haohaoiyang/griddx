from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from griddx.device_prediction import run_device_training
from griddx.economic_dispatch import run_dispatch
from griddx.paths import data_path
from griddx.station_assessment import run_station_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline griddx modeling pipeline.")
    parser.add_argument("--quick", action="store_true", help="Use smaller samples and fewer MLP epochs for environment checks.")
    parser.add_argument("--skip-device", action="store_true")
    parser.add_argument("--skip-station", action="store_true")
    parser.add_argument("--skip-dispatch", action="store_true")
    parser.add_argument("--skip-multi-discriminator", action="store_true")
    parser.add_argument("--skip-station-hierarchical", action="store_true")
    parser.add_argument("--device-csv", type=Path, default=data_path("model_base_device_day_line_or_load_like_enriched.csv"))
    parser.add_argument("--station-csv", type=Path, default=data_path("model_base_station_day_extract_enriched.csv"))
    parser.add_argument("--device-max-rows", type=int, default=120_000)
    parser.add_argument("--station-max-rows", type=int, default=0)
    parser.add_argument("--device-group", default="all")
    parser.add_argument("--models", default="logistic_regression,random_forest,extra_trees,hist_gradient_boosting")
    parser.add_argument("--mlp-epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--torch-device", default="auto", help="auto, cpu, cuda, cuda:<id>, or mps.")
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA device index, e.g. 0 for the first visible GPU.")
    parser.add_argument("--device-target-col", default=None, help="Real 0-3 device target column, when available.")
    parser.add_argument("--station-target-col", default=None, help="Real 0-3 station target column, when available.")
    parser.add_argument("--label-mode", choices=["auto", "enriched_weak", "cold_start"], default="auto")
    parser.add_argument("--split-strategy", choices=["group", "temporal", "random"], default="group")
    parser.add_argument("--dispatch-date", default=None)
    parser.add_argument("--supply-budget-mw", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if args.quick:
        args.device_max_rows = min(args.device_max_rows, 30_000)
        args.station_max_rows = args.station_max_rows or 4_000
        args.mlp_epochs = min(args.mlp_epochs, 5)

    if not args.skip_device:
        print("\n=== Device prediction models ===")
        run_device_training(
            csv_path=args.device_csv,
            max_rows=args.device_max_rows,
            group=args.device_group,
            models=models,
            train_mlp=True,
            mlp_epochs=args.mlp_epochs,
            torch_device=args.torch_device,
            gpu_id=args.gpu_id,
            batch_size=args.batch_size,
            target_col=args.device_target_col,
            label_mode=args.label_mode,
            split_strategy=args.split_strategy,
            train_multi_discriminator=not args.skip_multi_discriminator,
        )
    if not args.skip_station:
        print("\n=== Station assessment models ===")
        run_station_training(
            csv_path=args.station_csv,
            max_rows=args.station_max_rows,
            models=models,
            train_mlp=True,
            mlp_epochs=args.mlp_epochs,
            torch_device=args.torch_device,
            gpu_id=args.gpu_id,
            batch_size=args.batch_size,
            target_col=args.station_target_col,
            label_mode=args.label_mode,
            split_strategy=args.split_strategy,
            train_hierarchical=not args.skip_station_hierarchical,
        )
    if not args.skip_dispatch:
        print("\n=== Economic dispatch plan ===")
        run_dispatch(args.station_csv, date=args.dispatch_date, supply_budget_mw=args.supply_budget_mw)


if __name__ == "__main__":
    main()
