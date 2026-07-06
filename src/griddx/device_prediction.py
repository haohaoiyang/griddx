from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import read_csv_limited
from .features import build_device_features, modeling_columns
from .labels import make_device_cold_start_labels
from .model_zoo import save_metrics_summary, train_sklearn_models, train_torch_mlp
from .paths import data_path, ensure_output_dir


def build_device_training_table(
    csv_path: Path,
    max_rows: int,
    group: str | None = None,
    target_col: str | None = None,
) -> pd.DataFrame:
    df = read_csv_limited(csv_path, max_rows=max_rows)
    if group and group != "all" and "suggested_model_group" in df.columns:
        df = df[df["suggested_model_group"] == group].copy()
    features = build_device_features(df)
    if target_col and target_col in features.columns:
        features["state_label"] = features[target_col].astype(int)
        if "state_score" not in features.columns:
            features["state_score"] = features["state_label"] * 25.0
    else:
        features = make_device_cold_start_labels(features)
    return features


def run_device_training(
    csv_path: Path,
    max_rows: int = 120_000,
    group: str = "all",
    models: list[str] | None = None,
    train_mlp: bool = True,
    mlp_epochs: int = 18,
    torch_device: str = "auto",
    gpu_id: int | None = None,
    batch_size: int = 256,
) -> pd.DataFrame:
    output_dir = ensure_output_dir("device_prediction")
    table = build_device_training_table(csv_path, max_rows=max_rows, group=group)
    table.to_parquet(output_dir / "device_training_features.parquet", index=False)
    preview_cols = [
        col
        for col in ["date", "unified_device_id", "station_id", "suggested_model_group", "state_score", "state_label"]
        if col in table.columns
    ]
    table[preview_cols].head(5000).to_csv(output_dir / "device_label_preview.csv", index=False)

    numeric, categorical = modeling_columns(table, "state_label")
    results = train_sklearn_models(table, "state_label", numeric, categorical, output_dir, selected_models=models)
    if train_mlp:
        results.append(
            train_torch_mlp(
                table,
                "state_label",
                numeric,
                categorical,
                output_dir,
                epochs=mlp_epochs,
                batch_size=batch_size,
                torch_device=torch_device,
                gpu_id=gpu_id,
            )
        )
    summary = save_metrics_summary(results, output_dir)
    print(summary.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline device state prediction models.")
    parser.add_argument("--csv", type=Path, default=data_path("model_base_device_day.csv"))
    parser.add_argument("--max-rows", type=int, default=120_000, help="Use <=0 to read the full CSV.")
    parser.add_argument("--group", default="all", help="suggested_model_group to train on, or all.")
    parser.add_argument("--models", default="logistic_regression,random_forest,extra_trees,hist_gradient_boosting")
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--mlp-epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--torch-device", default="auto", help="auto, cpu, cuda, cuda:<id>, or mps.")
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA device index, e.g. 0 for the first visible GPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    run_device_training(
        csv_path=args.csv,
        max_rows=args.max_rows,
        group=args.group,
        models=models,
        train_mlp=not args.no_mlp,
        mlp_epochs=args.mlp_epochs,
        torch_device=args.torch_device,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
