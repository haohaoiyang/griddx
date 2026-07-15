from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import read_csv_limited
from .features import build_device_features, modeling_columns
from .labels import has_device_enriched_fields, make_device_cold_start_labels, make_device_enriched_weak_labels
from .model_zoo import (
    save_metrics_summary,
    train_sklearn_models,
    train_torch_mlp,
    train_torch_multi_discriminator,
)
from .paths import data_path, ensure_output_dir


def build_device_training_table(
    csv_path: Path,
    max_rows: int,
    group: str | None = None,
    target_col: str | None = None,
    label_mode: str = "auto",
) -> pd.DataFrame:
    df = read_csv_limited(csv_path, max_rows=max_rows)
    if group and group != "all" and "suggested_model_group" in df.columns:
        df = df[df["suggested_model_group"] == group].copy()
    features = build_device_features(df)
    mode = label_mode.lower()
    if target_col:
        if target_col not in features.columns:
            raise ValueError(f"Target column not found: {target_col}")
        features["state_label"] = features[target_col].astype(int)
        features["label_source"] = f"target:{target_col}"
        if "state_score" not in features.columns:
            features["state_score"] = features["state_label"] * 25.0
    elif mode in {"auto", "enriched_weak"} and has_device_enriched_fields(features):
        features = make_device_enriched_weak_labels(features)
    elif mode in {"auto", "cold_start"}:
        features = make_device_cold_start_labels(features)
        features["label_source"] = "cold_start_rule"
    else:
        raise ValueError("label_mode=enriched_weak requires the enriched history and maintenance columns.")
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
    target_col: str | None = None,
    label_mode: str = "auto",
    split_strategy: str = "group",
    train_multi_discriminator: bool = True,
) -> pd.DataFrame:
    output_dir = ensure_output_dir("device_prediction")
    table = build_device_training_table(
        csv_path,
        max_rows=max_rows,
        group=group,
        target_col=target_col,
        label_mode=label_mode,
    )
    table.to_parquet(output_dir / "device_training_features.parquet", index=False)
    preview_cols = [
        col
        for col in ["date", "unified_device_id", "station_id", "suggested_model_group", "state_score", "state_label"]
        if col in table.columns
    ]
    table[preview_cols].head(5000).to_csv(output_dir / "device_label_preview.csv", index=False)

    numeric, categorical = modeling_columns(table, "state_label")
    results = train_sklearn_models(
        table,
        "state_label",
        numeric,
        categorical,
        output_dir,
        selected_models=models,
        split_strategy=split_strategy,
        group_col="unified_device_id" if split_strategy == "group" else None,
    )
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
                split_strategy=split_strategy,
                group_col="unified_device_id" if split_strategy == "group" else None,
            )
        )
    if train_multi_discriminator:
        results.append(
            train_torch_multi_discriminator(
                table,
                "state_label",
                numeric,
                categorical,
                output_dir,
                epochs=mlp_epochs,
                batch_size=batch_size,
                torch_device=torch_device,
                gpu_id=gpu_id,
                split_strategy=split_strategy,
                group_col="unified_device_id" if split_strategy == "group" else None,
            )
        )
    metadata = {
        "csv": csv_path.name,
        "rows": len(table),
        "label_source": str(table["label_source"].iloc[0]) if "label_source" in table.columns else "unknown",
        "target_col": target_col,
        "split_strategy": split_strategy,
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "class_counts": {str(key): int(value) for key, value in table["state_label"].value_counts().sort_index().items()},
        "personalized_model": "DeviceAdaptiveMultiDiscriminator" if train_multi_discriminator else None,
        "device_profile_output": "device_expert_profiles.csv" if train_multi_discriminator else None,
    }
    (output_dir / "training_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = save_metrics_summary(results, output_dir)
    print(summary.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline device state prediction models.")
    parser.add_argument("--csv", type=Path, default=data_path("model_base_device_day_line_or_load_like_enriched.csv"))
    parser.add_argument("--max-rows", type=int, default=120_000, help="Use <=0 to read the full CSV.")
    parser.add_argument("--group", default="all", help="suggested_model_group to train on, or all.")
    parser.add_argument("--models", default="logistic_regression,random_forest,extra_trees,hist_gradient_boosting")
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--no-multi-discriminator", action="store_true")
    parser.add_argument("--mlp-epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--torch-device", default="auto", help="auto, cpu, cuda, cuda:<id>, or mps.")
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA device index, e.g. 0 for the first visible GPU.")
    parser.add_argument("--target-col", default=None, help="Real 0-3 target column. Overrides weak/cold-start labels.")
    parser.add_argument("--label-mode", choices=["auto", "enriched_weak", "cold_start"], default="auto")
    parser.add_argument("--split-strategy", choices=["group", "temporal", "random"], default="group")
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
        target_col=args.target_col,
        label_mode=args.label_mode,
        split_strategy=args.split_strategy,
        train_multi_discriminator=not args.no_multi_discriminator,
    )


if __name__ == "__main__":
    main()
