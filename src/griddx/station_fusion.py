from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import save_eval_artifacts
from .model_zoo import (
    TrainResult,
    build_preprocessor,
    describe_torch_device,
    make_train_test_indices,
    resolve_torch_device,
)


STATION_VIEW_NAMES = {
    "operation": "运行状态视图",
    "history_maintenance": "历史检修视图",
    "infrastructure": "基础设施视图",
}


def build_station_feature_views(feature_names: list[str] | np.ndarray) -> dict[str, list[int]]:
    names = [str(name).lower() for name in feature_names]
    patterns = {
        "operation": (
            "operation_coverage",
            "device_count",
            "voltage_",
            "current_",
            "active_power",
            "reactive_power",
            "switch_",
            "capacity_loading",
            "roll7",
            "diff_1d",
            "day_",
            "month",
        ),
        "history_maintenance": (
            "history_",
            "maintenance_",
            "unresolved_",
        ),
        "infrastructure": (
            "station_type",
            "voltage_level",
            "main_transformer",
            "lightning",
            "ice_area",
            "environment_",
        ),
    }
    all_indices = list(range(len(names)))
    views: dict[str, list[int]] = {}
    for view_name, tokens in patterns.items():
        indices = [index for index, name in enumerate(names) if any(token in name for token in tokens)]
        views[view_name] = indices or all_indices
    return views


class StationHierarchicalFusionNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        feature_views: dict[str, list[int]],
        gate_temperature: float = 0.90,
    ) -> None:
        super().__init__()
        self.view_names = list(feature_views)
        self.feature_views = feature_views
        self.gate_temperature = gate_temperature
        self.view_encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(len(feature_views[name]), 48),
                    nn.LayerNorm(48),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(48, 32),
                    nn.GELU(),
                )
                for name in self.view_names
            }
        )
        self.view_gate = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, len(self.view_names)),
        )
        self.fusion_norm = nn.LayerNorm(32)
        self.state_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(16, n_classes),
        )
        self.global_state_residual = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        view_weights = torch.softmax(self.view_gate(x) / self.gate_temperature, dim=1)
        embeddings = []
        for name in self.view_names:
            embeddings.append(self.view_encoders[name](x[:, self.feature_views[name]]))
        stacked = torch.stack(embeddings, dim=1)
        fused = self.fusion_norm((view_weights.unsqueeze(-1) * stacked).sum(dim=1))
        state_logits = self.global_state_residual(x) + 0.50 * self.state_head(fused)
        return state_logits, self.risk_head(fused).squeeze(1), view_weights


def _partition_name(values: pd.Series) -> str:
    unique = sorted(values.unique())
    return unique[0] if len(unique) == 1 else "mixed"


def save_station_hierarchical_profiles(
    model: StationHierarchicalFusionNetwork,
    preprocessor: Any,
    x: pd.DataFrame,
    df: pd.DataFrame,
    target_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_path: Path,
    device: torch.device,
    batch_size: int = 2048,
) -> pd.DataFrame:
    partition = np.full(len(df), "unused", dtype=object)
    partition[train_idx] = "train"
    partition[test_idx] = "test"
    parts: list[pd.DataFrame] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            stop = min(start + batch_size, len(df))
            batch_np = preprocessor.transform(x.iloc[start:stop]).astype("float32")
            logits, risk, weights = model(torch.from_numpy(batch_np).to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            risk_np = risk.cpu().numpy()
            weights_np = weights.cpu().numpy()
            meta = df.iloc[start:stop]
            part = pd.DataFrame(
                {
                    "station_id": meta["station_id"].astype("string").fillna("missing_station").to_numpy(),
                    "data_partition": partition[start:stop],
                    "state_label": meta[target_col].astype(float).to_numpy(),
                    "predicted_label": probabilities.argmax(axis=1),
                    "predicted_risk_score": 100.0 * risk_np,
                    "prediction_confidence": probabilities.max(axis=1),
                    "high_risk_probability": probabilities[:, 2:].sum(axis=1),
                }
            )
            if "state_score" in meta.columns:
                part["state_score"] = meta["state_score"].astype(float).to_numpy()
            for index, name in enumerate(model.view_names):
                part[f"view_weight_{name}"] = weights_np[:, index]
            parts.append(part)

    detail = pd.concat(parts, ignore_index=True)
    weight_cols = [f"view_weight_{name}" for name in model.view_names]
    mean_cols = [
        "state_label",
        "predicted_label",
        "predicted_risk_score",
        "prediction_confidence",
        "high_risk_probability",
        *weight_cols,
    ]
    if "state_score" in detail:
        mean_cols.append("state_score")
    profiles = detail.groupby("station_id", as_index=False)[mean_cols].mean()
    profiles["sample_count"] = detail.groupby("station_id").size().reindex(profiles["station_id"]).to_numpy()
    profiles["data_partition"] = (
        detail.groupby("station_id")["data_partition"]
        .agg(_partition_name)
        .reindex(profiles["station_id"])
        .to_numpy()
    )
    metadata_cols = [
        col for col in ["station_name", "station_type_code", "voltage_level_kv", "main_transformer_count"] if col in df
    ]
    if metadata_cols:
        metadata = df.groupby("station_id", as_index=False)[metadata_cols].first()
        profiles = profiles.merge(metadata, on="station_id", how="left")
    dominant = profiles[weight_cols].idxmax(axis=1).str.replace("view_weight_", "", regex=False)
    profiles["dominant_view_code"] = dominant
    profiles["dominant_view"] = dominant.map(STATION_VIEW_NAMES)
    profiles["view_concentration"] = profiles[weight_cols].max(axis=1)
    profiles = profiles.sort_values(["predicted_risk_score", "view_concentration"], ascending=False)
    profiles.to_csv(output_path, index=False)
    return profiles


def train_station_hierarchical_fusion(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
    output_dir: Path,
    epochs: int = 18,
    batch_size: int = 256,
    lr: float = 8e-4,
    random_state: int = 42,
    torch_device: str = "auto",
    gpu_id: int | None = None,
    split_strategy: str = "group",
    group_col: str | None = "station_id",
    date_col: str = "date",
) -> TrainResult:
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = True
    output_dir.mkdir(parents=True, exist_ok=True)

    x = df[numeric_features + categorical_features].copy()
    y = df[target_col].astype(int).to_numpy()
    risk_target = (df["state_score"].astype(float).clip(0, 100) / 100.0).to_numpy(dtype="float32")
    n_classes = int(np.max(y)) + 1
    train_idx, test_idx, split_metadata = make_train_test_indices(
        df,
        target_col,
        random_state=random_state,
        split_strategy=split_strategy,
        group_col=group_col,
        date_col=date_col,
    )
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    risk_train, risk_test = risk_target[train_idx], risk_target[test_idx]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    x_train_np = preprocessor.fit_transform(x_train).astype("float32")
    x_test_np = preprocessor.transform(x_test).astype("float32")
    feature_names = preprocessor.get_feature_names_out()
    feature_views = build_station_feature_views(feature_names)

    device = resolve_torch_device(torch_device=torch_device, gpu_id=gpu_id)
    device_text = describe_torch_device(device)
    model = StationHierarchicalFusionNetwork(
        n_features=x_train_np.shape[1],
        n_classes=n_classes,
        feature_views=feature_views,
    ).to(device)
    class_count = np.bincount(y_train, minlength=n_classes).astype("float32")
    class_weight = class_count.sum() / np.maximum(class_count, 1.0)
    class_weight = class_weight / class_weight.mean()
    state_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32, device=device))
    risk_loss_fn = nn.SmoothL1Loss(beta=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)

    train_ds = TensorDataset(
        torch.from_numpy(x_train_np),
        torch.from_numpy(y_train.astype("int64")),
        torch.from_numpy(risk_train),
    )
    use_cuda_loader = device.type == "cuda"
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=use_cuda_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_state_loss = 0.0
        total_risk_loss = 0.0
        for batch_x, batch_y, batch_risk in loader:
            batch_x = batch_x.to(device, non_blocking=use_cuda_loader)
            batch_y = batch_y.to(device, non_blocking=use_cuda_loader)
            batch_risk = batch_risk.to(device, non_blocking=use_cuda_loader)
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_risk, view_weights = model(batch_x)
            state_loss = state_loss_fn(logits, batch_y)
            risk_loss = risk_loss_fn(predicted_risk, batch_risk)
            mean_view = view_weights.mean(dim=0)
            uniform = torch.full_like(mean_view, 1.0 / len(model.view_names))
            balance_loss = torch.square(mean_view - uniform).sum()
            entropy = -(view_weights * torch.log(view_weights.clamp_min(1e-8))).sum(dim=1).mean()
            loss = state_loss + 0.35 * risk_loss + 0.15 * balance_loss + 0.003 * entropy
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            total_state_loss += state_loss.item() * batch_x.size(0)
            total_risk_loss += risk_loss.item() * batch_x.size(0)
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(
                f"station_hsf epoch={epoch:02d} loss={total_loss / len(train_ds):.4f} "
                f"state={total_state_loss / len(train_ds):.4f} "
                f"risk={total_risk_loss / len(train_ds):.4f} device={device_text}"
            )

    model.eval()
    with torch.no_grad():
        logits, predicted_risk, view_weights = model(
            torch.from_numpy(x_test_np).to(device, non_blocking=use_cuda_loader)
        )
        pred = logits.argmax(dim=1).cpu().numpy()
        predicted_risk_np = predicted_risk.cpu().numpy()
        view_np = view_weights.cpu().numpy()
    risk_mae = float(np.mean(np.abs(predicted_risk_np - risk_test)) * 100.0)
    risk_rmse = float(np.sqrt(np.mean(np.square(predicted_risk_np - risk_test))) * 100.0)
    view_usage = {
        name: round(float(view_np[:, index].mean()), 6) for index, name in enumerate(model.view_names)
    }
    extra_metrics: dict[str, Any] = {
        "torch_device": str(device),
        "torch_device_name": device_text,
        "torch_gpu_id": device.index if device.type == "cuda" else "",
        "risk_score_mae": risk_mae,
        "risk_score_rmse": risk_rmse,
        "view_count": len(model.view_names),
        "mean_view_concentration": float(view_np.max(axis=1).mean()),
        "view_mean_weights": json.dumps(view_usage, ensure_ascii=False),
    }
    extra_metrics.update(split_metadata)
    metrics = save_eval_artifacts(output_dir, "torch_station_hierarchical", y_test, pred, extra_metrics)

    save_station_hierarchical_profiles(
        model,
        preprocessor,
        x,
        df,
        target_col,
        train_idx,
        test_idx,
        output_dir / "station_hierarchical_profiles.csv",
        device,
    )
    artifact_path = output_dir / "torch_station_hierarchical.pt"
    torch.save(
        {
            "model_name": "HSF-Net",
            "model_class": "StationHierarchicalFusionNetwork",
            "model_state_dict": model.cpu().state_dict(),
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "transformed_feature_names": feature_names.tolist(),
            "feature_views": feature_views,
            "view_names": STATION_VIEW_NAMES,
            "n_classes": n_classes,
            "metrics": metrics,
            "torch_device_used": str(device),
            "torch_device_name": device_text,
        },
        artifact_path,
    )
    return TrainResult("torch_station_hierarchical", metrics, artifact_path)
