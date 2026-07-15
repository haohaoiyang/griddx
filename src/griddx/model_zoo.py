from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import save_eval_artifacts


@dataclass
class TrainResult:
    model_name: str
    metrics: dict[str, Any]
    artifact_path: Path


def make_train_test_indices(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.25,
    random_state: int = 42,
    split_strategy: str = "random",
    group_col: str | None = None,
    date_col: str = "date",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = np.arange(len(df))
    y = df[target_col].astype(int).to_numpy()
    strategy = split_strategy.lower()

    if strategy == "group":
        if not group_col or group_col not in df.columns:
            raise ValueError("group split requires an existing group_col.")
        groups = df[group_col].astype("string").fillna(pd.Series(indices, index=df.index).astype(str)).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(indices, y, groups))
        metadata: dict[str, Any] = {
            "split_strategy": "group",
            "split_group_col": group_col,
            "train_group_count": int(pd.Series(groups[train_idx]).nunique()),
            "test_group_count": int(pd.Series(groups[test_idx]).nunique()),
        }
    elif strategy == "temporal":
        if date_col not in df.columns:
            raise ValueError(f"temporal split requires date column: {date_col}")
        dates = pd.to_datetime(df[date_col], errors="coerce")
        unique_dates = np.array(sorted(dates.dropna().unique()))
        if len(unique_dates) < 2:
            raise ValueError("temporal split requires at least two distinct valid dates.")
        cutoff_position = min(max(int(np.floor(len(unique_dates) * (1.0 - test_size))), 1), len(unique_dates) - 1)
        cutoff = unique_dates[cutoff_position]
        train_idx = indices[(dates < cutoff).to_numpy()]
        test_idx = indices[(dates >= cutoff).to_numpy()]
        metadata = {
            "split_strategy": "temporal",
            "split_date_col": date_col,
            "test_start_date": str(pd.Timestamp(cutoff).date()),
        }
    elif strategy == "random":
        stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        metadata = {"split_strategy": "random"}
    else:
        raise ValueError("split_strategy must be one of: random, group, temporal.")

    metadata.update({"train_rows": int(len(train_idx)), "test_rows": int(len(test_idx))})
    return np.asarray(train_idx), np.asarray(test_idx), metadata


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def sklearn_model_specs(random_state: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": OneVsRestClassifier(
            LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced")
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=6,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.06,
            l2_regularization=0.05,
            random_state=random_state,
        ),
    }


def train_sklearn_models(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
    output_dir: Path,
    selected_models: list[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
    split_strategy: str = "random",
    group_col: str | None = None,
    date_col: str = "date",
) -> list[TrainResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_models = selected_models or list(sklearn_model_specs(random_state))
    x = df[numeric_features + categorical_features].copy()
    y = df[target_col].astype(int).to_numpy()
    train_idx, test_idx, split_metadata = make_train_test_indices(
        df,
        target_col,
        test_size=test_size,
        random_state=random_state,
        split_strategy=split_strategy,
        group_col=group_col,
        date_col=date_col,
    )
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    results: list[TrainResult] = []
    for name, estimator in sklearn_model_specs(random_state).items():
        if name not in selected_models:
            continue
        pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                ("model", estimator),
            ]
        )
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        metrics = save_eval_artifacts(output_dir, name, y_test, pred, split_metadata)
        artifact_path = output_dir / f"{name}.joblib"
        joblib.dump(pipe, artifact_path)
        results.append(TrainResult(name, metrics, artifact_path))
    return results


class SimpleMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


DISCRIMINATOR_NAMES = {
    "operation": "运行波动判别器",
    "history": "历史事件判别器",
    "maintenance": "检修恢复判别器",
    "family_profile": "设备画像判别器",
}


def build_discriminator_feature_groups(feature_names: list[str] | np.ndarray) -> dict[str, list[int]]:
    names = [str(name).lower() for name in feature_names]
    patterns = {
        "operation": (
            "current",
            "voltage",
            "active_power",
            "reactive_power",
            "frequency",
            "switch",
            "tap_position",
            "measurement",
            "coverage",
            "missing_flag",
            "diff_1d",
            "roll7",
            "roll30",
            "peer_z",
        ),
        "history": (
            "history_defect",
            "history_trip",
            "history_event",
        ),
        "maintenance": (
            "maintenance",
            "unresolved",
            "resolution_rate",
        ),
        "family_profile": (
            "family_",
            "device_type",
            "device_age",
            "manufacturer",
            "voltage_level",
            "operation_status",
            "station_type",
            "lightning",
            "ice_area",
        ),
    }
    all_indices = list(range(len(names)))
    groups: dict[str, list[int]] = {}
    for group_name, tokens in patterns.items():
        indices = [index for index, name in enumerate(names) if any(token in name for token in tokens)]
        groups[group_name] = indices or all_indices
    return groups


class DeviceAdaptiveMultiDiscriminator(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        feature_groups: dict[str, list[int]],
        gate_temperature: float = 0.85,
    ) -> None:
        super().__init__()
        self.expert_names = list(feature_groups)
        self.feature_groups = feature_groups
        self.gate_temperature = gate_temperature
        self.gate = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(self.expert_names)),
        )
        self.discriminators = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(len(feature_groups[name]), 64),
                    nn.LayerNorm(64),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(64, 32),
                    nn.GELU(),
                    nn.Linear(32, n_classes),
                )
                for name in self.expert_names
            }
        )
        self.global_residual = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_weights = torch.softmax(self.gate(x) / self.gate_temperature, dim=1)
        expert_logits = []
        for name in self.expert_names:
            indices = self.feature_groups[name]
            expert_logits.append(self.discriminators[name](x[:, indices]))
        stacked = torch.stack(expert_logits, dim=1)
        fused = (gate_weights.unsqueeze(-1) * stacked).sum(dim=1)
        return fused + 0.20 * self.global_residual(x), gate_weights


def resolve_torch_device(torch_device: str = "auto", gpu_id: int | None = None) -> torch.device:
    requested = (torch_device or "auto").lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        if ":" in requested:
            device = torch.device(requested)
            torch.cuda.set_device(device.index or 0)
            return device
        index = 0 if gpu_id is None else gpu_id
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"gpu_id={index} is invalid; visible CUDA device count is {torch.cuda.device_count()}.")
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "auto":
        raise ValueError("torch_device must be one of: auto, cpu, cuda, cuda:<id>, mps.")
    if torch.cuda.is_available():
        index = 0 if gpu_id is None else gpu_id
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"gpu_id={index} is invalid; visible CUDA device count is {torch.cuda.device_count()}.")
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_torch_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index or 0
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return "cpu"


def train_torch_mlp(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
    output_dir: Path,
    epochs: int = 18,
    batch_size: int = 256,
    lr: float = 1e-3,
    random_state: int = 42,
    torch_device: str = "auto",
    gpu_id: int | None = None,
    split_strategy: str = "random",
    group_col: str | None = None,
    date_col: str = "date",
) -> TrainResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = True
    x = df[numeric_features + categorical_features].copy()
    y = df[target_col].astype(int).to_numpy()
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

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    x_train_np = preprocessor.fit_transform(x_train).astype("float32")
    x_test_np = preprocessor.transform(x_test).astype("float32")

    device = resolve_torch_device(torch_device=torch_device, gpu_id=gpu_id)
    device_text = describe_torch_device(device)
    model = SimpleMLP(x_train_np.shape[1], n_classes).to(device)
    class_count = np.bincount(y_train, minlength=n_classes).astype("float32")
    class_weight = class_count.sum() / np.maximum(class_count, 1.0)
    class_weight = class_weight / class_weight.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(torch.from_numpy(x_train_np), torch.from_numpy(y_train.astype("int64")))
    use_cuda_loader = device.type == "cuda"
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=use_cuda_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=use_cuda_loader)
            batch_y = batch_y.to(device, non_blocking=use_cuda_loader)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(f"mlp epoch={epoch:02d} loss={total_loss / len(train_ds):.4f} device={device_text}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test_np).to(device, non_blocking=use_cuda_loader))
        pred = logits.argmax(dim=1).cpu().numpy()
    device_metrics: dict[str, Any] = {
        "torch_device": str(device),
        "torch_device_name": device_text,
        "torch_gpu_id": device.index if device.type == "cuda" else "",
    }
    device_metrics.update(split_metadata)
    metrics = save_eval_artifacts(output_dir, "torch_mlp", y_test, pred, device_metrics)
    artifact_path = output_dir / "torch_mlp.pt"
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "n_classes": n_classes,
            "metrics": metrics,
            "torch_device_requested": torch_device,
            "gpu_id": gpu_id,
            "torch_device_used": str(device),
            "torch_device_name": device_text,
        },
        artifact_path,
    )
    return TrainResult("torch_mlp", metrics, artifact_path)


def save_device_expert_profiles(
    model: DeviceAdaptiveMultiDiscriminator,
    preprocessor: ColumnTransformer,
    x: pd.DataFrame,
    df: pd.DataFrame,
    target_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_path: Path,
    device: torch.device,
    batch_size: int = 4096,
    id_col: str = "unified_device_id",
) -> pd.DataFrame:
    if id_col not in df.columns:
        return pd.DataFrame()

    partition = np.full(len(df), "unused", dtype=object)
    partition[train_idx] = "train"
    partition[test_idx] = "test"
    parts: list[pd.DataFrame] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            stop = min(start + batch_size, len(df))
            batch_np = preprocessor.transform(x.iloc[start:stop]).astype("float32")
            batch_tensor = torch.from_numpy(batch_np).to(device)
            logits, gates = model(batch_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            gate_np = gates.cpu().numpy()
            meta = df.iloc[start:stop]
            part = pd.DataFrame(
                {
                    id_col: meta[id_col].astype("string").fillna("missing_device").to_numpy(),
                    "data_partition": partition[start:stop],
                    "state_label": meta[target_col].astype(float).to_numpy(),
                    "predicted_label": probabilities.argmax(axis=1),
                    "prediction_confidence": probabilities.max(axis=1),
                    "high_risk_probability": probabilities[:, 2:].sum(axis=1) if probabilities.shape[1] > 2 else 0.0,
                    "gate_entropy": -(gate_np * np.log(np.clip(gate_np, 1e-8, 1.0))).sum(axis=1),
                }
            )
            if "state_score" in meta.columns:
                part["state_score"] = meta["state_score"].astype(float).to_numpy()
            for index, name in enumerate(model.expert_names):
                part[f"expert_weight_{name}"] = gate_np[:, index]
            parts.append(part)

    detail = pd.concat(parts, ignore_index=True)
    weight_cols = [f"expert_weight_{name}" for name in model.expert_names]
    mean_cols = [
        "state_label",
        "predicted_label",
        "prediction_confidence",
        "high_risk_probability",
        "gate_entropy",
        *weight_cols,
    ]
    if "state_score" in detail.columns:
        mean_cols.append("state_score")
    profiles = detail.groupby(id_col, as_index=False)[mean_cols].mean()
    profiles["sample_count"] = detail.groupby(id_col).size().reindex(profiles[id_col]).to_numpy()

    def partition_name(values: pd.Series) -> str:
        unique = sorted(values.unique())
        return unique[0] if len(unique) == 1 else "mixed"

    profiles["data_partition"] = (
        detail.groupby(id_col)["data_partition"].agg(partition_name).reindex(profiles[id_col]).to_numpy()
    )
    metadata_cols = [col for col in ["device_name", "station_id", "device_type_code", "voltage_level_kv"] if col in df]
    if metadata_cols:
        metadata = df.groupby(id_col, as_index=False)[metadata_cols].first()
        profiles = profiles.merge(metadata, on=id_col, how="left")

    dominant = profiles[weight_cols].idxmax(axis=1).str.replace("expert_weight_", "", regex=False)
    profiles["dominant_expert_code"] = dominant
    profiles["dominant_discriminator"] = dominant.map(DISCRIMINATOR_NAMES)
    profiles["personalization_strength"] = profiles[weight_cols].max(axis=1)
    profiles = profiles.sort_values(["high_risk_probability", "personalization_strength"], ascending=False)
    profiles.to_csv(output_path, index=False)
    return profiles


def train_torch_multi_discriminator(
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
    group_col: str | None = "unified_device_id",
    date_col: str = "date",
) -> TrainResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = True

    x = df[numeric_features + categorical_features].copy()
    y = df[target_col].astype(int).to_numpy()
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

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    x_train_np = preprocessor.fit_transform(x_train).astype("float32")
    x_test_np = preprocessor.transform(x_test).astype("float32")
    feature_names = preprocessor.get_feature_names_out()
    feature_groups = build_discriminator_feature_groups(feature_names)

    device = resolve_torch_device(torch_device=torch_device, gpu_id=gpu_id)
    device_text = describe_torch_device(device)
    model = DeviceAdaptiveMultiDiscriminator(
        n_features=x_train_np.shape[1],
        n_classes=n_classes,
        feature_groups=feature_groups,
    ).to(device)

    class_count = np.bincount(y_train, minlength=n_classes).astype("float32")
    class_weight = class_count.sum() / np.maximum(class_count, 1.0)
    class_weight = class_weight / class_weight.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    train_ds = TensorDataset(torch.from_numpy(x_train_np), torch.from_numpy(y_train.astype("int64")))
    use_cuda_loader = device.type == "cuda"
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=use_cuda_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=use_cuda_loader)
            batch_y = batch_y.to(device, non_blocking=use_cuda_loader)
            optimizer.zero_grad(set_to_none=True)
            logits, gates = model(batch_x)
            classification_loss = criterion(logits, batch_y)
            mean_gate = gates.mean(dim=0)
            uniform = torch.full_like(mean_gate, 1.0 / len(model.expert_names))
            balance_loss = torch.square(mean_gate - uniform).sum()
            gate_entropy = -(gates * torch.log(gates.clamp_min(1e-8))).sum(dim=1).mean()
            loss = classification_loss + 0.20 * balance_loss + 0.005 * gate_entropy
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            total_classification_loss += classification_loss.item() * batch_x.size(0)
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(
                f"multi_discriminator epoch={epoch:02d} "
                f"loss={total_loss / len(train_ds):.4f} "
                f"ce={total_classification_loss / len(train_ds):.4f} device={device_text}"
            )

    model.eval()
    with torch.no_grad():
        logits, gates = model(torch.from_numpy(x_test_np).to(device, non_blocking=use_cuda_loader))
        pred = logits.argmax(dim=1).cpu().numpy()
        gate_np = gates.cpu().numpy()

    expert_usage = {
        name: round(float(gate_np[:, index].mean()), 6) for index, name in enumerate(model.expert_names)
    }
    dominant_counts = np.bincount(gate_np.argmax(axis=1), minlength=len(model.expert_names))
    dominant_usage = {
        name: round(float(dominant_counts[index] / len(gate_np)), 6)
        for index, name in enumerate(model.expert_names)
    }
    gate_entropy = -(gate_np * np.log(np.clip(gate_np, 1e-8, 1.0))).sum(axis=1)
    extra_metrics: dict[str, Any] = {
        "torch_device": str(device),
        "torch_device_name": device_text,
        "torch_gpu_id": device.index if device.type == "cuda" else "",
        "discriminator_count": len(model.expert_names),
        "mean_gate_entropy": float(gate_entropy.mean()),
        "mean_personalization_strength": float(gate_np.max(axis=1).mean()),
        "expert_mean_weights": json.dumps(expert_usage, ensure_ascii=False),
        "dominant_expert_distribution": json.dumps(dominant_usage, ensure_ascii=False),
    }
    extra_metrics.update(split_metadata)
    metrics = save_eval_artifacts(output_dir, "torch_multi_discriminator", y_test, pred, extra_metrics)

    profile_path = output_dir / "device_expert_profiles.csv"
    save_device_expert_profiles(
        model,
        preprocessor,
        x,
        df,
        target_col,
        train_idx,
        test_idx,
        profile_path,
        device,
        batch_size=max(batch_size * 4, 2048),
    )

    artifact_path = output_dir / "torch_multi_discriminator.pt"
    torch.save(
        {
            "model_name": "DAMD-Net",
            "model_class": "DeviceAdaptiveMultiDiscriminator",
            "model_state_dict": model.cpu().state_dict(),
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "transformed_feature_names": feature_names.tolist(),
            "feature_groups": feature_groups,
            "discriminator_names": DISCRIMINATOR_NAMES,
            "n_classes": n_classes,
            "metrics": metrics,
            "torch_device_requested": torch_device,
            "gpu_id": gpu_id,
            "torch_device_used": str(device),
            "torch_device_name": device_text,
        },
        artifact_path,
    )
    return TrainResult("torch_multi_discriminator", metrics, artifact_path)


def save_metrics_summary(results: list[TrainResult], output_dir: Path) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"model": result.model_name, "artifact_path": str(result.artifact_path)}
        row.update(result.metrics)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["macro_f1", "high_risk_recall_label_ge_2"], ascending=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    return summary
