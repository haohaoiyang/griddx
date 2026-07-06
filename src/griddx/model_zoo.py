from __future__ import annotations

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
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import save_eval_artifacts


@dataclass
class TrainResult:
    model_name: str
    metrics: dict[str, float]
    artifact_path: Path


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
) -> list[TrainResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_models = selected_models or list(sklearn_model_specs(random_state))
    x = df[numeric_features + categorical_features].copy()
    y = df[target_col].astype(int).to_numpy()
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
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
        metrics = save_eval_artifacts(output_dir, name, y_test, pred)
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
) -> TrainResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = True
    x = df[numeric_features + categorical_features].copy()
    y = df[target_col].astype(int).to_numpy()
    n_classes = int(np.max(y)) + 1
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=stratify,
    )

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


def save_metrics_summary(results: list[TrainResult], output_dir: Path) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"model": result.model_name, "artifact_path": str(result.artifact_path)}
        row.update(result.metrics)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["macro_f1", "high_risk_recall_label_ge_2"], ascending=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    return summary
