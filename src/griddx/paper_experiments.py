"""Controlled experiments for the DAMD-Net / HSF-Net paper."""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import modeling_columns
from .model_zoo import (
    DeviceAdaptiveMultiDiscriminator, SimpleMLP, build_discriminator_feature_groups,
    build_preprocessor, make_train_test_indices, resolve_torch_device,
)
from .station_fusion import StationHierarchicalFusionNetwork, build_station_feature_views


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    removed: str | None = None
    uniform: bool = False
    residual: bool = True
    regularization: bool = True
    risk_task: bool = True
    class_weights: bool = True
    mlp: bool = False


def specifications(kind: str) -> list[ExperimentSpec]:
    branches = ("operation", "history", "maintenance", "family_profile") if kind == "device" else (
        "operation", "history_maintenance", "infrastructure"
    )
    specs = [ExperimentSpec("full")]
    specs += [ExperimentSpec(f"without_{key}", removed=key) for key in branches]
    specs += [ExperimentSpec("uniform_fusion", uniform=True),
              ExperimentSpec("without_residual", residual=False),
              ExperimentSpec("without_gate_regularization", regularization=False),
              ExperimentSpec("without_class_weights", class_weights=False)]
    if kind == "station":
        specs.append(ExperimentSpec("without_risk_task", risk_task=False))
    specs.append(ExperimentSpec("mlp", mlp=True, risk_task=False))
    return specs


class DeviceAblationNetwork(DeviceAdaptiveMultiDiscriminator):
    def __init__(self, n_features: int, groups: dict[str, list[int]], spec: ExperimentSpec):
        super().__init__(n_features, 4, {k: v for k, v in groups.items() if k != spec.removed})
        self.spec = spec
        if spec.uniform:
            self.gate = None
        if not spec.residual:
            self.global_residual = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = (x.new_full((len(x), len(self.expert_names)), 1.0 / len(self.expert_names))
                   if self.spec.uniform else torch.softmax(self.gate(x) / self.gate_temperature, dim=1))
        outputs = torch.stack([self.discriminators[k](x[:, self.feature_groups[k]])
                               for k in self.expert_names], dim=1)
        logits = (weights.unsqueeze(-1) * outputs).sum(dim=1)
        if self.spec.residual:
            logits = logits + 0.20 * self.global_residual(x)
        return logits, weights


class StationAblationNetwork(StationHierarchicalFusionNetwork):
    def __init__(self, n_features: int, groups: dict[str, list[int]], spec: ExperimentSpec):
        super().__init__(n_features, 4, {k: v for k, v in groups.items() if k != spec.removed})
        self.spec = spec
        if spec.uniform:
            self.view_gate = None
        if not spec.residual:
            self.global_state_residual = None
        if not spec.risk_task:
            self.risk_head = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = (x.new_full((len(x), len(self.view_names)), 1.0 / len(self.view_names))
                   if self.spec.uniform else torch.softmax(self.view_gate(x) / self.gate_temperature, dim=1))
        embeddings = torch.stack([self.view_encoders[k](x[:, self.feature_views[k]])
                                   for k in self.view_names], dim=1)
        fused = self.fusion_norm((weights.unsqueeze(-1) * embeddings).sum(dim=1))
        logits = 0.50 * self.state_head(fused)
        if self.spec.residual:
            logits = logits + self.global_state_residual(x)
        risk = self.risk_head(fused).squeeze(1) if self.spec.risk_task else x.new_zeros(len(x))
        return logits, risk, weights


def grouped_partitions(table: pd.DataFrame, group_col: str) -> dict[str, np.ndarray]:
    outer, test, _ = make_train_test_indices(
        table, "state_label", random_state=42, split_strategy="group", group_col=group_col
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=17)
    train_local, validation_local = next(splitter.split(outer, groups=table.iloc[outer][group_col]))
    parts = {"train": outer[train_local], "validation": outer[validation_local], "test": test}
    group_sets = {k: set(table.iloc[v][group_col].astype(str)) for k, v in parts.items()}
    assert not (group_sets["train"] & group_sets["validation"])
    assert not (group_sets["train"] & group_sets["test"])
    assert not (group_sets["validation"] & group_sets["test"])
    assert len(np.concatenate(list(parts.values()))) == len(table)
    return parts


def classification_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    positive = y >= 2
    predicted_positive = prediction >= 2
    tp = int((positive & predicted_positive).sum())
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y, prediction, labels=[0, 1, 2, 3], weights="quadratic")),
        "risk_precision": tp / max(int(predicted_positive.sum()), 1),
        "risk_recall": tp / max(int(positive.sum()), 1),
    }


def forward(model: nn.Module, x: torch.Tensor, kind: str, spec: ExperimentSpec):
    if spec.mlp:
        return model(x), x.new_zeros(len(x)), x.new_ones((len(x), 1))
    if kind == "device":
        logits, weights = model(x)
        return logits, x.new_zeros(len(x)), weights
    return model(x)


def components(logits, y, risk, risk_target, weights, class_weight, kind, spec):
    # Unreduced losses permit exact weighted CE aggregation across evaluation batches.
    ce = nn.functional.cross_entropy(logits, y, weight=class_weight, reduction="none")
    denominator = class_weight[y].sum() if class_weight is not None else y.new_tensor(len(y)).float()
    state = ce.sum() / denominator
    risk_loss = nn.functional.smooth_l1_loss(risk, risk_target, beta=0.10)
    balance = (weights.mean(0) - 1.0 / weights.shape[1]).square().sum()
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1).mean()
    risk_factor = 0.35 if kind == "station" and spec.risk_task and not spec.mlp else 0.0
    balance_factor = (0.20 if kind == "device" else 0.15) if spec.regularization and not spec.mlp else 0.0
    entropy_factor = (0.005 if kind == "device" else 0.003) if spec.regularization and not spec.mlp else 0.0
    total = state + risk_factor * risk_loss + balance_factor * balance + entropy_factor * entropy
    return total, state, risk_loss, balance, entropy


def evaluate(model, x, y, risks, class_weight, kind, spec, device, batch_size=4096):
    model.eval()
    values = [[], [], []]
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            outputs = forward(model, torch.from_numpy(x[start:start + batch_size]).to(device), kind, spec)
            for bucket, value in zip(values, outputs):
                bucket.append(value.cpu())
        logits, risk, weights = [torch.cat(bucket) for bucket in values]
        losses = components(logits, torch.from_numpy(y), risk, torch.from_numpy(risks), weights,
                            None if class_weight is None else class_weight.cpu(), kind, spec)
    probabilities = torch.softmax(logits, 1).numpy()
    metrics = classification_metrics(y, probabilities.argmax(1))
    metrics.update(dict(zip(("loss", "ce", "risk_loss", "balance", "entropy"), [float(v) for v in losses])))
    if kind == "station" and spec.risk_task and not spec.mlp:
        error = risk.numpy() * 100 - risks * 100
        metrics["risk_mae"] = float(np.abs(error).mean())
        metrics["risk_rmse"] = float(np.sqrt(np.square(error).mean()))
    return metrics, probabilities, risk.numpy(), weights.numpy()


def fit_one(kind, spec, seed, arrays, groups, parts, table, group_col, args, root):
    run_dir = root / kind / spec.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_torch_device(args.torch_device, args.gpu_id)
    feature_count = arrays["train"][0].shape[1]
    model = (SimpleMLP(feature_count, 4) if spec.mlp else
             DeviceAblationNetwork(feature_count, groups, spec) if kind == "device" else
             StationAblationNetwork(feature_count, groups, spec)).to(device)
    train_x, train_y, train_risk = arrays["train"]
    count = np.bincount(train_y, minlength=4).astype("float32")
    weight = count.sum() / np.maximum(count, 1)
    weight /= weight.mean()
    class_weight = torch.tensor(weight, device=device) if spec.class_weights else None
    learning_rate = 1e-3 if spec.mlp else 8e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4 if spec.mlp else 2e-4)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_risk))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                        pin_memory=device.type == "cuda")
    history = []
    best_ce, best_epoch, best_state = float("inf"), 0, None
    started = time.perf_counter()
    epochs = args.device_epochs if kind == "device" else args.station_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by, br in loader:
            bx, by, br = bx.to(device), by.to(device), br.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, risk, weights = forward(model, bx, kind, spec)
            loss = components(logits, by, risk, br, weights, class_weight, kind, spec)[0]
            loss.backward()
            optimizer.step()
        row = {"epoch": epoch}
        for partition in ("train", "validation"):
            metrics, _, _, _ = evaluate(model, *arrays[partition], class_weight, kind, spec, device)
            row.update({f"{partition}_{k}": v for k, v in metrics.items()})
        history.append(row)
        validation_ce = row["validation_ce"]
        if validation_ce < best_ce:
            best_ce, best_epoch = validation_ce, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{kind}/{spec.name}/s{seed} epoch={epoch}/{epochs} "
                  f"trainCE={row['train_ce']:.4f} valCE={validation_ce:.4f} "
                  f"valF1={row['validation_macro_f1']:.4f}", flush=True)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    metrics, probabilities, risk, weights = evaluate(model, *arrays["test"], class_weight, kind, spec, device)
    y = arrays["test"][1]
    result = {
        "kind": kind, "variant": spec.name, "seed": seed, "best_epoch": best_epoch,
        "best_validation_ce": best_ce, "epochs": epochs, "train_seconds": elapsed,
        "parameter_count": sum(p.numel() for p in model.parameters()), "device": str(device),
        **metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "test_predictions.npz", y_true=y, probabilities=probabilities,
                        risk_target=arrays["test"][2] * 100, risk_prediction=risk * 100, weights=weights)
    if not spec.mlp:
        keys = model.expert_names if kind == "device" else model.view_names
        profiles = pd.DataFrame(weights, columns=keys)
        profiles[group_col] = table.iloc[parts["test"]][group_col].astype(str).to_numpy()
        profiles.groupby(group_col).mean().to_csv(run_dir / "test_object_weights.csv")
    torch.save({"state_dict": best_state, "spec": asdict(spec), "feature_groups": groups,
                "feature_count": feature_count, "seed": seed, "metrics": result}, run_dir / "checkpoint.pt")
    print(f"DONE {kind}/{spec.name}/s{seed}: testF1={metrics['macro_f1']:.4f} "
          f"recall={metrics['risk_recall']:.4f} bestEpoch={best_epoch} seconds={elapsed:.1f}", flush=True)
    return result


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_task(kind, args, root):
    directory = "device_prediction" if kind == "device" else "station_assessment"
    source = args.project_root / "outputs/baseline_models" / directory / f"{kind}_training_features.parquet"
    table = pd.read_parquet(source).reset_index(drop=True)
    group_col = "unified_device_id" if kind == "device" else "station_id"
    parts = grouped_partitions(table, group_col)
    numeric, categorical = modeling_columns(table, "state_label")
    preprocessor = build_preprocessor(numeric, categorical)
    preprocessor.fit(table.iloc[parts["train"]][numeric + categorical])
    arrays = {}
    for name, indices in parts.items():
        frame = table.iloc[indices]
        matrix = preprocessor.transform(frame[numeric + categorical]).astype("float32")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite transformed features in {kind}/{name}")
        arrays[name] = (matrix, frame["state_label"].to_numpy(dtype="int64"),
                        frame["state_score"].clip(0, 100).to_numpy(dtype="float32") / 100)
    names = preprocessor.get_feature_names_out().tolist()
    groups = (build_discriminator_feature_groups(names) if kind == "device" else build_station_feature_views(names))
    task_dir = root / kind
    task_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, task_dir / "preprocessor.joblib")
    partition_frame = table[[group_col, "date", "state_label"]].copy()
    partition_frame["partition"] = ""
    for partition, indices in parts.items():
        partition_frame.loc[indices, "partition"] = partition
    partition_frame.to_csv(task_dir / "partitions.csv", index=False)
    metadata = {
        "source": source.name, "sha256": file_hash(source), "rows": len(table),
        "object_count": int(table[group_col].nunique()), "feature_count": len(names),
        "numeric_count": len(numeric), "categorical_count": len(categorical),
        "feature_groups": {k: len(v) for k, v in groups.items()},
        "date_min": str(pd.Timestamp(table["date"].min()).date()),
        "date_max": str(pd.Timestamp(table["date"].max()).date()),
        "days_per_object_min": int(table.groupby(group_col).size().min()),
        "days_per_object_max": int(table.groupby(group_col).size().max()),
        "partitions": {k: {"rows": len(v), "objects": int(table.iloc[v][group_col].nunique()),
                              "class_counts": np.bincount(arrays[k][1], minlength=4).tolist()}
                       for k, v in parts.items()},
        "feature_names": names,
    }
    (task_dir / "data_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return table, group_col, parts, groups, arrays, metadata


def run_experiments(args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "platform": platform.platform(), "seeds": args.seeds, "outer_split_seed": 42,
        "inner_split_seed": 17, "validation_fraction_of_outer_training": 0.20,
        "checkpoint_selection": "minimum validation weighted cross-entropy; no test-based selection",
        "batch_size": args.batch_size, "device_epochs": args.device_epochs,
        "station_epochs": args.station_epochs, "threads": args.threads,
        "module_sha256": file_hash(Path(__file__)), "tasks": {},
    }
    results = []
    for kind in args.tasks:
        table, group_col, parts, groups, arrays, metadata = prepare_task(kind, args, root)
        manifest["tasks"][kind] = metadata
        for spec in specifications(kind):
            if args.variants and spec.name not in args.variants:
                continue
            for seed in args.seeds:
                results.append(fit_one(kind, spec, seed, arrays, groups, parts, table, group_col, args, root))
                pd.DataFrame(results).to_csv(root / "all_runs.csv", index=False)
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    frame = pd.DataFrame(results)
    summary = frame.groupby(["kind", "variant"])[[
        "accuracy", "macro_f1", "qwk", "risk_precision", "risk_recall", "risk_mae", "risk_rmse",
        "parameter_count", "train_seconds", "best_epoch",
    ]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(root / "summary.csv")
    return frame
