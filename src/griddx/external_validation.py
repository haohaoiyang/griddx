from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .data import read_csv_limited
from .evaluation import STATE_NAMES
from .features import build_device_features
from .labels import DEVICE_ENRICHED_COLUMNS, make_device_enriched_weak_labels
from .model_zoo import DeviceAdaptiveMultiDiscriminator, SimpleMLP, describe_torch_device, resolve_torch_device


SKLEARN_MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
]
TORCH_MODEL_NAMES = ["torch_mlp", "torch_multi_discriminator"]
PROXY_LABEL_REQUIRED_COLUMNS = sorted(
    set(DEVICE_ENRICHED_COLUMNS)
    | {
        "defect_related_maintenance_count",
        "trip_related_maintenance_count",
        "family_device_count",
    }
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize type {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _to_device_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__missing_device__")


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Return quantile-binned PSI; NaN means the reference is not informative."""
    ref = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(ref) < 20 or len(cur) < 20:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    cut_points = np.unique(np.quantile(ref, quantiles)[1:-1])
    if len(cut_points) == 0:
        return float("nan")
    edges = np.concatenate(([-np.inf], cut_points, [np.inf]))
    ref_rate = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_rate = np.histogram(cur, bins=edges)[0] / len(cur)
    epsilon = 1e-6
    return float(np.sum((cur_rate - ref_rate) * np.log((cur_rate + epsilon) / (ref_rate + epsilon))))


def psi_level(psi: float) -> str:
    if not np.isfinite(psi):
        return "not_available"
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate_drift"
    return "major_drift"


def build_proxy_reference_labels(features: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Build weak-rule reference labels only for rows with a complete risk profile."""
    missing_columns = [col for col in PROXY_LABEL_REQUIRED_COLUMNS if col not in features.columns]
    labels = pd.Series(np.nan, index=features.index, dtype="float64")
    if missing_columns:
        return labels, pd.Series(False, index=features.index), {
            "label_source": "unavailable",
            "missing_proxy_columns": missing_columns,
            "reference_rows": 0,
            "excluded_rows": int(len(features)),
        }

    eligible = features[PROXY_LABEL_REQUIRED_COLUMNS].notna().all(axis=1)
    if eligible.any():
        weak_labels = make_device_enriched_weak_labels(features.loc[eligible].copy())
        labels.loc[eligible] = weak_labels["state_label"].astype(float)
    return labels, eligible, {
        "label_source": "proxy_enriched_weak_rule",
        "missing_proxy_columns": [],
        "reference_rows": int(eligible.sum()),
        "excluded_rows": int((~eligible).sum()),
        "excluded_rate": float((~eligible).mean()),
        "warning": "代理标签由当前数据的历史、检修、家族和运行字段按既有弱监督规则生成，不是真实故障结果。",
    }


def resolve_validation_labels(features: pd.DataFrame, target_col: str | None) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    if not target_col:
        return build_proxy_reference_labels(features)
    if target_col not in features.columns:
        raise ValueError(f"Target column not found in validation data: {target_col}")

    values = pd.to_numeric(features[target_col], errors="coerce")
    valid = values.notna() & values.between(0, 3)
    invalid_count = int((values.notna() & ~values.between(0, 3)).sum())
    labels = values.where(valid)
    return labels, valid, {
        "label_source": f"real_target:{target_col}",
        "reference_rows": int(valid.sum()),
        "excluded_rows": int((~valid).sum()),
        "excluded_rate": float((~valid).mean()),
        "invalid_label_rows": invalid_count,
        "warning": "该列被作为真实外部标签使用；请确认其产生时间晚于模型可用特征时间。",
    }


def resolve_device_scope(
    features: pd.DataFrame,
    profile_path: Path,
    partition: str = "test",
    device_col: str = "unified_device_id",
) -> tuple[pd.Series, dict[str, Any], set[str]]:
    if device_col not in features.columns:
        return pd.Series(True, index=features.index), {
            "scope": "all_rows_no_device_id",
            "warning": f"验证数据缺少 {device_col}，不能按训练分组选择设备。",
        }, set()

    device_ids = _to_device_id(features[device_col])
    all_devices = set(device_ids.unique().tolist())
    if partition == "all" or not profile_path.exists():
        warning = "包含训练设备；只能作为部署兼容性检查。" if partition == "all" else "没有设备分组文件；只能作为部署兼容性检查。"
        return pd.Series(True, index=features.index), {
            "scope": "all_devices",
            "validation_device_count": len(all_devices),
            "training_device_overlap_count": None,
            "warning": warning,
        }, set()

    profiles = pd.read_csv(profile_path)
    required = {device_col, "data_partition"}
    if not required.issubset(profiles.columns):
        return pd.Series(True, index=features.index), {
            "scope": "all_devices_missing_partition_columns",
            "validation_device_count": len(all_devices),
            "training_device_overlap_count": None,
            "warning": "设备分组文件缺少必要字段；只能作为部署兼容性检查。",
        }, set()

    profile_ids = _to_device_id(profiles[device_col])
    test_ids = set(profile_ids[profiles["data_partition"] == "test"].tolist())
    train_ids = set(profile_ids[profiles["data_partition"] == "train"].tolist())
    selected = device_ids.isin(test_ids)
    selected_devices = set(device_ids[selected].unique().tolist())
    return selected, {
        "scope": "saved_group_test_devices",
        "partition_file": profile_path.name,
        "recorded_test_device_count": len(test_ids),
        "recorded_train_device_count": len(train_ids),
        "validation_device_count": len(selected_devices),
        "validation_devices_missing_from_external_data": int(len(test_ids - all_devices)),
        "external_devices_not_in_partition_file": int(len(all_devices - (test_ids | train_ids))),
        "training_device_overlap_count": int(len(selected_devices & train_ids)),
        "warning": "仅选择保存的 group split 测试设备，避免与该次训练的设备 ID 重叠。",
    }, train_ids


def inspect_panel_overlap(external: pd.DataFrame, reference_csv: Path | None) -> dict[str, Any]:
    if reference_csv is None or not reference_csv.exists():
        return {"available": False, "reason": "reference_csv_not_found"}
    required = ["unified_device_id", "date"]
    if not set(required).issubset(external.columns):
        return {"available": False, "reason": "external_missing_panel_columns"}

    reference = read_csv_limited(reference_csv, usecols=required)
    if not set(required).issubset(reference.columns):
        return {"available": False, "reason": "reference_missing_panel_columns"}

    external_keys = pd.MultiIndex.from_frame(external[required].astype("string"))
    reference_keys = pd.MultiIndex.from_frame(reference[required].astype("string"))
    external_devices = set(external["unified_device_id"].dropna().astype("string"))
    reference_devices = set(reference["unified_device_id"].dropna().astype("string"))
    external_dates = set(pd.to_datetime(external["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d"))
    reference_dates = set(pd.to_datetime(reference["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d"))
    matching_key_rate = float(external_keys.isin(reference_keys).mean()) if len(external_keys) else float("nan")
    return {
        "available": True,
        "reference_csv": reference_csv.name,
        "external_device_count": len(external_devices),
        "reference_device_count": len(reference_devices),
        "device_overlap_rate": float(len(external_devices & reference_devices) / max(len(external_devices), 1)),
        "external_date_count": len(external_dates),
        "reference_date_count": len(reference_dates),
        "date_overlap_rate": float(len(external_dates & reference_dates) / max(len(external_dates), 1)),
        "matching_device_date_rate": matching_key_rate,
    }


def _extract_preprocessor_components(preprocessor: Any, categorical_features: list[str]) -> tuple[Any, Any]:
    cat_transformer = preprocessor.named_transformers_.get("cat")
    if cat_transformer is None:
        raise ValueError("The model preprocessor has no categorical transformer.")
    return cat_transformer.named_steps["imputer"], cat_transformer.named_steps["onehot"]


def categorical_oov_table(
    features: pd.DataFrame,
    preprocessor: Any,
    categorical_features: list[str],
) -> pd.DataFrame:
    imputer, onehot = _extract_preprocessor_components(preprocessor, categorical_features)
    rows: list[dict[str, Any]] = []
    for index, column in enumerate(categorical_features):
        if column not in features.columns:
            rows.append(
                {
                    "feature": column,
                    "status": "missing_feature",
                    "validation_rows": int(len(features)),
                    "missing_rows": int(len(features)),
                    "oov_rows": int(len(features)),
                    "oov_rate": 1.0,
                    "train_category_count": 0,
                    "validation_unique_count": 0,
                }
            )
            continue
        values = features[column].copy()
        missing = values.isna()
        fill_value = imputer.statistics_[index]
        filled = values.where(~missing, fill_value)
        train_categories = np.asarray(onehot.categories_[index])
        known = filled.isin(train_categories)
        rows.append(
            {
                "feature": column,
                "status": "ok",
                "validation_rows": int(len(values)),
                "missing_rows": int(missing.sum()),
                "oov_rows": int((~known).sum()),
                "oov_rate": float((~known).mean()),
                "train_category_count": int(len(train_categories)),
                "validation_unique_count": int(filled.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("oov_rate", ascending=False, na_position="last")


def numeric_drift_table(
    reference_features: pd.DataFrame | None,
    validation_features: pd.DataFrame,
    numeric_features: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in numeric_features:
        if column not in validation_features.columns:
            rows.append({"feature": column, "status": "missing_feature", "psi": float("nan"), "psi_level": "not_available"})
            continue
        if reference_features is None or column not in reference_features.columns:
            rows.append({"feature": column, "status": "reference_unavailable", "psi": float("nan"), "psi_level": "not_available"})
            continue
        reference = pd.to_numeric(reference_features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        current = pd.to_numeric(validation_features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        psi = population_stability_index(reference, current)
        rows.append(
            {
                "feature": column,
                "status": "ok",
                "reference_non_null": int(reference.notna().sum()),
                "validation_non_null": int(current.notna().sum()),
                "reference_missing_rate": float(reference.isna().mean()),
                "validation_missing_rate": float(current.isna().mean()),
                "missing_rate_change": float(current.isna().mean() - reference.isna().mean()),
                "reference_median": float(reference.median()) if reference.notna().any() else float("nan"),
                "validation_median": float(current.median()) if current.notna().any() else float("nan"),
                "psi": psi,
                "psi_level": psi_level(psi),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_validation_schema(model_dir: Path) -> tuple[Any, list[str], list[str], str]:
    for name in ["torch_multi_discriminator", "torch_mlp"]:
        path = model_dir / f"{name}.pt"
        if path.exists():
            artifact = _torch_load(path)
            return artifact["preprocessor"], artifact["numeric_features"], artifact["categorical_features"], name
    for name in SKLEARN_MODEL_NAMES:
        path = model_dir / f"{name}.joblib"
        if path.exists():
            pipeline = joblib.load(path)
            preprocessor = pipeline.named_steps["preprocess"]
            numeric = list(preprocessor.transformers_[0][2])
            categorical = list(preprocessor.transformers_[1][2])
            return preprocessor, numeric, categorical, name
    raise FileNotFoundError(f"No saved device model artifacts found in {model_dir}")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> dict[str, Any]:
    labels = list(range(n_classes))
    high_true = y_true >= 2
    high_pred = y_pred >= 2
    high_precision, high_recall, high_f1, _ = precision_recall_fscore_support(
        high_true, high_pred, average="binary", zero_division=0
    )
    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError:
        qwk = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": qwk,
        "high_risk_precision_label_ge_2": float(high_precision),
        "high_risk_recall_label_ge_2": float(high_recall),
        "high_risk_f1_label_ge_2": float(high_f1),
        "high_risk_support_label_ge_2": int(high_true.sum()),
        "majority_class_accuracy": float(np.max(np.bincount(y_true, minlength=n_classes)) / len(y_true)),
    }


def _iter_batches(frame: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


def _probability_summary(
    probabilities: np.ndarray,
    classes: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.arange(probabilities.shape[1]) if classes is None else np.asarray(classes)
    prediction = classes[probabilities.argmax(axis=1)] if prediction is None else np.asarray(prediction)
    class_positions = {int(label): index for index, label in enumerate(classes)}
    prediction_positions = np.asarray([class_positions.get(int(label), -1) for label in prediction])
    confidence = probabilities.max(axis=1)
    known_prediction = prediction_positions >= 0
    confidence[known_prediction] = probabilities[np.arange(len(probabilities))[known_prediction], prediction_positions[known_prediction]]
    high_risk = probabilities[:, classes >= 2].sum(axis=1) if (classes >= 2).any() else np.zeros(len(prediction))
    return prediction, confidence, high_risk


def predict_sklearn_model(pipeline: Any, features: pd.DataFrame, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    high_risks: list[np.ndarray] = []
    for batch in _iter_batches(features, batch_size):
        predicted_labels = np.asarray(pipeline.predict(batch)).astype(int)
        probabilities = np.asarray(pipeline.predict_proba(batch))
        classes = np.asarray(getattr(pipeline, "classes_", np.arange(probabilities.shape[1])))
        prediction, confidence, high_risk = _probability_summary(probabilities, classes, predicted_labels)
        predictions.append(prediction)
        confidences.append(confidence)
        high_risks.append(high_risk)
    return np.concatenate(predictions), np.concatenate(confidences), np.concatenate(high_risks)


def predict_torch_model(
    checkpoint: dict[str, Any],
    model_name: str,
    features: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    preprocessor = checkpoint["preprocessor"]
    n_features = len(checkpoint["transformed_feature_names"]) if model_name == "torch_multi_discriminator" else None
    if model_name == "torch_multi_discriminator":
        model = DeviceAdaptiveMultiDiscriminator(
            n_features=n_features or len(preprocessor.get_feature_names_out()),
            n_classes=int(checkpoint["n_classes"]),
            feature_groups=checkpoint["feature_groups"],
        )
    else:
        model = SimpleMLP(
            n_features=len(preprocessor.get_feature_names_out()),
            n_classes=int(checkpoint["n_classes"]),
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    high_risks: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _iter_batches(features, batch_size):
            transformed = preprocessor.transform(batch).astype("float32")
            batch_tensor = torch.from_numpy(transformed).to(device, non_blocking=device.type == "cuda")
            if model_name == "torch_multi_discriminator":
                logits, gate_weights = model(batch_tensor)
                gates.append(gate_weights.cpu().numpy())
            else:
                logits = model(batch_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            prediction, confidence, high_risk = _probability_summary(probabilities)
            predictions.append(prediction)
            confidences.append(confidence)
            high_risks.append(high_risk)
    return (
        np.concatenate(predictions),
        np.concatenate(confidences),
        np.concatenate(high_risks),
        np.concatenate(gates) if gates else None,
    )


def make_device_prediction_summary(
    features: pd.DataFrame,
    model_name: str,
    predictions: np.ndarray,
    confidences: np.ndarray,
    high_risks: np.ndarray,
    reference_labels: np.ndarray | None,
    gates: np.ndarray | None = None,
    device_col: str = "unified_device_id",
) -> pd.DataFrame:
    if device_col in features.columns:
        device_ids = _to_device_id(features[device_col])
    else:
        device_ids = pd.Series("all_devices", index=features.index, dtype="string")
    frame = pd.DataFrame(
        {
            device_col: device_ids.to_numpy(),
            "predicted_label": predictions,
            "prediction_confidence": confidences,
            "high_risk_probability": high_risks,
        }
    )
    if reference_labels is not None:
        frame["reference_label"] = reference_labels
        frame["reference_high_risk"] = (reference_labels >= 2).astype(int)
    if gates is not None:
        for index, name in enumerate(["operation", "history", "maintenance", "family_profile"]):
            frame[f"expert_weight_{name}"] = gates[:, index]

    aggregations: dict[str, Any] = {
        "predicted_label": "mean",
        "prediction_confidence": "mean",
        "high_risk_probability": "mean",
    }
    if reference_labels is not None:
        aggregations["reference_label"] = "mean"
        aggregations["reference_high_risk"] = "mean"
    if gates is not None:
        aggregations.update({column: "mean" for column in frame.columns if column.startswith("expert_weight_")})
    summary = frame.groupby(device_col, as_index=False).agg(aggregations)
    summary["sample_count"] = frame.groupby(device_col).size().reindex(summary[device_col]).to_numpy()
    summary.insert(0, "model", model_name)
    return summary


def _load_training_feature_snapshot(model_dir: Path, train_device_ids: set[str]) -> tuple[pd.DataFrame | None, str | None]:
    path = model_dir / "device_training_features.parquet"
    if not path.exists():
        return None, "training_feature_snapshot_not_found"
    try:
        table = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - environment dependent parquet engine
        return None, f"cannot_read_training_feature_snapshot:{type(exc).__name__}"
    if train_device_ids and "unified_device_id" in table.columns:
        ids = _to_device_id(table["unified_device_id"])
        table = table.loc[ids.isin(train_device_ids)].copy()
    return table, None


def _prediction_distribution(model_name: str, predictions: np.ndarray, n_classes: int) -> pd.DataFrame:
    counts = np.bincount(predictions, minlength=n_classes)
    return pd.DataFrame(
        {
            "model": model_name,
            "state_label": np.arange(n_classes),
            "state_name": [STATE_NAMES.get(index, str(index)) for index in range(n_classes)],
            "prediction_count": counts,
            "prediction_rate": counts / max(len(predictions), 1),
        }
    )


def run_external_device_validation(
    csv_path: Path,
    model_dir: Path,
    output_dir: Path,
    reference_csv: Path | None = None,
    target_col: str | None = None,
    device_partition: str = "test",
    max_rows: int = 0,
    batch_size: int = 8192,
    torch_device: str = "auto",
    gpu_id: int | None = None,
) -> pd.DataFrame:
    """Validate saved device models on a new table and save reproducible artifacts."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_csv_limited(csv_path, max_rows=max_rows)
    features = build_device_features(raw)
    preprocessor, numeric_features, categorical_features, schema_model = load_validation_schema(model_dir)
    model_features = numeric_features + categorical_features
    missing_model_features = [column for column in model_features if column not in features.columns]

    labels, label_mask, label_metadata = resolve_validation_labels(features, target_col)
    profile_path = model_dir / "device_expert_profiles.csv"
    scope_mask, scope_metadata, train_device_ids = resolve_device_scope(
        features, profile_path, partition=device_partition
    )
    scoring_mask = scope_mask & label_mask
    scope_features = features.loc[scope_mask].copy()
    scope_reference_labels = labels.loc[scope_mask].to_numpy()
    scoring_within_scope = label_mask.loc[scope_mask].to_numpy()
    scoring_labels = scope_reference_labels[scoring_within_scope].astype(int) if scoring_within_scope.any() else None

    input_schema = {
        "schema_model": schema_model,
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "missing_model_features": missing_model_features,
        "schema_compatible": not missing_model_features,
    }
    category_drift = categorical_oov_table(scope_features, preprocessor, categorical_features)
    training_features, training_snapshot_note = _load_training_feature_snapshot(model_dir, train_device_ids)
    numeric_drift = numeric_drift_table(training_features, scope_features, numeric_features)

    panel_overlap = inspect_panel_overlap(raw, reference_csv)
    date_values = pd.to_datetime(raw.get("date"), errors="coerce") if "date" in raw else pd.Series(dtype="datetime64[ns]")
    data_quality = {
        "input_csv": csv_path.name,
        "raw_row_count": int(len(raw)),
        "raw_column_count": int(len(raw.columns)),
        "feature_column_count": int(len(features.columns)),
        "device_count": int(_to_device_id(raw["unified_device_id"]).nunique()) if "unified_device_id" in raw else None,
        "station_count": int(raw["station_id"].astype("string").nunique()) if "station_id" in raw else None,
        "date_start": str(date_values.min().date()) if date_values.notna().any() else None,
        "date_end": str(date_values.max().date()) if date_values.notna().any() else None,
        "date_count": int(date_values.nunique()) if date_values.notna().any() else 0,
        "scope_row_count": int(scope_mask.sum()),
        "scoring_row_count": int(scoring_mask.sum()),
        "training_snapshot_note": training_snapshot_note,
        "label": label_metadata,
        "device_scope": scope_metadata,
        "panel_overlap": panel_overlap,
        "input_schema": input_schema,
        "category_oov_summary": {
            "features_with_any_oov": int((category_drift["oov_rate"] > 0).sum()),
            "features_with_oov_rate_ge_0_25": int((category_drift["oov_rate"] >= 0.25).sum()),
            "max_oov_rate": float(category_drift["oov_rate"].max()) if not category_drift.empty else 0.0,
        },
        "numeric_drift_summary": {
            "features_with_major_drift": int((numeric_drift["psi_level"] == "major_drift").sum()),
            "features_with_moderate_or_major_drift": int(
                numeric_drift["psi_level"].isin(["moderate_drift", "major_drift"]).sum()
            ),
        },
    }
    _write_json(output_dir / "external_validation_data_quality.json", data_quality)
    category_drift.to_csv(output_dir / "external_validation_categorical_oov.csv", index=False)
    numeric_drift.to_csv(output_dir / "external_validation_numeric_drift.csv", index=False)

    if missing_model_features:
        raise ValueError(f"Validation data is missing saved-model features: {missing_model_features}")
    if scope_features.empty:
        raise ValueError("No rows remain after the requested device-partition filter.")

    device = resolve_torch_device(torch_device=torch_device, gpu_id=gpu_id)
    metric_rows: list[dict[str, Any]] = []
    distributions: list[pd.DataFrame] = []
    device_summaries: list[pd.DataFrame] = []
    artifact_notes: list[dict[str, str]] = []
    n_classes = 4

    for model_name in SKLEARN_MODEL_NAMES:
        artifact_path = model_dir / f"{model_name}.joblib"
        if not artifact_path.exists():
            artifact_notes.append({"model": model_name, "status": "artifact_not_found"})
            continue
        pipeline = joblib.load(artifact_path)
        predictions, confidences, high_risks = predict_sklearn_model(pipeline, scope_features, batch_size)
        distributions.append(_prediction_distribution(model_name, predictions, n_classes))
        device_summaries.append(
            make_device_prediction_summary(
                scope_features,
                model_name,
                predictions,
                confidences,
                high_risks,
                scope_reference_labels if label_mask.any() else None,
            )
        )
        row: dict[str, Any] = {
            "model": model_name,
            "artifact": artifact_path.name,
            "scoring_type": label_metadata["label_source"],
            "scope_rows": int(len(scope_features)),
            "scoring_rows": int(scoring_within_scope.sum()),
            "torch_device": "",
        }
        if scoring_labels is not None and scoring_within_scope.any():
            score_predictions = predictions[scoring_within_scope]
            row.update(classification_metrics(scoring_labels, score_predictions, n_classes=n_classes))
            cm = confusion_matrix(scoring_labels, score_predictions, labels=list(range(n_classes)))
            pd.DataFrame(cm, index=[STATE_NAMES[index] for index in range(n_classes)], columns=[STATE_NAMES[index] for index in range(n_classes)]).to_csv(
                output_dir / f"external_validation_{model_name}_confusion_matrix.csv"
            )
        metric_rows.append(row)

    for model_name in TORCH_MODEL_NAMES:
        artifact_path = model_dir / f"{model_name}.pt"
        if not artifact_path.exists():
            artifact_notes.append({"model": model_name, "status": "artifact_not_found"})
            continue
        checkpoint = _torch_load(artifact_path)
        predictions, confidences, high_risks, gates = predict_torch_model(
            checkpoint, model_name, scope_features, device, batch_size
        )
        distributions.append(_prediction_distribution(model_name, predictions, n_classes))
        device_summaries.append(
            make_device_prediction_summary(
                scope_features,
                model_name,
                predictions,
                confidences,
                high_risks,
                scope_reference_labels if label_mask.any() else None,
                gates=gates,
            )
        )
        row = {
            "model": model_name,
            "artifact": artifact_path.name,
            "scoring_type": label_metadata["label_source"],
            "scope_rows": int(len(scope_features)),
            "scoring_rows": int(scoring_within_scope.sum()),
            "torch_device": describe_torch_device(device),
        }
        if scoring_labels is not None and scoring_within_scope.any():
            score_predictions = predictions[scoring_within_scope]
            row.update(classification_metrics(scoring_labels, score_predictions, n_classes=n_classes))
            cm = confusion_matrix(scoring_labels, score_predictions, labels=list(range(n_classes)))
            pd.DataFrame(cm, index=[STATE_NAMES[index] for index in range(n_classes)], columns=[STATE_NAMES[index] for index in range(n_classes)]).to_csv(
                output_dir / f"external_validation_{model_name}_confusion_matrix.csv"
            )
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty and "macro_f1" in metrics:
        metrics = metrics.sort_values(["macro_f1", "high_risk_recall_label_ge_2"], ascending=False, na_position="last")
    metrics.to_csv(output_dir / "external_validation_metrics.csv", index=False)
    if distributions:
        pd.concat(distributions, ignore_index=True).to_csv(output_dir / "external_validation_prediction_distribution.csv", index=False)
    if device_summaries:
        pd.concat(device_summaries, ignore_index=True).to_csv(output_dir / "external_validation_device_summary.csv", index=False)
    if artifact_notes:
        pd.DataFrame(artifact_notes).to_csv(output_dir / "external_validation_artifact_notes.csv", index=False)
    return metrics
