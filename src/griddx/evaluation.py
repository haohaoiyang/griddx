from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


STATE_NAMES = {
    0: "normal",
    1: "watch",
    2: "abnormal",
    3: "high_risk",
}


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    high_mask = y_true >= 2
    if high_mask.any():
        high_recall = float(((y_pred >= 2) & high_mask).sum() / high_mask.sum())
    else:
        high_recall = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "high_risk_recall_label_ge_2": high_recall,
    }


def save_eval_artifacts(
    output_dir: Path,
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_classification(y_true, y_pred)
    if extra_metrics:
        metrics.update(extra_metrics)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(output_dir / f"{name}_classification_report.csv")

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=[STATE_NAMES.get(i, i) for i in labels], columns=[STATE_NAMES.get(i, i) for i in labels]).to_csv(
        output_dir / f"{name}_confusion_matrix.csv"
    )
    return metrics
