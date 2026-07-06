from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_label(score: pd.Series, group: pd.Series | None = None) -> pd.Series:
    if group is None:
        rank = score.rank(method="average", pct=True)
    else:
        rank = score.groupby(group, observed=True).rank(method="average", pct=True)
    return pd.cut(
        rank.fillna(0),
        bins=[-0.01, 0.80, 0.95, 0.99, 1.01],
        labels=[0, 1, 2, 3],
    ).astype(int)


def robust_rank_score(df: pd.DataFrame, columns: list[str], group_col: str | None = None) -> pd.Series:
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return pd.Series(np.zeros(len(df)), index=df.index)
    values = df[valid_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    ranks = values.rank(pct=True).fillna(0.5)
    if group_col and group_col in df.columns:
        for col in valid_cols:
            ranks[col] = values[col].groupby(df[group_col], observed=True).rank(pct=True).fillna(0.5)
    return ranks.mean(axis=1)


def make_device_cold_start_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    deviation_cols = [
        col
        for col in out.columns
        if col.endswith("_peer_z") or col.endswith("_diff_1d") or col.endswith("_roll7_std")
    ]
    for col in deviation_cols:
        out[f"{col}_abs_for_score"] = out[col].abs()
    deviation_score = robust_rank_score(out, [f"{col}_abs_for_score" for col in deviation_cols], "suggested_model_group")

    physical_cols = [
        "current_3phase_abs",
        "voltage_abs",
        "active_power_3phase_abs",
        "reactive_power_3phase_abs",
        "reactive_active_ratio",
    ]
    physical_score = robust_rank_score(out, physical_cols, "suggested_model_group")

    coverage = out.get("measurement_coverage_ratio", pd.Series(1.0, index=out.index)).fillna(0.0)
    data_quality_risk = 1.0 - coverage.clip(0, 1)
    low_pf_risk = pd.Series(0.0, index=out.index)
    if "power_factor_proxy" in out.columns:
        low_pf_risk = (0.9 - out["power_factor_proxy"]).clip(lower=0).fillna(0)

    raw_score = 0.45 * deviation_score + 0.35 * physical_score + 0.15 * data_quality_risk + 0.05 * low_pf_risk
    group = out["suggested_model_group"] if "suggested_model_group" in out.columns else None
    out["state_score"] = 100 * raw_score.groupby(group, observed=True).rank(pct=True) if group is not None else 100 * raw_score.rank(pct=True)
    out["state_label"] = percentile_label(out["state_score"], group)
    return out


def make_station_cold_start_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    risk_components = []
    component_cols = [
        "voltage_spread",
        "current_peak_ratio",
        "active_power_abs",
        "reactive_active_ratio",
        "switch_action_rate",
        "active_power_abs_diff_1d",
        "current_3phase_mean_diff_1d",
    ]
    risk_components.append(robust_rank_score(out, component_cols))
    if "operation_coverage_rate" in out.columns:
        risk_components.append(1.0 - out["operation_coverage_rate"].clip(0, 1).fillna(1.0))
    if "device_count_with_operation" in out.columns:
        risk_components.append(robust_rank_score(out, ["device_count_with_operation"]))
    raw_score = sum(risk_components) / len(risk_components)
    out["state_score"] = 100 * raw_score.rank(pct=True)
    out["state_label"] = percentile_label(out["state_score"])
    return out
