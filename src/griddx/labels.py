from __future__ import annotations

import numpy as np
import pandas as pd


DEVICE_ENRICHED_COLUMNS = {
    "history_defect_count",
    "history_trip_count",
    "history_defect_level_code",
    "history_trip_level_code",
    "defect_maintenance_status_code",
    "trip_maintenance_status_code",
    "family_history_defect_count",
    "family_history_trip_count",
}

STATION_ENRICHED_COLUMNS = {
    "history_defect_count",
    "history_trip_count",
    "history_maintenance_count",
}


class RankReference:
    def __init__(self, distributions: list[dict[str, np.ndarray]] | None = None) -> None:
        self.fitting = distributions is None
        self.distributions = [] if distributions is None else distributions
        self.position = 0

    def rank(self, values: pd.Series, group: pd.Series | None = None) -> pd.Series:
        values = values.astype(float).replace([np.inf, -np.inf], np.nan)
        keys = group.astype("string").fillna("__missing__") if group is not None else None
        if self.fitting:
            reference = {"all": np.sort(values.dropna().to_numpy())}
            if keys is not None:
                for key in keys.unique():
                    reference[f"group:{key}"] = np.sort(values[keys == key].dropna().to_numpy())
            self.distributions.append(reference)
            return values.rank(pct=True) if keys is None else values.groupby(keys).rank(pct=True)
        if self.position >= len(self.distributions):
            raise ValueError("Rank reference does not match the scoring formula.")
        reference = self.distributions[self.position]
        self.position += 1
        result = pd.Series(np.nan, index=values.index, dtype=float)
        groups = [(None, pd.Series(True, index=values.index))] if keys is None else [
            (key, keys == key) for key in keys.unique()
        ]
        for key, mask in groups:
            sample = reference.get(f"group:{key}", reference["all"]) if key is not None else reference["all"]
            valid = mask & values.notna()
            if len(sample) and valid.any():
                current = values.loc[valid].to_numpy()
                left = np.searchsorted(sample, current, side="left")
                right = np.searchsorted(sample, current, side="right")
                result.loc[valid] = np.clip((left + right + 1.0) / (2.0 * len(sample)), 0, 1)
        return result


def _rank(values: pd.Series, group: pd.Series | None, reference: RankReference | None) -> pd.Series:
    if reference is not None:
        return reference.rank(values, group)
    return values.rank(pct=True) if group is None else values.groupby(group, observed=True).rank(pct=True)


def percentile_label(score: pd.Series, group: pd.Series | None = None, reference: RankReference | None = None) -> pd.Series:
    rank = _rank(score, group, reference)
    return pd.cut(
        rank.fillna(0),
        bins=[-0.01, 0.80, 0.95, 0.99, 1.01],
        labels=[0, 1, 2, 3],
    ).astype(int)


def robust_rank_score(df: pd.DataFrame, columns: list[str], group_col: str | None = None, reference: RankReference | None = None) -> pd.Series:
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return pd.Series(np.zeros(len(df)), index=df.index)
    values = df[valid_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    group = df[group_col] if group_col and group_col in df.columns else None
    ranks = pd.DataFrame({col: _rank(values[col], group, reference).fillna(0.5) for col in valid_cols}, index=df.index)
    return ranks.mean(axis=1)


def has_device_enriched_fields(df: pd.DataFrame) -> bool:
    return DEVICE_ENRICHED_COLUMNS.issubset(df.columns)


def has_station_enriched_fields(df: pd.DataFrame) -> bool:
    return STATION_ENRICHED_COLUMNS.issubset(df.columns)


def make_device_cold_start_labels(df: pd.DataFrame, reference: RankReference | None = None) -> pd.DataFrame:
    out = df.copy()
    deviation_cols = [
        col
        for col in out.columns
        if col.endswith("_peer_z") or col.endswith("_diff_1d") or col.endswith("_roll7_std")
    ]
    for col in deviation_cols:
        out[f"{col}_abs_for_score"] = out[col].abs()
    deviation_score = robust_rank_score(out, [f"{col}_abs_for_score" for col in deviation_cols], "suggested_model_group", reference)

    physical_cols = [
        "current_3phase_abs",
        "voltage_abs",
        "active_power_3phase_abs",
        "reactive_power_3phase_abs",
        "reactive_active_ratio",
    ]
    physical_score = robust_rank_score(out, physical_cols, "suggested_model_group", reference)

    coverage = out.get("measurement_coverage_ratio", pd.Series(1.0, index=out.index)).fillna(0.0)
    data_quality_risk = 1.0 - coverage.clip(0, 1)
    low_pf_risk = pd.Series(0.0, index=out.index)
    if "power_factor_proxy" in out.columns:
        low_pf_risk = (0.9 - out["power_factor_proxy"]).clip(lower=0).fillna(0)

    raw_score = 0.45 * deviation_score + 0.35 * physical_score + 0.15 * data_quality_risk + 0.05 * low_pf_risk
    group = out["suggested_model_group"] if "suggested_model_group" in out.columns else None
    out["state_score"] = 100 * _rank(raw_score, group, reference)
    out["state_label"] = percentile_label(out["state_score"], group, reference)
    return out


def make_station_cold_start_labels(df: pd.DataFrame, reference: RankReference | None = None) -> pd.DataFrame:
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
    risk_components.append(robust_rank_score(out, component_cols, reference=reference))
    if "operation_coverage_rate" in out.columns:
        risk_components.append(1.0 - out["operation_coverage_rate"].clip(0, 1).fillna(1.0))
    if "device_count_with_operation" in out.columns:
        risk_components.append(robust_rank_score(out, ["device_count_with_operation"], reference=reference))
    raw_score = sum(risk_components) / len(risk_components)
    out["state_score"] = 100 * _rank(raw_score, None, reference)
    out["state_label"] = percentile_label(out["state_score"], reference=reference)
    return out


def make_device_enriched_weak_labels(df: pd.DataFrame, reference: RankReference | None = None) -> pd.DataFrame:
    if not has_device_enriched_fields(df):
        missing = sorted(DEVICE_ENRICHED_COLUMNS - set(df.columns))
        raise ValueError(f"Device enriched weak labels require columns: {missing}")

    out = df.copy()
    defect_status = out["defect_maintenance_status_code"].map({0: 0.0, 1: 0.15, 2: 1.0}).fillna(0.5)
    trip_status = out["trip_maintenance_status_code"].map({0: 0.0, 1: 0.15, 2: 1.0}).fillna(0.5)
    unresolved_risk = 0.55 * defect_status + 0.45 * trip_status

    severity_risk = (
        0.55 * out["history_defect_level_code"].clip(0, 4) / 4.0
        + 0.45 * out["history_trip_level_code"].clip(0, 4) / 4.0
    )
    history_risk = robust_rank_score(
        out,
        ["history_defect_count", "history_trip_count", "unresolved_defect_count", "unresolved_trip_count"],
        "suggested_model_group",
        reference,
    )
    family_risk = robust_rank_score(
        out,
        ["family_history_defect_count", "family_history_trip_count", "family_event_count_per_device"],
        "suggested_model_group",
        reference,
    )

    deviation_cols = [
        col for col in out.columns if col.endswith("_peer_z") or col.endswith("_diff_1d") or col.endswith("_roll7_std")
    ]
    deviation_values = out[deviation_cols].abs().copy() if deviation_cols else pd.DataFrame(index=out.index)
    deviation_values.columns = [f"{col}_weak_abs" for col in deviation_values.columns]
    operation_risk = robust_rank_score(deviation_values, list(deviation_values.columns), reference=reference)
    age_risk = robust_rank_score(out, ["device_age_days"], "suggested_model_group", reference)
    coverage = out.get("measurement_coverage_ratio", pd.Series(1.0, index=out.index)).fillna(0.0).clip(0, 1)
    quality_risk = 1.0 - coverage

    raw_score = (
        0.28 * unresolved_risk
        + 0.18 * severity_risk
        + 0.12 * history_risk
        + 0.12 * family_risk
        + 0.20 * operation_risk
        + 0.05 * age_risk
        + 0.05 * quality_risk
    )
    group = out.get("suggested_model_group")
    out["state_score"] = 100 * _rank(raw_score, group, reference)
    out["state_label"] = percentile_label(out["state_score"], group, reference)
    out["label_source"] = "enriched_weak_rule"
    return out


def make_station_enriched_weak_labels(df: pd.DataFrame, reference: RankReference | None = None) -> pd.DataFrame:
    if not has_station_enriched_fields(df):
        missing = sorted(STATION_ENRICHED_COLUMNS - set(df.columns))
        raise ValueError(f"Station enriched weak labels require columns: {missing}")

    out = df.copy()
    operating = make_station_cold_start_labels(out, reference)["state_score"] / 100.0
    history_risk = robust_rank_score(
        out,
        ["history_defect_count", "history_trip_count", "history_event_count", "history_events_per_transformer"],
        reference=reference,
    )
    unresolved_risk = robust_rank_score(out, ["unresolved_event_count_proxy"], reference=reference)
    environment_risk = robust_rank_score(
        out,
        ["lightning_risk_level_code", "ice_area_level_code", "environment_risk_level"],
        reference=reference,
    )
    coverage = out.get("operation_coverage_rate", pd.Series(1.0, index=out.index)).fillna(0.0).clip(0, 1)
    quality_risk = 1.0 - coverage
    raw_score = (
        0.30 * history_risk
        + 0.20 * unresolved_risk
        + 0.15 * environment_risk
        + 0.30 * operating
        + 0.05 * quality_risk
    )
    out["state_score"] = 100 * _rank(raw_score, None, reference)
    out["state_label"] = percentile_label(out["state_score"], reference=reference)
    out["label_source"] = "enriched_weak_rule"
    return out
