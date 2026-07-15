from __future__ import annotations

import numpy as np
import pandas as pd

from .data import drop_empty_columns, parse_date_column


CORE_VALUE_COLUMNS = [
    "current_3phase",
    "voltage",
    "active_power_3phase",
    "reactive_power_3phase",
    "frequency",
    "tap_position",
    "switch_position",
]

PRESENT_COLUMNS = [
    "current_3phase_daily_present",
    "voltage_daily_present",
    "active_power_3phase_daily_present",
    "reactive_power_3phase_daily_present",
    "frequency_daily_present",
    "tap_position_daily_present",
    "switch_position_daily_present",
]

ID_COLUMNS = {
    "date",
    "unified_device_id",
    "device_name",
    "station_id",
    "station_name",
}

CATEGORICAL_COLUMNS = [
    "suggested_model_group",
    "station_id",
    "device_type_code",
    "manufacturer_code",
    "history_defect_type_code",
    "history_defect_reason_code",
    "history_trip_type_code",
    "history_trip_reason_code",
    "maintenance_relation_code",
    "family_history_defect_type_code",
    "family_history_defect_reason_code",
    "family_history_trip_type_code",
    "family_history_trip_reason_code",
    "station_type_code",
]


def _existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_date_column(df)
    if "date" not in df.columns:
        return df
    out = df.copy()
    min_date = out["date"].min()
    out["day_index"] = (out["date"] - min_date).dt.days.astype("float32")
    out["day_of_week"] = out["date"].dt.dayofweek.astype("float32")
    out["month"] = out["date"].dt.month.astype("float32")
    return out


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"active_power_3phase", "reactive_power_3phase"}.issubset(out.columns):
        active = out["active_power_3phase"].astype(float)
        reactive = out["reactive_power_3phase"].astype(float)
        apparent = np.sqrt(np.square(active) + np.square(reactive))
        out["apparent_power_proxy"] = apparent
        out["power_factor_proxy"] = np.where(apparent > 1e-6, np.abs(active) / apparent, np.nan)
        out["reactive_active_ratio"] = np.where(np.abs(active) > 1e-6, np.abs(reactive) / np.abs(active), np.nan)
    if {"active_power_3phase", "current_3phase"}.issubset(out.columns):
        out["active_current_ratio"] = np.where(
            np.abs(out["current_3phase"]) > 1e-6,
            out["active_power_3phase"] / out["current_3phase"],
            np.nan,
        )
    if {"voltage", "current_3phase"}.issubset(out.columns):
        out["voltage_current_product"] = out["voltage"] * out["current_3phase"]
    for col in _existing(out, CORE_VALUE_COLUMNS):
        out[f"{col}_abs"] = out[col].abs()
    present_cols = _existing(out, PRESENT_COLUMNS)
    if present_cols:
        out["measurement_coverage_ratio"] = out[present_cols].mean(axis=1)
    return out


def add_device_temporal_features(df: pd.DataFrame, windows: tuple[int, ...] = (7, 30)) -> pd.DataFrame:
    if "unified_device_id" not in df.columns or "date" not in df.columns:
        return df
    out = df.sort_values(["unified_device_id", "date"]).copy()
    value_cols = _existing(out, ["current_3phase", "voltage", "active_power_3phase", "reactive_power_3phase", "switch_position"])
    grouped = out.groupby("unified_device_id", sort=False)
    for col in value_cols:
        out[f"{col}_diff_1d"] = grouped[col].diff()
        for window in windows:
            roll = grouped[col].rolling(window=window, min_periods=2)
            out[f"{col}_roll{window}_mean"] = roll.mean().reset_index(level=0, drop=True)
            if window == 7:
                out[f"{col}_roll{window}_std"] = roll.std().reset_index(level=0, drop=True)
    return out


def add_station_peer_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "station_id", "suggested_model_group"}
    if not required.issubset(df.columns):
        return df
    out = df.copy()
    value_cols = _existing(out, ["current_3phase", "voltage", "active_power_3phase", "reactive_power_3phase"])
    peer_key = ["date", "station_id", "suggested_model_group"]
    for col in value_cols:
        peer_mean = out.groupby(peer_key, observed=True)[col].transform("mean")
        peer_std = out.groupby(peer_key, observed=True)[col].transform("std")
        out[f"{col}_peer_z"] = (out[col] - peer_mean) / peer_std.replace(0, np.nan)
    return out


def add_enriched_device_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"history_defect_count", "defect_related_maintenance_count"}.issubset(out.columns):
        defects = out["history_defect_count"].clip(lower=0)
        repaired = out["defect_related_maintenance_count"].clip(lower=0)
        out["unresolved_defect_count"] = (defects - repaired).clip(lower=0)
        out["defect_resolution_rate"] = np.where(defects > 0, repaired / defects, 1.0)
    if {"history_trip_count", "trip_related_maintenance_count"}.issubset(out.columns):
        trips = out["history_trip_count"].clip(lower=0)
        repaired = out["trip_related_maintenance_count"].clip(lower=0)
        out["unresolved_trip_count"] = (trips - repaired).clip(lower=0)
        out["trip_resolution_rate"] = np.where(trips > 0, repaired / trips, 1.0)
    if {"history_defect_count", "history_trip_count"}.issubset(out.columns):
        out["history_event_count"] = out["history_defect_count"].clip(lower=0) + out["history_trip_count"].clip(lower=0)
        out["history_event_count_log1p"] = np.log1p(out["history_event_count"])
    if {"history_defect_count", "history_defect_level_code"}.issubset(out.columns):
        out["history_defect_burden"] = out["history_defect_count"].clip(lower=0) * out[
            "history_defect_level_code"
        ].clip(lower=0)
    if {"history_trip_count", "history_trip_level_code"}.issubset(out.columns):
        out["history_trip_burden"] = out["history_trip_count"].clip(lower=0) * out[
            "history_trip_level_code"
        ].clip(lower=0)
    family_cols = {"family_history_defect_count", "family_history_trip_count", "family_device_count"}
    if family_cols.issubset(out.columns):
        family_events = out["family_history_defect_count"].clip(lower=0) + out["family_history_trip_count"].clip(lower=0)
        out["family_event_count_per_device"] = family_events / out["family_device_count"].replace(0, np.nan)
    if "device_age_days" in out.columns:
        out["device_age_years"] = out["device_age_days"].clip(lower=0) / 365.25
    return out


def add_enriched_station_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"history_defect_count", "history_trip_count"}.issubset(out.columns):
        out["history_event_count"] = out["history_defect_count"].clip(lower=0) + out["history_trip_count"].clip(lower=0)
        out["history_event_count_log1p"] = np.log1p(out["history_event_count"])
    if {"history_event_count", "history_maintenance_count"}.issubset(out.columns):
        maintenance = out["history_maintenance_count"].clip(lower=0)
        out["unresolved_event_count_proxy"] = (out["history_event_count"] - maintenance).clip(lower=0)
        out["maintenance_coverage_proxy"] = np.where(
            out["history_event_count"] > 0,
            np.minimum(maintenance / out["history_event_count"], 1.0),
            1.0,
        )
    if {"history_event_count", "main_transformer_count"}.issubset(out.columns):
        out["history_events_per_transformer"] = out["history_event_count"] / out["main_transformer_count"].replace(0, np.nan)
    if {"lightning_risk_level_code", "ice_area_level_code"}.issubset(out.columns):
        out["environment_risk_level"] = out[["lightning_risk_level_code", "ice_area_level_code"]].max(axis=1)
    if {"active_power_abs", "main_transformer_capacity_mva"}.issubset(out.columns):
        out["capacity_loading_proxy"] = out["active_power_abs"] / out["main_transformer_capacity_mva"].replace(0, np.nan)
    return out


def build_device_features(df: pd.DataFrame) -> pd.DataFrame:
    out = drop_empty_columns(df)
    out = add_calendar_features(out)
    out = add_physical_features(out)
    out = add_device_temporal_features(out)
    out = add_station_peer_features(out)
    out = add_enriched_device_features(out)
    return out


def build_station_features(df: pd.DataFrame) -> pd.DataFrame:
    out = drop_empty_columns(df)
    out = add_calendar_features(out)
    if {"voltage_max", "voltage_min"}.issubset(out.columns):
        out["voltage_spread"] = out["voltage_max"] - out["voltage_min"]
    if {"current_3phase_max", "current_3phase_mean"}.issubset(out.columns):
        out["current_peak_ratio"] = out["current_3phase_max"] / out["current_3phase_mean"].replace(0, np.nan)
    if {"active_power_3phase_sum", "reactive_power_3phase_sum"}.issubset(out.columns):
        out["active_power_abs"] = out["active_power_3phase_sum"].abs()
        out["reactive_active_ratio"] = out["reactive_power_3phase_sum"].abs() / out["active_power_3phase_sum"].abs().replace(0, np.nan)
    if {"switch_open_count", "switch_close_count", "device_count_with_operation"}.issubset(out.columns):
        out["switch_action_rate"] = (out["switch_open_count"] + out["switch_close_count"]) / out[
            "device_count_with_operation"
        ].replace(0, np.nan)
    if {"station_id", "date"}.issubset(out.columns):
        out = out.sort_values(["station_id", "date"]).copy()
        grouped = out.groupby("station_id", sort=False)
        for col in _existing(out, ["active_power_abs", "current_3phase_mean", "voltage_mean", "switch_action_rate"]):
            out[f"{col}_roll7_mean"] = grouped[col].rolling(window=7, min_periods=2).mean().reset_index(level=0, drop=True)
            out[f"{col}_diff_1d"] = grouped[col].diff()
    return add_enriched_station_features(out)


def modeling_columns(df: pd.DataFrame, target_col: str, extra_exclude: set[str] | None = None) -> tuple[list[str], list[str]]:
    exclude = set(ID_COLUMNS) | {target_col, "state_score", "state_label", "risk_score", "risk_level"}
    if extra_exclude:
        exclude |= extra_exclude
    categorical = [
        col
        for col in CATEGORICAL_COLUMNS
        if col in df.columns and col not in exclude and df[col].nunique(dropna=True) > 1
    ]
    numeric = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in exclude
        and col not in categorical
        and df[col].replace([np.inf, -np.inf], np.nan).notna().any()
        and df[col].nunique(dropna=True) > 1
    ]
    return numeric, categorical
