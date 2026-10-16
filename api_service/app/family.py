from __future__ import annotations

import hashlib
import math
import re
import statistics
from datetime import date
from typing import Any, Mapping

import pandas as pd


BATCH_COLUMNS = ("生产批次", "设备批次", "批次", "出厂批次")
FAMILY_ID_PREFIX = "FAM-"
FAMILY_RULE_NAME = "Family Defect Z-Score"
FAMILY_RULE_VERSION = "family-zscore-1"
FAMILY_KEY_DEFINITION = "manufacturer + model + batch"

_MISSING_MARKERS = {
    "",
    "-",
    "/",
    "0",
    "NULL",
    "NONE",
    "N/A",
    "NA",
    "未知",
    "不详",
    "资料不详",
    "其他",
    "其它",
    "无",
}


def clean_family_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_family_component(value: Any) -> str:
    text = clean_family_text(value)
    normalized = re.sub(r"[\s（）()]+", "", text).upper()
    if normalized in _MISSING_MARKERS or "不详" in normalized:
        return ""
    return normalized


def derive_production_batch(row: Mapping[str, Any]) -> tuple[str, str]:
    for column in BATCH_COLUMNS:
        value = normalize_family_component(row.get(column, ""))
        if value:
            return clean_family_text(row.get(column, "")), column

    manufacture_date = clean_family_text(row.get("出厂日期", ""))
    match = re.search(r"(19|20)\d{2}[-/.年]?(\d{1,2})?", manufacture_date)
    if not match:
        return "", ""
    year = match.group(0)[:4]
    month = match.group(2)
    if month:
        month_number = int(month)
        if 1 <= month_number <= 12:
            return f"{year}{month_number:02d}", "出厂日期(YYYYMM推导)"
    return year, "出厂日期(YYYY推导)"


def make_family_id(manufacturer: Any, model: Any, batch: Any) -> str:
    normalized = [
        normalize_family_component(manufacturer),
        normalize_family_component(model),
        normalize_family_component(batch),
    ]
    if not all(normalized):
        return ""
    digest = hashlib.sha256("|".join(normalized).encode("utf-8")).hexdigest()
    return f"{FAMILY_ID_PREFIX}{digest[:16].upper()}"


def build_family_alerts(
    family_frame: pd.DataFrame,
    predictions: list[dict[str, Any]],
    *,
    assessment_date: date,
    level: str | None = None,
    limit: int = 20,
    minimum_family_size: int = 3,
) -> dict[str, Any]:
    required = {
        "unified_device_id",
        "family_id",
        "family_manufacturer",
        "family_model",
        "family_batch",
        "family_batch_source",
        "family_device_count",
        "family_defect_device_count",
        "family_history_defect_count",
    }
    missing = sorted(required - set(family_frame.columns))
    if missing:
        raise ValueError(f"Family feature columns are missing: {missing}")

    selected = family_frame.copy()
    selected["unified_device_id"] = selected["unified_device_id"].astype(str)
    selected = selected.drop_duplicates("unified_device_id", keep="last")
    for column in (
        "family_id",
        "family_manufacturer",
        "family_model",
        "family_batch",
        "family_batch_source",
    ):
        selected[column] = selected[column].fillna("").astype(str).str.strip()
    selected = selected[
        (selected["family_id"] != "")
        & (selected["family_manufacturer"] != "")
        & (selected["family_model"] != "")
        & (selected["family_batch"] != "")
    ]

    prediction_by_device = {
        str(item["device_id"]): item for item in predictions if item.get("device_id")
    }
    families: list[dict[str, Any]] = []
    group_columns = [
        "family_id",
        "family_manufacturer",
        "family_model",
        "family_batch",
        "family_batch_source",
    ]
    for key, group in selected.groupby(group_columns, dropna=False, sort=False):
        family_id, manufacturer, model, batch, batch_source = key
        member_count_values = pd.to_numeric(
            group["family_device_count"], errors="coerce"
        ).dropna()
        member_count = int(member_count_values.max()) if not member_count_values.empty else len(group)
        if member_count < minimum_family_size:
            continue

        defect_values = pd.to_numeric(
            group["family_history_defect_count"], errors="coerce"
        ).dropna()
        if defect_values.empty:
            continue
        defect_count = int(defect_values.max())
        defect_device_values = pd.to_numeric(
            group["family_defect_device_count"], errors="coerce"
        ).dropna()
        if defect_device_values.empty:
            continue
        defect_device_count = min(int(defect_device_values.max()), member_count)
        defect_device_rate = defect_device_count / member_count
        defect_events_per_device = defect_count / member_count

        device_predictions = [
            prediction_by_device[device_id]
            for device_id in group["unified_device_id"].tolist()
            if device_id in prediction_by_device
        ]
        risk_probabilities = [
            float(item["high_risk_probability"]) for item in device_predictions
        ]
        risk_devices = [
            item for item in device_predictions if int(item.get("state_level", 0)) >= 2
        ]
        top_devices = sorted(
            (
                {
                    "device_id": str(item["device_id"]),
                    "station_id": item.get("station_id"),
                    "state_level": int(item["state_level"]),
                    "state_name": str(item["state_name"]),
                    "high_risk_probability": round(
                        float(item["high_risk_probability"]), 6
                    ),
                }
                for item in device_predictions
            ),
            key=lambda item: item["high_risk_probability"],
            reverse=True,
        )[:3]
        families.append(
            {
                "family_id": str(family_id),
                "manufacturer": str(manufacturer),
                "model": str(model),
                "batch": str(batch),
                "batch_source": str(batch_source),
                "device_count": member_count,
                "evaluated_device_count": len(device_predictions),
                "defect_count": defect_count,
                "defect_device_count": defect_device_count,
                "defect_rate": round(100.0 * defect_device_rate, 6),
                "defect_events_per_device": round(defect_events_per_device, 6),
                "_rate": defect_device_rate,
                "high_risk_device_count": len(risk_devices),
                "average_high_risk_probability": _mean(risk_probabilities),
                "max_high_risk_probability": max(risk_probabilities, default=0.0),
                "top_risk_devices": top_devices,
                "data_quality": (
                    "pass"
                    if len(device_predictions) >= member_count
                    and "推导" not in str(batch_source)
                    else "warning"
                ),
            }
        )

    rates = [item["_rate"] for item in families]
    mean_rate = statistics.fmean(rates) if rates else 0.0
    standard_deviation = statistics.pstdev(rates) if len(rates) >= 2 else 0.0
    alerts: list[dict[str, Any]] = []
    for item in families:
        z_score = (
            (item.pop("_rate") - mean_rate) / standard_deviation
            if standard_deviation > 0
            else 0.0
        )
        alert_level = _alert_level(z_score)
        if alert_level is None or (level is not None and alert_level != level):
            continue
        item["z_score"] = round(float(z_score), 6)
        item["level"] = alert_level
        item["warning_reason"] = (
            f"家族缺陷设备占比显著高于本次有效家族平均水平，Z-score={z_score:.2f}"
        )
        alerts.append(item)

    level_order = {"G": 0, "M": 1, "L": 2}
    alerts.sort(key=lambda item: (level_order[item["level"]], -item["z_score"]))
    return {
        "assessment_date": assessment_date,
        "rule_name": FAMILY_RULE_NAME,
        "rule_version": FAMILY_RULE_VERSION,
        "family_key_definition": FAMILY_KEY_DEFINITION,
        "minimum_family_size": minimum_family_size,
        "eligible_family_count": len(families),
        "alert_count": len(alerts),
        "items": alerts[:limit],
    }


def _alert_level(z_score: float) -> str | None:
    if z_score > 3.0:
        return "G"
    if z_score > 2.0:
        return "M"
    if z_score > 1.5:
        return "L"
    return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    value = statistics.fmean(values)
    return round(float(value), 6) if math.isfinite(value) else 0.0
