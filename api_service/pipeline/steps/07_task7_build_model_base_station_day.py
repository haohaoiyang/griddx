#!/usr/bin/env python3
"""Task 7: build model_base_station_day.csv from fact_operation_device_day_wide.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple


OUTPUT_COLUMNS = [
    "date",
    "station_id",
    "station_name",
    "device_count_with_operation",
    "operation_coverage_rate",
    "voltage_device_count",
    "current_device_count",
    "power_device_count",
    "switch_device_count",
    "voltage_mean",
    "voltage_max",
    "voltage_min",
    "current_3phase_mean",
    "current_3phase_max",
    "active_power_3phase_sum",
    "reactive_power_3phase_sum",
    "switch_open_count",
    "switch_close_count",
    "defect_count_today",
    "trip_count_today",
    "maintenance_count_today",
    "station_type_code",
    "voltage_level_kv",
    "main_transformer_count",
    "main_transformer_capacity_mva",
    "lightning_risk_level_code",
    "ice_area_level_code",
]

UNAVAILABLE_COLUMNS = [
    "defect_count_today",
    "trip_count_today",
    "maintenance_count_today",
    "station_type_code",
    "voltage_level_kv",
    "main_transformer_count",
    "main_transformer_capacity_mva",
    "lightning_risk_level_code",
    "ice_area_level_code",
]


def station_key(row: dict) -> Tuple[str, str]:
    return row.get("station_id", ""), row.get("station_name", "")


def is_present(value: str) -> bool:
    return value not in ("", None)


def to_float(value: str) -> Optional[float]:
    if not is_present(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def int_flag(row: dict, column: str) -> int:
    try:
        return int(float(row.get(column, "") or 0))
    except ValueError:
        return 0


def display_number(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def build_station_device_totals(input_path: Path) -> Dict[Tuple[str, str], int]:
    devices_by_station: Dict[Tuple[str, str], set] = {}
    with input_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = station_key(row)
            device_id = row.get("unified_device_id", "")
            if not device_id:
                continue
            devices_by_station.setdefault(key, set()).add(device_id)

    return {key: len(devices) for key, devices in devices_by_station.items()}


def blank_agg(date: str, station_id: str, station_name: str) -> dict:
    return {
        "date": date,
        "station_id": station_id,
        "station_name": station_name,
        "device_ids": set(),
        "voltage_device_count": 0,
        "current_device_count": 0,
        "power_device_count": 0,
        "switch_device_count": 0,
        "voltage_sum": 0.0,
        "voltage_count": 0,
        "voltage_max": None,
        "voltage_min": None,
        "current_sum": 0.0,
        "current_count": 0,
        "current_max": None,
        "active_sum": 0.0,
        "active_count": 0,
        "reactive_sum": 0.0,
        "reactive_count": 0,
        "switch_open_count": 0,
        "switch_close_count": 0,
    }


def update_min_max(current: Optional[float], value: float, fn) -> float:
    return value if current is None else fn(current, value)


def update_agg(agg: dict, row: dict) -> None:
    device_id = row.get("unified_device_id", "")
    if device_id:
        agg["device_ids"].add(device_id)

    voltage_present = int_flag(row, "voltage_daily_present")
    current_present = int_flag(row, "current_3phase_daily_present")
    active_present = int_flag(row, "active_power_3phase_daily_present")
    reactive_present = int_flag(row, "reactive_power_3phase_daily_present")
    switch_present = any(
        int_flag(row, column)
        for column in [
            "switch_position_daily_present",
            "switch_position_phase_a_daily_present",
            "switch_position_phase_b_daily_present",
            "switch_position_phase_c_daily_present",
        ]
    )

    if voltage_present:
        agg["voltage_device_count"] += 1
        value = to_float(row.get("voltage", ""))
        if value is not None:
            agg["voltage_sum"] += value
            agg["voltage_count"] += 1
            agg["voltage_max"] = update_min_max(agg["voltage_max"], value, max)
            agg["voltage_min"] = update_min_max(agg["voltage_min"], value, min)

    if current_present:
        agg["current_device_count"] += 1
        value = to_float(row.get("current_3phase", ""))
        if value is not None:
            agg["current_sum"] += value
            agg["current_count"] += 1
            agg["current_max"] = update_min_max(agg["current_max"], value, max)

    if active_present or reactive_present:
        agg["power_device_count"] += 1

    active_value = to_float(row.get("active_power_3phase", ""))
    if active_value is not None:
        agg["active_sum"] += active_value
        agg["active_count"] += 1

    reactive_value = to_float(row.get("reactive_power_3phase", ""))
    if reactive_value is not None:
        agg["reactive_sum"] += reactive_value
        agg["reactive_count"] += 1

    if switch_present:
        agg["switch_device_count"] += 1

    switch_value = to_float(row.get("switch_position", ""))
    if switch_value == 0:
        agg["switch_open_count"] += 1
    elif switch_value == 1:
        agg["switch_close_count"] += 1


def finalize_agg(agg: dict, station_device_totals: Dict[Tuple[str, str], int]) -> dict:
    key = (agg["station_id"], agg["station_name"])
    total_devices = station_device_totals.get(key, 0)
    device_count = len(agg["device_ids"])
    coverage = device_count / total_devices if total_devices else None

    out = {
        "date": agg["date"],
        "station_id": agg["station_id"],
        "station_name": agg["station_name"],
        "device_count_with_operation": device_count,
        "operation_coverage_rate": display_number(coverage),
        "voltage_device_count": agg["voltage_device_count"],
        "current_device_count": agg["current_device_count"],
        "power_device_count": agg["power_device_count"],
        "switch_device_count": agg["switch_device_count"],
        "voltage_mean": display_number(
            agg["voltage_sum"] / agg["voltage_count"]
            if agg["voltage_count"]
            else None
        ),
        "voltage_max": display_number(agg["voltage_max"]),
        "voltage_min": display_number(agg["voltage_min"]),
        "current_3phase_mean": display_number(
            agg["current_sum"] / agg["current_count"]
            if agg["current_count"]
            else None
        ),
        "current_3phase_max": display_number(agg["current_max"]),
        "active_power_3phase_sum": display_number(
            agg["active_sum"] if agg["active_count"] else None
        ),
        "reactive_power_3phase_sum": display_number(
            agg["reactive_sum"] if agg["reactive_count"] else None
        ),
        "switch_open_count": agg["switch_open_count"],
        "switch_close_count": agg["switch_close_count"],
    }

    for column in UNAVAILABLE_COLUMNS:
        out[column] = ""

    return out


def write_date_group(
    writer: csv.DictWriter,
    aggs: Dict[Tuple[str, str], dict],
    station_device_totals: Dict[Tuple[str, str], int],
) -> int:
    count = 0
    for key in sorted(aggs):
        writer.writerow(finalize_agg(aggs[key], station_device_totals))
        count += 1
    return count


def build_station_day(input_path: Path, output_path: Path) -> dict:
    station_device_totals = build_station_device_totals(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_read = 0
    rows_written = 0
    current_date = None
    aggs: Dict[Tuple[str, str], dict] = {}
    dates = set()

    with input_path.open(newline="", encoding="utf-8-sig") as in_f, output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for row in reader:
            rows_read += 1
            date = row.get("date", "")
            if not date:
                continue

            if current_date is None:
                current_date = date
            elif date != current_date:
                rows_written += write_date_group(writer, aggs, station_device_totals)
                aggs = {}
                current_date = date

            dates.add(date)
            key = station_key(row)
            if key not in aggs:
                aggs[key] = blank_agg(date, key[0], key[1])
            update_agg(aggs[key], row)

        if current_date is not None:
            rows_written += write_date_group(writer, aggs, station_device_totals)

    return {
        "rows_read": rows_read,
        "rows_written": rows_written,
        "date_count": len(dates),
        "station_count": len(station_device_totals),
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "生成任务7站点日基础表。当前只使用 fact_operation_device_day_wide.csv，"
            "缺陷/跳闸/检修和站点背景字段保留为空。"
        )
    )
    parser.add_argument(
        "--input",
        default="output/clean/02_operation/fact_operation_device_day_wide.csv",
        help="任务5设备日宽表路径",
    )
    parser.add_argument(
        "--output",
        default="output/clean/04_model_base/model_base_station_day.csv",
        help="任务7输出文件路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_station_day(Path(args.input), Path(args.output))
    print(f"输入行数：{stats['rows_read']}")
    print(f"输出行数：{stats['rows_written']}")
    print(f"日期数：{stats['date_count']}")
    print(f"站点数：{stats['station_count']}")
    print(f"输出文件：{stats['output_path']}")


if __name__ == "__main__":
    main()
