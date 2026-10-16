#!/usr/bin/env python3
"""Task 5: build fact_operation_device_day_wide.csv from merged DL/XML wide data."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SOURCE_TO_STANDARD = {
    "ThreePhaseCurrent（AMP）": ("current_3phase", "A"),
    "ZeroSequenceCurrent（AMP）": ("current_zero_sequence", "A"),
    "ThreePhaseActivePower（MW）": ("active_power_3phase", "MW"),
    "ThreePhaseReactivePower（MVAr）": ("reactive_power_3phase", "MVAr"),
    "Frequency（Hz）": ("frequency", "Hz"),
    "TapPosition（）": ("tap_position", ""),
    "Voltage（kV）": ("voltage", "kV"),
    "SwitchPosition（OPEN/CLOSE）": ("switch_position", ""),
    "SwitchPositionPhaseA（OPEN/CLOSE）": ("switch_position_phase_a", ""),
    "SwitchPositionPhaseB（OPEN/CLOSE）": ("switch_position_phase_b", ""),
    "SwitchPositionPhaseC（OPEN/CLOSE）": ("switch_position_phase_c", ""),
}

MEASUREMENT_TYPES = [standard for standard, _unit in SOURCE_TO_STANDARD.values()]

BASE_COLUMNS = [
    "date",
    "unified_device_id",
    "source_device_id",
    "device_name",
    "station_id",
    "station_name",
    "suggested_model_group",
    "measurement_type_count",
]


def date_from_time(value: str) -> str:
    if not value:
        return ""
    return value[:10]


def non_empty(value: str) -> bool:
    return value not in ("", None)


def infer_model_group(has_measurement: Dict[str, int], measurement_type_count: int) -> str:
    has_current = bool(
        has_measurement.get("current_3phase")
        or has_measurement.get("current_zero_sequence")
    )
    has_power = bool(
        has_measurement.get("active_power_3phase")
        or has_measurement.get("reactive_power_3phase")
    )
    has_voltage = bool(has_measurement.get("voltage"))
    has_frequency = bool(has_measurement.get("frequency"))
    has_tap = bool(has_measurement.get("tap_position"))
    has_switch = bool(has_measurement.get("switch_position"))
    has_phase_switch = bool(
        has_measurement.get("switch_position_phase_a")
        or has_measurement.get("switch_position_phase_b")
        or has_measurement.get("switch_position_phase_c")
    )

    if has_tap and (has_voltage or has_current):
        return "transformer_like"
    if has_phase_switch:
        return "phase_switch_like"
    if has_current or has_power:
        return "line_or_load_like"
    if (has_voltage or has_frequency) and not (has_current or has_power or has_tap or has_switch):
        return "bus_or_station_voltage_like"
    if has_switch and measurement_type_count == 1:
        return "switch_like"
    if measurement_type_count <= 1:
        return "insufficient_measurement"
    return "mixed_or_unknown"


def output_columns(measurement_types: Iterable[str]) -> List[str]:
    columns = list(BASE_COLUMNS)
    for measurement_type in measurement_types:
        columns.extend(
            [
                measurement_type,
                f"{measurement_type}_daily_present",
                f"{measurement_type}_missing_flag",
            ]
        )
    return columns


def read_header(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f))


def source_columns_present(header: List[str]) -> Dict[str, str]:
    return {
        source: standard
        for source, (standard, _unit) in SOURCE_TO_STANDARD.items()
        if source in header
    }


def build_device_availability(
    input_path: Path,
    source_to_standard: Dict[str, str],
) -> Dict[str, dict]:
    devices: Dict[str, dict] = {}

    with input_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            device_id = row.get("device_id", "")
            if not device_id:
                continue

            info = devices.setdefault(
                device_id,
                {
                    "source_device_id": device_id,
                    "device_name": row.get("device_name", ""),
                    "station_id": row.get("station_id", ""),
                    "station_name": row.get("station_name", ""),
                    "measurement_types": set(),
                },
            )

            if not info.get("device_name") and row.get("device_name"):
                info["device_name"] = row["device_name"]
            if not info.get("station_id") and row.get("station_id"):
                info["station_id"] = row["station_id"]
            if not info.get("station_name") and row.get("station_name"):
                info["station_name"] = row["station_name"]

            for source_col, standard_col in source_to_standard.items():
                if non_empty(row.get(source_col, "")):
                    info["measurement_types"].add(standard_col)

    for info in devices.values():
        measurement_types = info["measurement_types"]
        has_measurement = {mt: int(mt in measurement_types) for mt in MEASUREMENT_TYPES}
        info["measurement_type_count"] = len(measurement_types)
        info["has_measurement"] = has_measurement
        info["suggested_model_group"] = infer_model_group(
            has_measurement,
            len(measurement_types),
        )

    return devices


def blank_device_day(date: str, device_id: str, row: dict) -> dict:
    return {
        "date": date,
        "unified_device_id": device_id,
        "source_device_id": device_id,
        "device_name": row.get("device_name", ""),
        "station_id": row.get("station_id", ""),
        "station_name": row.get("station_name", ""),
    }


def merge_source_row_into_day(
    target: dict,
    source_row: dict,
    source_to_standard: Dict[str, str],
) -> None:
    for source_col, standard_col in source_to_standard.items():
        value = source_row.get(source_col, "")
        if non_empty(value):
            # If multiple records exist for one device on one day, keep the latest
            # non-empty value encountered in input order.
            target[standard_col] = value


def complete_output_row(row: dict, device_info: dict, measurement_types: List[str]) -> dict:
    out = dict(row)
    out["device_name"] = out.get("device_name") or device_info.get("device_name", "")
    out["station_id"] = out.get("station_id") or device_info.get("station_id", "")
    out["station_name"] = out.get("station_name") or device_info.get("station_name", "")
    out["suggested_model_group"] = device_info.get("suggested_model_group", "")
    out["measurement_type_count"] = device_info.get("measurement_type_count", 0)

    has_measurement = device_info.get("has_measurement", {})
    for measurement_type in measurement_types:
        present = int(non_empty(out.get(measurement_type, "")))
        out[f"{measurement_type}_daily_present"] = present
        out[f"{measurement_type}_missing_flag"] = int(
            has_measurement.get(measurement_type, 0) == 1 and present == 0
        )
    return out


def write_date_group(
    writer: csv.DictWriter,
    rows_by_device: Dict[str, dict],
    devices: Dict[str, dict],
    measurement_types: List[str],
) -> int:
    count = 0
    for device_id in sorted(rows_by_device):
        row = rows_by_device[device_id]
        device_info = devices.get(
            device_id,
            {
                "measurement_type_count": 0,
                "has_measurement": {},
                "suggested_model_group": "mixed_or_unknown",
            },
        )
        writer.writerow(complete_output_row(row, device_info, measurement_types))
        count += 1
    return count


def build_device_day_wide(input_path: Path, output_path: Path) -> dict:
    header = read_header(input_path)
    source_to_standard = source_columns_present(header)
    measurement_types = [
        standard
        for source, (standard, _unit) in SOURCE_TO_STANDARD.items()
        if source in source_to_standard
    ]

    devices = build_device_availability(input_path, source_to_standard)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    input_rows = 0
    date_count = 0
    current_date = None
    rows_by_device: Dict[str, dict] = {}

    with input_path.open(newline="", encoding="utf-8-sig") as in_f, output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(
            out_f,
            fieldnames=output_columns(measurement_types),
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in reader:
            input_rows += 1
            date = date_from_time(row.get("time", ""))
            device_id = row.get("device_id", "")
            if not date or not device_id:
                continue

            if current_date is None:
                current_date = date
                date_count = 1
            elif date != current_date:
                rows_written += write_date_group(
                    writer,
                    rows_by_device,
                    devices,
                    measurement_types,
                )
                rows_by_device = {}
                current_date = date
                date_count += 1

            day_row = rows_by_device.setdefault(
                device_id,
                blank_device_day(date, device_id, row),
            )
            merge_source_row_into_day(day_row, row, source_to_standard)

        if current_date is not None:
            rows_written += write_date_group(
                writer,
                rows_by_device,
                devices,
                measurement_types,
            )

    return {
        "input_rows": input_rows,
        "device_count": len(devices),
        "date_count": date_count,
        "measurement_type_count": len(measurement_types),
        "rows_written": rows_written,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成任务5设备日宽表 fact_operation_device_day_wide.csv"
    )
    parser.add_argument(
        "--input",
        default="output/merged.csv",
        help="上一步合并后的运行宽表，默认：output/merged.csv",
    )
    parser.add_argument(
        "--output",
        default="output/clean/02_operation/fact_operation_device_day_wide.csv",
        help="任务5输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_device_day_wide(Path(args.input), Path(args.output))
    print(f"输入行数：{stats['input_rows']}")
    print(f"设备数：{stats['device_count']}")
    print(f"日期数：{stats['date_count']}")
    print(f"量测类型数：{stats['measurement_type_count']}")
    print(f"输出行数：{stats['rows_written']}")
    print(f"输出文件：{stats['output_path']}")


if __name__ == "__main__":
    main()
