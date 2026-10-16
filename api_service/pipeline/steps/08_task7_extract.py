#!/usr/bin/env python3
"""Extract clean station-day rows from task7 model_base_station_day.csv."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


BASE_REQUIRED_COLUMNS = [
    "date",
    "station_id",
    "station_name",
    "device_count_with_operation",
    "operation_coverage_rate",
]

OPERATION_METRIC_COLUMNS = [
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
]

ABNORMAL_KEYWORDS = [
    "test",
    "sysfac",
    "sys_fac",
    "sysstation",
]


def has_value(value: str) -> bool:
    return value not in ("", None)


def is_normal_station(row: dict) -> bool:
    station_id = row.get("station_id", "")
    station_name = row.get("station_name", "")
    if not station_id or not station_name:
        return False

    text = f"{station_id} {station_name}".lower()
    if any(keyword in text for keyword in ABNORMAL_KEYWORDS):
        return False

    if not re.search(r"[\u4e00-\u9fff]", station_name):
        return False

    return True


def should_keep(row: dict, require_complete_operation_metrics: bool) -> bool:
    if not is_normal_station(row):
        return False

    if any(not has_value(row.get(column, "")) for column in BASE_REQUIRED_COLUMNS):
        return False

    if require_complete_operation_metrics:
        return all(has_value(row.get(column, "")) for column in OPERATION_METRIC_COLUMNS)

    return True


def extract_clean_rows(
    input_path: Path,
    output_path: Path,
    require_complete_operation_metrics: bool,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_read = 0
    rows_written = 0
    dropped_abnormal_station = 0
    dropped_empty_required = 0
    dropped_empty_operation_metric = 0
    stations = set()
    dates = set()

    with input_path.open(newline="", encoding="utf-8-sig") as in_f, output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as out_f:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} 没有表头")

        writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in reader:
            rows_read += 1

            if not is_normal_station(row):
                dropped_abnormal_station += 1
                continue

            if any(not has_value(row.get(column, "")) for column in BASE_REQUIRED_COLUMNS):
                dropped_empty_required += 1
                continue

            if require_complete_operation_metrics and any(
                not has_value(row.get(column, "")) for column in OPERATION_METRIC_COLUMNS
            ):
                dropped_empty_operation_metric += 1
                continue

            writer.writerow(row)
            rows_written += 1
            stations.add((row.get("station_id", ""), row.get("station_name", "")))
            if row.get("date"):
                dates.add(row["date"])

    return {
        "rows_read": rows_read,
        "rows_written": rows_written,
        "date_count": len(dates),
        "station_count": len(stations),
        "dropped_abnormal_station": dropped_abnormal_station,
        "dropped_empty_required": dropped_empty_required,
        "dropped_empty_operation_metric": dropped_empty_operation_metric,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "筛选 task7 站点日基础表：保留站点正常且没有空关键数据的行。"
            "缺陷/检修/站点背景等暂无来源字段不参与空值判断。"
        )
    )
    parser.add_argument(
        "--input",
        default="output/clean/04_model_base/model_base_station_day.csv",
        help="task7 原始输出表",
    )
    parser.add_argument(
        "--output",
        default="output/clean/04_model_base/model_base_station_day_extract.csv",
        help="筛选后的输出表",
    )
    parser.add_argument(
        "--require-complete-operation-metrics",
        action="store_true",
        help="要求运行聚合指标列都不为空；默认只要求基础站点字段不为空",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = extract_clean_rows(
        Path(args.input),
        Path(args.output),
        args.require_complete_operation_metrics,
    )
    print(f"输入行数：{stats['rows_read']}")
    print(f"输出行数：{stats['rows_written']}")
    print(f"日期数：{stats['date_count']}")
    print(f"站点数：{stats['station_count']}")
    print(f"剔除异常/空站点行数：{stats['dropped_abnormal_station']}")
    print(f"剔除基础字段空值行数：{stats['dropped_empty_required']}")
    print(f"剔除运行指标空值行数：{stats['dropped_empty_operation_metric']}")
    print(f"输出文件：{stats['output_path']}")


if __name__ == "__main__":
    main()
