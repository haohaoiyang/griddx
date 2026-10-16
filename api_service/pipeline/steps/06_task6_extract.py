#!/usr/bin/env python3
"""Extract a suggested_model_group subset from model_base_device_day.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def extract_group(input_path: Path, output_path: Path, group: str) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_read = 0
    rows_written = 0
    devices = set()
    dates = set()

    with input_path.open(newline="", encoding="utf-8-sig") as in_f, output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as out_f:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} 没有表头")
        if "suggested_model_group" not in reader.fieldnames:
            raise ValueError(f"{input_path} 缺少 suggested_model_group 字段")

        writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in reader:
            rows_read += 1
            if row.get("suggested_model_group") != group:
                continue

            writer.writerow(row)
            rows_written += 1

            if row.get("unified_device_id"):
                devices.add(row["unified_device_id"])
            if row.get("date"):
                dates.add(row["date"])

    return {
        "rows_read": rows_read,
        "rows_written": rows_written,
        "device_count": len(devices),
        "date_count": len(dates),
        "group": group,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 model_base_device_day.csv 中提取指定 suggested_model_group 分组"
    )
    parser.add_argument(
        "--input",
        default="output/clean/04_model_base/model_base_device_day.csv",
        help="任务6模型基础表路径",
    )
    parser.add_argument(
        "--output",
        default="output/clean/04_model_base/model_base_device_day_line_or_load_like.csv",
        help="筛选结果输出路径",
    )
    parser.add_argument(
        "--group",
        default="line_or_load_like",
        help="要提取的 suggested_model_group 值，默认：line_or_load_like",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = extract_group(Path(args.input), Path(args.output), args.group)
    print(f"输入行数：{stats['rows_read']}")
    print(f"提取分组：{stats['group']}")
    print(f"输出行数：{stats['rows_written']}")
    print(f"设备数：{stats['device_count']}")
    print(f"日期数：{stats['date_count']}")
    print(f"输出文件：{stats['output_path']}")


if __name__ == "__main__":
    main()
