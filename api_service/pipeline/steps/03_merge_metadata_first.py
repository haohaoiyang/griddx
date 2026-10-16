#!/usr/bin/env python3
"""Merge objects and relations first, then fill DT values into a device-level wide table."""

from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MEASUREMENT_TO_DEVICE = "cim:Measurement.MemberOf_PSR"
MEASUREMENT_TO_TYPE = "cim:Measurement.MeasurementType"
MEASUREMENT_TO_UNIT = "cim:Measurement.Unit"
EQUIPMENT_TO_CONTAINER = "cim:Equipment.MemberOf_EquipmentContainer"
CONNECTIVITY_NODE_TO_CONTAINER = "cim:ConnectivityNode.MemberOf_EquipmentContainer"
VOLTAGE_LEVEL_TO_SUBSTATION = "cim:VoltageLevel.MemberOf_Substation"
SUBSTATION_CONTAINS_VOLTAGE_LEVEL = "cim:Substation.Contains_VoltageLevels"
LINE_START_STATION = "cim:ACLineSegment.StartST"
LINE_END_STATION = "cim:ACLineSegment.EndST"

BASE_COLUMNS = ["time", "device_id", "device_name", "station_id", "station_name"]
MERGED_RELATION_COLUMNS = [
    "source_id",
    "source_type",
    "source_name",
    "source_description",
    "relation",
    "target_id",
    "target_type",
    "target_name",
    "target_description",
]


def read_objects(path: Path) -> Dict[str, dict]:
    objects: Dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            objects[row["object_id"]] = row
    return objects


def object_info(objects: Dict[str, dict], object_id: str) -> dict:
    obj = objects.get(object_id, {})
    return {
        "type": obj.get("object_type", ""),
        "name": obj.get("name", ""),
        "description": obj.get("description", ""),
    }


def merge_objects_relations(
    objects: Dict[str, dict],
    relations_path: Path,
    merged_output: Optional[Path] = None,
) -> List[dict]:
    """Enrich each relation row with source/target names from objects.csv."""
    merged_rows: List[dict] = []

    writer = None
    output_file = None
    if merged_output:
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        output_file = merged_output.open("w", newline="", encoding="utf-8-sig")
        writer = csv.DictWriter(output_file, fieldnames=MERGED_RELATION_COLUMNS)
        writer.writeheader()

    try:
        with relations_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                source_id = row["source_id"]
                target_id = row["target_id"]
                source = object_info(objects, source_id)
                target = object_info(objects, target_id)
                merged = {
                    "source_id": source_id,
                    "source_type": source["type"] or row.get("source_type", ""),
                    "source_name": source["name"],
                    "source_description": source["description"],
                    "relation": row["relation"],
                    "target_id": target_id,
                    "target_type": target["type"],
                    "target_name": target["name"],
                    "target_description": target["description"],
                }
                merged_rows.append(merged)
                if writer:
                    writer.writerow(merged)
    finally:
        if output_file:
            output_file.close()

    return merged_rows


def build_catalog_from_merged_relations(
    merged_relations: Iterable[dict],
) -> Tuple[Dict[str, dict], dict]:
    """Build measurement/device/station indexes from the already-merged relation table."""
    measurement_catalog: Dict[str, dict] = defaultdict(dict)
    equipment_container: Dict[str, str] = {}
    voltage_level_station: Dict[str, str] = {}
    line_stations: Dict[str, List[str]] = defaultdict(list)

    for row in merged_relations:
        source_id = row["source_id"]
        target_id = row["target_id"]
        relation = row["relation"]

        if relation == MEASUREMENT_TO_DEVICE:
            measurement_catalog[source_id].update(
                {
                    "device_id": target_id,
                    "device_name": row["target_name"],
                }
            )
        elif relation == MEASUREMENT_TO_TYPE:
            measurement_catalog[source_id].update(
                {
                    "type_id": target_id,
                    "type_name": row["target_name"],
                    "type_description": row["target_description"],
                }
            )
        elif relation == MEASUREMENT_TO_UNIT:
            measurement_catalog[source_id].update(
                {
                    "unit_id": target_id,
                    "unit_name": row["target_name"],
                }
            )
        elif relation in (EQUIPMENT_TO_CONTAINER, CONNECTIVITY_NODE_TO_CONTAINER):
            equipment_container[source_id] = target_id
        elif relation == VOLTAGE_LEVEL_TO_SUBSTATION:
            voltage_level_station[source_id] = target_id
        elif relation == SUBSTATION_CONTAINS_VOLTAGE_LEVEL:
            voltage_level_station.setdefault(target_id, source_id)
        elif relation == LINE_START_STATION:
            line_stations[source_id].insert(0, target_id)
        elif relation == LINE_END_STATION:
            line_stations[source_id].append(target_id)

    relation_indexes = {
        "equipment_container": equipment_container,
        "voltage_level_station": voltage_level_station,
        "line_stations": line_stations,
    }
    return dict(measurement_catalog), relation_indexes


def resolve_station(
    device_id: str,
    objects: Dict[str, dict],
    relation_indexes: dict,
) -> Tuple[str, str]:
    equipment_container = relation_indexes["equipment_container"]
    voltage_level_station = relation_indexes["voltage_level_station"]
    line_stations = relation_indexes["line_stations"]

    current = device_id
    seen = set()
    for _ in range(8):
        if not current or current in seen:
            break
        seen.add(current)

        current_obj = objects.get(current, {})
        current_type = current_obj.get("object_type", "")
        if current_type == "cim:Substation":
            return current, current_obj.get("name", "")

        if current_type == "cim:VoltageLevel":
            station_id = voltage_level_station.get(current, "")
            if station_id:
                return station_id, objects.get(station_id, {}).get("name", "")

        next_id = equipment_container.get(current, "")
        if next_id:
            current = next_id
            continue

        break

    for station_id in line_stations.get(device_id, []):
        if objects.get(station_id, {}).get("object_type") == "cim:Substation":
            return station_id, objects.get(station_id, {}).get("name", "")

    return "", ""


def attach_station_info(
    measurement_catalog: Dict[str, dict],
    objects: Dict[str, dict],
    relation_indexes: dict,
) -> Counter:
    stats = Counter()
    for meta in measurement_catalog.values():
        device_id = meta.get("device_id", "")
        type_id = meta.get("type_id", "")
        if not device_id or not type_id:
            stats["measurement_missing_device_or_type"] += 1
            continue

        station_id, station_name = resolve_station(device_id, objects, relation_indexes)
        meta["station_id"] = station_id
        meta["station_name"] = station_name
        if not station_id:
            stats["measurement_missing_station"] += 1

    return stats


def meas_type_sort_key(type_id: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", type_id)
    return (int(match.group(1)) if match else 10**9, type_id)


def type_label(row: dict, label: str) -> str:
    if label == "id":
        return row["object_id"]
    if label == "name":
        return row.get("name") or row.get("description") or row["object_id"]
    if label == "id_name":
        return f"{row['object_id']}_{row.get('name') or row.get('description') or 'value'}"
    return row.get("description") or row.get("name") or row["object_id"]


def most_common_unit_by_type(measurement_catalog: Dict[str, dict]) -> Dict[str, str]:
    counters: Dict[str, Counter] = defaultdict(Counter)
    for meta in measurement_catalog.values():
        type_id = meta.get("type_id", "")
        unit_name = meta.get("unit_name", "")
        if type_id and unit_name:
            counters[type_id][unit_name] += 1

    return {
        type_id: counter.most_common(1)[0][0]
        for type_id, counter in counters.items()
        if counter
    }


def make_type_columns(
    objects: Dict[str, dict],
    measurement_catalog: Dict[str, dict],
    label: str,
) -> Tuple[List[str], Dict[str, str]]:
    unit_by_type = most_common_unit_by_type(measurement_catalog)
    type_rows = [
        row
        for row in objects.values()
        if row.get("object_type") == "cim:MeasurementType"
    ]
    type_rows.sort(key=lambda row: meas_type_sort_key(row["object_id"]))

    ordered_columns: List[str] = []
    type_to_column: Dict[str, str] = {}
    used = Counter()

    for row in type_rows:
        base = type_label(row, label)
        unit = unit_by_type.get(row["object_id"], "")
        column_base = f"{base}（{unit}）"
        used[column_base] += 1
        column = column_base if used[column_base] == 1 else f"{column_base}_{row['object_id']}"

        ordered_columns.append(column)
        type_to_column[row["object_id"]] = column

    return ordered_columns, type_to_column


def parse_dt(path: Path) -> Tuple[str, List[Tuple[str, str]]]:
    time_value = ""
    section = ""
    headers: Dict[str, List[str]] = {}
    value_rows: List[Tuple[str, str]] = []

    with path.open(encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue

            if line.startswith("<") and not line.startswith("</"):
                section = line.strip("<>")
                continue
            if line.startswith("</"):
                section = ""
                continue
            if not section:
                continue

            parts = line.split("\t")
            marker = parts[0]
            values = parts[1:]
            if marker == "@":
                headers[section] = values
                continue
            if marker != "#":
                continue

            row = dict(zip(headers.get(section, []), values))
            if section == "system":
                time_value = row.get("Time", time_value)
            elif section in {"AnalogValue", "DiscreteValue"}:
                measurement_id = row.get("ID", "")
                value = row.get("Value", row.get("Status", ""))
                if measurement_id:
                    value_rows.append((measurement_id, value))

    if not time_value:
        raise ValueError(f"{path} 中没有解析到 <system> 的 Time 字段")

    return time_value, value_rows


def read_dt_time(path: Path) -> str:
    section = ""
    headers: Dict[str, List[str]] = {}

    with path.open(encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue

            if line.startswith("<") and not line.startswith("</"):
                section = line.strip("<>")
                continue
            if line.startswith("</"):
                section = ""
                continue
            if section != "system":
                continue

            parts = line.split("\t")
            marker = parts[0]
            values = parts[1:]
            if marker == "@":
                headers[section] = values
            elif marker == "#":
                row = dict(zip(headers.get(section, []), values))
                time_value = row.get("Time", "")
                if time_value:
                    return time_value

    raise ValueError(f"{path} 中没有解析到 <system> 的 Time 字段")


def expand_dt_paths(data_dir: Path, dt_args: Optional[List[str]]) -> List[Path]:
    if not dt_args:
        return sorted(data_dir.rglob("*.DT"))

    paths: List[Path] = []
    for item in dt_args:
        input_path = Path(item)
        if input_path.is_dir():
            paths.extend(input_path.rglob("*.DT"))
            continue

        matches = glob.glob(item, recursive=True)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(item))
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def fill_wide_rows(
    dt_paths: List[Path],
    measurement_catalog: Dict[str, dict],
    type_to_column: Dict[str, str],
) -> Tuple[Dict[Tuple[str, str], dict], Counter]:
    rows: Dict[Tuple[str, str], dict] = {}
    stats = Counter()

    for dt_path in dt_paths:
        time_value, value_rows = parse_dt(dt_path)
        stats["dt_files"] += 1

        for measurement_id, value in value_rows:
            meta = measurement_catalog.get(measurement_id)
            if not meta:
                stats["measurement_not_found"] += 1
                continue

            device_id = meta.get("device_id", "")
            type_id = meta.get("type_id", "")
            column = type_to_column.get(type_id, "")
            if not device_id or not column:
                stats["measurement_missing_device_or_type"] += 1
                continue

            key = (time_value, device_id)
            if key not in rows:
                rows[key] = {
                    "time": time_value,
                    "device_id": device_id,
                    "device_name": meta.get("device_name", ""),
                    "station_id": meta.get("station_id", ""),
                    "station_name": meta.get("station_name", ""),
                }

            if rows[key].get(column, "") != "":
                stats["duplicate_device_type_value"] += 1
                continue

            rows[key][column] = value
            stats["filled_values"] += 1

    return rows, stats


def collect_used_type_ids(
    dt_paths: List[Path],
    measurement_catalog: Dict[str, dict],
) -> Tuple[set, Counter]:
    used_type_ids = set()
    stats = Counter()

    for dt_path in dt_paths:
        _time_value, value_rows = parse_dt(dt_path)
        stats["dt_files_scanned"] += 1

        for measurement_id, value in value_rows:
            if value == "":
                continue

            meta = measurement_catalog.get(measurement_id)
            if not meta:
                stats["measurement_not_found_during_scan"] += 1
                continue

            type_id = meta.get("type_id", "")
            device_id = meta.get("device_id", "")
            if type_id and device_id:
                used_type_ids.add(type_id)

    return used_type_ids, stats


def write_wide_csv(path: Path, rows: Dict[Tuple[str, str], dict], type_columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = BASE_COLUMNS + type_columns
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])


def write_wide_csv_from_dt_files(
    path: Path,
    dt_paths: List[Path],
    measurement_catalog: Dict[str, dict],
    type_to_column: Dict[str, str],
    type_columns: List[str],
) -> Tuple[int, Counter]:
    dt_groups: Dict[str, List[Path]] = defaultdict(list)
    for dt_path in dt_paths:
        dt_groups[read_dt_time(dt_path)].append(dt_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = BASE_COLUMNS + type_columns
    total_rows = 0
    total_stats = Counter()

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for time_value in sorted(dt_groups):
            rows, stats = fill_wide_rows(
                dt_groups[time_value],
                measurement_catalog,
                type_to_column,
            )
            total_stats.update(stats)
            for key in sorted(rows):
                writer.writerow(rows[key])
                total_rows += 1

    return total_rows, total_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "先把 objects.csv 与 relations.csv 合并成对象关系表，"
            "再根据 DT 文件把值填入设备级宽表。"
        )
    )
    parser.add_argument("--data-dir", default="data", help="默认数据目录，默认：data")
    parser.add_argument("--objects", help="objects.csv 路径，默认：<data-dir>/objects.csv")
    parser.add_argument("--relations", help="relations.csv 路径，默认：<data-dir>/relations.csv")
    parser.add_argument(
        "--dt",
        nargs="*",
        help="DT 文件、目录或通配符；不传则递归读取 <data-dir>/**/*.DT",
    )
    parser.add_argument(
        "--output",
        default="output/merged_wide_metadata_first.csv",
        help="输出宽表 CSV 路径，默认：output/merged_wide_metadata_first.csv",
    )
    parser.add_argument(
        "--merged-relations-output",
        default="",
        help="可选：输出 objects+relations 合并后的中间表路径",
    )
    parser.add_argument(
        "--type-label",
        choices=["description", "name", "id", "id_name"],
        default="name",
        help="属性名使用 objects.csv 的哪个字段，默认：name",
    )
    parser.add_argument(
        "--keep-all-type-columns",
        action="store_true",
        help="保留 66 个全部数据类型列；默认只输出有数值的数据类型列",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    objects_path = Path(args.objects) if args.objects else data_dir / "objects.csv"
    relations_path = Path(args.relations) if args.relations else data_dir / "relations.csv"
    output_path = Path(args.output)
    merged_output = Path(args.merged_relations_output) if args.merged_relations_output else None
    dt_paths = expand_dt_paths(data_dir, args.dt)

    if not dt_paths:
        raise SystemExit(f"没有找到 DT 文件：{data_dir / '*.DT'}")

    objects = read_objects(objects_path)
    merged_relations = merge_objects_relations(objects, relations_path, merged_output)
    measurement_catalog, relation_indexes = build_catalog_from_merged_relations(merged_relations)
    index_stats = attach_station_info(measurement_catalog, objects, relation_indexes)
    type_columns, type_to_column = make_type_columns(objects, measurement_catalog, args.type_label)
    used_type_ids, scan_stats = collect_used_type_ids(dt_paths, measurement_catalog)
    if not args.keep_all_type_columns:
        type_columns = [
            column
            for type_id, column in type_to_column.items()
            if type_id in used_type_ids
        ]
    output_row_count, fill_stats = write_wide_csv_from_dt_files(
        output_path,
        dt_paths,
        measurement_catalog,
        type_to_column,
        type_columns,
    )

    print(f"objects+relations 合并关系行数：{len(merged_relations)}")
    if merged_output:
        print(f"objects+relations 中间表：{merged_output}")
    print(f"预扫描 DT 文件数：{scan_stats['dt_files_scanned']}")
    print(f"DT 文件数：{fill_stats['dt_files']}")
    print(f"输出行数：{output_row_count}")
    print(f"输出属性列数：{len(type_columns)}")
    print(f"填充值数：{fill_stats['filled_values']}")
    print(f"重复设备+属性值已跳过：{fill_stats['duplicate_device_type_value']}")
    print(f"DT 中未匹配到合并关系的测点：{fill_stats['measurement_not_found']}")
    print(f"未解析到站点的测点索引数：{index_stats['measurement_missing_station']}")
    print(f"宽表输出：{output_path}")


if __name__ == "__main__":
    main()
