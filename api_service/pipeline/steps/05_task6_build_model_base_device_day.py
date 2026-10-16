#!/usr/bin/env python3
"""Task 6: build model_base_device_day.csv from operation and background tables."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


MODEL_COLUMNS = [
    "date",
    "unified_device_id",
    "device_name",
    "station_id",
    "station_name",
    "suggested_model_group",
    "measurement_type_count",
    "current_3phase",
    "current_3phase_daily_present",
    "current_3phase_missing_flag",
    "voltage",
    "voltage_daily_present",
    "voltage_missing_flag",
    "active_power_3phase",
    "active_power_3phase_daily_present",
    "active_power_3phase_missing_flag",
    "reactive_power_3phase",
    "reactive_power_3phase_daily_present",
    "reactive_power_3phase_missing_flag",
    "frequency",
    "frequency_daily_present",
    "frequency_missing_flag",
    "tap_position",
    "tap_position_daily_present",
    "tap_position_missing_flag",
    "switch_position",
    "switch_position_daily_present",
    "switch_position_missing_flag",
    "device_type_code",
    "voltage_level_kv",
    "operation_status_code",
    "manufacturer_code",
    "device_age_days",
    "background_match_confidence",
    "defect_count_today",
    "trip_count_today",
    "maintenance_count_today",
]

BACKGROUND_COLUMNS = [
    "device_type_code",
    "voltage_level_kv",
    "operation_status_code",
    "manufacturer_code",
    "device_age_days",
    "background_match_confidence",
]

EVENT_COLUMNS = [
    "defect_count_today",
    "trip_count_today",
    "maintenance_count_today",
]

CODE_FIELDS = {
    "device_name_norm",
    "station_name_norm",
}

DEFAULT_BACKGROUND_TARGET_GROUP = "line_or_load_like"

BACKGROUND_VALUE_COLUMNS = [
    "device_type_code",
    "voltage_level_kv",
    "operation_status_code",
    "manufacturer_code",
    "device_age_days",
]


class BackgroundMatcher:
    def __init__(
        self,
        code_maps: Dict[str, Dict[str, str]],
        norm_pair_index: Dict[Tuple[str, str], dict],
        raw_pair_index: Dict[Tuple[str, str], dict],
        line_norm_device_index: Dict[str, dict],
        line_raw_device_index: Dict[str, dict],
        station_device_index: Dict[Tuple[str, str, str], dict],
        counts: Counter,
    ) -> None:
        self.code_maps = code_maps
        self.norm_pair_index = norm_pair_index
        self.raw_pair_index = raw_pair_index
        self.line_norm_device_index = line_norm_device_index
        self.line_raw_device_index = line_raw_device_index
        self.station_device_index = station_device_index
        self.counts = counts
        self.cache: Dict[Tuple[str, str, str], Optional[dict]] = {}

    def match(
        self,
        device_name: str,
        station_id: str,
        station_name: str,
    ) -> Optional[dict]:
        key = (device_name, station_id, station_name)
        if key in self.cache:
            return self.cache[key]

        device_norm = self.code_maps.get("device_name_norm", {}).get(device_name, "")
        station_norm = self.code_maps.get("station_name_norm", {}).get(station_name, "")

        match = None
        method = ""
        confidence = ""

        if device_norm and station_norm and (device_norm, station_norm) in self.norm_pair_index:
            method = "norm_pair"
            match = self.norm_pair_index[(device_norm, station_norm)]
            confidence = "1.00"
        elif (device_name, station_name) in self.raw_pair_index:
            method = "raw_pair"
            match = self.raw_pair_index[(device_name, station_name)]
            confidence = "1.00"
        elif device_norm and device_norm in self.line_norm_device_index:
            method = "line_norm_device"
            match = self.line_norm_device_index[device_norm]
            count = self.counts[("line_norm_device", device_norm)]
            confidence = "0.85" if count == 1 else "0.75"
        elif device_name in self.line_raw_device_index:
            method = "line_raw_device"
            match = self.line_raw_device_index[device_name]
            count = self.counts[("line_raw_device", device_name)]
            confidence = "0.85" if count == 1 else "0.75"
        elif (device_name, station_id, station_name) in self.station_device_index:
            method = "station_device_path"
            match = self.station_device_index[(device_name, station_id, station_name)]
            confidence = match.get("background_match_confidence", "0.85")

        if match:
            result = dict(match)
            result["background_match_confidence"] = confidence
            result["_match_method"] = method
        else:
            result = None

        self.cache[key] = result
        return result


def load_code_maps(code_path: Path) -> Dict[str, Dict[str, str]]:
    maps: Dict[str, Dict[str, str]] = {field: {} for field in CODE_FIELDS}
    if not code_path.exists():
        return maps

    with code_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"原始名称", "标准化编码", "字段"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{code_path} 缺少字段：{sorted(required)}")

        for row in reader:
            field = row.get("字段", "")
            raw_name = row.get("原始名称", "")
            norm_code = row.get("标准化编码", "")
            if field in maps and raw_name and norm_code:
                maps[field].setdefault(raw_name, norm_code)

    return maps


def collect_target_records(input_path: Path, target_group: str) -> set[Tuple[str, str, str]]:
    records: set[Tuple[str, str, str]] = set()
    with input_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if target_group and row.get("suggested_model_group") != target_group:
                continue
            records.add(
                (
                    row.get("device_name", ""),
                    row.get("station_id", ""),
                    row.get("station_name", ""),
                )
            )
    return records


def payload_from_background_row(row: dict) -> dict:
    return {column: row.get(column, "") for column in BACKGROUND_VALUE_COLUMNS}


def payload_completeness(payload: dict) -> int:
    return sum(1 for column in BACKGROUND_VALUE_COLUMNS if payload.get(column))


def first_put(index: Dict, key, payload: dict, count_key, counts: Counter) -> None:
    if key in ("", None):
        return
    if isinstance(key, tuple) and any(part in ("", None) for part in key):
        return
    counts[count_key] += 1
    index.setdefault(key, payload)


def station_alias(name: str) -> str:
    value = (name or "").strip()
    value = re.sub(r"^(?:\d+(?:\.\d+)?kV|\d+KV)", "", value)
    return value.replace("变电站", "站")


def classify_target_device(name: str) -> Tuple[str, list[str]]:
    value = (name or "").strip()
    body = re.sub(r"^\d+\s*kV", "", value, flags=re.IGNORECASE)
    if not value:
        return "empty", []
    if re.fullmatch(r"\d{3,4}", value):
        return "number", [value]

    number_match = re.search(r"#(\d{2,4})D?", body)
    if number_match and "压电容器" in value:
        return "capacitor", [number_match.group(1)]
    if number_match and "压电抗器" in value:
        return "reactor", [number_match.group(1)]

    transformer_match = re.search(r"#(\d+)Z变", body)
    if transformer_match and "绕组" in value:
        return "winding", [transformer_match.group(1)]

    numbers = re.findall(r"(\d{3,4})", body)
    if value.endswith("线路") and numbers:
        return "line_interval", [numbers[-1]]
    if "开关" in value and numbers:
        return "switch", [numbers[-1]]
    if "线" in value:
        return "line_name", [value]
    return "other", numbers[-1:] if numbers else [value]


def station_candidate_score(row: dict, kind: str, tokens: list[str]) -> int:
    device_name = row.get("device_name", "")
    full_path = row.get("full_path", "")
    source_table = row.get("source_table", "")
    device_type = row.get("device_type", "")

    if kind == "line_name":
        if source_table == "DM_输电线路" and device_name == tokens[0]:
            return 95
        return 0

    if kind == "winding":
        if not tokens:
            return 0
        transformer_no = tokens[0]
        if source_table == "DM_主网功能位置" and device_name == f"#{transformer_no}主变压器":
            return 93
        if source_table == "DM_主网功能位置" and device_name == f"#{transformer_no}主变间隔":
            return 88
        if (
            source_table == "DM_主网设备"
            and f"#{transformer_no}主变变" in device_name
            and "间隔" in device_name
        ):
            return 84
        return 0

    if not tokens:
        return 0

    token = tokens[0]
    if kind in ("number", "line_interval", "switch"):
        if token not in device_name and token not in full_path:
            return 0
        if source_table == "DM_主网功能位置" and token in device_name and "线路开关间隔" in device_type:
            return 92
        if source_table == "DM_主网设备" and token in device_name and "间隔" in device_name:
            return 90
        if source_table == "DM_主网功能位置" and device_name == f"{token}开关":
            return 87
        if (
            source_table == "DM_主网设备"
            and f"{token}开关" in full_path
            and ("断路器" in device_type or "断路器" in device_name)
        ):
            return 82
        return 0

    if kind in ("capacitor", "reactor"):
        if token not in device_name and token not in full_path:
            return 0
        expected_name = "电容" if kind == "capacitor" else "电抗"
        if source_table == "DM_主网功能位置" and expected_name in device_name and "间隔" in device_name:
            return 94
        if source_table == "DM_主网功能位置" and device_name == f"{token}开关":
            return 84
        if (
            source_table == "DM_主网设备"
            and f"{token}开关" in full_path
            and ("断路器" in device_type or "断路器" in device_name)
        ):
            return 80
        return 0

    return 0


def confidence_from_score(score: int) -> str:
    if score >= 92:
        return "0.90"
    if score >= 87:
        return "0.85"
    return "0.80"


def load_unique_station_aliases(background_path: Path) -> set[str]:
    aliases: Dict[str, set[str]] = defaultdict(set)

    with background_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("source_table") != "DM_变电站":
                continue
            alias = station_alias(row.get("station_name", ""))
            if alias:
                aliases[alias].add(row.get("station_name", ""))

    return {alias for alias, names in aliases.items() if len(names) == 1}


def voltage_hint_from_device_name(device_name: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*kV", device_name or "", flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def build_allowed_station_names(
    background_path: Path,
    target_records: Iterable[Tuple[str, str, str]],
) -> Dict[Tuple[str, str], set[str]]:
    target_records = list(target_records)
    dm_station_names: Dict[str, set[str]] = defaultdict(set)
    dm_station_voltage: Dict[str, float] = {}

    with background_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("source_table") != "DM_变电站":
                continue

            name = row.get("station_name", "")
            alias = station_alias(name)
            if not alias:
                continue

            dm_station_names[alias].add(name)
            try:
                dm_station_voltage[name] = float(row.get("voltage_level_kv", ""))
            except ValueError:
                pass

    max_voltage_by_target_station: Dict[Tuple[str, str], float] = {}
    for device, station_id, station_name in target_records:
        voltage = voltage_hint_from_device_name(device)
        if voltage is None:
            continue
        key = (station_name, station_id)
        max_voltage_by_target_station[key] = max(
            voltage,
            max_voltage_by_target_station.get(key, 0.0),
        )

    allowed: Dict[Tuple[str, str], set[str]] = {}
    for _device, station_id, station_name in target_records:
        key = (station_name, station_id)
        if key in allowed:
            continue

        names = dm_station_names.get(station_name, set())
        if len(names) == 1:
            allowed[key] = set(names)
            continue

        max_voltage = max_voltage_by_target_station.get(key)
        if max_voltage is None:
            allowed[key] = set()
            continue

        exact = {
            name
            for name in names
            if dm_station_voltage.get(name) == max_voltage
        }
        allowed[key] = exact if len(exact) == 1 else set()

    return allowed


def build_background_matcher(
    background_path: Path,
    code_path: Path,
    target_records: Iterable[Tuple[str, str, str]],
) -> Optional[BackgroundMatcher]:
    if not background_path.exists():
        return None

    code_maps = load_code_maps(code_path)
    target_records = set(target_records)
    target_pairs = {
        (device, station)
        for device, _station_id, station in target_records
    }
    target_device_names = {device for device, _station in target_pairs if device}
    target_device_norms = {
        code_maps["device_name_norm"][device]
        for device in target_device_names
        if device in code_maps["device_name_norm"]
    }
    target_records_by_station: Dict[
        str,
        list[Tuple[str, str, str, str, list[str]]],
    ] = defaultdict(list)
    for device, station_id, station in target_records:
        kind, tokens = classify_target_device(device)
        target_records_by_station[station].append(
            (device, station_id, station, kind, tokens)
        )

    allowed_station_names = build_allowed_station_names(background_path, target_records)
    norm_pair_keys = {
        (
            code_maps["device_name_norm"].get(device, ""),
            code_maps["station_name_norm"].get(station, ""),
        )
        for device, station in target_pairs
    }
    norm_pair_keys = {key for key in norm_pair_keys if key[0] and key[1]}

    norm_pair_index: Dict[Tuple[str, str], dict] = {}
    raw_pair_index: Dict[Tuple[str, str], dict] = {}
    line_norm_device_index: Dict[str, dict] = {}
    line_raw_device_index: Dict[str, dict] = {}
    station_device_index: Dict[Tuple[str, str, str], dict] = {}
    counts: Counter = Counter()

    with background_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "device_name",
            "device_name_norm",
            "station_name",
            "station_name_norm",
            "source_table",
            *BACKGROUND_VALUE_COLUMNS,
        }
        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"{background_path} 缺少字段：{missing}")

        for row in reader:
            payload = payload_from_background_row(row)
            norm_pair = (row.get("device_name_norm", ""), row.get("station_name_norm", ""))
            raw_pair = (row.get("device_name", ""), row.get("station_name", ""))
            device_norm = row.get("device_name_norm", "")
            device_name = row.get("device_name", "")

            if norm_pair in norm_pair_keys:
                first_put(
                    norm_pair_index,
                    norm_pair,
                    payload,
                    ("norm_pair", norm_pair),
                    counts,
                )
            if raw_pair in target_pairs:
                first_put(
                    raw_pair_index,
                    raw_pair,
                    payload,
                    ("raw_pair", raw_pair),
                    counts,
                )
            if row.get("source_table") == "DM_输电线路":
                if device_norm in target_device_norms:
                    first_put(
                        line_norm_device_index,
                        device_norm,
                        payload,
                        ("line_norm_device", device_norm),
                        counts,
                    )
                if device_name in target_device_names:
                    first_put(
                        line_raw_device_index,
                        device_name,
                        payload,
                        ("line_raw_device", device_name),
                        counts,
                    )

            alias = station_alias(row.get("station_name", ""))
            if alias in target_records_by_station:
                for target_device, target_station_id, target_station_name, kind, tokens in (
                    target_records_by_station[alias]
                ):
                    allowed_names = allowed_station_names.get(
                        (target_station_name, target_station_id),
                        set(),
                    )
                    if row.get("station_name", "") not in allowed_names:
                        continue

                    score = station_candidate_score(row, kind, tokens)
                    if score <= 0:
                        continue

                    candidate = payload_from_background_row(row)
                    candidate["background_match_confidence"] = confidence_from_score(score)
                    candidate["_score"] = score
                    key = (target_device, target_station_id, target_station_name)
                    existing = station_device_index.get(key)
                    if existing is None or (
                        score,
                        payload_completeness(candidate),
                    ) > (
                        int(existing.get("_score", 0)),
                        payload_completeness(existing),
                    ):
                        station_device_index[key] = candidate

    return BackgroundMatcher(
        code_maps=code_maps,
        norm_pair_index=norm_pair_index,
        raw_pair_index=raw_pair_index,
        line_norm_device_index=line_norm_device_index,
        line_raw_device_index=line_raw_device_index,
        station_device_index=station_device_index,
        counts=counts,
    )


def build_model_base(
    input_path: Path,
    output_path: Path,
    background_path: Path,
    code_path: Path,
    background_target_group: str,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_records = collect_target_records(input_path, background_target_group)
    matcher = build_background_matcher(background_path, code_path, target_records)

    rows_read = 0
    rows_written = 0
    background_matched_rows = 0
    background_unmatched_rows = 0
    match_methods: Counter = Counter()
    dates = set()
    devices = set()

    with input_path.open(newline="", encoding="utf-8-sig") as in_f, output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=MODEL_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for row in reader:
            rows_read += 1
            out = {column: row.get(column, "") for column in MODEL_COLUMNS}

            # These event fields need defect/trip/maintenance inputs.
            for column in EVENT_COLUMNS:
                out[column] = ""

            if matcher:
                background = matcher.match(
                    out.get("device_name", ""),
                    out.get("station_id", ""),
                    out.get("station_name", ""),
                )
                if background:
                    for column in BACKGROUND_COLUMNS:
                        out[column] = background.get(column, "")
                    background_matched_rows += 1
                    match_methods[background.get("_match_method", "")] += 1
                else:
                    for column in BACKGROUND_COLUMNS:
                        out[column] = ""
                    out["background_match_confidence"] = "0.00"
                    background_unmatched_rows += 1
            else:
                for column in BACKGROUND_COLUMNS:
                    out[column] = ""

            writer.writerow(out)
            rows_written += 1

            if out.get("date"):
                dates.add(out["date"])
            if out.get("unified_device_id"):
                devices.add(out["unified_device_id"])

    return {
        "rows_read": rows_read,
        "rows_written": rows_written,
        "date_count": len(dates),
        "device_count": len(devices),
        "background_enabled": matcher is not None,
        "background_matched_rows": background_matched_rows,
        "background_unmatched_rows": background_unmatched_rows,
        "background_unique_matched_pairs": len(
            [value for value in (matcher.cache.values() if matcher else []) if value]
        ),
        "background_target_group": background_target_group,
        "match_methods": dict(match_methods),
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "生成任务6设备日模型基础表。当前版本只使用 fact_operation_device_day_wide.csv，"
            "并在有 DM 设备统一视图和编码对照表时补充背景字段。"
        )
    )
    parser.add_argument(
        "--input",
        default="output/clean/02_operation/fact_operation_device_day_wide.csv",
        help="任务5输出文件路径",
    )
    parser.add_argument(
        "--output",
        default="output/clean/04_model_base/model_base_device_day.csv",
        help="任务6输出文件路径",
    )
    parser.add_argument(
        "--background",
        default="data/DM_设备统一视图_norm.csv",
        help="DM 设备统一视图路径；文件不存在时背景字段保留为空",
    )
    parser.add_argument(
        "--code-map",
        default="data/编码对照表.csv",
        help="编码对照表路径，用于把设备名/站点名转换为标准化编码",
    )
    parser.add_argument(
        "--background-target-group",
        default=DEFAULT_BACKGROUND_TARGET_GROUP,
        help=(
            "需要补背景字段的 suggested_model_group；默认只补 line_or_load_like，"
            "避免对其他分组做过宽的路径匹配"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_model_base(
        Path(args.input),
        Path(args.output),
        Path(args.background),
        Path(args.code_map),
        args.background_target_group,
    )
    print(f"输入行数：{stats['rows_read']}")
    print(f"输出行数：{stats['rows_written']}")
    print(f"日期数：{stats['date_count']}")
    print(f"设备数：{stats['device_count']}")
    print(f"背景表启用：{stats['background_enabled']}")
    print(f"背景匹配行数：{stats['background_matched_rows']}")
    print(f"背景未匹配行数：{stats['background_unmatched_rows']}")
    print(f"背景匹配设备站点数：{stats['background_unique_matched_pairs']}")
    print(f"背景匹配目标分组：{stats['background_target_group']}")
    print(f"背景匹配方式：{stats['match_methods']}")
    print(f"输出文件：{stats['output_path']}")


if __name__ == "__main__":
    main()
