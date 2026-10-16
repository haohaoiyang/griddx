#!/usr/bin/env python3
"""用真实主网、缺陷、检修和跳闸数据增强 7.15 设备日表。"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_service.app.family import (
    derive_production_batch,
    make_family_id,
    normalize_family_component,
)


DEFAULT_ROOT = Path("data/7.15数据生成")
DEFAULT_SOURCE = DEFAULT_ROOT / "处理后数据"
DEFAULT_OUTPUT = Path("output/7.15数据生成")

DEVICE_EVENT_COLUMNS = [
    "history_defect_count",
    "history_trip_count",
    "history_maintenance_count",
    "history_defect_level_code",
    "history_defect_type_code",
    "history_defect_reason_code",
    "history_trip_level_code",
    "history_trip_type_code",
    "history_trip_reason_code",
    "defect_related_maintenance_count",
    "trip_related_maintenance_count",
    "defect_maintenance_status_code",
    "trip_maintenance_status_code",
    "maintenance_relation_code",
    "family_device_count",
    "family_defect_device_count",
    "family_history_defect_count",
    "family_history_trip_count",
    "family_history_maintenance_count",
    "family_history_defect_level_code",
    "family_history_defect_type_code",
    "family_history_defect_reason_code",
    "family_history_trip_level_code",
    "family_history_trip_type_code",
    "family_history_trip_reason_code",
]

FAMILY_IDENTITY_COLUMNS = [
    "family_id",
    "family_manufacturer",
    "family_model",
    "family_batch",
    "family_batch_source",
]

DROP_COLUMNS = {
    "background_match_confidence",
    "defect_count_today",
    "trip_count_today",
    "maintenance_count_today",
}

MEASUREMENT_COLUMNS = [
    "current_3phase",
    "voltage",
    "active_power_3phase",
    "reactive_power_3phase",
    "frequency",
    "tap_position",
    "switch_position",
]

LEVEL_CODES = {"其他": 1, "一般": 2, "普通": 2, "重大": 3, "紧急": 4}


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def normalize_identity(value: str) -> str:
    return clean_text(value).upper()


def normalize_station(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"^[-+±]?\d+(?:\.\d+)?\s*[kK][vV]", "", text)
    text = text.replace("变电站", "").replace("站", "")
    return re.sub(r"[\s/()（）#]", "", text)


def device_key(row: dict[str, str]) -> str:
    return row.get("unified_device_id") or "|".join(
        [row.get("station_id", ""), row.get("station_name", ""), row.get("device_name", "")]
    )


def parse_date(value: str) -> Optional[date]:
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def parse_voltage(value: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[kK][vV]", value or "")
    if not match:
        return ""
    numeric = float(match.group(1))
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def strip_voltage_tokens(value: str) -> str:
    return re.sub(r"[-+±]?\d+(?:\.\d+)?\s*[kK][vV]", "", value or "", flags=re.I)


def extract_business_number(device_name: str) -> str:
    numbers = re.findall(r"\d+", strip_voltage_tokens(device_name))
    return numbers[-1] if numbers else ""


def exact_number_pattern(number: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\d){re.escape(number)}(?!\d)")


def classify_semantic(device_name: str) -> str:
    name = device_name or ""
    if "电容" in name:
        return "capacitor"
    if "电抗" in name or "抗器" in name:
        return "reactor"
    if "主变" in name or "Z变" in name or "绕组" in name:
        return "transformer"
    if "母联" in name or "M联" in name:
        return "bus_coupler"
    if "分段" in name or "F段" in name:
        return "section"
    if re.search(r"PT|电压互感器", name, flags=re.I):
        return "voltage_transformer"
    if "电机" in name:
        return "motor"
    if "负荷" in name:
        return "load"
    if "线" in name or "线路" in name:
        return "line"
    if "开关" in name:
        return "switch"
    if re.fullmatch(r"\d{2,5}", clean_text(name)):
        return "switch_like"
    return "other"


def semantic_matches_text(semantic: str, text: str, number: str) -> bool:
    corpus = text or ""
    if semantic == "capacitor":
        return "电容" in corpus
    if semantic == "reactor":
        return "电抗" in corpus or "抗器" in corpus
    if semantic == "transformer":
        return "主变" in corpus or "变压器" in corpus
    if semantic == "bus_coupler":
        return "母联" in corpus or "M联" in corpus
    if semantic == "section":
        return "分段" in corpus or "F段" in corpus
    if semantic == "voltage_transformer":
        return bool(re.search(r"PT|电压互感器", corpus, flags=re.I))
    if semantic == "motor":
        return "电机" in corpus
    if semantic == "load":
        return "负荷" in corpus or "线" in corpus
    if semantic == "line":
        return "线" in corpus
    if semantic in {"switch", "switch_like"}:
        return bool(
            re.search(
                rf"(?<!\d){re.escape(number)}(?!\d)[^/;，,]*?(?:开关|断路器)",
                corpus,
            )
        )
    return True


BREAKER_TARGET_SEMANTICS = {
    "switch",
    "switch_like",
    "capacitor",
    "reactor",
    "bus_coupler",
    "section",
}


def is_breaker_candidate(candidate: dict[str, str]) -> bool:
    path_leaf = (candidate.get("path", "").rstrip("/").split("/") or [""])[-1]
    type_leaf = (candidate.get("device_type_name", "").split("/") or [""])[-1]
    return any(
        "断路器" in text
        for text in (candidate.get("device_name", ""), type_leaf, path_leaf)
    )


def normalize_match_path(path: str) -> str:
    return re.sub(r"\s+", "", path or "").replace("／", "/").rstrip("/")


def candidate_path_rank(path: str, number: str) -> tuple[int, int]:
    pattern = exact_number_pattern(number)
    best = (99, 99)
    segments = (path or "").split("/")
    for distance, segment in enumerate(reversed(segments)):
        if not pattern.search(segment):
            continue
        if "开关" in segment or "断路器" in segment:
            kind_rank = 0
        elif any(
            word in segment
            for word in ("主变", "变压器", "电容", "电抗", "抗器", "线路", "PT", "电压互感器")
        ):
            kind_rank = 1
        elif "间隔" in segment:
            kind_rank = 2
        else:
            kind_rank = 3
        best = min(best, (kind_rank, distance))
    return best


def has_operation(row: dict[str, str]) -> bool:
    for column, value in row.items():
        if column.endswith("_daily_present"):
            try:
                if float(value or 0) > 0:
                    return True
            except ValueError:
                pass
    return any(clean_text(row.get(column, "")) for column in MEASUREMENT_COLUMNS)


def read_target_devices(path: Path) -> tuple[dict[str, dict], set[date]]:
    devices: dict[str, dict] = {}
    dates: set[date] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            row_date = parse_date(row.get("date", ""))
            if row_date:
                dates.add(row_date)
            key = device_key(row)
            if not key or key in devices or not row_date:
                continue
            name = row.get("device_name", "")
            devices[key] = {
                "device_key": key,
                "unified_device_id": row.get("unified_device_id", ""),
                "device_name": name,
                "station_id": row.get("station_id", ""),
                "station_name": row.get("station_name", ""),
                "station_norm": normalize_station(row.get("station_name", "")),
                "business_number": extract_business_number(name),
                "semantic": classify_semantic(name),
                "target_voltage": parse_voltage(name),
            }
    return devices, dates


def read_dm_candidates(path: Path, devices: dict[str, dict]) -> tuple[dict, set[str]]:
    station_codes: dict[str, set[str]] = defaultdict(set)
    for device in devices.values():
        if device["station_norm"] and device["business_number"]:
            station_codes[device["station_norm"]].add(device["business_number"])

    candidates: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    dm_stations: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            station = normalize_station(row.get("所属变电站", ""))
            dm_stations.add(station)
            wanted = station_codes.get(station)
            if not wanted:
                continue
            path_text = row.get("全路径", "")
            path_numbers = set(re.findall(r"\d+", path_text))
            for number in wanted & path_numbers:
                production_batch, batch_source = derive_production_batch(row)
                candidate = {
                    "device_id": clean_text(row.get("设备ID", "")),
                    "identity_code": clean_text(row.get("设备身份编码", "")),
                    "device_name": clean_text(row.get("设备名称", "")),
                    "device_model": clean_text(row.get("设备型号", "")),
                    "device_type_name": clean_text(row.get("设备类别名称", "")),
                    "voltage_level": clean_text(row.get("电压等级", "")),
                    "operation_status": clean_text(row.get("运行状态", "")),
                    "manufacturer": clean_text(
                        row.get("厂家现用名", "") or row.get("厂家名称", "")
                    ),
                    "commission_date": clean_text(row.get("投运日期", "")),
                    "production_batch": production_batch,
                    "batch_source": batch_source,
                    "station_name": clean_text(row.get("所属变电站", "")),
                    "path": path_text,
                }
                candidate["path_rank"] = candidate_path_rank(path_text, number)
                candidate_id = (
                    candidate["device_id"]
                    or candidate["identity_code"]
                    or "|".join([candidate["device_name"], candidate["path"]])
                )
                candidates[(station, number)].setdefault(candidate_id, candidate)
    return candidates, dm_stations


def match_devices(
    devices: dict[str, dict], candidates: dict, dm_stations: set[str]
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    matches: dict[str, dict] = {}
    results: list[dict] = []
    issues: list[dict] = []

    for key, device in devices.items():
        station = device["station_norm"]
        number = device["business_number"]
        layer1: list[dict] = []
        semantic_candidates: list[dict] = []
        role_candidates: list[dict] = []
        best_candidates: list[dict] = []
        applied_rules: list[str] = []

        if not number:
            status = "no_number"
        elif station not in dm_stations:
            status = "station_not_found"
        else:
            layer1 = list(candidates.get((station, number), {}).values())
            if not layer1:
                status = "number_not_found"
            else:
                target_voltage = device["target_voltage"]
                if target_voltage:
                    same_voltage = [
                        item
                        for item in layer1
                        if parse_voltage(item["voltage_level"]) == target_voltage
                    ]
                    if same_voltage:
                        layer1 = same_voltage

                semantic_candidates = [
                    item
                    for item in layer1
                    if semantic_matches_text(
                        device["semantic"],
                        "|".join(
                            [item["path"], item["device_name"], item["device_type_name"]]
                        ),
                        number,
                    )
                ]
                if not semantic_candidates:
                    status = "semantic_not_found"
                else:
                    role_candidates = semantic_candidates
                    if device["semantic"] in BREAKER_TARGET_SEMANTICS:
                        breaker_candidates = [
                            item for item in semantic_candidates if is_breaker_candidate(item)
                        ]
                        if breaker_candidates:
                            role_candidates = breaker_candidates
                            if len(role_candidates) < len(semantic_candidates):
                                applied_rules.append("breaker_role_filter")

                    best_rank = min(item["path_rank"] for item in role_candidates)
                    best_candidates = [
                        item for item in role_candidates if item["path_rank"] == best_rank
                    ]
                    if len(best_candidates) > 1:
                        normalized_paths = {
                            normalize_match_path(item["path"])
                            for item in best_candidates
                        }
                        running_candidates = [
                            item
                            for item in best_candidates
                            if item["operation_status"] == "运行"
                        ]
                        if len(normalized_paths) == 1 and len(running_candidates) == 1:
                            best_candidates = running_candidates
                            applied_rules.append("same_path_running_preferred")

                    if len(best_candidates) == 1:
                        status = "matched"
                        matches[key] = best_candidates[0]
                    else:
                        status = "ambiguous"

        selected = matches.get(key, {})
        results.append(
            {
                "match_status": status,
                "device_key": key,
                "unified_device_id": device["unified_device_id"],
                "station_name": device["station_name"],
                "device_name": device["device_name"],
                "business_number": number,
                "semantic": device["semantic"],
                "applied_rules": "+".join(applied_rules),
                "layer1_candidate_count": len(layer1),
                "semantic_candidate_count": len(semantic_candidates),
                "role_candidate_count": len(role_candidates),
                "best_candidate_count": len(best_candidates),
                "matched_device_id": selected.get("device_id", ""),
                "matched_identity_code": selected.get("identity_code", ""),
                "matched_device_name": selected.get("device_name", ""),
                "matched_device_model": selected.get("device_model", ""),
                "matched_manufacturer": selected.get("manufacturer", ""),
                "matched_path": selected.get("path", ""),
            }
        )

        if status == "matched":
            continue
        issue_candidates = best_candidates or role_candidates or semantic_candidates or layer1
        if not issue_candidates:
            issue_candidates = [{}]
        for candidate in issue_candidates:
            issues.append(
                {
                    "match_status": status,
                    "device_key": key,
                    "unified_device_id": device["unified_device_id"],
                    "station_name": device["station_name"],
                    "device_name": device["device_name"],
                    "business_number": number,
                    "semantic": device["semantic"],
                    "applied_rules": "+".join(applied_rules),
                    "candidate_path_rank": "|".join(
                        str(value) for value in candidate.get("path_rank", ())
                    ),
                    "candidate_device_id": candidate.get("device_id", ""),
                    "candidate_identity_code": candidate.get("identity_code", ""),
                    "candidate_device_name": candidate.get("device_name", ""),
                    "candidate_device_model": candidate.get("device_model", ""),
                    "candidate_voltage_level": candidate.get("voltage_level", ""),
                    "candidate_operation_status": candidate.get("operation_status", ""),
                    "candidate_manufacturer": candidate.get("manufacturer", ""),
                    "candidate_path": candidate.get("path", ""),
                }
            )
    return matches, results, issues


class CodeRegistry:
    def __init__(self, existing_path: Path):
        self.path = existing_path
        self.maps: dict[str, dict[str, int]] = defaultdict(dict)
        self.meta: dict[tuple[str, str], tuple[str, str]] = {}
        if existing_path.exists():
            with existing_path.open(newline="", encoding="utf-8-sig") as file:
                for row in csv.DictReader(file):
                    field = row.get("field_name", "")
                    value = row.get("source_value", "")
                    code = row.get("code", "")
                    if field and value and code.isdigit() and int(code) > 0:
                        self.maps[field][value] = int(code)
                        self.meta[(field, value)] = (
                            row.get("source_table", ""),
                            row.get("source_column", ""),
                        )

    def register_many(
        self, field: str, values: Iterable[str], source_table: str, source_column: str
    ) -> None:
        mapping = self.maps[field]
        next_code = max(mapping.values(), default=0) + 1
        for value in sorted({clean_text(value) for value in values if clean_text(value)}):
            if value not in mapping:
                mapping[value] = next_code
                next_code += 1
            self.meta[(field, value)] = (source_table, source_column)

    def code(self, field: str, value: str) -> int:
        return self.maps.get(field, {}).get(clean_text(value), 0)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["field_name", "code", "source_value", "source_table", "source_column"]
        fixed_rows = [
            ("operation_status_code", 0, "当日无运行量测"),
            ("operation_status_code", 1, "当日有运行量测"),
            ("history_defect_level_code", 0, "无缺陷"),
            ("history_defect_level_code", 1, "其他"),
            ("history_defect_level_code", 2, "一般/普通"),
            ("history_defect_level_code", 3, "重大"),
            ("history_defect_level_code", 4, "紧急"),
            ("history_trip_level_code", 0, "固定等级"),
            ("family_history_trip_level_code", 0, "固定等级"),
            ("defect_maintenance_status_code", 0, "无缺陷"),
            ("defect_maintenance_status_code", 1, "有缺陷且有对应检修"),
            ("defect_maintenance_status_code", 2, "有缺陷但无对应检修"),
            ("trip_maintenance_status_code", 0, "无跳闸"),
            ("trip_maintenance_status_code", 1, "有跳闸且有对应检修"),
            ("trip_maintenance_status_code", 2, "有跳闸但无对应检修"),
            ("maintenance_relation_code", 0, "无关联检修"),
            ("maintenance_relation_code", 1, "关联缺陷"),
            ("maintenance_relation_code", 2, "关联跳闸"),
            ("maintenance_relation_code", 3, "同时关联缺陷和跳闸"),
        ]
        with self.path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for field in sorted(self.maps):
                for value, code in sorted(self.maps[field].items(), key=lambda item: item[1]):
                    table, column = self.meta.get((field, value), ("", ""))
                    writer.writerow(
                        {
                            "field_name": field,
                            "code": code,
                            "source_value": value,
                            "source_table": table,
                            "source_column": column,
                        }
                    )
            for field, code, value in fixed_rows:
                writer.writerow(
                    {
                        "field_name": field,
                        "code": code,
                        "source_value": value,
                        "source_table": "规则",
                        "source_column": "",
                    }
                )


def build_physical_profiles(
    devices: dict[str, dict], matches: dict[str, dict]
) -> tuple[dict[str, str], dict[str, dict]]:
    target_to_physical: dict[str, str] = {}
    profiles: dict[str, dict] = {}
    for target_key, candidate in matches.items():
        physical_key = (
            candidate.get("device_id")
            or normalize_identity(candidate.get("identity_code", ""))
            or target_key
        )
        target_to_physical[target_key] = physical_key
        if physical_key not in profiles:
            profiles[physical_key] = {
                **candidate,
                "physical_key": physical_key,
                "target_keys": [],
            }
        profiles[physical_key]["target_keys"].append(target_key)
    return target_to_physical, profiles


def load_defects(path: Path, profiles: dict[str, dict]) -> dict[str, list[dict]]:
    identity_to_physical: dict[str, set[str]] = defaultdict(set)
    for physical_key, profile in profiles.items():
        identity = normalize_identity(profile.get("identity_code", ""))
        if identity:
            identity_to_physical[identity].add(physical_key)

    events: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            physical_keys = identity_to_physical.get(normalize_identity(row.get("设备编码", "")))
            if not physical_keys:
                continue
            event_id = clean_text(row.get("缺陷编号", "")) or f"row-{row_number}"
            event_date = (
                parse_date(row.get("发现时间", ""))
                or parse_date(row.get("创建时间", ""))
                or parse_date(row.get("填报时间", ""))
                or date.min
            )
            level_text = clean_text(row.get("缺陷等级", ""))
            event = {
                "event_id": event_id,
                "date": event_date,
                "level": LEVEL_CODES.get(level_text, 0),
                "type_text": clean_text(row.get("缺陷类型", "")),
                "reason_text": clean_text(
                    row.get("缺陷原因", "") or row.get("缺陷原因类别", "")
                ),
            }
            for physical_key in physical_keys:
                events[physical_key].setdefault(event_id, event)
    return {
        key: sorted(items.values(), key=lambda item: (item["date"], item["event_id"]))
        for key, items in events.items()
    }


def trip_targets(
    devices: dict[str, dict], target_to_physical: dict[str, str]
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    targets: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for target_key, physical_key in target_to_physical.items():
        device = devices[target_key]
        station = device["station_norm"]
        number = device["business_number"]
        if station and number:
            targets[(station, number)].append((physical_key, device["semantic"]))
    return targets


def load_trip_file(
    path: Path,
    encoding: str,
    targets: dict[tuple[str, str], list[tuple[str, str]]],
    station_column: str,
    text_column: str,
    type_column: str,
    reason_column: str,
    reason_sub_column: str = "",
) -> dict[str, list[dict]]:
    events: dict[str, dict[str, dict]] = defaultdict(dict)
    station_numbers: dict[str, set[str]] = defaultdict(set)
    for station, number in targets:
        station_numbers[station].add(number)

    with path.open(newline="", encoding=encoding, errors="replace") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            station = normalize_station(row.get(station_column, ""))
            wanted = station_numbers.get(station)
            if not wanted:
                continue
            text = row.get(text_column, "")
            number_text = strip_voltage_tokens(text)
            row_numbers = set(re.findall(r"\d+", number_text))
            event_id = clean_text(row.get("跳闸单编号", "")) or f"{path.name}-row-{row_number}"
            event_date = parse_date(row.get("跳闸时间", "")) or date.min
            reason = clean_text(row.get(reason_column, ""))
            reason_sub = clean_text(row.get(reason_sub_column, "")) if reason_sub_column else ""
            if reason_sub:
                reason = f"{reason}/{reason_sub}" if reason else reason_sub
            event = {
                "event_id": event_id,
                "date": event_date,
                "type_text": clean_text(row.get(type_column, "")),
                "reason_text": reason,
            }
            for number in wanted & row_numbers:
                for physical_key, semantic in targets[(station, number)]:
                    corpus = "|".join(
                        [text, row.get(type_column, ""), row.get("故障设备", "")]
                    )
                    if not semantic_matches_text(semantic, corpus, number):
                        continue
                    events[physical_key].setdefault(event_id, event)
    return {
        key: sorted(items.values(), key=lambda item: (item["date"], item["event_id"]))
        for key, items in events.items()
    }


def merge_event_maps(*maps: dict[str, list[dict]]) -> dict[str, list[dict]]:
    merged: dict[str, dict[str, dict]] = defaultdict(dict)
    for event_map in maps:
        for key, events in event_map.items():
            for event in events:
                merged[key].setdefault(event["event_id"], event)
    return {
        key: sorted(items.values(), key=lambda item: (item["date"], item["event_id"]))
        for key, items in merged.items()
    }


def normalize_maintenance_match_text(value: str) -> str:
    text = clean_text(value).replace("变电站", "站").replace("Z变", "主变")
    return re.sub(r"[\s/\\()（）#\[\]【】;；,，。、]", "", text).upper()


def load_maintenance(
    main_path: Path,
    device_path: Path,
    profiles: dict[str, dict],
    devices: dict[str, dict],
    target_to_physical: dict[str, str],
) -> dict[str, list[dict]]:
    device_id_to_physical: dict[str, set[str]] = defaultdict(set)
    for physical_key, profile in profiles.items():
        device_id = clean_text(profile.get("device_id", ""))
        if device_id:
            device_id_to_physical[device_id].add(physical_key)

    station_number_targets: dict[tuple[str, str], list[str]] = defaultdict(list)
    number_targets: dict[str, list[str]] = defaultdict(list)
    for target_key, physical_key in target_to_physical.items():
        device = devices[target_key]
        station = device["station_norm"]
        number = device["business_number"]
        if station and number:
            station_number_targets[(station, number)].append(target_key)
            number_targets[number].append(target_key)

    station_variants = {
        station: normalize_maintenance_match_text(f"{station}站")
        for station, _number in station_number_targets
    }

    order_aliases: dict[str, dict] = {}
    maintenance: dict[str, dict[str, dict]] = defaultdict(dict)
    with main_path.open(newline="", encoding="utf-8-sig") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            order_id = clean_text(row.get("ID", "") or row.get("YWID", "") or row.get("OBJ_ID", ""))
            if not order_id:
                order_id = f"main-row-{row_number}"
            completion_date = (
                parse_date(row.get("SBGZWGSJ", ""))
                or parse_date(row.get("PZGZJSSJ", ""))
                or parse_date(row.get("SQGZJSSJ", ""))
                or parse_date(row.get("DATAMTIME", ""))
            )
            order = {
                "order_id": order_id,
                "date": completion_date or date.min,
                "undated": completion_date is None,
                "status": clean_text(row.get("GZC", "")),
            }
            for column in ("ID", "YWID", "OBJ_ID"):
                alias = clean_text(row.get(column, ""))
                if alias:
                    order_aliases[alias] = order

            if order["status"] == "已作废":
                continue

            current_station = ""
            raw_targets = row.get("TDJXSB", "")
            for raw_segment in re.split(r"[;；，,。、\n]+", raw_targets or ""):
                segment = clean_text(raw_segment)
                if not segment:
                    continue

                normalized_segment = normalize_maintenance_match_text(segment)
                station_hits = [
                    station
                    for station, variant in station_variants.items()
                    if variant and variant in normalized_segment
                ]
                if len(station_hits) == 1:
                    current_station = station_hits[0]
                elif len(station_hits) > 1:
                    current_station = ""

                numbers = set(re.findall(r"\d+", strip_voltage_tokens(segment)))
                if not numbers:
                    continue

                candidate_keys: set[str] = set()
                if current_station:
                    for number in numbers:
                        candidate_keys.update(
                            station_number_targets.get((current_station, number), [])
                        )
                else:
                    for number in numbers:
                        for target_key in number_targets.get(number, []):
                            target_name = devices[target_key]["device_name"]
                            name_core = normalize_maintenance_match_text(
                                strip_voltage_tokens(target_name)
                            )
                            if len(name_core) >= 4 and name_core in normalized_segment:
                                candidate_keys.add(target_key)

                matched_keys: set[str] = set()
                for target_key in candidate_keys:
                    device = devices[target_key]
                    number = device["business_number"]
                    if not semantic_matches_text(device["semantic"], segment, number):
                        continue
                    if (
                        len(number) <= 2
                        and device["semantic"] == "transformer"
                        and not re.search(
                            rf"#?{re.escape(number)}(?:号)?(?:主变|Z变)",
                            segment,
                            flags=re.I,
                        )
                    ):
                        continue
                    matched_keys.add(target_key)

                physical_keys = {
                    target_to_physical[target_key] for target_key in matched_keys
                }
                if len(physical_keys) == 1:
                    physical_key = next(iter(physical_keys))
                    maintenance[physical_key].setdefault(order["order_id"], order)

    # TDJXSB is the primary matching field. Exact SBID linkage is retained as a
    # high-confidence supplement when its parent maintenance order is available.
    with device_path.open(newline="", encoding="utf-8-sig") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            physical_keys = device_id_to_physical.get(clean_text(row.get("SBID", "")))
            if not physical_keys:
                continue
            order = None
            for column in ("SSSJ", "YWID", "OBJ_ID"):
                alias = clean_text(row.get(column, ""))
                if alias and alias in order_aliases:
                    order = order_aliases[alias]
                    break
            if order and order["status"] == "已作废":
                continue
            if not order:
                continue
            for physical_key in physical_keys:
                maintenance[physical_key].setdefault(order["order_id"], order)
    return {
        key: sorted(items.values(), key=lambda item: (item["date"], item["order_id"]))
        for key, items in maintenance.items()
    }


def most_common_code(values: Iterable[int]) -> int:
    counter = Counter(value for value in values if value)
    if not counter:
        return 0
    highest = max(counter.values())
    return min(value for value, count in counter.items() if count == highest)


def paired_order_ids(events: list[dict], orders: list[dict]) -> set[str]:
    event_dates = sorted(event["date"] for event in events)
    paired: set[str] = set()
    event_index = 0
    dated_orders = [order for order in orders if not order.get("undated")]
    undated_orders = [order for order in orders if order.get("undated")]
    for order in sorted(dated_orders, key=lambda item: (item["date"], item["order_id"])):
        if event_index < len(event_dates) and event_dates[event_index] <= order["date"]:
            paired.add(order["order_id"])
            event_index += 1
    for order in sorted(undated_orders, key=lambda item: item["order_id"]):
        if event_index < len(event_dates):
            paired.add(order["order_id"])
            event_index += 1
    return paired


def build_stats_functions(
    profiles: dict[str, dict],
    defects: dict[str, list[dict]],
    trips: dict[str, list[dict]],
    maintenance: dict[str, list[dict]],
    registry: CodeRegistry,
):
    device_cache: dict[tuple[str, date], dict] = {}

    family_members: dict[tuple, list[str]] = defaultdict(list)
    physical_to_family: dict[str, tuple] = {}
    family_metadata: dict[tuple, dict[str, str]] = {}
    for physical_key, profile in profiles.items():
        model = normalize_family_component(profile.get("device_model", ""))
        manufacturer = normalize_family_component(profile.get("manufacturer", ""))
        batch = normalize_family_component(profile.get("production_batch", ""))
        if model and manufacturer and batch:
            family = ("manufacturer_model_batch", manufacturer, model, batch)
            family_metadata.setdefault(
                family,
                {
                    "family_id": make_family_id(manufacturer, model, batch),
                    "family_manufacturer": clean_text(profile.get("manufacturer", "")),
                    "family_model": clean_text(profile.get("device_model", "")),
                    "family_batch": clean_text(profile.get("production_batch", "")),
                    "family_batch_source": clean_text(profile.get("batch_source", "")),
                },
            )
        else:
            family = ("single", physical_key)
            family_metadata[family] = {column: "" for column in FAMILY_IDENTITY_COLUMNS}
        physical_to_family[physical_key] = family
        family_members[family].append(physical_key)

    def device_stats(physical_key: str, target_date: date) -> dict:
        cache_key = (physical_key, target_date)
        if cache_key in device_cache:
            return device_cache[cache_key]

        defect_events = [
            event for event in defects.get(physical_key, []) if event["date"] <= target_date
        ]
        trip_events = [
            event for event in trips.get(physical_key, []) if event["date"] <= target_date
        ]
        orders = [
            event
            for event in maintenance.get(physical_key, [])
            if event.get("undated") or event["date"] <= target_date
        ]
        defect_orders = paired_order_ids(defect_events, orders)
        trip_orders = paired_order_ids(trip_events, orders)

        defect_count = len(defect_events)
        trip_count = len(trip_events)
        stats = {
            "history_defect_count": defect_count,
            "history_trip_count": trip_count,
            "history_maintenance_count": len(orders),
            "history_defect_level_code": max(
                (event["level"] for event in defect_events), default=0
            ),
            "history_defect_type_code": most_common_code(
                registry.code("history_defect_type_code", event["type_text"])
                for event in defect_events
            ),
            "history_defect_reason_code": most_common_code(
                registry.code("history_defect_reason_code", event["reason_text"])
                for event in defect_events
            ),
            "history_trip_level_code": 0,
            "history_trip_type_code": most_common_code(
                registry.code("history_trip_type_code", event["type_text"])
                for event in trip_events
            ),
            "history_trip_reason_code": most_common_code(
                registry.code("history_trip_reason_code", event["reason_text"])
                for event in trip_events
            ),
            "defect_related_maintenance_count": len(defect_orders),
            "trip_related_maintenance_count": len(trip_orders),
            "defect_maintenance_status_code": 0
            if defect_count == 0
            else (1 if defect_orders else 2),
            "trip_maintenance_status_code": 0
            if trip_count == 0
            else (1 if trip_orders else 2),
            "maintenance_relation_code": 3
            if defect_orders and trip_orders
            else (1 if defect_orders else (2 if trip_orders else 0)),
        }
        device_cache[cache_key] = stats
        return stats

    family_cache: dict[tuple[tuple, date], dict] = {}

    def family_stats(physical_key: str, target_date: date) -> dict:
        family = physical_to_family[physical_key]
        cache_key = (family, target_date)
        if cache_key in family_cache:
            return family_cache[cache_key]
        members = family_members[family]
        member_stats = [device_stats(member, target_date) for member in members]

        defect_events = [
            event
            for member in members
            for event in defects.get(member, [])
            if event["date"] <= target_date
        ]
        trip_events = [
            event
            for member in members
            for event in trips.get(member, [])
            if event["date"] <= target_date
        ]
        stats = {
            **family_metadata[family],
            "family_device_count": len(members),
            "family_defect_device_count": sum(
                item["history_defect_count"] > 0 for item in member_stats
            ),
            "family_history_defect_count": sum(
                item["history_defect_count"] for item in member_stats
            ),
            "family_history_trip_count": sum(
                item["history_trip_count"] for item in member_stats
            ),
            "family_history_maintenance_count": sum(
                item["history_maintenance_count"] for item in member_stats
            ),
            "family_history_defect_level_code": max(
                (item["history_defect_level_code"] for item in member_stats), default=0
            ),
            "family_history_defect_type_code": most_common_code(
                registry.code("history_defect_type_code", event["type_text"])
                for event in defect_events
            ),
            "family_history_defect_reason_code": most_common_code(
                registry.code("history_defect_reason_code", event["reason_text"])
                for event in defect_events
            ),
            "family_history_trip_level_code": 0,
            "family_history_trip_type_code": most_common_code(
                registry.code("history_trip_type_code", event["type_text"])
                for event in trip_events
            ),
            "family_history_trip_reason_code": most_common_code(
                registry.code("history_trip_reason_code", event["reason_text"])
                for event in trip_events
            ),
        }
        family_cache[cache_key] = stats
        return stats

    return device_stats, family_stats


def output_columns(input_path: Path) -> list[str]:
    with input_path.open(newline="", encoding="utf-8-sig") as file:
        header = next(csv.reader(file))
    enrichment_columns = {*FAMILY_IDENTITY_COLUMNS, *DEVICE_EVENT_COLUMNS}
    columns = [
        column
        for column in header
        if column not in DROP_COLUMNS and column not in enrichment_columns
    ]
    insert_at = columns.index("device_age_days") + 1 if "device_age_days" in columns else len(columns)
    return (
        columns[:insert_at]
        + FAMILY_IDENTITY_COLUMNS
        + DEVICE_EVENT_COLUMNS
        + columns[insert_at:]
    )


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def enrich_table(
    input_path: Path,
    output_path: Path,
    devices: dict[str, dict],
    target_to_physical: dict[str, str],
    profiles: dict[str, dict],
    registry: CodeRegistry,
    device_stats,
    family_stats,
) -> dict:
    fields = output_columns(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    invalid_rows = 0
    matched_rows = 0

    with input_path.open(newline="", encoding="utf-8-sig") as input_file, output_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as output_file:
        reader = csv.DictReader(input_file)
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            target_date = parse_date(row.get("date", ""))
            key = device_key(row)
            if not target_date or not key or key not in devices:
                invalid_rows += 1
                continue

            enriched = dict(row)
            enriched["operation_status_code"] = "1" if has_operation(row) else "0"
            physical_key = target_to_physical.get(key)
            if not physical_key:
                for column in (
                    "device_type_code",
                    "voltage_level_kv",
                    "manufacturer_code",
                    "device_age_days",
                    *FAMILY_IDENTITY_COLUMNS,
                    *DEVICE_EVENT_COLUMNS,
                ):
                    enriched[column] = ""
            else:
                matched_rows += 1
                profile = profiles[physical_key]
                enriched["device_type_code"] = str(
                    registry.code("device_type_code", profile.get("device_type_name", ""))
                ) or ""
                if enriched["device_type_code"] == "0":
                    enriched["device_type_code"] = ""
                enriched["voltage_level_kv"] = parse_voltage(profile.get("voltage_level", ""))
                enriched["manufacturer_code"] = str(
                    registry.code("manufacturer_code", profile.get("manufacturer", ""))
                )
                if enriched["manufacturer_code"] == "0":
                    enriched["manufacturer_code"] = ""
                commission_date = parse_date(profile.get("commission_date", ""))
                age = (target_date - commission_date).days if commission_date else None
                enriched["device_age_days"] = str(age) if age is not None and age >= 0 else ""

                stats = device_stats(physical_key, target_date)
                stats.update(family_stats(physical_key, target_date))
                for column in (*FAMILY_IDENTITY_COLUMNS, *DEVICE_EVENT_COLUMNS):
                    enriched[column] = str(stats[column])

            # 用户确认跳闸等级不从源表推导，设备和家族等级均固定为 0。
            enriched["history_trip_level_code"] = "0"
            enriched["family_history_trip_level_code"] = "0"

            writer.writerow(enriched)
            rows_written += 1

    return {
        "rows_written": rows_written,
        "invalid_rows": invalid_rows,
        "matched_rows": matched_rows,
        "columns": len(fields),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用真实源表生成 7.15 设备历史和家族字段")
    parser.add_argument(
        "--input", default=str(DEFAULT_ROOT / "model_base_device_day_line_or_load_like_.csv")
    )
    parser.add_argument("--dm-device", default=str(DEFAULT_SOURCE / "DM_主网设备.csv"))
    parser.add_argument("--defect", default=str(DEFAULT_SOURCE / "DM_缺陷数据.csv"))
    parser.add_argument("--maintenance-main", default=str(DEFAULT_SOURCE / "DM_检修单主表_JXSQD.csv"))
    parser.add_argument(
        "--maintenance-device", default=str(DEFAULT_SOURCE / "DM_检修单从表_TD_JXSBLB.csv")
    )
    parser.add_argument("--trip-substation", default=str(DEFAULT_SOURCE / "DM_变电跳闸记录.csv"))
    parser.add_argument("--trip-line", default=str(DEFAULT_SOURCE / "DM_输电跳闸记录.csv"))
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT / "model_base_device_day_line_or_load_like_real.csv")
    )
    parser.add_argument("--match-log", default=str(DEFAULT_OUTPUT / "device_match_result.csv"))
    parser.add_argument("--match-issues", default=str(DEFAULT_OUTPUT / "device_match_issues.csv"))
    parser.add_argument("--code-map", default=str(DEFAULT_OUTPUT / "real_field_code_map.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    match_log_path = Path(args.match_log)
    match_issues_path = Path(args.match_issues)
    code_map_path = Path(args.code_map)

    devices, target_dates = read_target_devices(input_path)
    candidates, dm_stations = read_dm_candidates(Path(args.dm_device), devices)
    matches, match_results, match_issues = match_devices(devices, candidates, dm_stations)
    target_to_physical, profiles = build_physical_profiles(devices, matches)

    defects = load_defects(Path(args.defect), profiles)
    targets = trip_targets(devices, target_to_physical)
    substation_trips = load_trip_file(
        Path(args.trip_substation),
        "gb18030",
        targets,
        "故障设备所在变电站",
        "跳闸开关",
        "设备分类",
        "跳闸原因分类",
    )
    line_trips = load_trip_file(
        Path(args.trip_line),
        "gb18030",
        targets,
        "故障设备所在变电站",
        "跳闸线路",
        "线路分类",
        "跳闸原因分类",
        "跳闸原因子类",
    )
    trips = merge_event_maps(substation_trips, line_trips)
    maintenance = load_maintenance(
        Path(args.maintenance_main),
        Path(args.maintenance_device),
        profiles,
        devices,
        target_to_physical,
    )

    registry = CodeRegistry(code_map_path)
    registry.register_many(
        "device_type_code",
        (profile.get("device_type_name", "") for profile in profiles.values()),
        "DM_主网设备",
        "设备类别名称",
    )
    registry.register_many(
        "manufacturer_code",
        (profile.get("manufacturer", "") for profile in profiles.values()),
        "DM_主网设备",
        "厂家现用名/厂家名称",
    )
    registry.register_many(
        "history_defect_type_code",
        (event["type_text"] for events in defects.values() for event in events),
        "DM_缺陷数据",
        "缺陷类型",
    )
    registry.register_many(
        "history_defect_reason_code",
        (event["reason_text"] for events in defects.values() for event in events),
        "DM_缺陷数据",
        "缺陷原因/缺陷原因类别",
    )
    registry.register_many(
        "history_trip_type_code",
        (event["type_text"] for events in trips.values() for event in events),
        "DM_变电跳闸记录/DM_输电跳闸记录",
        "设备分类/线路分类",
    )
    registry.register_many(
        "history_trip_reason_code",
        (event["reason_text"] for events in trips.values() for event in events),
        "DM_变电跳闸记录/DM_输电跳闸记录",
        "跳闸原因分类/跳闸原因子类",
    )
    registry.write()

    device_stats, family_stats = build_stats_functions(
        profiles, defects, trips, maintenance, registry
    )
    output_stats = enrich_table(
        input_path,
        output_path,
        devices,
        target_to_physical,
        profiles,
        registry,
        device_stats,
        family_stats,
    )

    write_rows(match_log_path, match_results, list(match_results[0].keys()))
    write_rows(match_issues_path, match_issues, list(match_issues[0].keys()))

    status_counts = Counter(row["match_status"] for row in match_results)
    print(f"目标唯一设备数：{len(devices)}")
    print(f"目标日期数：{len(target_dates)}")
    print(f"物理设备匹配数：{len(profiles)}")
    print("匹配状态：" + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))
    print(f"缺陷设备数：{len(defects)}")
    print(f"跳闸设备数：{len(trips)}")
    print(f"检修设备数：{len(maintenance)}")
    print(f"输出行数：{output_stats['rows_written']}")
    print(f"无效输入行数：{output_stats['invalid_rows']}")
    print(f"输出列数：{output_stats['columns']}")
    print(f"主输出：{output_path}")
    print(f"匹配日志：{match_log_path}")
    print(f"匹配问题：{match_issues_path}")
    print(f"编码对照：{code_map_path}")


if __name__ == "__main__":
    main()
