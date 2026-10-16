from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from api_service.app.config import get_settings
from api_service.app.family import make_family_id
from api_service.app.feature_store import FeatureStore


def main() -> int:
    settings = get_settings()
    bundled_seed = settings.api_service_root / "demo/seed"
    bundled_device = bundled_seed / "device_features.csv"
    bundled_station = bundled_seed / "station_features.csv"
    if bundled_device.is_file() and bundled_station.is_file():
        device_source = bundled_device
        station_source = bundled_station
    else:
        device_source = (
            settings.project_root
            / "data/model_base_device_day_line_or_load_like_real.csv"
        )
        station_source = (
            settings.project_root
            / "data/model_base_station_day_extract_enriched.csv"
        )
    if not device_source.is_file() or not station_source.is_file():
        raise FileNotFoundError(
            "Demo seed CSV files are missing from api_service/demo/seed/."
        )

    device = pd.read_csv(device_source, low_memory=False)
    station = pd.read_csv(station_source, low_memory=False)
    device["date"] = pd.to_datetime(device["date"], errors="coerce")
    station["date"] = pd.to_datetime(station["date"], errors="coerce")
    dates = sorted(set(device["date"].dropna()) & set(station["date"].dropna()))[:7]
    if not dates:
        raise ValueError("Demo source CSV files have no overlapping dates.")
    first_date = dates[0]
    device_ids = (
        device.loc[device["date"] == first_date, "unified_device_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(12)
        .tolist()
    )
    station_ids = (
        station.loc[station["date"] == first_date, "station_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(5)
        .tolist()
    )
    demo_root = settings.runtime_root / "demo_seed"
    demo_root.mkdir(parents=True, exist_ok=True)
    device_demo = device[
        device["date"].isin(dates) & device["unified_device_id"].astype(str).isin(device_ids)
    ].copy()
    family_examples = [
        ("南瑞继保", "RCS-931", "202201", 6, 3, 3, 12),
        ("许继电气", "WXH-803A", "202103", 2, 1, 1, 8),
        ("国电南自", "PSL-603U", "202205", 0, 0, 0, 4),
        ("四方股份", "CSC-103", "202112", 0, 0, 0, 2),
    ]
    family_by_device = {
        device_id: family_examples[min(index // 3, len(family_examples) - 1)]
        for index, device_id in enumerate(device_ids)
    }
    for device_id, family in family_by_device.items():
        manufacturer, model, batch, defects, defect_devices, trips, maintenance = family
        mask = device_demo["unified_device_id"].astype(str) == device_id
        device_demo.loc[mask, "family_id"] = make_family_id(
            manufacturer, model, batch
        )
        device_demo.loc[mask, "family_manufacturer"] = manufacturer
        device_demo.loc[mask, "family_model"] = model
        device_demo.loc[mask, "family_batch"] = batch
        device_demo.loc[mask, "family_batch_source"] = "demo_seed"
        device_demo.loc[mask, "family_device_count"] = 3
        device_demo.loc[mask, "family_defect_device_count"] = defect_devices
        device_demo.loc[mask, "family_history_defect_count"] = defects
        device_demo.loc[mask, "family_history_trip_count"] = trips
        device_demo.loc[mask, "family_history_maintenance_count"] = maintenance
    station_demo = station[
        station["date"].isin(dates) & station["station_id"].astype(str).isin(station_ids)
    ].copy()
    device_target = demo_root / "device_features.csv"
    station_target = demo_root / "station_features.csv"
    device_demo.to_csv(device_target, index=False, encoding="utf-8-sig")
    station_demo.to_csv(station_target, index=False, encoding="utf-8-sig")

    manifest = FeatureStore(settings.published_root).publish(
        device_csv=device_target,
        station_csv=station_target,
        batch_id="demo-seed",
        business_date=str(pd.Timestamp(dates[-1]).date()),
        run_mode="full",
        source_job_id="demo-seed",
    )
    bundled_batch = settings.api_service_root / "demo/batches/demo-raw-20260201"
    inbox_batch = settings.inbox_root / "demo-raw-20260201"
    if inbox_batch.exists():
        shutil.rmtree(inbox_batch)
    shutil.copytree(
        bundled_batch,
        inbox_batch,
        ignore=shutil.ignore_patterns("READY.json"),
    )
    shutil.copy2(bundled_batch / "READY.json", inbox_batch / "READY.example.json")
    print(f"Published demo feature version: {manifest['version']}")
    print(f"Assessment dates: {dates[0].date()} to {dates[-1].date()}")
    print(f"Device IDs: {', '.join(device_ids[:3])}")
    print(f"Station IDs: {', '.join(station_ids[:3])}")
    print(f"Raw cleaning batch copied to: {inbox_batch}")
    print("READY marker is stored as READY.example.json to avoid changing demo features during API tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
