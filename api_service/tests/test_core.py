from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from api_service.app.database import JobStore
from api_service.app.family import (
    build_family_alerts,
    derive_production_batch,
    make_family_id,
)
from api_service.app.feature_store import FeatureStore
from api_service.pipeline.db_export import DB_EXPORT_FILES, export_database_tables


class FeatureStoreTest(unittest.TestCase):
    def test_publish_and_read_active_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "device.csv"
            station = root / "station.csv"
            pd.DataFrame(
                [{"date": "2026-02-01", "unified_device_id": "D1"}]
            ).to_csv(device, index=False)
            pd.DataFrame([{"date": "2026-02-01", "station_id": "S1"}]).to_csv(
                station, index=False
            )
            store = FeatureStore(root / "published")
            manifest = store.publish(
                device_csv=device,
                station_csv=station,
                batch_id="test",
                business_date="2026-02-01",
                run_mode="operation_only",
                source_job_id="job-test",
            )
            self.assertEqual(manifest["device_rows"], 1)
            self.assertEqual(manifest["station_rows"], 1)
            self.assertEqual(store.get_active_manifest()["version"], manifest["version"])


class JobStoreTest(unittest.TestCase):
    def test_job_is_idempotent_and_claimed_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "jobs.db")
            first, created = store.create_job(
                request_id="request-1",
                batch_id="batch-1",
                business_date="2026-02-01",
                run_mode="operation_only",
            )
            second, created_again = store.create_job(
                request_id="request-1",
                batch_id="batch-1",
                business_date="2026-02-01",
                run_mode="operation_only",
            )
            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(first["job_id"], second["job_id"])
            claimed = store.claim_next_job("worker-test")
            self.assertEqual(claimed["job_id"], first["job_id"])
            self.assertIsNone(store.claim_next_job("worker-other"))


class DatabaseExportTest(unittest.TestCase):
    def test_all_six_logical_tables_are_exported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "source.db"
            connection = sqlite3.connect(database)
            table_map: dict[str, str] = {}
            for index, logical_name in enumerate(DB_EXPORT_FILES):
                table_name = f"source_{index}"
                connection.execute(f"CREATE TABLE {table_name} (id INTEGER, value TEXT)")
                connection.execute(f"INSERT INTO {table_name} VALUES (1, 'ok')")
                table_map[logical_name] = table_name
            connection.commit()
            connection.close()
            exported = export_database_tables(
                database_url=f"sqlite:///{database}",
                table_map=table_map,
                output_dir=root / "exports",
            )
            self.assertEqual(set(exported), set(DB_EXPORT_FILES))
            self.assertTrue(all(path.is_file() for path in exported.values()))


class FamilyAlertTest(unittest.TestCase):
    def test_batch_uses_explicit_value_then_manufacture_month(self) -> None:
        self.assertEqual(
            derive_production_batch({"批次": "2024-A", "出厂日期": "2023-01-01"}),
            ("2024-A", "批次"),
        )
        self.assertEqual(
            derive_production_batch({"出厂日期": "2023-07-18"}),
            ("202307", "出厂日期(YYYYMM推导)"),
        )

    def test_family_alert_returns_manufacturer_model_batch_triplet(self) -> None:
        family_specs = [
            ("厂家甲", "型号A", "202201", 6, 3),
            ("厂家乙", "型号B", "202202", 2, 1),
            ("厂家丙", "型号C", "202203", 0, 0),
            ("厂家丁", "型号D", "202204", 0, 0),
        ]
        rows: list[dict] = []
        predictions: list[dict] = []
        for family_index, (manufacturer, model, batch, defects, defect_devices) in enumerate(
            family_specs
        ):
            family_id = make_family_id(manufacturer, model, batch)
            for member_index in range(3):
                device_id = f"D{family_index}-{member_index}"
                rows.append(
                    {
                        "unified_device_id": device_id,
                        "family_id": family_id,
                        "family_manufacturer": manufacturer,
                        "family_model": model,
                        "family_batch": batch,
                        "family_batch_source": "批次",
                        "family_device_count": 3,
                        "family_defect_device_count": defect_devices,
                        "family_history_defect_count": defects,
                    }
                )
                predictions.append(
                    {
                        "device_id": device_id,
                        "station_id": f"S{family_index}",
                        "state_level": 2 if family_index == 0 else 0,
                        "state_name": "abnormal" if family_index == 0 else "normal",
                        "high_risk_probability": 0.8 if family_index == 0 else 0.1,
                    }
                )

        result = build_family_alerts(
            pd.DataFrame(rows),
            predictions,
            assessment_date=date(2026, 2, 7),
            level="L",
        )
        self.assertEqual(result["eligible_family_count"], 4)
        self.assertEqual(result["alert_count"], 1)
        alert = result["items"][0]
        self.assertEqual(
            (alert["manufacturer"], alert["model"], alert["batch"]),
            ("厂家甲", "型号A", "202201"),
        )
        self.assertEqual(alert["family_id"], make_family_id("厂家甲", "型号A", "202201"))
        self.assertEqual(alert["device_count"], 3)
        self.assertEqual(alert["defect_count"], 6)
        self.assertEqual(alert["defect_device_count"], 3)
        self.assertEqual(alert["defect_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
