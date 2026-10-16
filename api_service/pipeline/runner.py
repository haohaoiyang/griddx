from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from api_service.app.config import Settings
from api_service.app.feature_store import FeatureStore
from api_service.pipeline.db_export import export_database_tables


StageCallback = Callable[[str], None]


class PipelineRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.steps_root = settings.api_service_root / "pipeline" / "steps"
        self.feature_store = FeatureStore(settings.published_root)

    def run(self, job: dict[str, Any], on_stage: StageCallback | None = None) -> dict[str, Any]:
        notify = on_stage or (lambda _stage: None)
        batch_id = str(job["batch_id"])
        batch_dir = self.settings.inbox_root / batch_id
        xml_files, dt_files = self._validate_inputs(batch_dir)

        job_root = self.settings.jobs_root / str(job["job_id"])
        metadata_dir = job_root / "metadata"
        middle_dir = job_root / "middle"
        final_dir = job_root / "final"
        db_dir = job_root / "db_exports"
        for directory in (metadata_dir, middle_dir, final_dir):
            directory.mkdir(parents=True, exist_ok=True)
        log_path = job_root / "pipeline.log"

        notify("extracting_xml")
        self._run_step(
            "02_extract_objects_relations.py",
            [str(xml_files[0]), "--output-dir", str(metadata_dir)],
            log_path,
        )

        merged = middle_dir / "merged.csv"
        notify("merging_dt")
        merge_args = [
            "--objects",
            str(metadata_dir / "objects.csv"),
            "--relations",
            str(metadata_dir / "relations.csv"),
            "--output",
            str(merged),
            "--merged-relations-output",
            str(middle_dir / "merged_relations.csv"),
            "--type-label",
            "name",
            "--dt",
            *[str(path) for path in dt_files],
        ]
        self._run_step("03_merge_metadata_first.py", merge_args, log_path)

        fact_device = middle_dir / "fact_operation_device_day_wide.csv"
        notify("building_device_daily")
        self._run_step(
            "04_task5_build_operation_device_day_wide.py",
            ["--input", str(merged), "--output", str(fact_device)],
            log_path,
        )

        exported: dict[str, Path] = {}
        if str(job["run_mode"]) == "full":
            notify("exporting_database")
            exported = export_database_tables(
                database_url=self.settings.database_url,
                table_map=self.settings.database_table_map,
                output_dir=db_dir,
            )

        task6 = middle_dir / "model_base_device_day.csv"
        background = exported.get("background", db_dir / "missing_background.csv")
        code_map = exported.get("code_map", db_dir / "missing_code_map.csv")
        notify("building_device_model_table")
        self._run_step(
            "05_task6_build_model_base_device_day.py",
            [
                "--input",
                str(fact_device),
                "--output",
                str(task6),
                "--background",
                str(background),
                "--code-map",
                str(code_map),
            ],
            log_path,
        )
        task6_extract = final_dir / "device_features_operation.csv"
        self._run_step(
            "06_task6_extract.py",
            [
                "--input",
                str(task6),
                "--output",
                str(task6_extract),
                "--group",
                "line_or_load_like",
            ],
            log_path,
        )

        station_base = middle_dir / "model_base_station_day.csv"
        station_extract = final_dir / "station_features.csv"
        notify("building_station_daily")
        self._run_step(
            "07_task7_build_model_base_station_day.py",
            ["--input", str(fact_device), "--output", str(station_base)],
            log_path,
        )
        self._run_step(
            "08_task7_extract.py",
            ["--input", str(station_base), "--output", str(station_extract)],
            log_path,
        )

        device_final = task6_extract
        if str(job["run_mode"]) == "full":
            notify("enriching_device_history")
            device_final = final_dir / "device_features.csv"
            self._run_step(
                "09_task_7_15_real_fields.py",
                [
                    "--input",
                    str(task6_extract),
                    "--dm-device",
                    str(exported["dm_device"]),
                    "--defect",
                    str(exported["defect"]),
                    "--maintenance-main",
                    str(exported["maintenance_main"]),
                    "--maintenance-device",
                    str(exported["maintenance_device"]),
                    "--trip-substation",
                    str(exported["trip_substation"]),
                    "--trip-line",
                    str(exported["trip_line"]),
                    "--output",
                    str(device_final),
                    "--match-log",
                    str(final_dir / "device_match_result.csv"),
                    "--match-issues",
                    str(final_dir / "device_match_issues.csv"),
                    "--code-map",
                    str(final_dir / "real_field_code_map.csv"),
                ],
                log_path,
            )

        self._validate_business_date(
            device_final, station_extract, str(job["business_date"])
        )
        notify("publishing")
        manifest = self.feature_store.publish(
            device_csv=device_final,
            station_csv=station_extract,
            batch_id=batch_id,
            business_date=str(job["business_date"]),
            run_mode=str(job["run_mode"]),
            source_job_id=str(job["job_id"]),
        )
        result = {
            "feature_version": manifest["version"],
            "device_rows": manifest["device_rows"],
            "station_rows": manifest["station_rows"],
            "business_date": manifest["business_date"],
            "run_mode": manifest["run_mode"],
            "log_file": str(log_path.relative_to(self.settings.runtime_root)),
        }
        (job_root / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    def _run_step(self, script_name: str, arguments: list[str], log_path: Path) -> None:
        script = self.steps_root / script_name
        if not script.is_file():
            raise FileNotFoundError(f"Pipeline script does not exist: {script}")
        command = [sys.executable, str(script), *arguments]
        environment = os.environ.copy()
        environment.setdefault("PYTHONUTF8", "1")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(command)}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=self.settings.project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
                check=False,
            )
        if completed.returncode != 0:
            tail = self._log_tail(log_path)
            raise RuntimeError(
                f"Pipeline step {script_name} failed with exit code "
                f"{completed.returncode}.\n{tail}"
            )

    @staticmethod
    def _validate_inputs(batch_dir: Path) -> tuple[list[Path], list[Path]]:
        if not batch_dir.is_dir():
            raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")
        xml_files = sorted((batch_dir / "xml").glob("*.xml"))
        xml_files.extend(sorted((batch_dir / "xml").glob("*.XML")))
        dt_files = sorted((batch_dir / "dt").glob("*.DT"))
        dt_files.extend(sorted((batch_dir / "dt").glob("*.dt")))
        if len(xml_files) != 1:
            raise ValueError(
                f"Batch {batch_dir.name} must contain exactly one XML file; found {len(xml_files)}."
            )
        if not dt_files:
            raise ValueError(f"Batch {batch_dir.name} contains no DT files.")
        empty = [str(path.name) for path in [*xml_files, *dt_files] if path.stat().st_size == 0]
        if empty:
            raise ValueError(f"Batch contains empty source files: {empty}")
        return xml_files, dt_files

    @staticmethod
    def _log_tail(log_path: Path, line_count: int = 30) -> str:
        try:
            return "".join(log_path.read_text(encoding="utf-8").splitlines(True)[-line_count:])
        except OSError:
            return "Pipeline log could not be read."

    @staticmethod
    def _validate_business_date(
        device_csv: Path, station_csv: Path, business_date: str
    ) -> None:
        requested = pd.Timestamp(business_date).normalize()
        for kind, path in (("device", device_csv), ("station", station_csv)):
            dates = pd.to_datetime(
                pd.read_csv(path, usecols=["date"])["date"], errors="coerce"
            ).dt.normalize()
            if requested not in set(dates.dropna()):
                raise ValueError(
                    f"Requested business_date={business_date} is absent from the {kind} output."
                )
