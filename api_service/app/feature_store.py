from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd


class FeatureDataNotReadyError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FeatureStore:
    """Versioned CSV feature store with an atomically replaced active manifest."""

    def __init__(self, published_root: Path) -> None:
        self.root = published_root
        self.versions_root = self.root / "versions"
        self.active_path = self.root / "active.json"
        self._lock = RLock()
        self._cache_version: str | None = None
        self._device_cache: pd.DataFrame | None = None
        self._station_cache: pd.DataFrame | None = None
        self.versions_root.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        *,
        device_csv: Path,
        station_csv: Path,
        batch_id: str,
        business_date: str,
        run_mode: str,
        source_job_id: str,
    ) -> dict[str, Any]:
        self._validate_csv(device_csv, {"date", "unified_device_id"}, "device")
        self._validate_csv(station_csv, {"date", "station_id"}, "station")

        version = (
            f"{business_date.replace('-', '')}-{batch_id}-"
            f"{datetime.now(UTC).strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        temporary_dir = self.versions_root / f".{version}.tmp"
        final_dir = self.versions_root / version
        temporary_dir.mkdir(parents=True, exist_ok=False)
        try:
            device_target = temporary_dir / "device_features.csv"
            station_target = temporary_dir / "station_features.csv"
            shutil.copy2(device_csv, device_target)
            shutil.copy2(station_csv, station_target)
            device_rows = self._count_rows(device_target)
            station_rows = self._count_rows(station_target)
            manifest = {
                "version": version,
                "batch_id": batch_id,
                "business_date": business_date,
                "run_mode": run_mode,
                "source_job_id": source_job_id,
                "published_at": _utc_now(),
                "device_csv": f"versions/{version}/device_features.csv",
                "station_csv": f"versions/{version}/station_features.csv",
                "device_rows": device_rows,
                "station_rows": station_rows,
                "device_sha256": file_sha256(device_target),
                "station_sha256": file_sha256(station_target),
            }
            (temporary_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary_dir, final_dir)
            active_tmp = self.root / f".active.{uuid.uuid4().hex}.tmp"
            active_tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(active_tmp, self.active_path)
            return manifest
        except Exception:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            raise

    def get_active_manifest(self) -> dict[str, Any]:
        if not self.active_path.exists():
            raise FeatureDataNotReadyError(
                "No feature version has been published. Run the data worker first."
            )
        with self._lock:
            manifest = json.loads(self.active_path.read_text(encoding="utf-8"))
        for key in ("device_csv", "station_csv", "version"):
            if key not in manifest:
                raise FeatureDataNotReadyError(f"Active feature manifest is missing {key}.")
        return manifest

    def get_active_paths(self) -> tuple[Path, Path, dict[str, Any]]:
        manifest = self.get_active_manifest()
        device_path = self.root / manifest["device_csv"]
        station_path = self.root / manifest["station_csv"]
        if not device_path.is_file() or not station_path.is_file():
            raise FeatureDataNotReadyError(
                "Active feature manifest points to missing feature files."
            )
        return device_path, station_path, manifest

    def read_device_features(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        device_path, _, manifest = self.get_active_paths()
        with self._lock:
            self._reset_cache_if_version_changed(str(manifest["version"]))
            if self._device_cache is None:
                self._device_cache = self._read_csv(device_path)
            return self._device_cache, manifest

    def read_station_features(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        _, station_path, manifest = self.get_active_paths()
        with self._lock:
            self._reset_cache_if_version_changed(str(manifest["version"]))
            if self._station_cache is None:
                self._station_cache = self._read_csv(station_path)
            return self._station_cache, manifest

    def _reset_cache_if_version_changed(self, version: str) -> None:
        if self._cache_version == version:
            return
        self._cache_version = version
        self._device_cache = None
        self._station_cache = None

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, low_memory=False)
        if "date" not in frame.columns:
            raise FeatureDataNotReadyError(f"Feature file has no date column: {path.name}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        return frame

    @staticmethod
    def _validate_csv(path: Path, required: set[str], kind: str) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{kind} feature file is missing or empty: {path}")
        preview = pd.read_csv(path, nrows=1)
        columns = set(preview.columns)
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"{kind} feature file is missing columns: {missing}")
        if preview.empty:
            raise ValueError(f"{kind} feature file contains no data rows: {path}")

    @staticmethod
    def _count_rows(path: Path) -> int:
        return sum(
            len(chunk)
            for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False)
        )
