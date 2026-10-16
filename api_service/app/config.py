from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


API_SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(API_SERVICE_ROOT / ".env")


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    api_service_root: Path
    runtime_root: Path
    inbox_root: Path
    jobs_root: Path
    published_root: Path
    state_db_path: Path
    device_model_path: Path
    station_model_path: Path
    torch_device: str
    gpu_id: int | None
    max_sync_objects: int
    worker_poll_seconds: float
    database_url: str
    database_table_map: dict[str, str]

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.runtime_root,
            self.inbox_root,
            self.jobs_root,
            self.published_root,
            self.state_db_path.parent,
            self.runtime_root / "logs",
            self.runtime_root / "demo_responses",
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = _path_env("GRIDDX_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
    api_service_root = _path_env("GRIDDX_API_ROOT", API_SERVICE_ROOT)
    runtime_root = _path_env("GRIDDX_API_RUNTIME", api_service_root / "runtime")
    gpu_raw = os.getenv("GPU_ID", "")
    table_map_raw = os.getenv("DB_TABLE_MAP", "{}")
    try:
        table_map = json.loads(table_map_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("DB_TABLE_MAP must be a valid JSON object.") from exc
    if not isinstance(table_map, dict):
        raise ValueError("DB_TABLE_MAP must be a JSON object mapping logical names to tables.")

    settings = Settings(
        project_root=project_root,
        api_service_root=api_service_root,
        runtime_root=runtime_root,
        inbox_root=runtime_root / "inbox",
        jobs_root=runtime_root / "jobs",
        published_root=runtime_root / "published",
        state_db_path=runtime_root / "state" / "service.db",
        device_model_path=_path_env(
            "DEVICE_MODEL_PATH",
            (
                api_service_root / "models/device_damd_net.pt"
                if (api_service_root / "models/device_damd_net.pt").is_file()
                else project_root
                / "outputs/baseline_models/device_prediction/torch_multi_discriminator.pt"
            ),
        ),
        station_model_path=_path_env(
            "STATION_MODEL_PATH",
            (
                api_service_root / "models/station_hsf_net.pt"
                if (api_service_root / "models/station_hsf_net.pt").is_file()
                else project_root
                / "outputs/baseline_models/station_assessment/torch_station_hierarchical.pt"
            ),
        ),
        torch_device=os.getenv("TORCH_DEVICE", "auto"),
        gpu_id=int(gpu_raw) if gpu_raw not in ("", None) else None,
        max_sync_objects=_int_env("MAX_SYNC_OBJECTS", 10_000),
        worker_poll_seconds=_float_env("WORKER_POLL_SECONDS", 2.0),
        database_url=os.getenv("DATABASE_URL", ""),
        database_table_map={str(key): str(value) for key, value in table_map.items()},
    )
    settings.ensure_runtime_dirs()
    return settings


def reset_settings_cache() -> None:
    get_settings.cache_clear()
