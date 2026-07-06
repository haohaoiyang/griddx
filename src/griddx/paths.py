from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DATA_ROOT = Path(os.getenv("GRIDDX_DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "baseline_models"


def data_path(name: str) -> Path:
    return DEFAULT_DATA_ROOT / name


def ensure_output_dir(*parts: str) -> Path:
    path = OUTPUT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
