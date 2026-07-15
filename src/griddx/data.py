from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_csv_limited(path: str | Path, max_rows: int | None = None, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if max_rows is not None and max_rows > 0:
        df = pd.read_csv(path, nrows=max_rows, usecols=usecols, low_memory=False)
    else:
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
    return df.dropna(how="all").reset_index(drop=True)


def parse_date_column(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    if column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def drop_empty_columns(df: pd.DataFrame, keep: set[str] | None = None) -> pd.DataFrame:
    keep = keep or set()
    empty = [col for col in df.columns if col not in keep and df[col].isna().all()]
    return df.drop(columns=empty)


def safe_numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    exclude_set = set(exclude)
    return [col for col in df.select_dtypes(include="number").columns if col not in exclude_set]
