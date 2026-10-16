from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, select
from sqlalchemy.engine import Engine


DB_EXPORT_FILES = {
    "dm_device": "DM_主网设备.csv",
    "defect": "DM_缺陷数据.csv",
    "maintenance_main": "DM_检修单主表_JXSQD.csv",
    "maintenance_device": "DM_检修单从表_TD_JXSBLB.csv",
    "trip_substation": "DM_变电跳闸记录.csv",
    "trip_line": "DM_输电跳闸记录.csv",
}


def export_database_tables(
    *,
    database_url: str,
    table_map: dict[str, str],
    output_dir: Path,
    chunk_size: int = 100_000,
) -> dict[str, Path]:
    """Export configured relational tables to the exact CSV contract used by the pipeline."""

    if not database_url:
        raise ValueError("DATABASE_URL is required when run_mode=full.")
    missing = sorted(set(DB_EXPORT_FILES) - set(table_map))
    if missing:
        raise ValueError(f"DB_TABLE_MAP is missing logical tables: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        return {
            logical_name: _export_one(
                engine,
                source_name=table_map[logical_name],
                target=output_dir / target_name,
                chunk_size=chunk_size,
            )
            for logical_name, target_name in DB_EXPORT_FILES.items()
        }
    finally:
        engine.dispose()


def _export_one(
    engine: Engine,
    *,
    source_name: str,
    target: Path,
    chunk_size: int,
) -> Path:
    schema, table_name = _split_table_name(source_name)
    table = Table(table_name, MetaData(), schema=schema, autoload_with=engine)
    first = True
    with engine.connect() as connection:
        chunks = pd.read_sql_query(select(table), connection, chunksize=chunk_size)
        for frame in chunks:
            frame.to_csv(
                target,
                mode="w" if first else "a",
                header=first,
                index=False,
                encoding="utf-8-sig",
            )
            first = False
    if first:
        pd.DataFrame(columns=[column.name for column in table.columns]).to_csv(
            target, index=False, encoding="utf-8-sig"
        )
    return target


def _split_table_name(value: str) -> tuple[str | None, str]:
    parts = value.rsplit(".", maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (None, value)
