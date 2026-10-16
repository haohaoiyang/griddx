from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


TERMINAL_JOB_STATES = {"succeeded", "failed"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_job (
                    job_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    batch_id TEXT NOT NULL,
                    business_date TEXT NOT NULL,
                    run_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error_message TEXT,
                    worker_id TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_pipeline_job_status_created "
                "ON pipeline_job(status, created_at)"
            )
            connection.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        value = dict(row)
        value["payload"] = json.loads(value.pop("payload_json") or "{}")
        value["result"] = json.loads(value.pop("result_json") or "null")
        return value

    def create_job(
        self,
        *,
        request_id: str,
        batch_id: str,
        business_date: str,
        run_mode: str,
        force: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        self.initialize()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM pipeline_job WHERE request_id = ?", (request_id,)
            ).fetchone()
            if existing and not force:
                return self._row_to_dict(existing) or {}, False
            effective_request_id = request_id
            if existing and force:
                effective_request_id = f"{request_id}:retry:{uuid.uuid4().hex[:8]}"
            now = utc_now()
            job_id = f"job-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            payload = {
                "request_id": effective_request_id,
                "batch_id": batch_id,
                "business_date": business_date,
                "run_mode": run_mode,
            }
            connection.execute(
                """
                INSERT INTO pipeline_job (
                    job_id, request_id, batch_id, business_date, run_mode,
                    status, current_stage, payload_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', 'queued', ?, ?, ?)
                """,
                (
                    job_id,
                    effective_request_id,
                    batch_id,
                    business_date,
                    run_mode,
                    json.dumps(payload, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM pipeline_job WHERE job_id = ?", (job_id,)
            ).fetchone()
            return self._row_to_dict(row) or {}, True

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM pipeline_job WHERE job_id = ?", (job_id,)
            ).fetchone()
            return self._row_to_dict(row)

    def claim_next_job(self, worker_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM pipeline_job WHERE status = 'pending' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            now = utc_now()
            changed = connection.execute(
                """
                UPDATE pipeline_job
                SET status = 'running', current_stage = 'validating', worker_id = ?,
                    started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE job_id = ? AND status = 'pending'
                """,
                (worker_id, now, now, row["job_id"]),
            ).rowcount
            if changed != 1:
                connection.rollback()
                return None
            connection.commit()
            claimed = connection.execute(
                "SELECT * FROM pipeline_job WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            return self._row_to_dict(claimed)

    def set_stage(self, job_id: str, stage: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE pipeline_job SET current_stage = ?, updated_at = ? WHERE job_id = ?",
                (stage, utc_now(), job_id),
            )
            connection.commit()

    def succeed(self, job_id: str, result: dict[str, Any]) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE pipeline_job
                SET status = 'succeeded', current_stage = 'published', result_json = ?,
                    error_message = NULL, finished_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(result, ensure_ascii=False), now, now, job_id),
            )
            connection.commit()

    def fail(self, job_id: str, message: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE pipeline_job
                SET status = 'failed', current_stage = 'failed', error_message = ?,
                    finished_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (message[-8000:], now, now, job_id),
            )
            connection.commit()

