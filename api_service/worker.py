from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import time
from pathlib import Path

from api_service.app.config import get_settings
from api_service.app.database import JobStore
from api_service.pipeline.runner import PipelineRunner


LOGGER = logging.getLogger("griddx.data_worker")


class DataWorker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = JobStore(self.settings.state_db_path)
        self.store.initialize()
        self.runner = PipelineRunner(self.settings)
        self.worker_id = f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}"
        self.running = True

    def request_stop(self, *_args: object) -> None:
        self.running = False

    def run(self, *, once: bool = False) -> int:
        while self.running:
            self.discover_ready_batches()
            job = self.store.claim_next_job(self.worker_id)
            if job is not None:
                self.process_job(job)
                if once:
                    return 0
                continue
            if once:
                return 0
            time.sleep(self.settings.worker_poll_seconds)
        return 0

    def discover_ready_batches(self) -> None:
        for marker in sorted(self.settings.inbox_root.glob("*/READY.json")):
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                batch_id = marker.parent.name
                business_date = str(payload["business_date"])
                run_mode = str(payload.get("run_mode", "operation_only"))
                if run_mode not in {"operation_only", "full"}:
                    raise ValueError("run_mode must be operation_only or full")
                request_id = str(payload.get("request_id") or f"auto:{batch_id}:{business_date}:{run_mode}")
                _, created = self.store.create_job(
                    request_id=request_id,
                    batch_id=batch_id,
                    business_date=business_date,
                    run_mode=run_mode,
                )
                if created:
                    LOGGER.info("Discovered READY batch %s", batch_id)
            except Exception:
                LOGGER.exception("Invalid READY marker: %s", marker)

    def process_job(self, job: dict[str, object]) -> None:
        job_id = str(job["job_id"])
        LOGGER.info("Processing %s for batch %s", job_id, job["batch_id"])
        try:
            result = self.runner.run(
                job, on_stage=lambda stage: self.store.set_stage(job_id, stage)
            )
        except Exception as exc:
            LOGGER.exception("Pipeline job %s failed", job_id)
            self.store.fail(job_id, str(exc))
            return
        self.store.succeed(job_id, result)
        LOGGER.info("Pipeline job %s succeeded", job_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the asynchronous data cleaning worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = DataWorker()
    signal.signal(signal.SIGTERM, worker.request_stop)
    signal.signal(signal.SIGINT, worker.request_stop)
    return worker.run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
