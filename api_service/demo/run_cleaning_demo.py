from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Any


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    unique = int(time.time())
    created = request(
        "POST",
        "/api/data-jobs",
        {
            "request_id": f"demo-cleaning-e2e-{unique}",
            "batch_id": "demo-raw-20260201",
            "business_date": "2026-02-05",
            "run_mode": "operation_only",
            "force": False,
        },
    )
    job_id = created["data"]["job_id"]
    print(f"Created {job_id}")
    for _ in range(10):
        subprocess.run(
            [sys.executable, "-m", "api_service.worker", "--once"],
            check=True,
        )
        job = request("GET", f"/api/data-jobs/{job_id}")
        status = job["data"]["status"]
        print(f"status={status}, stage={job['data']['current_stage']}")
        if status == "succeeded":
            print(json.dumps(job, ensure_ascii=False, indent=2))
            print("Cleaning demo passed. Run prepare_demo_data again to restore the model API demo dataset.")
            return 0
        if status == "failed":
            raise RuntimeError(job["data"]["error_message"])
    raise TimeoutError(f"Job {job_id} was not completed after 10 worker iterations.")


if __name__ == "__main__":
    raise SystemExit(main())
