from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
OUTPUT_DIR = Path(os.getenv("DEMO_RESPONSE_DIR", "api_service/runtime/demo_responses"))


def call(
    name: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "X-Request-ID": f"smoke-{name}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            status = response.status
    except urllib.error.HTTPError as exc:
        result = json.loads(exc.read().decode("utf-8"))
        raise RuntimeError(f"{name} failed with HTTP {exc.code}: {result}") from exc
    if status < 200 or status >= 300 or result.get("code") != "0":
        raise RuntimeError(f"{name} returned an invalid response: HTTP {status}, {result}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"PASS {method:4s} {path} -> HTTP {status}")
    return result


def query(path: str, **parameters: str | int) -> str:
    return f"{path}?{urllib.parse.urlencode(parameters)}"


def main() -> int:
    call("health_live", "GET", "/health/live")
    call("health_ready", "GET", "/health/ready")
    call("models", "GET", "/api/models")
    call(
        "device_assessment",
        "POST",
        "/api/device/assessment",
        {
            "request_id": "demo-device-001",
            "assessment_date": "2026-02-07",
            "device_ids": ["AFZ.511C", "AFZ.512C"],
        },
    )
    call(
        "device_query",
        "GET",
        query(
            "/api/device/query",
            assessment_date="2026-02-07",
            device_id="AFZ.511C",
        ),
    )
    call(
        "device_profile",
        "GET",
        query(
            "/api/device/profile",
            device_id="AFZ.511C",
            start_date="2026-02-01",
            end_date="2026-02-07",
        ),
    )
    call(
        "family_alerts",
        "GET",
        query(
            "/api/defect/family-alerts",
            assessment_date="2026-02-07",
            limit=20,
        ),
    )
    call(
        "station_assessment",
        "POST",
        "/api/station/assessment",
        {
            "request_id": "demo-station-001",
            "assessment_date": "2026-02-07",
            "station_ids": ["SS-AFZ", "SS-BAZ"],
        },
    )
    call(
        "station_query",
        "GET",
        query(
            "/api/station/query",
            assessment_date="2026-02-07",
            station_id="SS-AFZ",
        ),
    )
    call(
        "station_profile",
        "GET",
        query(
            "/api/station/profile",
            station_id="SS-AFZ",
            start_date="2026-02-01",
            end_date="2026-02-07",
        ),
    )
    call(
        "risk_overview",
        "GET",
        query("/api/risk/overview", assessment_date="2026-02-07", top_n=3),
    )
    call(
        "economic_dispatch",
        "POST",
        "/api/economic-dispatch",
        {
            "request_id": "demo-dispatch-001",
            "assessment_date": "2026-02-07",
            "station_ids": ["SS-AFZ", "SS-BAZ"],
            "supply_budget_mw": 10.0,
            "economic_parameters": {
                "load_growth_rate": 0.08,
                "capacity_multiplier": 1.25,
                "dispatch_cost_base": 0.30,
                "risk_cost_base": 0.20,
                "risk_cost_weight": 1.10,
                "shortfall_cost_base": 1.25,
                "shortfall_criticality_weight": 1.75,
            },
        },
    )
    job = call(
        "data_job_create",
        "POST",
        "/api/data-jobs",
        {
            "request_id": "demo-cleaning-001",
            "batch_id": "demo-raw-20260201",
            "business_date": "2026-02-05",
            "run_mode": "operation_only",
            "force": False,
        },
    )
    call("data_job_query", "GET", f"/api/data-jobs/{job['data']['job_id']}")
    print(f"All API smoke tests passed. Responses: {OUTPUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise
