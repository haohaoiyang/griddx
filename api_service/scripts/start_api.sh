#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${API_HOST:-127.0.0.1}"
PORT="${API_PORT:-8000}"

exec "${PYTHON_BIN}" -m uvicorn api_service.app.main:app --host "${HOST}" --port "${PORT}" --workers 1
