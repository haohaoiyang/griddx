#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/api_service/compose.yaml"
ENV_FILE="${PROJECT_ROOT}/api_service/.env"

cd "${PROJECT_ROOT}"

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${PROJECT_ROOT}/api_service/.env.example" "${ENV_FILE}"
fi

docker compose -f "${COMPOSE_FILE}" build model-api
docker compose -f "${COMPOSE_FILE}" stop data-worker >/dev/null 2>&1 || true
docker compose -f "${COMPOSE_FILE}" run --rm --no-deps \
  model-api python -m api_service.demo.prepare_demo_data
docker compose -f "${COMPOSE_FILE}" up -d model-api

ready=false
for _attempt in $(seq 1 90); do
  if docker compose -f "${COMPOSE_FILE}" exec -T model-api \
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)" \
    >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 1
done

if [[ "${ready}" != true ]]; then
  docker compose -f "${COMPOSE_FILE}" logs --no-color --tail=100
  printf 'model-api did not become ready within 90 seconds.\n' >&2
  exit 1
fi

docker compose -f "${COMPOSE_FILE}" exec -T \
  model-api python api_service/demo/run_smoke_test.py
docker compose -f "${COMPOSE_FILE}" up -d data-worker
docker compose -f "${COMPOSE_FILE}" ps

printf '\nSwagger: http://127.0.0.1:8000/docs\n'
