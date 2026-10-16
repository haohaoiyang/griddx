FROM python:3.11-slim-bookworm

ARG PYTORCH_VERSION=2.11.0
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    GRIDDX_PROJECT_ROOT=/app \
    GRIDDX_API_ROOT=/app/api_service \
    GRIDDX_API_RUNTIME=/app/api_service/runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api_service/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir \
        "torch==${PYTORCH_VERSION}" --index-url "${PYTORCH_INDEX_URL}"

COPY src /app/src
COPY api_service /app/api_service

RUN mkdir -p /app/api_service/runtime \
    && useradd --create-home --uid 10001 modelsvc \
    && chown -R modelsvc:modelsvc /app/api_service/runtime

USER modelsvc

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

CMD ["python", "-m", "uvicorn", "api_service.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
