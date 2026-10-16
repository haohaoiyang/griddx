from __future__ import annotations

import uuid
import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from api_service.app.config import get_settings
from api_service.app.database import JobStore
from api_service.app.dispatch_service import DispatchService
from api_service.app.feature_store import FeatureDataNotReadyError, FeatureStore
from api_service.app.model_service import (
    AssessmentDateNotFoundError,
    ModelDataError,
    ModelService,
    ObjectNotFoundError,
)
from api_service.app.schemas import (
    ApiResponse,
    DataJobData,
    DataJobRequest,
    DeviceAssessmentData,
    DevicePredictRequest,
    DeviceProfileData,
    DispatchData,
    DispatchRequest,
    FamilyAlertData,
    RiskOverviewData,
    StationAssessmentData,
    StationPredictRequest,
    StationProfileData,
)


LOGGER = logging.getLogger("model_api")


def _request_id(value: str | None = None) -> str:
    return value or f"req-{uuid.uuid4().hex[:16]}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    feature_store = FeatureStore(settings.published_root)
    job_store = JobStore(settings.state_db_path)
    job_store.initialize()
    app.state.settings = settings
    app.state.feature_store = feature_store
    app.state.job_store = job_store
    app.state.model_service = ModelService(settings, feature_store)
    app.state.dispatch_service = DispatchService(settings, feature_store)
    yield


app = FastAPI(
    title="Data and Model API",
    summary="Daily power-grid data processing and trained-model inference service.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "code": "INVALID_REQUEST",
            "message": "Request validation failed.",
            "request_id": _request_id(request.headers.get("X-Request-ID")),
            "data": {"errors": jsonable_encoder(exc.errors())},
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    code = "DATA_NOT_FOUND" if exc.status_code == 404 else "INVALID_REQUEST"
    return _error_response(request, exc.status_code, code, str(exc.detail))


@app.exception_handler(FeatureDataNotReadyError)
async def feature_not_ready_handler(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(request, 503, "FEATURE_DATA_NOT_READY", str(exc))


@app.exception_handler(ObjectNotFoundError)
@app.exception_handler(AssessmentDateNotFoundError)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(request, 404, "DATA_NOT_FOUND", str(exc))


@app.exception_handler(ModelDataError)
@app.exception_handler(ValueError)
async def bad_data_handler(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(request, 400, "INVALID_DATA", str(exc))


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled API error", exc_info=exc)
    return _error_response(
        request, 500, "INTERNAL_ERROR", "Unexpected internal service error."
    )


def _error_response(
    request: Request, status_code: int, code: str, message: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code,
            "message": message,
            "request_id": _request_id(request.headers.get("X-Request-ID")),
            "data": None,
        },
    )


def _job_data(job: dict[str, Any]) -> DataJobData:
    return DataJobData(
        job_id=job["job_id"],
        batch_id=job["batch_id"],
        business_date=job["business_date"],
        run_mode=job["run_mode"],
        status=job["status"],
        current_stage=job["current_stage"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        result=job.get("result"),
        error_message=job.get("error_message"),
    )


@app.get("/health/live", tags=["health"])
def live(x_request_id: str | None = Header(default=None)) -> ApiResponse[dict[str, str]]:
    return ApiResponse(
        request_id=_request_id(x_request_id), data={"status": "up", "service": "model-api"}
    )


@app.get("/health/ready", tags=["health"])
def ready(request: Request, x_request_id: str | None = Header(default=None)) -> ApiResponse[dict[str, Any]]:
    manifest = request.app.state.feature_store.get_active_manifest()
    return ApiResponse(
        request_id=_request_id(x_request_id),
        data={
            "status": "ready",
            "feature_version": manifest["version"],
            "business_date": manifest["business_date"],
        },
    )


@app.get("/api/models", tags=["models"])
def models(request: Request, x_request_id: str | None = Header(default=None)) -> ApiResponse[dict[str, Any]]:
    return ApiResponse(
        request_id=_request_id(x_request_id),
        data=request.app.state.model_service.model_info(),
    )


@app.post(
    "/api/data-jobs",
    tags=["data"],
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ApiResponse[DataJobData],
)
def create_data_job(payload: DataJobRequest, request: Request) -> ApiResponse[DataJobData]:
    batch_dir = request.app.state.settings.inbox_root / payload.batch_id
    if not batch_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Batch directory does not exist: {payload.batch_id}")
    job, _created = request.app.state.job_store.create_job(
        request_id=payload.request_id,
        batch_id=payload.batch_id,
        business_date=str(payload.business_date),
        run_mode=payload.run_mode,
        force=payload.force,
    )
    return ApiResponse(request_id=payload.request_id, data=_job_data(job))


@app.get(
    "/api/data-jobs/{job_id}",
    tags=["data"],
    response_model=ApiResponse[DataJobData],
)
def get_data_job(
    job_id: str,
    request: Request,
    x_request_id: str | None = Header(default=None),
) -> ApiResponse[DataJobData]:
    job = request.app.state.job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Data job was not found.")
    return ApiResponse(request_id=_request_id(x_request_id), data=_job_data(job))


@app.post(
    "/api/device/assessment",
    tags=["device"],
    response_model=ApiResponse[DeviceAssessmentData],
)
def assess_devices(
    payload: DevicePredictRequest, request: Request
) -> ApiResponse[DeviceAssessmentData]:
    items = request.app.state.model_service.assess_devices(
        payload.assessment_date, payload.device_ids
    )
    _enforce_result_limit(request, len(items))
    model = request.app.state.model_service.device_model
    return ApiResponse(
        request_id=payload.request_id,
        data=DeviceAssessmentData(
            assessment_date=payload.assessment_date,
            model_name=model.name,
            model_version=model.version,
            items=items,
        ),
    )


@app.get(
    "/api/device/query",
    tags=["device"],
    response_model=ApiResponse[DeviceAssessmentData],
)
def query_device(
    request: Request,
    assessment_date: date,
    device_id: str = Query(min_length=1),
    x_request_id: str | None = Header(default=None),
) -> ApiResponse[DeviceAssessmentData]:
    request_id = _request_id(x_request_id)
    items = request.app.state.model_service.assess_devices(assessment_date, [device_id])
    model = request.app.state.model_service.device_model
    return ApiResponse(
        request_id=request_id,
        data=DeviceAssessmentData(
            assessment_date=assessment_date,
            model_name=model.name,
            model_version=model.version,
            items=items,
        ),
    )


@app.get(
    "/api/device/profile",
    tags=["device"],
    response_model=ApiResponse[DeviceProfileData],
)
def device_profile(
    request: Request,
    device_id: str = Query(min_length=1),
    start_date: date = Query(),
    end_date: date = Query(),
    x_request_id: str | None = Header(default=None),
) -> ApiResponse[DeviceProfileData]:
    data = request.app.state.model_service.device_profile(device_id, start_date, end_date)
    return ApiResponse(request_id=_request_id(x_request_id), data=DeviceProfileData(**data))


@app.get(
    "/api/defect/family-alerts",
    tags=["defect"],
    response_model=ApiResponse[FamilyAlertData],
)
def family_alerts(
    request: Request,
    assessment_date: date,
    level: Literal["G", "M", "L"] | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    x_request_id: str | None = Header(default=None),
) -> ApiResponse[FamilyAlertData]:
    data = request.app.state.model_service.family_alerts(
        assessment_date,
        level=level,
        limit=limit,
    )
    return ApiResponse(
        request_id=_request_id(x_request_id),
        data=FamilyAlertData(**data),
    )


@app.post(
    "/api/station/assessment",
    tags=["station"],
    response_model=ApiResponse[StationAssessmentData],
)
def assess_stations(
    payload: StationPredictRequest, request: Request
) -> ApiResponse[StationAssessmentData]:
    items = request.app.state.model_service.assess_stations(
        payload.assessment_date, payload.station_ids
    )
    _enforce_result_limit(request, len(items))
    model = request.app.state.model_service.station_model
    return ApiResponse(
        request_id=payload.request_id,
        data=StationAssessmentData(
            assessment_date=payload.assessment_date,
            model_name=model.name,
            model_version=model.version,
            items=items,
        ),
    )


@app.get(
    "/api/station/query",
    tags=["station"],
    response_model=ApiResponse[StationAssessmentData],
)
def query_station(
    request: Request,
    assessment_date: date,
    station_id: str = Query(min_length=1),
    x_request_id: str | None = Header(default=None),
) -> ApiResponse[StationAssessmentData]:
    request_id = _request_id(x_request_id)
    items = request.app.state.model_service.assess_stations(assessment_date, [station_id])
    model = request.app.state.model_service.station_model
    return ApiResponse(
        request_id=request_id,
        data=StationAssessmentData(
            assessment_date=assessment_date,
            model_name=model.name,
            model_version=model.version,
            items=items,
        ),
    )


@app.get(
    "/api/station/profile",
    tags=["station"],
    response_model=ApiResponse[StationProfileData],
)
def station_profile(
    request: Request,
    station_id: str = Query(min_length=1),
    start_date: date = Query(),
    end_date: date = Query(),
    x_request_id: str | None = Header(default=None),
) -> ApiResponse[StationProfileData]:
    data = request.app.state.model_service.station_profile(station_id, start_date, end_date)
    return ApiResponse(request_id=_request_id(x_request_id), data=StationProfileData(**data))


@app.get(
    "/api/risk/overview",
    tags=["risk"],
    response_model=ApiResponse[RiskOverviewData],
)
def risk_overview(
    request: Request,
    assessment_date: date,
    top_n: int = Query(default=5, ge=1, le=50),
    x_request_id: str | None = Header(default=None),
) -> ApiResponse[RiskOverviewData]:
    devices = request.app.state.model_service.assess_devices(assessment_date)
    stations = request.app.state.model_service.assess_stations(assessment_date)
    _enforce_result_limit(request, len(devices))
    device_counts = _state_counts(devices)
    station_counts = _state_counts(stations)
    top_devices = sorted(
        (
            {
                "device_id": item["device_id"],
                "station_id": item["station_id"],
                "state_name": item["state_name"],
                "high_risk_probability": item["high_risk_probability"],
            }
            for item in devices
        ),
        key=lambda item: item["high_risk_probability"],
        reverse=True,
    )[:top_n]
    top_stations = sorted(
        (
            {
                "station_id": item["station_id"],
                "station_name": item["station_name"],
                "state_name": item["state_name"],
                "risk_score": item["risk_score"],
            }
            for item in stations
        ),
        key=lambda item: item["risk_score"],
        reverse=True,
    )[:top_n]
    data = RiskOverviewData(
        assessment_date=assessment_date,
        device_state_counts=device_counts,
        station_state_counts=station_counts,
        average_device_high_risk_probability=_mean(
            item["high_risk_probability"] for item in devices
        ),
        average_station_risk_score=_mean(item["risk_score"] for item in stations),
        top_risk_devices=top_devices,
        top_risk_stations=top_stations,
    )
    return ApiResponse(request_id=_request_id(x_request_id), data=data)


@app.post(
    "/api/economic-dispatch",
    tags=["dispatch"],
    response_model=ApiResponse[DispatchData],
)
def economic_dispatch(
    payload: DispatchRequest, request: Request
) -> ApiResponse[DispatchData]:
    data = request.app.state.dispatch_service.dispatch(
        assessment_date=payload.assessment_date,
        station_ids=payload.station_ids,
        supply_budget_mw=payload.supply_budget_mw,
        economic_parameters=payload.economic_parameters.model_dump(),
        request_overrode_parameters=bool(payload.economic_parameters.model_fields_set),
    )
    _enforce_result_limit(request, len(data["items"]))
    return ApiResponse(request_id=payload.request_id, data=DispatchData(**data))


def _enforce_result_limit(request: Request, count: int) -> None:
    maximum = request.app.state.settings.max_sync_objects
    if count > maximum:
        raise ValueError(
            f"Synchronous result contains {count} objects; configured maximum is {maximum}. "
            "Provide an explicit object ID list or raise MAX_SYNC_OBJECTS after capacity testing."
        )


def _state_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"normal": 0, "watch": 0, "abnormal": 0, "high_risk": 0}
    for item in items:
        counts[str(item["state_name"])] += 1
    return counts


def _mean(values: Any) -> float:
    materialized = list(values)
    return round(float(np.mean(materialized)), 6) if materialized else 0.0
