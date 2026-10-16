from __future__ import annotations

from datetime import date
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


T = TypeVar("T")
StateName = Literal["normal", "watch", "abnormal", "high_risk"]


class ApiResponse(BaseModel, Generic[T]):
    code: str = "0"
    message: str = "success"
    request_id: str
    data: T


class DevicePredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req-device-20260205-001",
                "assessment_date": "2026-02-05",
                "device_ids": ["AFZ.511C", "AFZ.512C"],
            }
        }
    )
    request_id: str = Field(min_length=1, max_length=64)
    assessment_date: date
    device_ids: list[str] | None = Field(default=None, max_length=100)

    @field_validator("device_ids")
    @classmethod
    def validate_device_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("device_ids must contain at least one ID or be null.")
        if value is not None and len(value) != len(set(value)):
            raise ValueError("device_ids must not contain duplicates.")
        return value


class StationPredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req-station-20260205-001",
                "assessment_date": "2026-02-05",
                "station_ids": ["SS-AFZ", "SS-BAZ"],
            }
        }
    )
    request_id: str = Field(min_length=1, max_length=64)
    assessment_date: date
    station_ids: list[str] | None = Field(default=None, max_length=100)

    @field_validator("station_ids")
    @classmethod
    def validate_station_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("station_ids must contain at least one ID or be null.")
        if value is not None and len(value) != len(set(value)):
            raise ValueError("station_ids must not contain duplicates.")
        return value


class DataJobRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req-clean-demo-001",
                "batch_id": "demo-raw-20260201",
                "business_date": "2026-02-01",
                "run_mode": "operation_only",
                "force": False,
            }
        }
    )
    request_id: str = Field(min_length=1, max_length=96)
    batch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    business_date: date
    run_mode: Literal["operation_only", "full"] = "operation_only"
    force: bool = False


class DataJobData(BaseModel):
    job_id: str
    batch_id: str
    business_date: date
    run_mode: str
    status: str
    current_stage: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error_message: str | None = None


class DeviceAssessmentItem(BaseModel):
    device_id: str
    station_id: str | None
    state_level: int = Field(ge=0, le=3)
    state_name: StateName
    confidence: float = Field(ge=0, le=1)
    high_risk_probability: float = Field(ge=0, le=1)
    dominant_discriminator: str
    discriminator_weights: dict[str, float]
    data_quality: Literal["pass", "warning"]


class DeviceAssessmentData(BaseModel):
    assessment_date: date
    model_name: str
    model_version: str
    items: list[DeviceAssessmentItem]


class StationAssessmentItem(BaseModel):
    station_id: str
    station_name: str | None
    state_level: int = Field(ge=0, le=3)
    state_name: StateName
    risk_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    high_risk_probability: float = Field(ge=0, le=1)
    dominant_view: str
    view_weights: dict[str, float]
    data_quality: Literal["pass", "warning"]


class StationAssessmentData(BaseModel):
    assessment_date: date
    model_name: str
    model_version: str
    items: list[StationAssessmentItem]


class DeviceProfileData(BaseModel):
    device_id: str
    station_id: str | None
    start_date: date
    end_date: date
    sample_count: int = Field(ge=1)
    average_high_risk_probability: float = Field(ge=0, le=1)
    average_confidence: float = Field(ge=0, le=1)
    dominant_discriminator: str
    average_discriminator_weights: dict[str, float]


class FamilyRiskDevice(BaseModel):
    device_id: str
    station_id: str | None
    state_level: int = Field(ge=0, le=3)
    state_name: StateName
    high_risk_probability: float = Field(ge=0, le=1)


class FamilyAlertItem(BaseModel):
    family_id: str
    manufacturer: str
    model: str
    batch: str
    batch_source: str
    device_count: int = Field(ge=3)
    evaluated_device_count: int = Field(ge=0)
    defect_count: int = Field(ge=0)
    defect_device_count: int = Field(ge=0)
    defect_rate: float = Field(ge=0, le=100)
    defect_events_per_device: float = Field(ge=0)
    z_score: float
    high_risk_device_count: int = Field(ge=0)
    average_high_risk_probability: float = Field(ge=0, le=1)
    max_high_risk_probability: float = Field(ge=0, le=1)
    level: Literal["G", "M", "L"]
    warning_reason: str
    top_risk_devices: list[FamilyRiskDevice]
    data_quality: Literal["pass", "warning"]


class FamilyAlertData(BaseModel):
    assessment_date: date
    rule_name: str
    rule_version: str
    family_key_definition: str
    minimum_family_size: int = Field(ge=3)
    eligible_family_count: int = Field(ge=0)
    alert_count: int = Field(ge=0)
    device_model_name: str
    device_model_version: str
    items: list[FamilyAlertItem]


class StationProfileData(BaseModel):
    station_id: str
    station_name: str | None
    start_date: date
    end_date: date
    sample_count: int = Field(ge=1)
    average_risk_score: float = Field(ge=0, le=100)
    average_high_risk_probability: float = Field(ge=0, le=1)
    average_confidence: float = Field(ge=0, le=1)
    dominant_view: str
    average_view_weights: dict[str, float]


class RiskOverviewData(BaseModel):
    assessment_date: date
    device_state_counts: dict[str, int]
    station_state_counts: dict[str, int]
    average_device_high_risk_probability: float
    average_station_risk_score: float
    top_risk_devices: list[dict[str, Any]]
    top_risk_stations: list[dict[str, Any]]


class EconomicParameters(BaseModel):
    load_growth_rate: float = Field(default=0.08, ge=0, le=1)
    capacity_multiplier: float = Field(default=1.25, gt=0, le=5)
    dispatch_cost_base: float = Field(default=0.30, ge=0)
    risk_cost_base: float = Field(default=0.20, ge=0)
    risk_cost_weight: float = Field(default=1.10, ge=0)
    shortfall_cost_base: float = Field(default=1.25, ge=0)
    shortfall_criticality_weight: float = Field(default=1.75, ge=0)


class DispatchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req-dispatch-20260205-001",
                "assessment_date": "2026-02-05",
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
            }
        }
    )
    request_id: str = Field(min_length=1, max_length=64)
    assessment_date: date
    station_ids: list[str] | None = Field(default=None, max_length=100)
    supply_budget_mw: float = Field(ge=0)
    economic_parameters: EconomicParameters = Field(default_factory=EconomicParameters)

    @field_validator("station_ids")
    @classmethod
    def validate_station_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("station_ids must contain at least one ID or be null.")
        if value is not None and len(value) != len(set(value)):
            raise ValueError("station_ids must not contain duplicates.")
        return value


class DispatchItem(BaseModel):
    station_id: str
    station_name: str | None
    max_adjustable_mw: float
    combined_risk_index: float
    recommended_allocation_mw: float
    unserved_adjustment_mw: float
    adjustment_service_rate: float
    dispatch_priority: int
    dispatch_reason: str


class DispatchData(BaseModel):
    assessment_date: date
    model_name: str
    solver_status: str
    station_count: int
    supply_budget_mw: float
    required_adjustment_mw: float
    allocated_adjustment_mw: float
    unserved_adjustment_mw: float
    parameter_source: str
    economic_parameters: dict[str, float]
    items: list[DispatchItem]
