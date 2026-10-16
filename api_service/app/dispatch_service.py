from __future__ import annotations

import sys
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from api_service.app.config import Settings
from api_service.app.feature_store import FeatureStore
from api_service.app.model_service import ObjectNotFoundError


class DispatchService:
    def __init__(self, settings: Settings, feature_store: FeatureStore) -> None:
        source_root = str(settings.project_root / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from griddx.economic_dispatch import prepare_dispatch_table, solve_dispatch

        self._prepare_dispatch_table = prepare_dispatch_table
        self._solve_dispatch = solve_dispatch
        self.feature_store = feature_store

    def dispatch(
        self,
        *,
        assessment_date: date,
        station_ids: list[str] | None,
        supply_budget_mw: float,
        economic_parameters: dict[str, float],
        request_overrode_parameters: bool,
    ) -> dict[str, Any]:
        _, station_path, _manifest = self.feature_store.get_active_paths()
        table = self._prepare_dispatch_table(station_path, str(assessment_date))
        if station_ids is not None:
            available = set(table["station_id"].dropna().astype(str))
            missing = [value for value in station_ids if value not in available]
            if missing:
                raise ObjectNotFoundError(
                    f"Unknown station IDs for dispatch date {assessment_date}: {missing}"
                )
            table = table[table["station_id"].astype(str).isin(station_ids)].copy()
        if table.empty:
            raise ValueError("No stations are available for economic dispatch.")

        table = self._apply_economic_parameters(table, economic_parameters)
        plan = self._solve_dispatch(table, supply_budget_mw=supply_budget_mw)
        items: list[dict[str, Any]] = []
        for _, row in plan.iterrows():
            items.append(
                {
                    "station_id": str(row["station_id"]),
                    "station_name": self._optional_text(row.get("station_name")),
                    "max_adjustable_mw": self._number(row.get("max_adjustable_mw")),
                    "combined_risk_index": self._number(row.get("combined_risk_index")),
                    "recommended_allocation_mw": self._number(
                        row.get("recommended_allocation_mw")
                    ),
                    "unserved_adjustment_mw": self._number(
                        row.get("unserved_adjustment_mw")
                    ),
                    "adjustment_service_rate": self._number(
                        row.get("adjustment_service_rate")
                    ),
                    "dispatch_priority": int(row.get("dispatch_priority", 0)),
                    "dispatch_reason": str(row.get("dispatch_reason", "")),
                }
            )
        return {
            "assessment_date": assessment_date,
            "model_name": "RA-MOD",
            "solver_status": str(plan["solver_status"].iloc[0]),
            "station_count": int(len(plan)),
            "supply_budget_mw": self._number(supply_budget_mw),
            "required_adjustment_mw": self._number(plan["required_adjustment_mw"].sum()),
            "allocated_adjustment_mw": self._number(
                plan["recommended_allocation_mw"].sum()
            ),
            "unserved_adjustment_mw": self._number(
                plan["unserved_adjustment_mw"].sum()
            ),
            "parameter_source": (
                "request_override" if request_overrode_parameters else "built_in_baseline"
            ),
            "economic_parameters": {
                key: self._number(value) for key, value in economic_parameters.items()
            },
            "items": items,
        }

    @staticmethod
    def _apply_economic_parameters(
        table: pd.DataFrame, parameters: dict[str, float]
    ) -> pd.DataFrame:
        work = table.copy()
        current = work["current_load_mw_proxy"].fillna(0).clip(lower=0)
        base_capacity = work["capacity_mw_proxy"].fillna(0) / 1.25
        work["capacity_mw_proxy"] = (
            base_capacity * parameters["capacity_multiplier"]
        ).clip(lower=0)
        work["forecast_demand_mw"] = (
            current * (1.0 + parameters["load_growth_rate"])
        ).clip(lower=0.1)
        derate = np.select(
            [
                work["state_label"] == 0,
                work["state_label"] == 1,
                work["state_label"] == 2,
                work["state_label"] >= 3,
            ],
            [1.00, 0.85, 0.60, 0.25],
            default=0.75,
        )
        work["required_adjustment_mw"] = (
            work["forecast_demand_mw"] - current
        ).clip(lower=0.05)
        available = (work["capacity_mw_proxy"] * derate - current).clip(lower=0)
        work["risk_adjusted_headroom_mw"] = (
            available * work["maintenance_availability_factor"]
        )
        work["max_adjustable_mw"] = np.minimum(
            work["required_adjustment_mw"], work["risk_adjusted_headroom_mw"]
        )
        base_dispatch_component = work["dispatch_cost_per_mw"] - 0.30
        work["dispatch_cost_per_mw"] = (
            parameters["dispatch_cost_base"] + base_dispatch_component
        ).clip(lower=0)
        work["risk_cost_per_mw"] = (
            parameters["risk_cost_base"]
            + parameters["risk_cost_weight"] * work["combined_risk_index"]
        )
        work["shortfall_cost_per_mw"] = (
            parameters["shortfall_cost_base"]
            + parameters["shortfall_criticality_weight"]
            * work["station_criticality_index"]
        )
        work["marginal_net_value"] = (
            work["shortfall_cost_per_mw"]
            - work["dispatch_cost_per_mw"]
            - work["risk_cost_per_mw"]
        )
        work["unit_benefit"] = work["shortfall_cost_per_mw"]
        work["net_value"] = work["marginal_net_value"]
        return work

    @staticmethod
    def _number(value: Any) -> float:
        number = float(value)
        return round(number, 6) if np.isfinite(number) else 0.0

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        return None if value is None or pd.isna(value) else str(value)
