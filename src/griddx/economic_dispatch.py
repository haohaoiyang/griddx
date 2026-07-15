from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .features import build_station_features
from .labels import has_station_enriched_fields, make_station_cold_start_labels, make_station_enriched_weak_labels
from .paths import data_path, ensure_output_dir


STATE_LABELS = {
    0: "normal",
    1: "watch",
    2: "abnormal",
    3: "high_risk",
}

DISPATCH_MODEL_NAME = "RA-MOD"


def prepare_dispatch_table(station_csv: Path, date: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(station_csv, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if date:
        selected_date = pd.to_datetime(date)
    else:
        selected_date = df["date"].max()
    day = df[df["date"] == selected_date].copy()
    if day.empty:
        raise ValueError(f"No station rows found for date={selected_date.date()}")
    features = build_station_features(day)
    if has_station_enriched_fields(features):
        features = make_station_enriched_weak_labels(features)
    else:
        features = make_station_cold_start_labels(features)

    active = features.get("active_power_3phase_sum", pd.Series(0.0, index=features.index)).abs().fillna(0)
    historical = df.copy()
    historical["active_abs"] = historical["active_power_3phase_sum"].abs()
    p95 = historical.groupby("station_id")["active_abs"].quantile(0.95)
    features["current_load_mw_proxy"] = active
    features["capacity_mw_proxy"] = features["station_id"].map(p95).fillna(active.quantile(0.95)).clip(lower=1.0) * 1.25
    features["forecast_demand_mw"] = (active * 1.08).clip(lower=0.1)

    state_risk = features["state_score"].fillna(50) / 100.0
    history_risk = features.get("history_event_count", pd.Series(0.0, index=features.index)).rank(pct=True).fillna(0.5)
    maintenance_gap_risk = features.get(
        "unresolved_event_count_proxy", pd.Series(0.0, index=features.index)
    ).rank(pct=True).fillna(0.5)
    environment_risk = features.get(
        "environment_risk_level", pd.Series(0.0, index=features.index)
    ).rank(pct=True).fillna(0.5)
    combined_risk = 0.45 * state_risk + 0.20 * history_risk + 0.20 * maintenance_gap_risk + 0.15 * environment_risk
    features["combined_risk_index"] = combined_risk.clip(0, 1)
    features["risk_penalty"] = 0.15 + 0.85 * features["combined_risk_index"]
    features["maintenance_availability_factor"] = (1.0 - 0.35 * maintenance_gap_risk).clip(0.65, 1.0)
    features["state_name"] = features["state_label"].map(STATE_LABELS)
    derate = np.select(
        [features["state_label"] == 0, features["state_label"] == 1, features["state_label"] == 2, features["state_label"] >= 3],
        [1.00, 0.85, 0.60, 0.25],
        default=0.75,
    )
    features["required_adjustment_mw"] = (
        features["forecast_demand_mw"] - features["current_load_mw_proxy"]
    ).clip(lower=0.05)
    available = (features["capacity_mw_proxy"] * derate - features["current_load_mw_proxy"]).clip(lower=0)
    features["risk_adjusted_headroom_mw"] = available * features["maintenance_availability_factor"]
    features["max_adjustable_mw"] = np.minimum(
        features["required_adjustment_mw"], features["risk_adjusted_headroom_mw"]
    )

    load_rank = features["forecast_demand_mw"].rank(pct=True).fillna(0.5)
    coverage = features.get("operation_coverage_rate", pd.Series(1.0, index=features.index)).fillna(1.0).clip(0, 1)
    response_potential = features.get("switch_action_rate", pd.Series(0.0, index=features.index)).rank(pct=True).fillna(0.5)
    voltage_rank = features.get("voltage_level_kv", pd.Series(0.0, index=features.index)).rank(pct=True).fillna(0.5)
    transformer_rank = features.get(
        "main_transformer_count", pd.Series(0.0, index=features.index)
    ).rank(pct=True).fillna(0.5)
    features["station_criticality_index"] = (
        0.45 * voltage_rank + 0.35 * load_rank + 0.20 * transformer_rank
    ).clip(0, 1)
    features["dispatch_cost_per_mw"] = (
        0.30 + 0.20 * load_rank + 0.20 * (1.0 - response_potential) + 0.15 * (1.0 - coverage)
    )
    features["risk_cost_per_mw"] = 0.20 + 1.10 * features["combined_risk_index"]
    features["shortfall_cost_per_mw"] = 1.25 + 1.75 * features["station_criticality_index"]
    features["marginal_net_value"] = (
        features["shortfall_cost_per_mw"]
        - features["dispatch_cost_per_mw"]
        - features["risk_cost_per_mw"]
    )
    features["unit_benefit"] = features["shortfall_cost_per_mw"]
    features["net_value"] = features["marginal_net_value"]
    return features


def solve_dispatch(table: pd.DataFrame, supply_budget_mw: float | None = None) -> pd.DataFrame:
    work = table.copy()
    if supply_budget_mw is None:
        supply_budget_mw = float(work["max_adjustable_mw"].sum() * 0.55)
    if supply_budget_mw < 0:
        raise ValueError("supply_budget_mw must be non-negative.")
    station_count = len(work)
    allocation_bounds = [(0, float(value)) for value in work["max_adjustable_mw"].fillna(0)]
    shortfall_bounds = [(0, float(value)) for value in work["required_adjustment_mw"].fillna(0)]

    if station_count == 0:
        work["recommended_allocation_mw"] = 0.0
        work["unserved_adjustment_mw"] = 0.0
        work["solver_status"] = "empty input"
        return work

    dispatch_objective = (
        work["dispatch_cost_per_mw"] + work["risk_cost_per_mw"]
    ).fillna(0).to_numpy(dtype=float)
    shortfall_objective = work["shortfall_cost_per_mw"].fillna(0).to_numpy(dtype=float)
    c = np.concatenate([dispatch_objective, shortfall_objective])

    budget_row = np.concatenate([np.ones(station_count), np.zeros(station_count)])
    balance_rows = np.zeros((station_count, 2 * station_count), dtype=float)
    for index in range(station_count):
        balance_rows[index, index] = -1.0
        balance_rows[index, station_count + index] = -1.0
    a_ub = np.vstack([budget_row, balance_rows])
    b_ub = np.concatenate(
        [np.array([supply_budget_mw]), -work["required_adjustment_mw"].to_numpy(dtype=float)]
    )
    bounds = allocation_bounds + shortfall_bounds

    result = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if result.success:
        allocation = np.clip(result.x[:station_count], 0, None)
        shortfall = np.clip(result.x[station_count:], 0, None)
        solver_status = "optimal"
    else:
        allocation = greedy_dispatch(work, supply_budget_mw)
        shortfall = (work["required_adjustment_mw"].to_numpy(dtype=float) - allocation).clip(min=0)
        solver_status = f"greedy fallback: {result.message}"
        print(f"linprog failed, used greedy fallback: {result.message}")
    work["recommended_allocation_mw"] = allocation
    work["unserved_adjustment_mw"] = shortfall
    work["adjustment_service_rate"] = np.where(
        work["required_adjustment_mw"] > 1e-9,
        work["recommended_allocation_mw"] / work["required_adjustment_mw"],
        1.0,
    )
    work["dispatch_energy_cost"] = work["recommended_allocation_mw"] * work["dispatch_cost_per_mw"]
    work["risk_exposure_cost"] = work["recommended_allocation_mw"] * work["risk_cost_per_mw"]
    work["shortfall_loss_cost"] = work["unserved_adjustment_mw"] * work["shortfall_cost_per_mw"]
    work["total_objective_cost"] = (
        work["dispatch_energy_cost"] + work["risk_exposure_cost"] + work["shortfall_loss_cost"]
    )
    work["solver_status"] = solver_status
    work["supply_budget_mw"] = supply_budget_mw
    work["dispatch_priority"] = work["marginal_net_value"].rank(ascending=False, method="dense").astype(int)
    work["dispatch_reason"] = np.where(
        work["max_adjustable_mw"] <= 1e-9,
        "risk or capacity constraint leaves no dispatch headroom",
        np.where(
            work["recommended_allocation_mw"] > 1e-9,
            "allocated by avoided shortfall loss under cost and risk constraints",
            "not allocated because marginal risk-adjusted value is lower",
        ),
    )
    return work.sort_values(["recommended_allocation_mw", "marginal_net_value"], ascending=False)


def greedy_dispatch(table: pd.DataFrame, supply_budget_mw: float) -> np.ndarray:
    allocation = np.zeros(len(table), dtype=float)
    remaining = supply_budget_mw
    order = table["marginal_net_value"].fillna(-1e9).sort_values(ascending=False).index
    loc = {idx: i for i, idx in enumerate(table.index)}
    for idx in order:
        if remaining <= 1e-9:
            break
        if float(table.loc[idx, "marginal_net_value"]) <= 0:
            continue
        upper = float(table.loc[idx, "max_adjustable_mw"])
        take = min(upper, remaining)
        allocation[loc[idx]] = take
        remaining -= take
    return allocation


def run_dispatch(station_csv: Path, date: str | None = None, supply_budget_mw: float | None = None) -> pd.DataFrame:
    output_dir = ensure_output_dir("economic_dispatch")
    table = prepare_dispatch_table(station_csv, date)
    plan = solve_dispatch(table, supply_budget_mw=supply_budget_mw)
    cols = [
        "date",
        "station_id",
        "station_name",
        "state_name",
        "state_score",
        "current_load_mw_proxy",
        "capacity_mw_proxy",
        "forecast_demand_mw",
        "required_adjustment_mw",
        "risk_adjusted_headroom_mw",
        "max_adjustable_mw",
        "combined_risk_index",
        "station_criticality_index",
        "maintenance_availability_factor",
        "dispatch_cost_per_mw",
        "risk_cost_per_mw",
        "shortfall_cost_per_mw",
        "marginal_net_value",
        "recommended_allocation_mw",
        "unserved_adjustment_mw",
        "adjustment_service_rate",
        "dispatch_energy_cost",
        "risk_exposure_cost",
        "shortfall_loss_cost",
        "total_objective_cost",
        "dispatch_priority",
        "solver_status",
        "dispatch_reason",
    ]
    existing = [col for col in cols if col in plan.columns]
    plan[existing].to_csv(output_dir / "dispatch_plan.csv", index=False)
    summary = {
        "model_name": DISPATCH_MODEL_NAME,
        "date": str(pd.Timestamp(plan["date"].iloc[0]).date()),
        "solver_status": str(plan["solver_status"].iloc[0]),
        "station_count": int(len(plan)),
        "supply_budget_mw": float(plan["supply_budget_mw"].iloc[0]),
        "required_adjustment_mw": float(plan["required_adjustment_mw"].sum()),
        "allocated_adjustment_mw": float(plan["recommended_allocation_mw"].sum()),
        "unserved_adjustment_mw": float(plan["unserved_adjustment_mw"].sum()),
        "total_objective_cost": float(plan["total_objective_cost"].sum()),
    }
    (output_dir / "dispatch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(plan[existing].head(20).to_string(index=False))
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a station-level economic dispatch plan.")
    parser.add_argument("--station-csv", type=Path, default=data_path("model_base_station_day_extract_enriched.csv"))
    parser.add_argument("--date", default=None)
    parser.add_argument("--supply-budget-mw", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dispatch(args.station_csv, date=args.date, supply_budget_mw=args.supply_budget_mw)


if __name__ == "__main__":
    main()
