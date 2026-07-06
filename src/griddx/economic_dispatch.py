from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .features import build_station_features
from .labels import make_station_cold_start_labels
from .paths import data_path, ensure_output_dir


STATE_LABELS = {
    0: "normal",
    1: "watch",
    2: "abnormal",
    3: "high_risk",
}


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
    features = make_station_cold_start_labels(build_station_features(day))

    active = features.get("active_power_3phase_sum", pd.Series(0.0, index=features.index)).abs().fillna(0)
    historical = df.copy()
    historical["active_abs"] = historical["active_power_3phase_sum"].abs()
    p95 = historical.groupby("station_id")["active_abs"].quantile(0.95)
    features["current_load_mw_proxy"] = active
    features["capacity_mw_proxy"] = features["station_id"].map(p95).fillna(active.quantile(0.95)).clip(lower=1.0) * 1.25
    features["forecast_demand_mw"] = (active * 1.08).clip(lower=0.1)

    risk = features["state_score"].fillna(50) / 100.0
    features["risk_penalty"] = 0.15 + 0.85 * risk
    features["state_name"] = features["state_label"].map(STATE_LABELS)
    derate = np.select(
        [features["state_label"] == 0, features["state_label"] == 1, features["state_label"] == 2, features["state_label"] >= 3],
        [1.00, 0.85, 0.60, 0.25],
        default=0.75,
    )
    available = (features["capacity_mw_proxy"] * derate - features["current_load_mw_proxy"]).clip(lower=0)
    features["max_adjustable_mw"] = np.minimum(features["forecast_demand_mw"], available)

    load_rank = features["forecast_demand_mw"].rank(pct=True).fillna(0.5)
    coverage = features.get("operation_coverage_rate", pd.Series(1.0, index=features.index)).fillna(1.0).clip(0, 1)
    response_potential = features.get("switch_action_rate", pd.Series(0.0, index=features.index)).rank(pct=True).fillna(0.5)
    features["unit_benefit"] = 0.45 * load_rank + 0.25 * coverage + 0.20 * response_potential + 0.10
    features["net_value"] = features["unit_benefit"] - 0.55 * features["risk_penalty"]
    return features


def solve_dispatch(table: pd.DataFrame, supply_budget_mw: float | None = None) -> pd.DataFrame:
    work = table.copy()
    if supply_budget_mw is None:
        supply_budget_mw = float(work["max_adjustable_mw"].sum() * 0.55)
    bounds = [(0, float(max_value)) for max_value in work["max_adjustable_mw"].fillna(0)]
    c = -work["net_value"].fillna(0).to_numpy(dtype=float)
    a_ub = np.ones((1, len(work)))
    b_ub = np.array([supply_budget_mw], dtype=float)

    if len(work) == 0 or sum(upper for _, upper in bounds) <= 0:
        work["recommended_allocation_mw"] = 0.0
        work["dispatch_reason"] = "no adjustable capacity"
        return work

    result = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if result.success:
        allocation = result.x
    else:
        allocation = greedy_dispatch(work, supply_budget_mw)
        print(f"linprog failed, used greedy fallback: {result.message}")
    work["recommended_allocation_mw"] = allocation
    work["dispatch_priority"] = work["net_value"].rank(ascending=False, method="dense").astype(int)
    work["dispatch_reason"] = np.where(
        work["state_label"] >= 3,
        "high risk station, strongly derated",
        np.where(work["recommended_allocation_mw"] > 0, "allocated by net economic value under risk constraints", "not selected"),
    )
    return work.sort_values(["recommended_allocation_mw", "net_value"], ascending=False)


def greedy_dispatch(table: pd.DataFrame, supply_budget_mw: float) -> np.ndarray:
    allocation = np.zeros(len(table), dtype=float)
    remaining = supply_budget_mw
    order = table["net_value"].fillna(-1e9).sort_values(ascending=False).index
    loc = {idx: i for i, idx in enumerate(table.index)}
    for idx in order:
        if remaining <= 1e-9:
            break
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
        "max_adjustable_mw",
        "unit_benefit",
        "risk_penalty",
        "net_value",
        "recommended_allocation_mw",
        "dispatch_priority",
        "dispatch_reason",
    ]
    existing = [col for col in cols if col in plan.columns]
    plan[existing].to_csv(output_dir / "dispatch_plan.csv", index=False)
    print(plan[existing].head(20).to_string(index=False))
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a station-level economic dispatch plan.")
    parser.add_argument("--station-csv", type=Path, default=data_path("model_base_station_day_extract.csv"))
    parser.add_argument("--date", default=None)
    parser.add_argument("--supply-budget-mw", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dispatch(args.station_csv, date=args.date, supply_budget_mw=args.supply_budget_mw)


if __name__ == "__main__":
    main()
