# RA-MOD：风险约束多目标经济调度模型

## 1. 模型定位

经济调度模型命名为 RA-MOD（Risk-Aware Multi-Objective Dispatch），代码入口位于 `economic_dispatch.py`。

RA-MOD 接收站点负荷、容量、状态风险、历史事件、检修可用性和站点重要度，在有限增量供电预算下确定每个站点的推荐调电量。模型不再进行简单风险排序，而是显式权衡：

1. 调电执行成本。
2. 向风险站点增加承载带来的风险暴露成本。
3. 未满足负荷增长需求造成的供电缺额损失。

## 2. 问题定义

站点下一时段负荷由当前负荷乘以增长系数得到。若所有站点都需要增量供电，但可用调电预算不足，就需要回答：

- 哪些站点应优先获得增量供电？
- 高风险或检修未闭环站点应如何降额？
- 高电压等级、高负荷、主变较多的关键站点缺额损失如何体现？
- 调电成本、运行风险和供电保障之间如何统一到同一个目标函数？

RA-MOD 将这些问题转化为带连续变量的线性规划，由 SciPy HiGHS 求解器计算全局最优解。

## 3. 输入与代理参数

当前数据尚未包含真实节点电价、购电成本和停电损失，因此模型使用可替换的代理参数。代理参数全部在输出中单独保留，后续可以直接替换成经济调研数据。

### 3.1 负荷与容量

```text
current_load = abs(active_power_3phase_sum)
capacity = 站点历史有功绝对值 P95 * 1.25
forecast_demand = current_load * 1.08
required_adjustment = forecast_demand - current_load
```

其中 `required_adjustment_mw` 表示下一时段需要额外调入的增量功率。

### 3.2 综合风险指数

```text
combined_risk =
    0.45 * station_state_risk
  + 0.20 * history_event_risk
  + 0.20 * unresolved_maintenance_risk
  + 0.15 * environment_risk
```

状态等级进一步形成容量降额系数：正常 1.00、关注 0.85、异常 0.60、高风险 0.25。未处理检修风险还会形成 0.65 至 1.00 的检修可用系数。

风险调整后可调上限为：

```text
risk_adjusted_headroom =
    max(capacity * state_derate - current_load, 0)
    * maintenance_availability

max_adjustable = min(required_adjustment, risk_adjusted_headroom)
```

### 3.3 站点重要度

站点重要度用于衡量供电缺额的影响：

```text
criticality =
    0.45 * voltage_level_rank
  + 0.35 * load_rank
  + 0.20 * transformer_count_rank
```

高电压等级、高负荷和主变数量较多的站点具有更高缺额损失系数。

## 4. 决策变量

对每个站点 `i` 设置两个连续决策变量：

- `x_i`：推荐增量调电量，即 `recommended_allocation_mw`。
- `s_i`：未满足增量需求，即 `unserved_adjustment_mw`。

如果共有 `N` 个站点，线性规划包含 `2N` 个连续变量。

## 5. 三类目标成本

### 5.1 调电执行成本

```text
dispatch_cost_per_mw =
    0.30
  + 0.20 * load_rank
  + 0.20 * (1 - response_potential)
  + 0.15 * (1 - operation_coverage)
```

负荷较高、开关响应潜力较弱或数据覆盖率较低时，调电执行成本提高。

### 5.2 风险暴露成本

```text
risk_cost_per_mw = 0.20 + 1.10 * combined_risk
```

该成本随站点风险上升，用于抑制向高风险站点继续增加承载。

### 5.3 供电缺额损失

```text
shortfall_cost_per_mw = 1.25 + 1.75 * criticality
```

关键站点未获得增量供电时具有更高损失，从而在预算不足时获得更高优先级。

## 6. 目标函数

RA-MOD 最小化总目标成本：

```text
minimize sum_i(
    dispatch_cost_i * x_i
  + risk_cost_i * x_i
  + shortfall_cost_i * s_i
)
```

也可以从边际价值角度解释：

```text
marginal_net_value_i =
    shortfall_cost_i
  - dispatch_cost_i
  - risk_cost_i
```

边际净价值越高，说明向该站点调入 1 MW 相比保留缺额更有价值。

## 7. 约束条件

### 7.1 总调电预算

```text
sum_i(x_i) <= supply_budget_mw
```

### 7.2 单站点风险容量约束

```text
0 <= x_i <= max_adjustable_i
```

### 7.3 需求平衡约束

```text
x_i + s_i >= required_adjustment_i
```

### 7.4 缺额边界

```text
0 <= s_i <= required_adjustment_i
```

这些约束保证模型始终可行。即使某站点因风险或容量限制无法调入功率，也可以通过缺额变量记录未满足需求及其经济损失。

## 8. 求解方法

模型使用 `scipy.optimize.linprog(method="highs")`。HiGHS 是成熟的线性规划求解器，能够给出全局最优解和求解状态。

若求解器异常，代码会按边际净价值从高到低执行贪心回退，并继续满足总预算和单站点上限。正常实验结果的 `solver_status` 为 `optimal`。

## 9. 当前实验结果

2026-05-06 的 77 个站点调度结果：

| 指标 | 结果 |
| --- | ---: |
| 增量供电预算 | 258.238 MW |
| 总增量需求 | 673.135 MW |
| 推荐调电量 | 258.238 MW |
| 未满足增量需求 | 414.897 MW |
| 总代理目标成本 | 1,299.754 |
| 求解状态 | optimal |

该结果表示在当前代理成本和风险约束下，预算已全部分配。高风险、容量不足或边际风险调整价值较低的站点保留为缺额，由 `shortfall_loss_cost` 显式记录影响。

## 10. 输出文件

```text
outputs/baseline_models/economic_dispatch/
├── dispatch_plan.csv
└── dispatch_summary.json
```

`dispatch_plan.csv` 包含每个站点的需求、风险调整容量、三类单位成本、调电量、缺额量、服务率、三类实际成本、优先级和调度原因。

## 11. 推荐项目表述

> 构建 RA-MOD 风险约束多目标经济调度模型，将站点状态风险、历史事件、检修可用性和站点重要度映射为风险成本与供电缺额损失。在总调电预算、单站点风险容量和需求平衡约束下，通过线性规划联合优化调电执行成本、风险暴露成本和缺额损失，形成可解释的站点级调电方案。

## 12. 当前限制与真实经济数据替换

当前成本系数是基于负荷、风险和重要度构造的代理值，不能直接用于生产调度。经济调研完成后应依次替换：

1. `dispatch_cost_per_mw`：替换为分时购电成本、启停成本、网损和调节成本。
2. `risk_cost_per_mw`：替换为故障概率乘事故损失期望值。
3. `shortfall_cost_per_mw`：替换为不同用户类型的失供电价值和停电损失。
4. `max_adjustable_mw`：替换为潮流计算、线路热稳限额和主变真实可用容量。

保持线性规划框架不变即可完成真实参数升级，不需要立即引入复杂强化学习。
