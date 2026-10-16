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

## 3. 输入与基础经济参数

当前 RA-MOD 是可直接运行的基础参数化经济模型。由于现阶段没有完整的节点电价、购电成本和停电损失数据，程序内置了一套硬编码的默认经济场景；默认值集中管理并带有版本号，不分散在接口或业务逻辑中。

| 参数 | 基础默认值 | 作用 |
| --- | ---: | --- |
| `load_growth_rate` | 0.08 | 下一时段负荷增长率 |
| `capacity_multiplier` | 1.25 | 历史负荷 P95 到容量代理值的放大系数 |
| `default_budget_ratio` | 0.55 | 未指定预算时，可调上限合计的预算比例 |
| `dispatch_cost_base` | 0.30 | 调电执行基础成本 |
| `risk_cost_base` | 0.20 | 风险暴露基础成本 |
| `risk_cost_weight` | 1.10 | 综合风险对应的成本权重 |
| `shortfall_cost_base` | 1.25 | 供电缺额基础损失 |
| `shortfall_criticality_weight` | 1.75 | 站点重要度对应的缺额损失权重 |

FastAPI 调度接口允许直接输入 `supply_budget_mw`、全局 `economic_parameters` 和 `station_parameter_overrides`。参数优先级为：站点级请求参数、全局请求参数、数据库 B10 参数、程序默认参数。请求未携带经济参数时，系统自动使用基础默认场景；传入参数时仅覆盖指定项，其余项继续使用默认值。

每次计算都会保存最终生效参数、`parameter_source` 和 `economic_parameter_version`，因此硬编码基础场景与外部覆盖场景都可以复现和审计。

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
| 基础场景总目标成本 | 1,299.754 |
| 求解状态 | optimal |

该结果表示在当前基础经济参数和风险约束下，预算已全部分配。高风险、容量不足或边际风险调整价值较低的站点保留为缺额，由 `shortfall_loss_cost` 显式记录影响。

## 10. 输出文件

```text
outputs/baseline_models/economic_dispatch/
├── dispatch_plan.csv
└── dispatch_summary.json
```

`dispatch_plan.csv` 包含每个站点的需求、风险调整容量、三类单位成本、调电量、缺额量、服务率、三类实际成本、优先级和调度原因。

## 11. 推荐项目表述

> 构建 RA-MOD 风险约束多目标经济调度模型，将站点状态风险、历史事件、检修可用性和站点重要度映射为风险成本与供电缺额损失。在总调电预算、单站点风险容量和需求平衡约束下，通过线性规划联合优化调电执行成本、风险暴露成本和缺额损失，形成可解释的站点级调电方案。

## 12. 参数升级与生产边界

基础版的成本系数由负荷、风险和重要度构造，适合完成流程联调、方案比较和辅助决策。它们不是实际电价或事故损失值，因此当前结果保持 `decision_type=advisory`，不直接作为自动控制指令。

经济调研完成后，可通过 API 参数或 B10 数据表依次覆盖：

1. `dispatch_cost_per_mw`：替换为分时购电成本、启停成本、网损和调节成本。
2. `risk_cost_per_mw`：替换为故障概率乘事故损失期望值。
3. `shortfall_cost_per_mw`：替换为不同用户类型的失供电价值和停电损失。
4. `max_adjustable_mw`：替换为潮流计算、线路热稳限额和主变真实可用容量。

参数覆盖发生在优化求解前，模型仍执行相同的边界检查和线性规划。保持现有框架不变即可完成真实参数升级，不需要重新训练模型，也不需要立即引入复杂强化学习。
