# 基础建模报告

## 背景

当前南网数据已经形成设备日级宽表和站点日级聚合表，可以支撑设备状态预测、站点状态评估和经济调电优化的基础建模。由于缺陷、跳闸、检修和人工复核等级等监督标签暂未补齐，本阶段采用冷启动评分构造伪标签，用于跑通实验流程和建立多模型对比框架。

## 建模链路

```text
CSV 数据
-> 数据清洗
-> 特征工程
-> 冷启动状态评分
-> 四级状态标签
-> 多模型训练与评估
-> 站点风险结果
-> 经济调电优化
```

## 设备状态预测

设备模型输入来自 `model_base_device_day.csv`。特征包括当日量测、数据质量标记、滚动统计、同站同类偏离和物理语义特征。当前输出为四级状态标签：

| 标签 | 含义 |
| --- | --- |
| 0 | 正常 |
| 1 | 关注 |
| 2 | 异常 |
| 3 | 高风险 |

当前设备侧 quick 结果中，`hist_gradient_boosting` 的 Macro-F1 最高，为 0.8818。

## 站点状态评估

站点模型输入来自 `model_base_station_day_extract.csv`。特征包括站点设备数量、运行覆盖率、电压 spread、电流峰均比、有功无功汇总、开关动作率和站点近 7 日变化。

当前站点侧 quick 结果中，`hist_gradient_boosting` 的 Macro-F1 最高，为 0.8010。

## 模型清单

| 模型 | 类型 | 作用 |
| --- | --- | --- |
| `logistic_regression` | 线性模型 | 可解释基线 |
| `random_forest` | Bagging 树模型 | 非线性表格基线 |
| `extra_trees` | 随机树模型 | 树模型稳定性对照 |
| `hist_gradient_boosting` | Boosting 树模型 | 当前主力表格基线 |
| `torch_mlp` | PyTorch 神经网络 | GPU 训练链路和深度模型基线 |

## PyTorch 设备选择

神经网络训练支持：

```text
auto -> cuda -> mps -> cpu
```

常用命令：

```bash
python scripts/run_baseline_pipeline.py --torch-device cuda --gpu-id 0
```

在 Linux 3090 服务器上，`--gpu-id 0` 表示使用当前进程可见的第 0 张 CUDA GPU。

## 经济调电模型

经济调电模块使用线性规划，目标为最大化净经济价值：

```text
净经济价值 = 供电收益 - 运行风险惩罚
```

约束包括：

- 总供电预算。
- 站点可调容量上限。
- 站点状态风险降额。
- 高风险站点限制新增负荷。

输出为：

```text
outputs/baseline_models/economic_dispatch/dispatch_plan.csv
```

## 当前限制

1. 当前标签是伪标签，不是正式业务标签。
2. 设备静态台账字段暂不完整。
3. 区域经济数据、峰谷电价、需求响应收益尚未接入。
4. 当前神经网络是 MLP，后续需要补充 LSTM、TCN、Transformer 等时序模型。

## 后续方向

优先补充真实标签和业务侧经济参数，然后重新训练模型。树模型适合作为第一阶段表格主力模型，PyTorch 模型适合在 3090 服务器上做序列模型和大样本实验。
