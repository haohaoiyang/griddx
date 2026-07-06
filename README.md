# griddx：南网运行状态评估与经济调电基础建模工具

`griddx` 是基于当前南网项目数据搭建的基础建模代码仓库，用于支撑后续设备状态预测、站点状态评估、风险预警和经济调电优化实验。当前阶段重点是把数据读取、特征构造、多模型对比、PyTorch GPU 训练和调电优化流程先跑通，形成后续接入真实标签后的实验底座。

## 项目现状

当前已经实现三条主线：

1. 设备级状态预测：基于设备日级运行宽表，预测单台设备的状态等级。
2. 站点级状态评估：基于站点日级聚合表，评估站点整体运行状态。
3. 经济调电优化：综合站点状态、负荷需求、容量代理和风险惩罚，输出站点供电分配建议。

目前数据中的缺陷、跳闸、检修等正式监督标签仍为空，因此当前代码采用“冷启动评分 + 伪标签 + 多模型对比”的方案。它可以用于环境验证、流程验证和模型基线对比，但不能直接视为最终业务模型效果。

## 目录结构

```text
griddx/
├── README.md
├── configs/
│   ├── environment-macos.yml
│   └── environment-linux-cu128.yml
├── docs/
│   └── baseline_modeling_report.md
├── examples/
│   ├── grid_fault_demo.py
│   └── test.py
├── scripts/
│   └── run_baseline_pipeline.py
├── src/griddx/
│   ├── data.py
│   ├── features.py
│   ├── labels.py
│   ├── model_zoo.py
│   ├── device_prediction.py
│   ├── station_assessment.py
│   ├── economic_dispatch.py
│   ├── evaluation.py
│   └── paths.py
└── outputs/
    ├── baseline_models/
    └── demo/
```

## 环境配置

### macOS Apple Silicon

```bash
cd ~/Desktop/griddx
mamba env create -f configs/environment-macos.yml
conda activate griddx
```

### Linux + RTX 3090

```bash
cd ~/Desktop/griddx
mamba env create -f configs/environment-linux-cu128.yml
conda activate griddx
```

如果服务器 CUDA 驱动不支持 `cu128`，需要根据服务器 `nvidia-smi` 显示的驱动版本，把环境文件中的 PyTorch CUDA wheel 源切换为 `cu126` 或 `cu118`。

## 数据路径

默认读取南网资料目录：

```text
/Users/joyhanzhu/Desktop/南网
```

在 Linux 服务器上可通过环境变量指定数据目录：

```bash
export GRIDDX_DATA_ROOT=/path/to/nanwang_data
```

默认使用的数据文件：

| 文件 | 用途 |
| --- | --- |
| `model_base_device_day.csv` | 设备日级主表 |
| `model_base_device_day_line_or_load_like.csv` | 线路/负荷类设备子表 |
| `model_base_station_day.csv` | 站点日级聚合表 |
| `model_base_station_day_extract.csv` | 站点日级完整抽取表 |

## 快速运行

在本机快速验证全流程：

```bash
cd ~/Desktop/griddx
conda activate griddx
python scripts/run_baseline_pipeline.py --quick --torch-device auto
```

`--quick` 会限制样本量并缩短神经网络训练轮数，用于验证环境和代码链路是否正常。

## Linux 服务器 GPU 使用方式

自动选择设备：

```bash
python scripts/run_baseline_pipeline.py --torch-device auto
```

设备选择优先级：

```text
cuda -> mps -> cpu
```

指定第 0 张可见 GPU：

```bash
python scripts/run_baseline_pipeline.py \
  --device-max-rows 300000 \
  --mlp-epochs 50 \
  --batch-size 1024 \
  --torch-device cuda \
  --gpu-id 0
```

指定第 1 张可见 GPU：

```bash
python scripts/run_baseline_pipeline.py \
  --device-max-rows 300000 \
  --mlp-epochs 50 \
  --batch-size 1024 \
  --torch-device cuda \
  --gpu-id 1
```

如果使用 `CUDA_VISIBLE_DEVICES=2` 暴露物理第 2 张卡，那么代码里的 `--gpu-id 0` 表示当前进程可见的第 0 张卡，也就是物理第 2 张卡：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_baseline_pipeline.py \
  --torch-device cuda \
  --gpu-id 0
```

## 单独运行模块

设备预测：

```bash
PYTHONPATH=src python -m griddx.device_prediction \
  --max-rows 120000 \
  --mlp-epochs 18 \
  --torch-device auto
```

站点评估：

```bash
PYTHONPATH=src python -m griddx.station_assessment \
  --mlp-epochs 18 \
  --torch-device auto
```

经济调电：

```bash
PYTHONPATH=src python -m griddx.economic_dispatch
```

## 当前使用的模型

### 传统机器学习模型

| 模型 | 代码名 | 选择依据 |
| --- | --- | --- |
| 逻辑回归 One-vs-Rest | `logistic_regression` | 作为线性可解释基线，便于判断特征是否有基本区分度 |
| 随机森林 | `random_forest` | 对表格特征鲁棒，能处理非线性关系 |
| 极端随机树 | `extra_trees` | 比随机森林更随机，用于观察树模型稳定性 |
| 直方图梯度提升树 | `hist_gradient_boosting` | 适合结构化表格数据，是当前主力基线 |

### PyTorch 神经网络

| 模型 | 代码名 | 结构 |
| --- | --- | --- |
| 多层感知机 | `torch_mlp` | Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear |

神经网络使用 `CrossEntropyLoss(class_weight)` 处理类别不均衡，优化器为 `AdamW`。代码已经支持 macOS MPS、Linux CUDA 和 CPU。

## 当前 quick 结果摘要

quick 命令：

```bash
python scripts/run_baseline_pipeline.py --quick --torch-device auto
```

设备侧结果：

| 模型 | Accuracy | Macro-F1 | QWK | 高风险召回 |
| --- | ---: | ---: | ---: | ---: |
| `hist_gradient_boosting` | 0.9947 | 0.8818 | 0.9520 | 0.8305 |
| `random_forest` | 0.9833 | 0.8032 | 0.8283 | 0.8644 |
| `extra_trees` | 0.9444 | 0.6683 | 0.6783 | 0.8644 |
| `torch_mlp` | 0.9548 | 0.5751 | 0.7171 | 0.8475 |
| `logistic_regression` | 0.9699 | 0.5296 | 0.7243 | 0.4407 |

站点侧结果：

| 模型 | Accuracy | Macro-F1 | QWK | 高风险召回 |
| --- | ---: | ---: | ---: | ---: |
| `hist_gradient_boosting` | 0.9580 | 0.8010 | 0.9346 | 0.7000 |
| `random_forest` | 0.9350 | 0.7717 | 0.9059 | 0.7600 |
| `extra_trees` | 0.8700 | 0.5860 | 0.8122 | 0.7800 |
| `logistic_regression` | 0.8810 | 0.5691 | 0.7947 | 0.5800 |
| `torch_mlp` | 0.8400 | 0.4931 | 0.7052 | 0.7000 |

当前 quick 结果中，设备侧和站点侧表现最好的都是 `hist_gradient_boosting`。这符合预期：当前数据是结构化表格特征，且标签是由规则评分生成的冷启动伪标签，树模型通常会更占优。神经网络当前主要用于验证 PyTorch 训练链路和后续 GPU 实验框架。

## 经济调电模型

经济调电模块当前使用线性规划建模，目标是：

```text
最大化 供电收益 - 风险惩罚
```

主要输入：

- 站点状态等级。
- 站点状态风险分。
- 当前负荷代理值。
- 容量代理值。
- 可调负荷代理值。
- 运行覆盖率和开关动作率等收益侧代理特征。

主要约束：

- 总供电预算约束。
- 单站点可调容量约束。
- 高风险站点降额约束。
- 异常站点不鼓励新增承载压力。

输出文件：

```text
outputs/baseline_models/economic_dispatch/dispatch_plan.csv
```

## 输出目录

设备模型输出：

```text
outputs/baseline_models/device_prediction/
```

站点模型输出：

```text
outputs/baseline_models/station_assessment/
```

经济调电输出：

```text
outputs/baseline_models/economic_dispatch/
```

早期环境验证 demo 输出：

```text
outputs/demo/
```

## 后续开发建议

1. 接入正式业务标签：缺陷、跳闸、检修、人工复核等级、南网正式状态等级。
2. 按 `suggested_model_group` 分组训练设备模型。
3. 增加 LightGBM、XGBoost、CatBoost 作为正式表格建模主力候选。
4. 增加 LSTM、TCN、Transformer Encoder 等时序神经网络。
5. 补充主变容量、线路容量、区域负荷、峰谷电价、需求响应收益等经济调电真实输入。
6. 增加 SHAP 或特征重要性输出，解释每个设备和站点的主要触发因素。

## 重要说明

当前模型效果只代表代码链路和冷启动伪标签上的表现，不代表最终业务准确率。真正的模型结论需要等正式标签和业务校验数据接入后重新评估。
