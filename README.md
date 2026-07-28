# griddx：电网运行状态评估与经济调电基础建模工具

`griddx` 是一个面向电网运行数据的基础建模项目，用于支撑设备状态预测、站点状态评估、风险预警和经济调电优化实验。当前版本已接入 enriched 设备日表和站点日表，支持历史缺陷、跳闸、检修、家族设备、设备台账和站点环境风险特征。

> enriched 表中的历史事件和台账字段是按设备或站点生成的静态画像，同一对象在 90 天内保持不变，且部分字段来自随机编码。当前默认采用“enriched 弱监督风险评分 + 分组切分 + 多模型对比”，用于状态评估规则验证，不等同于未来真实故障预测。

## 当前三层建模方案

项目目前形成设备、站点、经济调度三层模型：

| 层级 | 技术名称 | 解决的问题 | 主要输出 |
| --- | --- | --- | --- |
| 设备 | DAMD-Net 设备自适应多判别器网络 | 表达每台设备在运行、历史、检修和家族画像上的差异 | 四级状态、设备风险概率、四判别器权重 |
| 站点 | HSF-Net 站点分层多视角融合网络 | 融合站点运行、历史检修和基础设施风险 | 四级状态、连续风险分、三视图权重 |
| 经济 | RA-MOD 风险约束多目标调度模型 | 在调电成本、风险成本和供电缺额损失之间优化 | 站点调电量、缺额量、成本分解和优先级 |

```text
单设备日级数据
-> DAMD-Net 设备差异化状态识别
-> HSF-Net 站点多视角风险评估
-> RA-MOD 风险约束经济调度
```

详细技术文档：

- [DAMD-Net 设备自适应多判别器网络](docs/device_personalized_multi_discriminator.md)
- [HSF-Net 站点分层多视角状态融合网络](docs/station_hierarchical_fusion_model.md)
- [RA-MOD 风险约束多目标经济调度模型](docs/risk_aware_economic_dispatch_model.md)
- [设备模型 real 数据验证报告](docs/device_model_real_data_validation_report.md)

## 目录结构

```text
griddx/
├── README.md                         # 项目首页文档，说明安装、运行、模型和结果
├── .gitignore                        # Git 忽略规则，排除 data/、outputs/、缓存等本地文件
├── configs/                          # 环境配置文件
│   ├── environment-macos.yml         # macOS / Apple Silicon 环境
│   └── environment-linux-cu128.yml   # Linux / CUDA 环境，适合 3090 服务器
├── data/                             # 本地数据目录，不上传 Git
│   ├── model_base_device_day.csv
│   ├── model_base_device_day_line_or_load_like.csv
│   ├── model_base_device_day_line_or_load_like_enriched.csv
│   ├── model_base_device_day_line_or_load_like_real.csv
│   ├── model_base_station_day.csv
│   └── model_base_station_day_extract_enriched.csv
├── docs/                             # 更详细的建模说明文档
│   ├── baseline_modeling_report.md
│   ├── enriched_modeling_report.md   # 新数据分析、改进方案和结果边界
│   ├── device_personalized_multi_discriminator.md # DAMD-Net 设备模型
│   ├── station_hierarchical_fusion_model.md        # HSF-Net 站点模型
│   ├── risk_aware_economic_dispatch_model.md       # RA-MOD 经济模型
│   └── device_model_real_data_validation_report.md # real 数据验证报告
├── examples/                         # 示例脚本和早期环境验证脚本
│   ├── grid_fault_demo.py
│   └── test.py
├── scripts/                          # 命令行入口脚本
│   ├── run_baseline_pipeline.py      # 一键运行设备模型、站点模型和经济调电
│   └── validate_external_device_data.py # 验证已保存设备模型
├── src/griddx/                       # Python 源码包
│   ├── __init__.py                   # 包初始化文件
│   ├── data.py                       # CSV 读取、日期解析、空列处理
│   ├── features.py                   # 设备和站点特征工程
│   ├── labels.py                     # 冷启动状态评分和四级伪标签生成
│   ├── model_zoo.py                  # 机器学习、MLP 和设备个性化多判别器
│   ├── device_prediction.py          # 设备级状态预测训练入口
│   ├── station_assessment.py         # 站点级状态评估训练入口
│   ├── station_fusion.py             # HSF-Net 网络、双任务训练和站点画像
│   ├── economic_dispatch.py          # 经济调电优化模型
│   ├── evaluation.py                 # 指标评估、混淆矩阵和报告输出
│   ├── external_validation.py        # real 数据验证、漂移检查和结果汇总
│   └── paths.py                      # 项目路径和数据路径配置
├── tests/                             # 数据特征、标签和切分测试
│   ├── test_enriched_pipeline.py
│   └── test_external_validation.py
└── outputs/                          # 本地实验输出目录，不上传 Git
    ├── baseline_models/              # baseline 模型、指标和调电结果
    └── demo/                         # 早期 demo 输出
```

## 环境配置

### macOS Apple Silicon

```bash
cd griddx
mamba env create -f configs/environment-macos.yml
conda activate griddx
```

### Linux + RTX 3090

```bash
cd griddx
mamba env create -f configs/environment-linux-cu128.yml
conda activate griddx
```

如果服务器 CUDA 驱动不支持 `cu128`，需要根据服务器 `nvidia-smi` 显示的驱动版本，把环境文件中的 PyTorch CUDA wheel 源切换为 `cu126` 或 `cu118`。

## 数据准备

本项目默认从项目内的 `data/` 目录读取 CSV 数据：

```text
data/
├── model_base_device_day.csv
├── model_base_device_day_line_or_load_like.csv
├── model_base_device_day_line_or_load_like_enriched.csv
├── model_base_device_day_line_or_load_like_real.csv
├── model_base_station_day.csv
└── model_base_station_day_extract_enriched.csv
```

`data/` 已写入 `.gitignore`，不会上传到 GitHub。

如果数据不放在项目目录内，也可以通过环境变量指定：

```bash
export GRIDDX_DATA_ROOT=/path/to/your/data
```

## 快速运行

快速验证全流程：

```bash
cd griddx
conda activate griddx
python scripts/run_baseline_pipeline.py --quick --torch-device auto
```

`--quick` 会限制样本量并缩短神经网络训练轮数，用于验证环境和代码链路。

默认参数会自动识别 enriched 字段，生成弱监督状态标签，并按设备 ID、站点 ID 分组切分。这样同一个对象不会同时出现在训练集和测试集中。

## Linux 服务器 GPU 使用

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

如果使用 `CUDA_VISIBLE_DEVICES` 限制可见显卡，例如只暴露物理第 2 张卡：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_baseline_pipeline.py \
  --torch-device cuda \
  --gpu-id 0
```

此时 `--gpu-id 0` 表示当前进程可见的第 0 张 GPU。

## 单独运行模块

设备状态预测：

```bash
PYTHONPATH=src python -m griddx.device_prediction \
  --csv data/model_base_device_day_line_or_load_like_enriched.csv \
  --max-rows 120000 \
  --mlp-epochs 18 \
  --label-mode enriched_weak \
  --split-strategy group \
  --torch-device auto
```

站点状态评估：

```bash
PYTHONPATH=src python -m griddx.station_assessment \
  --csv data/model_base_station_day_extract_enriched.csv \
  --mlp-epochs 18 \
  --label-mode enriched_weak \
  --split-strategy group \
  --torch-device auto
```

经济调电：

```bash
PYTHONPATH=src python -m griddx.economic_dispatch
```

## real 数据模型验证

最新 real 设备日表已用于验证保存的六个设备模型。主验证限定为训练时 group split 保存的 1,433 台测试设备，共 115,560 条可评分记录。

| 模型 | Macro-F1 | QWK | 风险精确率 | 风险召回率 |
| --- | ---: | ---: | ---: | ---: |
| `torch_multi_discriminator` | **0.3812** | **0.5801** | 0.9931 | 0.3033 |
| `torch_mlp` | 0.3664 | 0.5116 | 0.9928 | 0.1867 |
| `logistic_regression` | 0.2841 | 0.3631 | 0.1868 | **0.8741** |
| `random_forest` | 0.2663 | 0.3421 | 1.0000 | 0.1817 |
| `extra_trees` | 0.2620 | 0.3257 | 1.0000 | 0.1651 |
| `hist_gradient_boosting` | 0.2313 | 0.3023 | 1.0000 | 0.1246 |

DAMD-Net 的代理标签综合效果最好，但 real 表没有未来真实故障标签，且与 enriched 源表的设备和日期面板完全重叠。以上结果只能说明真实画像替换后的弱监督一致性，不能表述为真实故障预测准确率。完整边界、混淆矩阵、设备判别器差异及数据漂移见 [real 数据验证报告](docs/device_model_real_data_validation_report.md)。

复现验证：

```bash
python scripts/validate_external_device_data.py \
  --csv data/model_base_device_day_line_or_load_like_real.csv \
  --reference-csv data/model_base_device_day_line_or_load_like_enriched.csv \
  --device-partition test \
  --torch-device auto
```

验证脚本会输出六模型指标、四级混淆矩阵、类别未见率、数值 PSI 和每台设备的 DAMD-Net 判别器权重。结果保存在 `outputs/baseline_models/external_validation/`，不会上传 Git。

## 当前使用的模型

### 传统机器学习模型

| 模型 | 代码名 | 作用 |
| --- | --- | --- |
| 逻辑回归 One-vs-Rest | `logistic_regression` | 线性可解释基线 |
| 随机森林 | `random_forest` | 非线性表格数据基线 |
| 极端随机树 | `extra_trees` | 树模型稳定性对照 |
| 直方图梯度提升树 | `hist_gradient_boosting` | 当前主力表格模型基线 |

### PyTorch 神经网络

| 模型 | 代码名 | 结构 |
| --- | --- | --- |
| 多层感知机 | `torch_mlp` | Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear |
| DAMD-Net | `torch_multi_discriminator` | 四视角 MLP 判别器 + 设备自适应门控 + 加权融合 |
| HSF-Net | `torch_station_hierarchical` | 三视角编码器 + 门控融合 + 状态分类和风险回归双任务 |

神经网络使用 `CrossEntropyLoss(class_weight)` 处理类别不均衡，优化器为 `AdamW`。代码支持 macOS MPS、Linux CUDA 和 CPU。

多判别器分别处理运行波动、历史事件、检修恢复和设备画像。门控网络为每台设备生成独立的四判别器权重，因此不同设备拥有不同的判别路径。详细结构和推荐项目表述见 [设备个性化多判别器模型](docs/device_personalized_multi_discriminator.md)。

## enriched quick 结果摘要

quick 命令：

```bash
python scripts/run_baseline_pipeline.py \
  --quick \
  --label-mode enriched_weak \
  --split-strategy group \
  --torch-device auto
```

设备侧结果：

| 模型 | Accuracy | Macro-F1 | QWK | 高风险召回 |
| --- | ---: | ---: | ---: | ---: |
| `torch_mlp` | 0.9460 | 0.8632 | 0.9215 | 0.8710 |
| `torch_multi_discriminator` | 0.9428 | 0.8142 | 0.9151 | 0.8817 |
| `hist_gradient_boosting` | 0.9522 | 0.7982 | 0.9157 | 0.7661 |
| `random_forest` | 0.9388 | 0.7951 | 0.9096 | 0.8038 |
| `extra_trees` | 0.9363 | 0.7767 | 0.9000 | 0.7957 |
| `logistic_regression` | 0.9251 | 0.6381 | 0.8772 | 0.5430 |

站点侧结果：

| 模型 | Accuracy | Macro-F1 | QWK | 高风险召回 |
| --- | ---: | ---: | ---: | ---: |
| `torch_mlp` | 0.8662 | 0.5486 | 0.8039 | 0.7429 |
| `torch_station_hierarchical` | 0.8422 | 0.4773 | 0.6818 | 0.5143 |
| `hist_gradient_boosting` | 0.8547 | 0.4959 | 0.7379 | 0.4857 |
| `extra_trees` | 0.8614 | 0.4898 | 0.7751 | 0.6571 |
| `random_forest` | 0.8547 | 0.4477 | 0.7603 | 0.4286 |
| `logistic_regression` | 0.8056 | 0.3260 | 0.4642 | 0.0000 |

设备测试集包含 1,433 个未参与训练的设备，站点测试集包含 20 个未参与训练的站点。普通 MLP 的 Macro-F1 最高，多判别器的高风险召回更高，并为全部 5,729 台设备输出独立专家画像。结果位于 `outputs/baseline_models/device_prediction/device_expert_profiles.csv`。

这些指标衡量的是模型复现弱监督风险规则的能力，不是未来缺陷或跳闸的业务命中率。

## RA-MOD 经济调度模型

RA-MOD 使用 HiGHS 线性规划求解器，联合最小化：

```text
调电执行成本 + 风险暴露成本 + 供电缺额损失
```

主要输入：

- 站点状态风险、历史事件、未处理检修和环境风险。
- 当前负荷、预测负荷和风险调整容量。
- 电压等级、负荷水平和主变数量形成的站点重要度。
- 调电单位成本、风险单位成本和供电缺额单位损失。

主要约束：

- 总增量调电预算约束。
- 单站点风险调整容量约束。
- 调电量与未满足需求的平衡约束。
- 高风险和检修未闭环站点降额约束。

输出文件：

```text
outputs/baseline_models/economic_dispatch/dispatch_plan.csv
outputs/baseline_models/economic_dispatch/dispatch_summary.json
```

当前 77 个站点实验中，增量预算为 258.238 MW，总增量需求为 673.135 MW，求解器状态为 `optimal`。这些数值仍基于代理经济参数，等待经济调研数据替换。

## 后续开发建议

1. 接入按日期变化的正式业务标签，例如用 D 日及以前信息预测 D+1 至 D+7 的缺陷或跳闸。
2. 按 `suggested_model_group` 分组训练设备模型。
3. 为每台设备构建滚动 7 天和 30 天设备原型，增强门控网络的动态个性化能力。
4. 在真实时序标签接入后增加轻量 TCN，暂不优先堆叠大型 Transformer。
5. 补充主变容量、线路容量、区域负荷、峰谷电价、需求响应收益等经济调电真实输入。
6. 增加置信度校准和主要触发因素输出，形成可复核的单设备风险清单。

## 重要说明

当前模型效果只代表 enriched 弱监督规则上的泛化表现，不代表最终业务准确率。随机生成的厂家、设备年龄和历史事件画像可以验证代码，但不应进入正式结论。真正的模型结论需要使用带事件日期、实体映射和业务复核的标签重新训练。
