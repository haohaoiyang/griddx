# 基础建模报告

## 当前状态

项目已经完成设备状态评估、站点状态评估、多模型对比、PyTorch GPU 训练和经济调电原型。默认数据已切换为 enriched 设备日表和站点日表。

完整的数据检查、弱监督标签公式、分组评估方式和最新实验结果见 [enriched_modeling_report.md](enriched_modeling_report.md)。

## 建模链路

```text
Enriched CSV
-> 数据清洗
-> 量测、时序、历史事件和检修特征
-> Enriched 弱监督四级状态标签
-> 按设备或站点分组切分
-> DAMD-Net 设备个性化多判别器
-> HSF-Net 站点分层融合与连续风险评分
-> RA-MOD 风险约束多目标经济调度
```

## 模型清单

| 模型 | 类型 | 作用 |
| --- | --- | --- |
| `logistic_regression` | 线性模型 | 可解释基线 |
| `random_forest` | Bagging 树模型 | 非线性表格基线 |
| `extra_trees` | 随机树模型 | 树模型稳定性对照 |
| `hist_gradient_boosting` | Boosting 树模型 | 表格模型基线 |
| `torch_mlp` | PyTorch 神经网络 | 通用 CPU、MPS、CUDA 深度学习基线 |
| `torch_multi_discriminator` | DAMD-Net | 为每台设备生成个性化判别器权重和风险路径 |
| `torch_station_hierarchical` | HSF-Net | 联合输出站点状态、风险分和三视图权重 |
| `RA-MOD` | 线性规划 | 联合优化调电成本、风险成本与供电缺额损失 |

## PyTorch 设备选择

自动选择顺序为 CUDA、Apple MPS、CPU。在 Linux 3090 服务器上指定第 0 张可见 GPU：

```bash
python scripts/run_baseline_pipeline.py \
  --torch-device cuda \
  --gpu-id 0
```

## 当前限制

1. enriched 历史和台账字段按对象静态生成，部分字段为随机编码。
2. 当前标签是弱监督状态标签，不是未来真实缺陷或跳闸标签。
3. 站点只有 77 个独立样本对象，跨站点评估波动较大。
4. 调电模块使用收益、容量和风险代理值，尚未接入潮流约束及真实经济参数。

## 正确解释

当前实验用于验证特征、模型、GPU 和优化链路。模型指标表示对弱监督状态规则的复现和跨对象泛化能力，不代表生产环境中的故障预测准确率或可直接执行的调电策略。
