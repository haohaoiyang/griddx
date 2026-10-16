from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

COLORS = ["#197C80", "#C14F4D", "#D69B32", "#66718C"]
NAMES = {"device": "DAMD-Net", "station": "HSF-Net"}
LABELS = {
    "full": "Full model", "without_residual": "w/o global residual",
    "without_class_weights": "w/o class weighting", "without_infrastructure": "w/o infrastructure",
    "without_operation": "w/o operation", "without_history_maintenance": "w/o history / maintenance",
}
CN = {
    "full": "完整模型", "without_residual": "移除全局残差",
    "without_class_weights": "移除类别权重", "without_infrastructure": "移除基础设施分支",
    "without_operation": "移除运行分支", "without_history_maintenance": "移除历史检修分支",
}
STATES = ["Normal", "Watch", "Abnormal", "High risk"]


def save(fig, args, name):
    for extension in ("png", "svg"):
        fig.savefig(args.figure_dir / f"{name}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summary(frame, kind):
    metrics = ["accuracy", "macro_f1", "qwk", "risk_precision", "risk_recall", "risk_mae", "risk_rmse"]
    result = frame[frame.kind == kind].groupby("variant", sort=False)[metrics].agg(["mean", "std"])
    result.columns = [f"{a}_{b}" for a, b in result.columns]
    return result


def display(row, metric, digits=4):
    return f"{row[f'{metric}_mean']:.{digits}f} ± {row[f'{metric}_std']:.{digits}f}"


def predictions(args, kind, seed=42):
    return np.load(args.output_dir / kind / "full" / f"seed_{seed}" / "test_predictions.npz")


def ranking_experiment(args, manifest):
    rows = []
    curves = {}
    for kind in NAMES:
        curves[kind] = []
        for seed in manifest["seeds"]:
            data = predictions(args, kind, seed)
            reference = data["y_true"] >= 2
            scores = data["probabilities"][:, 2:].sum(1)
            order = np.argsort(-scores, kind="stable")
            cumulative = np.cumsum(reference[order])
            n = len(reference)
            for fraction in (.05, .10, .20):
                count = int(np.ceil(fraction * n))
                positives = int(cumulative[count-1])
                precision = positives / count
                rows.append({"kind": kind, "seed": seed, "fraction": fraction, "selected_records": count,
                             "risk_records": int(reference.sum()), "precision": precision,
                             "recall": positives / int(reference.sum()), "lift": precision / reference.mean()})
            fractions = np.linspace(0, 1, 201)
            counts = np.ceil(fractions * n).astype(int)
            gain = np.array([0 if count == 0 else cumulative[count-1] / reference.sum() for count in counts])
            curves[kind].append(gain)
    ranking = pd.DataFrame(rows)
    ranking.to_csv(args.output_dir / "risk_ranking_metrics.csv", index=False)
    return ranking, curves


def plot_results(args, frame, manifest, ranking, curves):
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "figure.dpi": 110, "savefig.dpi": 240, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                         "svg.fonttype": "none"})
    stats = {kind: summary(frame, kind) for kind in NAMES}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), layout="constrained")
    for ax, kind in zip(axes, NAMES):
        hs = [pd.read_csv(args.output_dir / kind / "full" / f"seed_{seed}" / "history.csv")
              for seed in manifest["seeds"]]
        values = np.array([h.train_ce for h in hs])
        mean, sd = values.mean(0), values.std(0, ddof=1)
        ax.plot(hs[0].epoch, mean, color=COLORS[0], label="Training CE")
        ax.fill_between(hs[0].epoch, np.maximum(mean-sd, 0), mean+sd, color=COLORS[0], alpha=.18, label="Seed SD")
        ax.set(xlabel="Epoch", ylabel="Training weighted cross-entropy", title=NAMES[kind], ylim=(0, None))
        ax.legend()
    save(fig, args, "01_training_convergence")

    selections = [
        ("device", ["full", "without_residual"], ["macro_f1", "qwk", "risk_recall"], "02_device_residual"),
        ("device", ["full", "without_class_weights"], ["risk_recall"], "03_device_risk_detection"),
        ("station", ["full", "without_infrastructure", "without_residual", "without_class_weights"],
         ["macro_f1", "qwk", "risk_recall"], "04_station_key_modules"),
    ]
    for kind, keys, metrics, name in selections:
        fig, axes = plt.subplots(1, len(metrics), figsize=(max(5, 4.1*len(metrics)), 3.9), layout="constrained", squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            data = stats[kind].loc[keys]
            ax.bar(np.arange(len(keys)), data[f"{metric}_mean"], yerr=data[f"{metric}_std"],
                   capsize=4, color=[COLORS[0]]+["#A8B7BC"]*(len(keys)-1))
            ax.set_xticks(range(len(keys)), [LABELS[k] for k in keys], rotation=22, ha="right")
            lower = min(0, float((data[f"{metric}_mean"]-data[f"{metric}_std"]).min()))
            upper = max(1, float((data[f"{metric}_mean"]+data[f"{metric}_std"]).max()))
            ax.set(ylabel=metric.replace("_", " "), title=NAMES[kind], ylim=(lower-.025, upper+.025))
        save(fig, args, name)

    keys = ["full", "without_operation", "without_history_maintenance"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    for ax, metric in zip(axes, ("risk_mae", "risk_rmse")):
        data = stats["station"].loc[keys]
        ax.bar(range(3), data[f"{metric}_mean"], yerr=data[f"{metric}_std"], capsize=4,
               color=[COLORS[0], "#A8B7BC", "#A8B7BC"])
        ax.set_xticks(range(3), [LABELS[k] for k in keys], rotation=20, ha="right")
        ax.set(ylabel=f"{metric.replace('_', ' ')} (0-100 score)", title="HSF-Net: view contributions", ylim=(0, None))
    save(fig, args, "05_station_risk_views")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, kind in zip(axes, NAMES):
        data = predictions(args, kind)
        cm = confusion_matrix(data["y_true"], data["probabilities"].argmax(1), labels=range(4))
        normalized = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i,j]}\n{normalized[i,j]:.1%}", ha="center", va="center", fontsize=9,
                        color="white" if normalized[i,j] > .5 else "black")
        ax.set_xticks(range(4), STATES, rotation=25, ha="right")
        ax.set_yticks(range(4), STATES)
        ax.set(xlabel="Predicted", ylabel="Weak reference label", title=f"{NAMES[kind]} (seed 42)")
        ax.grid(False)
    save(fig, args, "06_confusion_matrices")

    data = predictions(args, "station")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    density = axes[0].hexbin(data["risk_target"], data["risk_prediction"], gridsize=24, mincnt=1, cmap="YlGnBu")
    fig.colorbar(density, ax=axes[0], label="Record count", shrink=.8)
    axes[0].plot([0, 100], [0, 100], linestyle="--", color=COLORS[1])
    axes[0].set(xlabel="Weak reference risk score", ylabel="Predicted risk score", xlim=(0, 100), ylim=(0, 100),
                title="HSF-Net risk regression (seed 42)")
    axes[1].hist(data["risk_prediction"]-data["risk_target"], bins=30, color=COLORS[0])
    axes[1].axvline(0, linestyle="--", color=COLORS[1])
    axes[1].set(xlabel="Predicted - reference score", ylabel="Record count", title="Risk score residuals")
    save(fig, args, "07_station_risk_fit")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
    for ax, kind, prefix in zip(axes, NAMES, ("D", "S")):
        weights = pd.read_csv(args.output_dir / kind / "full/seed_42/test_object_weights.csv", index_col=0).sort_index()
        chosen = weights.iloc[np.linspace(0, len(weights)-1, min(24, len(weights)), dtype=int)]
        hm = ax.imshow(chosen, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_yticks(range(len(chosen)), [f"{prefix}{i+1:02d}" for i in range(len(chosen))], fontsize=8)
        ax.set_xticks(range(len(chosen.columns)), [c.replace("_", " ") for c in chosen.columns], rotation=25, ha="right")
        ax.set(title=NAMES[kind], ylabel="Anonymized held-out object")
        ax.grid(False)
    fig.colorbar(hm, ax=axes, label="Object mean gating weight", shrink=.85)
    save(fig, args, "08_object_weights")

    histories = [pd.read_csv(args.output_dir / "station/full" / f"seed_{s}" / "history.csv") for s in manifest["seeds"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, metric, label in zip(axes, ("risk_loss", "risk_mae"), ("Smooth L1 risk loss", "Risk score MAE")):
        for partition, color in zip(("train", "validation"), COLORS):
            values = np.array([h[f"{partition}_{metric}"] for h in histories])
            ax.plot(histories[0].epoch, values.mean(0), color=color, label=partition.title())
        ax.set(xlabel="Epoch", ylabel=label, title="HSF-Net auxiliary risk task", ylim=(0, None))
        ax.legend()
    save(fig, args, "09_risk_learning")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, kind in zip(axes, NAMES):
        values = np.array(curves[kind])
        xs = np.linspace(0, 100, values.shape[1])
        mean, sd = values.mean(0)*100, values.std(0, ddof=1)*100
        ax.plot(xs, mean, color=COLORS[0], label=NAMES[kind])
        ax.fill_between(xs, np.maximum(mean-sd, 0), np.minimum(mean+sd, 100), color=COLORS[0], alpha=.18)
        ax.plot([0, 100], [0, 100], color=COLORS[1], linestyle="--", label="Random ranking expectation")
        ax.set(xlabel="Records selected by risk rank (%)", ylabel="Reference risk records covered (%)",
               xlim=(0, 100), ylim=(0, 101), title=f"{NAMES[kind]}: cumulative risk coverage")
        ax.legend(fontsize=8, loc="lower right")
    save(fig, args, "10_risk_ranking_coverage")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for ax, kind in zip(axes, NAMES):
        data = ranking[ranking.kind == kind].groupby("fraction").lift.agg(["mean", "std"])
        ax.bar(range(len(data)), data["mean"], yerr=data["std"], capsize=4, color=COLORS[0])
        ax.axhline(1, color=COLORS[1], linestyle="--", label="Random expectation")
        ax.set_xticks(range(len(data)), [f"Top {f:.0%}" for f in data.index])
        ax.set(ylabel="Risk concentration lift", title=NAMES[kind], ylim=(0, None))
        ax.legend()
    save(fig, args, "11_risk_ranking_lift")


def metric_table(data, keys, metrics):
    titles = {"macro_f1": "Macro-F1", "qwk": "QWK", "risk_recall": "风险召回率",
              "risk_mae": "风险分 MAE", "risk_rmse": "风险分 RMSE"}
    lines = ["| 配置 | " + " | ".join(titles[k] for k in metrics) + " |",
             "| --- | " + " | ".join("---:" for _ in metrics) + " |"]
    for key in keys:
        lines.append("| " + " | ".join([CN[key]]+[display(data.loc[key], m, 2 if m.startswith('risk_m') or m == 'risk_rmse' else 4)
                                                     for m in metrics]) + " |")
    return "\n".join(lines)


def make_report(args, frame, manifest, ranking):
    device, station = summary(frame, "device"), summary(frame, "station")
    task = manifest["tasks"]
    parts = ["| 层级 | 分区 | 对象数 | 日记录数 | 正常 / 关注 / 异常 / 高危 |", "| --- | --- | ---: | ---: | --- |"]
    for kind, label in (("device", "设备"), ("station", "站点")):
        for key, name in (("train", "训练"), ("validation", "验证"), ("test", "测试")):
            p = task[kind]["partitions"][key]
            parts.append(f"| {label} | {name} | {p['objects']} | {p['rows']} | " + " / ".join(map(str, p['class_counts'])) + " |")
    rank_rows = ["| 层级 | 优先选取比例 | 风险覆盖率 | 选中记录风险占比 | 相对随机排序富集倍数 |",
                 "| --- | ---: | ---: | ---: | ---: |"]
    for (kind, fraction), group in ranking.groupby(["kind", "fraction"], sort=False):
        items = ["设备日" if kind == "device" else "站点日", f"{fraction:.0%}"]
        items += [f"{group[m].mean():.4f} ± {group[m].std(ddof=1):.4f}" for m in ("recall", "precision", "lift")]
        rank_rows.append("| " + " | ".join(items) + " |")
    text = f'''# 实验结果

## 1. 实验设置

### 1.1 实验任务与数据

实验围绕设备状态分级和站点状态评估，分析 DAMD-Net 的残差路径与风险检出能力，以及 HSF-Net 的基础设施视图、残差路径、类别权重与连续风险评分。实验还通过风险排序检验模型对高风险记录的优先筛选能力。

设备数据包含 {task['device']['rows']:,} 条设备日记录、{task['device']['object_count']:,} 台设备，日期为 {task['device']['date_min']} 至 {task['device']['date_max']}；每台设备有 {task['device']['days_per_object_min']}～{task['device']['days_per_object_max']} 条记录。站点数据包含 {task['station']['rows']:,} 条站点日记录、{task['station']['object_count']} 个站点，日期为 {task['station']['date_min']} 至 {task['station']['date_max']}。

状态标签为弱监督规则生成的正常、关注、异常、高危四级标签；第 2、3 类合并为风险正类。站点连续风险评分目标为 0～100 分的弱监督风险分。指标反映对状态评估规则的拟合与跨对象泛化表现。

### 1.2 数据划分

设备按 unified_device_id、站点按 station_id 分组。GroupShuffleSplit 的外层测试比例为 25%，随机种子为 42；其余对象中按组抽取 20% 作为验证集，随机种子为 17。三个分区的对象互不重叠，各消融配置使用相同分区。

{chr(10).join(parts)}

站点验证分区没有第 3 级样本，因此验证指标对最高风险类的覆盖有限。填补、标准化和 One-Hot 编码仅在训练分区拟合，设备输入为 {task['device']['feature_count']} 维，站点输入为 {task['station']['feature_count']} 维。标签在配置间保持一致，分支消融期间不重新计算。

### 1.3 训练参数与评价指标

| 参数 | 取值 |
| --- | --- |
| 运行环境 | Python {manifest['python']}；PyTorch {manifest['torch']}；CPU |
| 优化器 | AdamW |
| 学习率 | DAMD-Net、HSF-Net：0.0008 |
| weight decay | 0.0002 |
| 批大小 | {manifest['batch_size']} |
| 最大训练轮数 | 设备 {manifest['device_epochs']}；站点 {manifest['station_epochs']} |
| 权重选择 | 验证集分类交叉熵最小轮次 |
| 训练种子 | 42、43、44 |
| 结果汇总 | 三次训练均值 ± 样本标准差 |
| 分类评价 | Macro-F1、二次加权 Kappa（QWK）和风险召回率 |
| 连续评分评价 | MAE、RMSE，均以 0～100 分为尺度 |

三个训练种子用于衡量初始化与批次顺序的波动，数据划分保持固定。测试集在训练和权重选择完成后评分。结果中的均值差异不等同于统计显著性。

## 2. 训练收敛

![状态分类训练损失](images/experiments/01_training_convergence.png)

两种模型的训练分类损失随迭代下降，说明网络能够学习多源输入与弱监督状态标签之间的映射。实线为三个种子的训练集加权交叉熵均值，阴影为样本标准差；这些曲线用于观察训练优化过程，泛化能力由测试指标评价。

![风险回归训练过程](images/experiments/09_risk_learning.png)

HSF-Net 同时优化状态分类与连续评分，风险 Smooth L1 损失在 0～1 归一化尺度计算，MAE 在 0～100 分尺度报告。训练与验证风险误差随迭代下降，表明辅助风险头能够学习连续风险排序信息；最终权重仍按验证分类交叉熵选择。

## 3. DAMD-Net 关键模块分析

### 3.1 全局残差路径

完整模型将四个专家判别器的门控融合输出与 0.20 倍全输入残差相加。消融配置删除全局残差头，其余专家和训练参数保持一致，并从头训练。

{metric_table(device, ['full', 'without_residual'], ['macro_f1', 'qwk', 'risk_recall'])}

![设备全局残差消融](images/experiments/02_device_residual.png)

保留全局残差后，Macro-F1 提高 {(device.loc['full'].macro_f1_mean-device.loc['without_residual'].macro_f1_mean)*100:.2f} 个百分点，QWK 提高 {(device.loc['full'].qwk_mean-device.loc['without_residual'].qwk_mean)*100:.2f} 个百分点。残差路径为专家融合提供全局特征补充，有助于保持状态分级的一致性。

### 3.2 类别权重与风险检出

该实验以少数风险类召回为评价目标，对照类别加权交叉熵与普通交叉熵，检验类别权重对风险检出的作用。

{metric_table(device, ['full', 'without_class_weights'], ['risk_recall'])}

![设备风险检出能力](images/experiments/03_device_risk_detection.png)

类别加权后的风险召回率为 {display(device.loc['full'], 'risk_recall')}，比无类别权重配置提高 {(device.loc['full'].risk_recall_mean-device.loc['without_class_weights'].risk_recall_mean)*100:.2f} 个百分点，支持在少数类风险筛查任务中采用类别权重。该结论限定于风险召回，不表示所有分类指标或误报表现均提升。

## 4. HSF-Net 关键模块分析

### 4.1 基础设施视图、全局残差与类别权重

分别删除基础设施编码器、全局残差头，以及交叉熵类别权重，对照完整 HSF-Net。删除视图时保留其他分支，并将剩余门控权重归一化。

{metric_table(station, ['full', 'without_infrastructure', 'without_residual', 'without_class_weights'], ['macro_f1', 'qwk', 'risk_recall'])}

![站点关键模块消融](images/experiments/04_station_key_modules.png)

完整模型的 Macro-F1 为 {display(station.loc['full'], 'macro_f1')}，风险召回为 {display(station.loc['full'], 'risk_recall')}。移除基础设施分支后，Macro-F1 下降 {(station.loc['full'].macro_f1_mean-station.loc['without_infrastructure'].macro_f1_mean)*100:.2f} 个百分点；移除全局残差后下降 {(station.loc['full'].macro_f1_mean-station.loc['without_residual'].macro_f1_mean)*100:.2f} 个百分点。去掉类别权重后，三个种子的风险召回均为 0。这些对照说明基础设施专门表示、全局路径和类别权重对当前站点分级任务具有作用。

### 4.2 多视图表示与连续风险评分

通过分别删除运行视图和历史检修视图，分析专门编码器对连续风险分回归的作用。

{metric_table(station, ['full', 'without_operation', 'without_history_maintenance'], ['risk_mae', 'risk_rmse'])}

![多视图风险评分消融](images/experiments/05_station_risk_views.png)

完整模型的 MAE 为 {display(station.loc['full'], 'risk_mae', 2)} 分，RMSE 为 {display(station.loc['full'], 'risk_rmse', 2)} 分。移除运行视图后 MAE 增至 {station.loc['without_operation'].risk_mae_mean:.2f} 分，移除历史检修视图后增至 {station.loc['without_history_maintenance'].risk_mae_mean:.2f} 分，说明运行和检修信息的专门表示有利于连续风险刻画。

分支消融检验的是专门编码器的结构贡献，全输入门控和残差仍可读取相应特征；它不等同于完全删除对应数据源。删除模块也会改变参数量，结果表示该结构调整的整体效应。

## 5. 风险优先排序实验

将测试记录按模型输出的高风险概率降序排列，分别选取前 5%、10%、20%，统计其中覆盖的风险正类比例。高风险概率为第 2、3 类预测概率之和，标签按相同口径合并。选取比例预先固定，不通过测试标签优化阈值。

{chr(10).join(rank_rows)}

![风险累计覆盖率](images/experiments/10_risk_ranking_coverage.png)

![风险记录富集倍数](images/experiments/11_risk_ranking_lift.png)

富集倍数定义为“选中记录中的风险占比 / 整体测试集风险占比”。随机排序期望为 1；超过 1 表示模型将风险记录集中到靠前位置。结果支持将模型输出用于风险复核优先级排序。

排序单位为设备日或站点日记录，同一对象不同日期分别计数。这里评价整个离线测试集合的排序效果，不代表去重后的设备巡检覆盖率，也不直接等同于单日告警效果。

## 6. 状态输出与对象差异

![四级状态混淆矩阵](images/experiments/06_confusion_matrices.png)

图中使用 seed=42 的完整模型，单元格同时显示数量和按参考类别归一化的比例，用于展示四级状态的识别情况。

![站点连续风险分](images/experiments/07_station_risk_fit.png)

HSF-Net 在四级状态之外输出连续风险分。左图比较预测分与弱监督参考分，右图展示评分残差；参考分不是实际事故概率。

![设备与站点权重分布](images/experiments/08_object_weights.png)

不同对象形成不同的专家或视图组合，体现自适应融合对对象差异的表示能力。权重取测试对象跨日期均值；设备按 ID 排序后等距选取最多 24 台，站点展示全部 20 个测试对象，图中编号为匿名序号。门控权重表示模型内部使用程度，不作为事故原因的因果证据。

## 7. 实验复现

```bash
conda activate griddx
python scripts/run_paper_experiments.py --torch-device cpu --seeds 42 43 44
python scripts/summarize_paper_experiments.py
```

输入为 `outputs/baseline_models/` 下的设备和站点训练特征表。若尚未生成，可执行 `python scripts/run_baseline_pipeline.py --quick --skip-dispatch --torch-device cpu`。

Linux GPU 环境可使用 `--torch-device cuda --gpu-id 0` 指定可见 GPU 编号。运行配置、数据指纹、逐轮指标、模型权重和所有配置的测试结果保存在 `outputs/paper_experiments/`，其中 `risk_ranking_metrics.csv` 保存风险排序实验指标。图片以 PNG 和 SVG 格式保存在 `docs/images/experiments/`。
'''
    args.report.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate experimental results and key-module figures.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/paper_experiments")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "docs/images/experiments")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/experimental_results.md")
    args = parser.parse_args()
    frame = pd.read_csv(args.output_dir / "all_runs.csv")
    manifest = json.loads((args.output_dir / "run_manifest.json").read_text())
    if len(frame) != 60 or frame.duplicated(["kind", "variant", "seed"]).any():
        raise ValueError("Expected 60 unique training runs.")
    ranking, curves = ranking_experiment(args, manifest)
    plot_results(args, frame, manifest, ranking, curves)
    make_report(args, frame, manifest, ranking)
    print(f"Report: {args.report}; 11 figure pairs; ranking metrics: {args.output_dir / 'risk_ranking_metrics.csv'}")


if __name__ == "__main__":
    main()
