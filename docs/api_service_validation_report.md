# API 服务与 Docker 验证报告

## 1. 验证结论

本次已完成三条链路的实际运行：

1. `XML + DT -> 独立 Worker -> 设备日/站点日数据 -> 版本发布`。
2. `FastAPI -> 已训练设备/站点模型 -> 画像/风险汇总 -> 经济调度`。
3. `Docker Compose -> model-api + data-worker -> 共享持久化目录`。

演示 XML/DT 批次产出 10 条设备日记录和 5 条站点日记录，Docker Worker 新任务最终为 `succeeded/published`。全部 14 个 HTTP 联调请求在宿主机与容器内均调用成功，其中家族性预警接口按“厂家 + 型号 + 批次”三元组返回家族定位信息。

CPU 镜像已在本机实际启动；Linux `x86_64` / CUDA 12.8 GPU 镜像已实际跨架构构建，且可加载两份模型产物。由于本机没有 NVIDIA GPU，3090 CUDA 实机推理仍需在目标 Linux 服务器验收。

## 2. 验证环境

| 项目 | 实际环境 |
| --- | --- |
| 宿主机 | macOS / Apple Silicon |
| Docker Client/Server | 29.7.2 / 29.7.2 |
| Docker Compose | 5.3.1 |
| 本机 Conda Python / PyTorch | 3.11.15 / 2.12.0 |
| CPU 容器 | Linux `arm64`，Python 3.11.15，PyTorch `2.11.0+cpu` |
| GPU 镜像 | Linux `amd64`，Python 3.11，PyTorch `2.11.0+cu128`，CUDA runtime 12.8 |
| FastAPI | 0.141.1 |
| Uvicorn | 0.52.1 |
| Pydantic | 2.13.4 |
| SQLAlchemy | 2.0.51 |
| NumPy / Pandas | 2.4.6 / 3.0.3 |
| SciPy / scikit-learn | 1.17.1 / 1.9.0 |

## 3. Docker 镜像与进程验证

| 镜像 | 架构 | 实际大小 | 镜像 ID | 结果 |
| --- | --- | ---: | --- | --- |
| `power-grid-model-api:1.0.0-cpu` | `linux/arm64` | 362,774,600 B（约 363 MB） | `515418ca6224` | 构建、启动、推理成功 |
| `power-grid-model-api:1.0.0-cu128` | `linux/amd64` | 4,191,402,390 B（约 4.19 GB） | `f19929a8af4c` | 构建、PyTorch 导入、模型加载成功 |

CPU Compose 运行状态：

```text
model-api   Up (healthy)   0.0.0.0:8000->8000/tcp
data-worker Up
```

GPU 镜像内实测输出：

```text
machine=x86_64
torch=2.11.0+cu128
cuda_runtime=12.8
cuda_available=False
```

`cuda_available=False` 是因为此检查运行于无 NVIDIA GPU 的 Mac，不代表镜像缺少 CUDA 运行库。GPU 镜像已以 CPU 回退方式实际加载 DAMD-Net 和 HSF-Net。

## 4. 模型产物验证

| 模型 | 文件 | SHA-256 | 加载结果 |
| --- | --- | --- | --- |
| DAMD-Net | `api_service/models/device_damd_net.pt` | `9e1e625b5935fc56854bb173410d616f7383f5460b8cef964d13532efc35fb49` | 成功 |
| HSF-Net | `api_service/models/station_hsf_net.pt` | `56be09ba8581aeb5030aa005fef86c144528eb00c90c4dad0a37f5f79d19501d` | 成功 |
| RA-MOD | Python 线性优化模块 | `rule-lp-1` | 成功 |

服务启动时已完成两个 PyTorch 模型、scikit-learn 预处理器和经济求解器的加载。

## 5. XML/DT 异步清洗验证

### 5.1 输入

| 输入 | 数量 | 内容 |
| --- | ---: | --- |
| CIM/RDF XML | 1 | 1 个站点、2 台设备、8 个测点及关系 |
| DT | 5 | 2026-02-01 至 2026-02-05，电流、电压、有功和无功功率 |
| 模式 | 1 | `operation_only` |

### 5.2 Docker Worker 处理结果

```json
{
  "status": "succeeded",
  "current_stage": "published",
  "result": {
    "feature_version": "20260205-demo-raw-20260201-091347-8d92a9",
    "device_rows": 10,
    "station_rows": 5,
    "business_date": "2026-02-05",
    "run_mode": "operation_only"
  }
}
```

已验证 XML 对象/关系解析、DT 测点关联、设备日聚合、设备分组、站点日聚合、站点清洗、业务日期校验和原子发布。

## 6. DB 读取层验证

使用本地 SQLite 构建 6 张模拟源表，已验证 SQLAlchemy 可按 `DB_TABLE_MAP` 读取并导出为六个现有清洗脚本所需的逻辑数据集：

```text
DM_主网设备
DM_缺陷数据
DM_检修单主表_JXSQD
DM_检修单从表_TD_JXSBLB
DM_变电跳闸记录
DM_输电跳闸记录
```

生产 DB 类型、表/视图名和驱动尚未确定，本次未对真实 DB 执行 `full` 模式。该项必须在 DB 信息确定后使用只读账号验收。

## 7. API 回归验证

| 方法 | 路由 | 实际状态 |
| --- | --- | --- |
| GET | `/health/live` | 200 |
| GET | `/health/ready` | 200 |
| GET | `/api/models` | 200 |
| POST | `/api/device/assessment` | 200 |
| GET | `/api/device/query` | 200 |
| GET | `/api/device/profile` | 200 |
| POST | `/api/station/assessment` | 200 |
| GET | `/api/station/query` | 200 |
| GET | `/api/station/profile` | 200 |
| GET | `/api/risk/overview` | 200 |
| GET | `/api/defect/family-alerts` | 200 |
| POST | `/api/economic-dispatch` | 200 |
| POST | `/api/data-jobs` | 202 |
| GET | `/api/data-jobs/{job_id}` | 200 |

家族性预警已验证厂家、型号、批次三元组定位、家族成员统计、缺陷设备占比、缺陷事件强度、Z-score 分级和高风险设备明细。经济调度已验证内置参数与请求覆盖参数两种情况，线性规划均返回 `solver_status=optimal`。

## 8. 自动测试

```text
test_all_six_logical_tables_are_exported ... ok
test_publish_and_read_active_version ... ok
test_job_is_idempotent_and_claimed_atomically ... ok
test_batch_derivation ... ok
test_family_tuple_and_alert ... ok

Ran 5 tests
OK
```

## 9. 可重复的 Docker 验证

```bash
bash api_service/scripts/docker_demo.sh
```

或手动执行：

```bash
docker compose -f api_service/compose.yaml build model-api
docker compose -f api_service/compose.yaml run --rm --no-deps \
  model-api python -m api_service.demo.prepare_demo_data
docker compose -f api_service/compose.yaml up -d
docker compose -f api_service/compose.yaml exec -T \
  model-api python api_service/demo/run_smoke_test.py
```

## 10. 尚需在目标环境验收的部分

| 项目 | 原因 |
| --- | --- |
| RTX 3090 CUDA 实机推理 | GPU 镜像已构建，但本机无 NVIDIA GPU；需验收驱动、Container Toolkit 和 `cuda_available=True` |
| 真实 DB `full` 模式 | DB 类型、连接方式和六张逻辑表映射尚待确认 |
| 大数据量性能 | 需根据目标服务器、真实日批次和 SLA 压测 |

上述待验收项不影响当前 CPU Docker 全链路联调，但不能在对应环境未确定时宣称已完成生产性能或 3090 实机验收。
