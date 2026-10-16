# API 服务快速运行

`api_service/` 是与业务系统对接的独立版本，包含 FastAPI 模型服务、异步数据清洗 Worker、已训练模型、Docker 配置和可执行演示。

## Docker 一键演示

在项目根目录执行：

```bash
bash api_service/scripts/docker_demo.sh
```

脚本会依次完成 CPU 镜像构建、演示数据发布、API/Worker 启动和 14 个 HTTP 联调请求。成功后打开 `http://127.0.0.1:8000/docs` 即可使用 Swagger。

家族缺陷预警接口：

```bash
curl "http://127.0.0.1:8000/api/defect/family-alerts?assessment_date=2026-02-07&limit=20"
```

接口按“厂家 + 型号 + 批次”三元组定位家族。数据处理优先使用真实批次字段；没有独立批次时，按出厂日期归一为 `YYYYMM`，并通过响应字段 `batch_source` 明确标记推导来源。

## Docker 手动启动

首次运行：

```bash
cp api_service/.env.example api_service/.env
docker compose -f api_service/compose.yaml build model-api
docker compose -f api_service/compose.yaml run --rm --no-deps \
  model-api python -m api_service.demo.prepare_demo_data
docker compose -f api_service/compose.yaml up -d
```

验证状态与接口：

```bash
docker compose -f api_service/compose.yaml ps
curl http://127.0.0.1:8000/health/ready
docker compose -f api_service/compose.yaml exec -T \
  model-api python api_service/demo/run_smoke_test.py
```

常用管理命令：

```bash
docker compose -f api_service/compose.yaml logs -f --tail=100
docker compose -f api_service/compose.yaml restart
docker compose -f api_service/compose.yaml down
```

## Linux 3090 GPU 启动

宿主机安装 NVIDIA 驱动与 NVIDIA Container Toolkit 后，在 `api_service/.env` 中设置需要使用的物理 GPU：

```text
NVIDIA_GPU_ID=0
```

然后执行：

```bash
docker compose \
  -f api_service/compose.yaml \
  -f api_service/compose.gpu.yaml \
  up --build -d
```

`NVIDIA_GPU_ID` 选择宿主机物理显卡。只暴露一张卡时，它在容器内会映射为 `cuda:0`，因此容器内 `GPU_ID` 固定为 `0`。Mac 不能运行 NVIDIA CUDA 推理，但可构建 `linux/amd64` GPU 镜像。

## XML/DT 投递

原始文件按批次放入：

```text
api_service/runtime/inbox/<batch_id>/
├── xml/
│   └── *.xml
├── dt/
│   └── *.DT
└── READY.json
```

先写入 XML/DT，文件全部到齐后再以原子重命名方式放入 `READY.json`。Worker 默认每 2 秒扫描，也可调用 `POST /api/data-jobs` 手动建立任务。

## 本机 Python 调试

```bash
conda activate griddx
python -m pip install -r api_service/requirements.txt
python -m api_service.demo.prepare_demo_data
python -m uvicorn api_service.app.main:app --host 127.0.0.1 --port 8000
```

另开终端执行：

```bash
conda activate griddx
python api_service/demo/run_smoke_test.py
python api_service/demo/run_cleaning_demo.py
```

完整 Docker 用法、离线交付和故障排查见 [`docs/docker_deployment_guide.md`](../docs/docker_deployment_guide.md)。
