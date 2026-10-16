# Docker 部署与使用说明

## 1. Docker 在本项目中做什么

Docker 把 Python 3.11、FastAPI、PyTorch、科学计算库、项目代码和已训练模型封装成同一个镜像。部署时从该镜像启动两个独立进程：

| 容器 | 作用 | 是否对外开放端口 |
| --- | --- | --- |
| `model-api` | 运行 FastAPI，读取已处理特征并调用已训练模型推理 | 是，默认 `8000` |
| `data-worker` | 异步读取 XML/DT/DB、清洗、聚合并发布特征版本 | 否 |

两个容器共享宿主机 `api_service/runtime/` 目录。删除或重建容器不会删除该目录中的原始批次、任务状态和特征版本。

## 2. 已提供的镜像配置

| 用途 | 镜像标签 | 架构 | PyTorch | 本次实际构建 |
| --- | --- | --- | --- | --- |
| Mac/普通 Linux 联调 | `power-grid-model-api:1.0.0-cpu` | 跟随构建机 | `2.11.0+cpu` | 成功，Apple Silicon 镜像约 363 MB |
| RTX 3090 生产部署 | `power-grid-model-api:1.0.0-cu128` | `linux/amd64` | `2.11.0+cu128` | 成功，镜像约 4.19 GB |

GPU 镜像已实际验证 `torch.version.cuda == "12.8"`，且 DAMD-Net 与 HSF-Net 模型产物可正常加载。当前 Mac 没有 NVIDIA GPU，因此 `torch.cuda.is_available()` 的最终验收必须在 3090 Linux 服务器执行。

## 3. Mac 或 CPU 环境快速启动

进入项目根目录：

```bash
cp api_service/.env.example api_service/.env
docker compose -f api_service/compose.yaml build model-api
```

首次联调时发布演示特征：

```bash
docker compose -f api_service/compose.yaml run --rm --no-deps \
  model-api python -m api_service.demo.prepare_demo_data
```

启动 API 与 Worker：

```bash
docker compose -f api_service/compose.yaml up -d
```

检查容器：

```bash
docker compose -f api_service/compose.yaml ps
```

正常情况下，`model-api` 显示 `healthy`，`data-worker` 显示 `Up`。打开 `http://127.0.0.1:8000/docs` 可直接调试全部 API。

纯 Docker 联调，不依赖宿主机 Conda：

```bash
docker compose -f api_service/compose.yaml exec -T \
  model-api python api_service/demo/run_smoke_test.py
```

也可直接执行已提供的一键脚本：

```bash
bash api_service/scripts/docker_demo.sh
```

## 4. Linux RTX 3090 GPU 启动

### 4.1 宿主机前置条件

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Ubuntu 22.04/24.04 LTS `x86_64` |
| NVIDIA GPU | RTX 3090 24 GB 或同级 |
| NVIDIA 驱动 | CUDA 12.8 兼容，最低 `570.26`，建议 R570 且不低于 `570.124.06` |
| 容器运行时 | Docker Engine + Compose + NVIDIA Container Toolkit |

先验证宿主机 GPU：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

第二条命令必须能在容器中显示 3090，否则先修复驱动或 NVIDIA Container Toolkit。

### 4.2 选择 GPU 并启动

在 `api_service/.env` 中设置物理 GPU 编号：

```text
NVIDIA_GPU_ID=0
```

启动 GPU 版：

```bash
docker compose \
  -f api_service/compose.yaml \
  -f api_service/compose.gpu.yaml \
  up --build -d
```

如需宿主机第 2 张物理卡，将 `NVIDIA_GPU_ID` 设为 `1`。只向容器暴露一张卡时，它在容器内总是 `cuda:0`。

验证 PyTorch 与 GPU：

```bash
docker compose \
  -f api_service/compose.yaml \
  -f api_service/compose.gpu.yaml \
  exec -T model-api python api_service/scripts/verify_environment.py
```

预期关键输出为：

```text
torch=2.11.0+cu128
cuda_available=True
cuda_runtime=12.8
gpu_count=1
gpu_0=NVIDIA GeForce RTX 3090
```

## 5. 数据与持久化

`api_service/runtime/` 是宿主机持久化目录，不会打入镜像，也不会上传 Git。生产 Linux 中容器用户 UID 为 `10001`，首次部署前建议执行：

```bash
mkdir -p api_service/runtime
sudo chown -R 10001:10001 api_service/runtime
```

原始批次目录：

```text
api_service/runtime/inbox/<batch_id>/
├── xml/*.xml
├── dt/*.DT
└── READY.json
```

`READY.json` 是文件全部到齐后的触发标记。Worker 异步完成清洗后，将新特征写入 `runtime/published/versions/`，再原子切换 `runtime/published/active.json`。API 仅读取活跃版本，不在请求时现场清洗。

## 6. 常用运维命令

| 操作 | 命令 |
| --- | --- |
| 查看状态 | `docker compose -f api_service/compose.yaml ps` |
| 追踪日志 | `docker compose -f api_service/compose.yaml logs -f --tail=100` |
| 只看 API 日志 | `docker compose -f api_service/compose.yaml logs -f model-api` |
| 只看 Worker 日志 | `docker compose -f api_service/compose.yaml logs -f data-worker` |
| 重启 | `docker compose -f api_service/compose.yaml restart` |
| 停止并删除容器 | `docker compose -f api_service/compose.yaml down` |
| 重新构建 | `docker compose -f api_service/compose.yaml build --no-cache model-api` |

`down` 不会删除宿主机 `api_service/runtime/`。不要对该目录使用未经确认的删除命令。

## 7. 离线服务器交付

如果目标服务器无法访问镜像源，可在可联网的 `linux/amd64` 机器构建后导出：

```bash
docker save power-grid-model-api:1.0.0-cu128 \
  | gzip > power-grid-model-api-1.0.0-cu128.tar.gz
```

将压缩包、项目代码和配置文件交付到服务器后导入：

```bash
gunzip -c power-grid-model-api-1.0.0-cu128.tar.gz | docker load
docker compose \
  -f api_service/compose.yaml \
  -f api_service/compose.gpu.yaml \
  up -d --no-build
```

镜像压缩包体积较大，不应提交到 Git。代码仓库保留 Dockerfile、Compose 和锁定依赖，用于重建同版本镜像。

## 8. 常见问题

### `docker: command not found`

Mac Docker Desktop 已安装但 CLI 未进入 `PATH`时，可建立符号链接：

```bash
ln -s /Applications/Docker.app/Contents/Resources/bin/docker \
  /opt/homebrew/bin/docker
```

本机已完成该配置，`docker compose version` 可正常执行。

### `health/ready` 返回未就绪

这表示已训练模型或活跃特征版本不完整。联调环境可重新运行演示数据发布命令；生产环境应查看 Worker 日志和最近数据任务状态。

### 端口 `8000` 被占用

在 `api_service/.env` 中设置其他宿主机端口：

```text
API_PORT=8001
```

容器内仍使用 `8000`，访问地址改为 `http://127.0.0.1:8001/docs`。

### Worker 无法写入文件

检查 `api_service/runtime/` 的 UID/GID 权限。生产 Linux 默认容器用户为 `10001`。
