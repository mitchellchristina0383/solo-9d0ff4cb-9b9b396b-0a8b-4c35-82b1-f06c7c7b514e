# GPU 分区改配调度模拟器

推理节点要腾出更多小分区跑轻量模型。本服务用 FastAPI 模拟两张 GPU，支持按规格
领用/归还分区、值班员发起改配、模拟驱动异步回报，并严格保证在跑任务不受影响、
调度只按驱动确认的实际布局恢复。

## 一、模型与规则

### 资源模型

- 两张卡：`gpu-0`、`gpu-1`；每张卡只有两种布局：
  - `large`：一个大分区（`<gpu>-p0`，size=`large`）
  - `split`：两个小分区（`<gpu>-p0`、`<gpu>-p1`，size=`small`）
- 任务按规格精确领用：`large` 任务只能占大分区，`small` 任务只能占小分区；
  任务结束按 `task_id` 归还分区。
- 初始状态：两张卡均为 `large` 布局、`active`。

### 卡状态机

| 状态 | 含义 | 是否接新任务 |
| --- | --- | --- |
| `active` | 正常调度 | 是 |
| `draining` | 已发起改配，停接新任务，等在跑任务归还分区 | 否（在跑任务保留） |
| `reconfiguring` | 分区已全部归还，已交给模拟驱动改配，等回报 | 否 |
| `disabled` | 驱动回报失败、实际布局未知，停用等待本次批次补报 | 否 |

流转规则：

1. 值班员对 `active` 卡发起改配（`POST /reconfig`），生成新的改配批次号
   `generation` 并记录目标布局：
   - 卡空闲 → 直接进入 `reconfiguring`；
   - 卡上有在跑任务 → 进入 `draining`，任务照常跑完；最后一个分区归还时
     自动进入 `reconfiguring`。
2. `draining/reconfiguring/disabled` 期间该卡不参与任何新任务分配，
   **另一张卡照常分配**；所有卡都给不出分区时返回 `503`。
3. 模拟驱动通过 `POST /driver/report` 回报：
   - `success`：必须带 `actual_layout`。服务按驱动确认的实际布局重建分区并恢复
     `active`——**调度只认本次驱动确认的布局，不认申请时的目标布局**；
   - `failed`：实际布局未知，卡进入/保持 `disabled`（分区列表清空，避免按
     未确认的布局分配），等待同批次补报；补报 `success` 后才恢复。
4. **旧改配回报不能改变当前状态**：回报的 `generation` 与卡上当前批次不一致
   （更旧或未来的批次号）一律 `409` 拒收；批次已结束后的重复回报同样拒收。
5. 一张卡同时只允许一个进行中的改配批次，非 `active` 状态再次发起改配返回 `409`。

所有写操作在同一把异步锁内串行执行，不存在并发交错导致的状态破坏。

## 二、目录结构

```
app/
  models.py    # 枚举与 Pydantic 请求/响应模型
  service.py   # 调度状态机（GPU/分区数据结构、领用归还、改配、驱动回报）
  main.py      # FastAPI 路由与异常映射
tests/
  test_scheduler.py   # 13 个端到端测试（TestClient 打真实路由）
scripts/
  smoke.sh            # 真实 uvicorn + curl 的冒烟脚本
Dockerfile
docker-compose.yml
requirements.txt / requirements-dev.txt
```

## 三、启动方式（Docker Compose）

```bash
docker compose up --build
# 服务监听 http://localhost:8000
# 交互式文档： http://localhost:8000/docs
# OpenAPI：    http://localhost:8000/openapi.json
```

容器带 healthcheck（`GET /health`）。本机无 Docker 时也可直接用 Python 运行：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 四、API 一览（均可在 /docs 操作）

| 方法/路径 | 说明 |
| --- | --- |
| `GET /gpus` | 两张卡的实时状态（状态、已确认布局、批次号、目标布局、分区占用） |
| `GET /gpus/{gpu_id}` | 单卡状态 |
| `POST /tasks/acquire` | 按 `size` 领用分区，可带 `task_id` 或由服务端生成 |
| `POST /tasks/release` | 任务结束归还分区；draining 卡借此自动推进到 reconfiguring |
| `POST /reconfig` | 值班员发起改配，返回 `generation`（驱动回报必须带同一批次号） |
| `POST /driver/report` | 模拟驱动回报成功（带 `actual_layout`）或失败 |
| `POST /admin/reset` | 重置为初始状态（测试用） |
| `GET /health` | 健康检查 |

### /docs 手工演示序列

```bash
# 1. gpu-0 上跑一个大任务
curl -X POST localhost:8000/tasks/acquire -H 'Content-Type: application/json' \
  -d '{"size":"large","task_id":"job-a"}'
# 2. 发起改配：gpu-0 进入 draining，停接新任务
curl -X POST localhost:8000/reconfig -H 'Content-Type: application/json' \
  -d '{"gpu_id":"gpu-0","target_layout":"split"}'
# 3. 新任务只能落到 gpu-1
curl -X POST localhost:8000/tasks/acquire -H 'Content-Type: application/json' \
  -d '{"size":"large","task_id":"job-b"}'
# 4. job-a 正常结束归还，gpu-0 自动交给驱动（reconfiguring）
curl -X POST localhost:8000/tasks/release -H 'Content-Type: application/json' \
  -d '{"task_id":"job-a"}'
# 5. 模拟驱动回报成功，确认实际布局为 split，随后即可领小分区
curl -X POST localhost:8000/driver/report -H 'Content-Type: application/json' \
  -d '{"gpu_id":"gpu-0","generation":1,"status":"success","actual_layout":"split"}'
# 失败演练：把 status 换成 failed（不带 actual_layout）→ disabled；
# 再用同一 generation 补报 success + actual_layout 即恢复。
```

## 五、实测结果

### 自动化测试（13 项，全部通过）

```
13 passed in 1.00s
```

覆盖：初始布局、领还分区、无匹配分区 503、空闲卡直接 reconfiguring、
**新申请与改配交错**（停用卡停接/另一张照常/归还后自动交驱动/按确认布局恢复）、
**失败后停用与同批次补报**、重复失败保持停用、**旧批次与未来批次回报拒收**、
批次结束后重复回报拒收、success 缺 actual_layout 拒收、重复 task_id、
同布局改配拒绝、未知 GPU/任务 404、draining 期间在跑任务保留。

### 真实进程冒烟（uvicorn + curl，`scripts/smoke.sh`）

实测关键结论：

1. 交错场景：gpu-0 占用中发起改配 → `draining`；期间新大任务落 gpu-1，
   再申请返回 `503 当前没有空闲的 large 分区`；job-a 归还后 gpu-0 自动转
   `reconfiguring`；驱动 success 确认 split 后出现两个 small 分区，两个轻量
   任务各领一个，第三个返回 `503`。
2. 失败补报：驱动 `failed` → gpu-0 变 `disabled`、`partitions=[]`、批次号保留；
   停用期间新大任务仍由 gpu-1 承接；同批次再次 failed 保持停用；同批次
   `success` 补报带 actual_layout 后恢复 `active` 并重建分区。
3. 旧回报拒收：批次 #2 进行中拿批次 #1 回报 →
   `409 回报批次 #1 与当前批次 #2 不一致，旧改配回报一律忽略`，
   卡保持 `reconfiguring`、已确认布局不变；批次号 99 等未来批次同样拒收。

## 六、余项与边界说明

- 状态保存在进程内存中，重启即回到初始布局；`/admin/reset` 仅用于测试，
  生产化时需替换为持久化存储并加鉴权。
- 两张卡、两种布局按题面固定；卡数与布局类型如需扩展，改 `SchedulerService`
  初始化与 `_build_partitions` 即可。
- 未实现任务超时/抢占：draining 完全等待任务主动归还，符合“在跑任务照常结束”
  的要求。
- 当前环境未安装 Docker，Dockerfile 与 docker-compose.yml 已按 Compose 规范
  编写并通过 YAML 校验，实际容器启动需在有 Docker 的机器上执行
  `docker compose up --build`；所有功能已在本机 venv + uvicorn 实测通过。
