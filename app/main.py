"""FastAPI 入口：在 /docs 可完成改配、领还分区与模拟驱动回报。"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .models import (
    AcquireRequest,
    AcquireResponse,
    DriverReportRequest,
    GPUOut,
    GPUState,
    MessageOut,
    PartitionOut,
    ReconfigRequest,
    ReconfigResponse,
    ReleaseRequest,
)
from .service import GPU, SchedulerError, SchedulerService

app = FastAPI(
    title="GPU 分区改配调度模拟器",
    version="1.0.0",
    description=(
        "模拟两张 GPU（gpu-0、gpu-1），每张卡可切成一个大分区（large）或两个小分区（split）。"
        "任务按规格领用分区、结束归还；值班员发起改配后该卡停接新任务，"
        "分区全部归还才交给模拟驱动，另一张卡照常分配。"
    ),
)

service = SchedulerService()


@app.exception_handler(SchedulerError)
async def scheduler_error_handler(_: Request, exc: SchedulerError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def _gpu_to_out(gpu: GPU) -> GPUOut:
    return GPUOut(
        id=gpu.id,
        state=gpu.state,
        layout=gpu.layout,
        generation=gpu.generation,
        pending_target=gpu.pending_target,
        partitions=[
            PartitionOut(id=p.id, size=p.size, occupied_by=p.occupied_by)
            for p in gpu.partitions
        ],
    )


TAG_GPU = "GPU 状态查询"
TAG_TASK = "任务：领用 / 归还分区"
TAG_RECONFIG = "值班员：发起改配"
TAG_DRIVER = "模拟驱动：改配回报"
TAG_ADMIN = "测试辅助"


@app.get("/gpus", response_model=list[GPUOut], tags=[TAG_GPU], summary="查看两张卡的实时状态")
async def list_gpus() -> list[GPUOut]:
    return [_gpu_to_out(g) for g in await service.snapshot()]


@app.get("/gpus/{gpu_id}", response_model=GPUOut, tags=[TAG_GPU], summary="查看单卡状态")
async def get_gpu(gpu_id: str) -> GPUOut:
    return _gpu_to_out(await service.get_gpu(gpu_id))


@app.post(
    "/tasks/acquire",
    response_model=AcquireResponse,
    tags=[TAG_TASK],
    summary="按规格领用分区（large 领大分区，small 领小分区）",
    responses={409: {"model": MessageOut}, 503: {"model": MessageOut}},
)
async def acquire_partition(req: AcquireRequest) -> AcquireResponse:
    tid, gpu, part = await service.acquire(req.size, req.task_id)
    return AcquireResponse(task_id=tid, gpu_id=gpu.id, partition_id=part.id)


@app.post(
    "/tasks/release",
    response_model=MessageOut,
    tags=[TAG_TASK],
    summary="任务结束归还分区",
    responses={404: {"model": MessageOut}},
)
async def release_partition(req: ReleaseRequest) -> MessageOut:
    gpu, part = await service.release(req.task_id)
    return MessageOut(detail=f"任务 {req.task_id} 已归还 {gpu.id}/{part.id}")


@app.post(
    "/reconfig",
    response_model=ReconfigResponse,
    tags=[TAG_RECONFIG],
    summary="发起改配：该卡立即停接新任务，分区全部归还后交给模拟驱动",
    responses={409: {"model": MessageOut}, 404: {"model": MessageOut}},
)
async def request_reconfig(req: ReconfigRequest) -> ReconfigResponse:
    gpu = await service.request_reconfig(req.gpu_id, req.target_layout)
    return ReconfigResponse(
        gpu_id=gpu.id,
        state=gpu.state,
        generation=gpu.generation,
        target_layout=req.target_layout,
    )


@app.post(
    "/driver/report",
    response_model=GPUOut,
    tags=[TAG_DRIVER],
    summary="模拟驱动回报改配结果（批次号必须与改配响应一致）",
    description=(
        "- success：必须带 actual_layout，调度只按驱动确认的实际布局恢复分配；\n"
        "- failed：布局未知，卡继续停用（disabled），等待同批次补报；\n"
        "- 旧批次（generation 不一致）回报一律拒收，不改变当前分配状态。"
    ),
    responses={409: {"model": MessageOut}, 404: {"model": MessageOut}},
)
async def driver_report(req: DriverReportRequest) -> GPUOut:
    gpu = await service.driver_report(
        req.gpu_id, req.generation, req.status.value, req.actual_layout
    )
    return _gpu_to_out(gpu)


@app.post("/admin/reset", response_model=MessageOut, tags=[TAG_ADMIN], summary="重置为初始状态（测试用）")
async def reset_state() -> MessageOut:
    await service.reset()
    return MessageOut(detail="已重置：两张卡均为 large 大分区、ACTIVE、批次归零")


@app.get("/health", tags=[TAG_ADMIN], summary="健康检查")
async def health() -> dict[str, str]:
    return {"status": "ok"}
