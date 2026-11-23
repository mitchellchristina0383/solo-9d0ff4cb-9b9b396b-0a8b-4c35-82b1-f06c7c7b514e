"""核心调度状态机：两张 GPU、分区领用/归还、停用改配与驱动回报。

状态机（单张卡）：

    ACTIVE ──发起改配且卡上有占用──▶ DRAINING ──分区全部归还──▶ RECONFIGURING
      │                                                              │
      └──────────发起改配且卡已空闲──────────────────────────────▶ RECONFIGURING
                                                                     │
                              驱动 success（批次一致，带实际布局） ──┴──▶ ACTIVE
                              驱动 failed（批次一致） ─────────────────▶ DISABLED

    DISABLED ──本次批次补报 success（带实际布局）──▶ ACTIVE
             ──本次批次补报 failed ──────────────▶ 继续 DISABLED

关键规则：
* DRAINING / RECONFIGURING / DISABLED 的卡不参与新任务分配；另一张卡照常分配。
* 调度只按驱动 success 回报确认的实际布局重建分区并恢复分配。
* 回报的 generation 与卡上当前批次不一致时一律拒收（旧改配回报不能改变当前状态）。
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .models import GPUState, LayoutType, PartitionSize


class SchedulerError(Exception):
    """可预期的调度错误，由 API 层映射为 4xx。"""

    status_code = 400

    def __init__(self, detail: str, status_code: Optional[int] = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class NotFoundError(SchedulerError):
    status_code = 404


class ConflictError(SchedulerError):
    status_code = 409


@dataclass
class Partition:
    id: str
    size: PartitionSize
    occupied_by: Optional[str] = None


@dataclass
class GPU:
    id: str
    state: GPUState = GPUState.ACTIVE
    layout: LayoutType = LayoutType.LARGE
    generation: int = 0
    pending_target: Optional[LayoutType] = None
    partitions: list[Partition] = field(default_factory=list)

    def free_partitions(self) -> list[Partition]:
        return [p for p in self.partitions if p.occupied_by is None]

    def is_idle(self) -> bool:
        return all(p.occupied_by is None for p in self.partitions)


def _build_partitions(gpu_id: str, layout: LayoutType) -> list[Partition]:
    if layout is LayoutType.LARGE:
        return [Partition(id=f"{gpu_id}-p0", size=PartitionSize.LARGE)]
    return [
        Partition(id=f"{gpu_id}-p0", size=PartitionSize.SMALL),
        Partition(id=f"{gpu_id}-p1", size=PartitionSize.SMALL),
    ]


class SchedulerService:
    """内存态调度器。所有公开方法都在同一把异步锁内串行执行。"""

    def __init__(self, gpu_ids: tuple[str, ...] = ("gpu-0", "gpu-1")) -> None:
        self._lock = asyncio.Lock()
        self._gpus: dict[str, GPU] = {}
        for gid in gpu_ids:
            # 初始布局：每张卡一个大分区
            self._gpus[gid] = GPU(
                id=gid,
                state=GPUState.ACTIVE,
                layout=LayoutType.LARGE,
                generation=0,
                pending_target=None,
                partitions=_build_partitions(gid, LayoutType.LARGE),
            )
        # task_id -> (gpu_id, partition_id)
        self._tasks: dict[str, tuple[str, str]] = {}

    # ---------- 内部工具 ----------

    def _get_gpu(self, gpu_id: str) -> GPU:
        gpu = self._gpus.get(gpu_id)
        if gpu is None:
            raise NotFoundError(f"未知 GPU: {gpu_id}")
        return gpu

    @staticmethod
    def _pump(gpu: GPU) -> None:
        """停用中的卡分区全部归还后，交给模拟驱动（进入 RECONFIGURING）。"""
        if gpu.state is GPUState.DRAINING and gpu.is_idle():
            gpu.state = GPUState.RECONFIGURING

    # ---------- 查询 ----------

    async def snapshot(self) -> list[GPU]:
        async with self._lock:
            return list(self._gpus.values())

    async def get_gpu(self, gpu_id: str) -> GPU:
        async with self._lock:
            return self._get_gpu(gpu_id)

    # ---------- 任务领用 / 归还 ----------

    async def acquire(self, size: PartitionSize, task_id: Optional[str]) -> tuple[str, GPU, Partition]:
        async with self._lock:
            if task_id is not None and task_id in self._tasks:
                raise ConflictError(f"任务 {task_id} 已持有分区，需先归还")
            tid = task_id or f"task-{uuid.uuid4().hex[:12]}"

            # 只有 ACTIVE 卡接新任务；按分区规格精确匹配，卡间按 id 顺序遍历
            for gpu in self._gpus.values():
                if gpu.state is not GPUState.ACTIVE:
                    continue
                for part in gpu.partitions:
                    if part.size is size and part.occupied_by is None:
                        part.occupied_by = tid
                        self._tasks[tid] = (gpu.id, part.id)
                        return tid, gpu, part

            raise ConflictError(
                f"当前没有空闲的 {size.value} 分区（停用中的卡不参与分配）",
                status_code=503,
            )

    async def release(self, task_id: str) -> tuple[GPU, Partition]:
        async with self._lock:
            location = self._tasks.pop(task_id, None)
            if location is None:
                raise NotFoundError(f"未知任务或分区已归还: {task_id}")
            gpu_id, part_id = location
            gpu = self._get_gpu(gpu_id)
            part = next((p for p in gpu.partitions if p.id == part_id), None)
            if part is None or part.occupied_by != task_id:
                # 理论上不会出现：任务表与分区占用保持一致
                raise ConflictError("任务占用记录与分区状态不一致")
            part.occupied_by = None
            self._pump(gpu)
            return gpu, part

    # ---------- 改配 ----------

    async def request_reconfig(self, gpu_id: str, target: LayoutType) -> GPU:
        async with self._lock:
            gpu = self._get_gpu(gpu_id)
            if gpu.state is not GPUState.ACTIVE:
                raise ConflictError(
                    f"{gpu_id} 处于 {gpu.state.value} 状态，已有进行中或待补报的改配批次 "
                    f"#{gpu.generation}，不能重复发起改配"
                )
            if target is gpu.layout:
                raise ConflictError(
                    f"目标布局 {target.value} 与当前已确认布局相同，无需改配"
                )
            gpu.generation += 1
            gpu.pending_target = target
            if gpu.is_idle():
                # 卡上没有在跑任务，立即交给模拟驱动
                gpu.state = GPUState.RECONFIGURING
            else:
                # 停接新任务，等分区全部归还
                gpu.state = GPUState.DRAINING
            return gpu

    # ---------- 模拟驱动回报 ----------

    async def driver_report(
        self,
        gpu_id: str,
        generation: int,
        status: str,
        actual_layout: Optional[LayoutType],
    ) -> GPU:
        async with self._lock:
            gpu = self._get_gpu(gpu_id)

            # 旧批次回报：无论成功失败都不能改变当前分配状态
            if generation != gpu.generation:
                raise ConflictError(
                    f"回报批次 #{generation} 与当前批次 #{gpu.generation} 不一致，"
                    "旧改配回报一律忽略"
                )

            if gpu.state is GPUState.RECONFIGURING:
                if status == "success":
                    if actual_layout is None:
                        raise ConflictError(
                            "success 回报必须携带驱动确认的 actual_layout"
                        )
                    # 调度只按驱动确认的实际布局重建分区并恢复分配
                    gpu.layout = actual_layout
                    gpu.partitions = _build_partitions(gpu.id, actual_layout)
                    gpu.state = GPUState.ACTIVE
                    gpu.pending_target = None
                    return gpu
                # failed：布局未知，继续停用，等待本次批次补报
                gpu.state = GPUState.DISABLED
                gpu.partitions = []
                return gpu

            if gpu.state is GPUState.DISABLED:
                if status == "success":
                    if actual_layout is None:
                        raise ConflictError(
                            "success 补报必须携带驱动确认的 actual_layout"
                        )
                    gpu.layout = actual_layout
                    gpu.partitions = _build_partitions(gpu.id, actual_layout)
                    gpu.state = GPUState.ACTIVE
                    gpu.pending_target = None
                    return gpu
                # 补报仍然失败：布局依旧未知，继续停用等待下一次补报
                return gpu

            raise ConflictError(
                f"{gpu_id} 当前为 {gpu.state.value} 状态，没有等待驱动回报的改配"
            )

    # ---------- 测试辅助 ----------

    async def reset(self) -> None:
        async with self._lock:
            for gid, gpu in list(self._gpus.items()):
                gpu.state = GPUState.ACTIVE
                gpu.layout = LayoutType.LARGE
                gpu.generation = 0
                gpu.pending_target = None
                gpu.partitions = _build_partitions(gid, LayoutType.LARGE)
            self._tasks.clear()
