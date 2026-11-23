"""Pydantic 对外数据模型与枚举。"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PartitionSize(str, Enum):
    LARGE = "large"   # 大分区，只能跑大规格任务
    SMALL = "small"   # 小分区，跑轻量模型


class LayoutType(str, Enum):
    LARGE = "large"   # 一个大分区
    SPLIT = "split"   # 两个小分区


class GPUState(str, Enum):
    ACTIVE = "active"                  # 正常接任务
    DRAINING = "draining"              # 已停接新任务，等待在跑任务归还分区
    RECONFIGURING = "reconfiguring"    # 分区已全部归还，已交给模拟驱动改配
    DISABLED = "disabled"              # 改配失败且实际布局未知，停用等待补报


class ReportStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class PartitionOut(BaseModel):
    id: str
    size: PartitionSize
    occupied_by: Optional[str] = Field(
        default=None, description="占用该分区的任务 ID；空闲为 null"
    )


class GPUOut(BaseModel):
    id: str
    state: GPUState
    layout: LayoutType = Field(description="当前已确认布局")
    generation: int = Field(description="改配批次号，驱动回报必须携带且与当前批次一致")
    pending_target: Optional[LayoutType] = Field(
        default=None, description="当前批次申请的目标布局；无待完成改配时为 null"
    )
    partitions: list[PartitionOut]


class AcquireRequest(BaseModel):
    size: PartitionSize = Field(description="任务规格：large 领大分区，small 领小分区")
    task_id: Optional[str] = Field(
        default=None, description="可选任务 ID；不传则由服务端生成"
    )


class AcquireResponse(BaseModel):
    task_id: str
    gpu_id: str
    partition_id: str


class ReleaseRequest(BaseModel):
    task_id: str


class ReconfigRequest(BaseModel):
    gpu_id: str
    target_layout: LayoutType


class ReconfigResponse(BaseModel):
    gpu_id: str
    state: GPUState
    generation: int
    target_layout: LayoutType


class DriverReportRequest(BaseModel):
    gpu_id: str
    generation: int = Field(description="回报对应的改配批次号（取自改配响应）")
    status: ReportStatus
    actual_layout: Optional[LayoutType] = Field(
        default=None,
        description="status=success 时必填：驱动确认的实际布局，调度只按它恢复分配",
    )


class MessageOut(BaseModel):
    detail: str
