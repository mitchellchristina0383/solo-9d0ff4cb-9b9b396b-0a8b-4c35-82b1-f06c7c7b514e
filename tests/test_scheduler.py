"""端到端状态机测试（通过 FastAPI TestClient 打真实 HTTP 路由）。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as c:
        c.post("/admin/reset")
        yield c


def gpu_state(client: TestClient, gpu_id: str) -> dict:
    return client.get(f"/gpus/{gpu_id}").json()


def test_initial_layout_is_one_large_partition_per_gpu(client: TestClient) -> None:
    gpus = client.get("/gpus").json()
    assert len(gpus) == 2
    for g in gpus:
        assert g["state"] == "active"
        assert g["layout"] == "large"
        assert len(g["partitions"]) == 1
        assert g["partitions"][0]["size"] == "large"
        assert g["generation"] == 0


def test_acquire_and_release_large_partition(client: TestClient) -> None:
    r = client.post("/tasks/acquire", json={"size": "large", "task_id": "t1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task_id"] == "t1"
    assert body["gpu_id"] == "gpu-0"  # 按卡顺序分配
    assert body["partition_id"] == "gpu-0-p0"

    g0 = gpu_state(client, "gpu-0")
    assert g0["partitions"][0]["occupied_by"] == "t1"

    r = client.post("/tasks/release", json={"task_id": "t1"})
    assert r.status_code == 200
    g0 = gpu_state(client, "gpu-0")
    assert g0["partitions"][0]["occupied_by"] is None


def test_small_task_has_no_partition_until_split_layout(client: TestClient) -> None:
    # 初始都是大分区，小规格任务领不到
    r = client.post("/tasks/acquire", json={"size": "small", "task_id": "s1"})
    assert r.status_code == 503


def test_reconfig_idle_card_goes_straight_to_reconfiguring(client: TestClient) -> None:
    r = client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "split"})
    assert r.status_code == 200
    body = r.json()
    assert body["generation"] == 1
    assert body["state"] == "reconfiguring"

    g0 = gpu_state(client, "gpu-0")
    assert g0["pending_target"] == "split"


def test_interleaved_new_requests_and_reconfig(client: TestClient) -> None:
    """新申请与改配交错：停用卡不接新任务，另一张卡照常分配。"""
    # 1) gpu-0 被一个大任务占住
    r = client.post("/tasks/acquire", json={"size": "large", "task_id": "job-a"})
    assert r.status_code == 200
    assert r.json()["gpu_id"] == "gpu-0"

    # 2) 值班员发起 gpu-0 改配 -> DRAINING，停接新任务
    r = client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "split"})
    assert r.status_code == 200
    assert r.json()["state"] == "draining"
    assert r.json()["generation"] == 1

    # 3) 改配进行中再发起改配 -> 409
    r = client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "split"})
    assert r.status_code == 409

    # 4) 新的大任务只能落到 gpu-1（gpu-0 停接）
    r = client.post("/tasks/acquire", json={"size": "large", "task_id": "job-b"})
    assert r.status_code == 200
    assert r.json()["gpu_id"] == "gpu-1"

    # 5) 两张卡都没空的大分区了 -> 503
    r = client.post("/tasks/acquire", json={"size": "large", "task_id": "job-c"})
    assert r.status_code == 503

    # 6) job-a 结束归还，gpu-0 分区全部归还 -> 自动交给驱动 RECONFIGURING
    r = client.post("/tasks/release", json={"task_id": "job-a"})
    assert r.status_code == 200
    g0 = gpu_state(client, "gpu-0")
    assert g0["state"] == "reconfiguring"

    # 7) 驱动 success，确认实际布局为 split，按确认布局恢复分配
    r = client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 1, "status": "success", "actual_layout": "split"},
    )
    assert r.status_code == 200
    g0 = r.json()
    assert g0["state"] == "active"
    assert g0["layout"] == "split"
    assert len(g0["partitions"]) == 2
    assert all(p["size"] == "small" for p in g0["partitions"])

    # 8) 轻量模型任务现在可以领到小分区，且两个小分区都能分出去
    r1 = client.post("/tasks/acquire", json={"size": "small", "task_id": "s1"})
    r2 = client.post("/tasks/acquire", json={"size": "small", "task_id": "s2"})
    assert r1.status_code == r2.status_code == 200
    assert {r1.json()["partition_id"], r2.json()["partition_id"]} == {"gpu-0-p0", "gpu-0-p1"}

    # 9) 小分区占满后再来小任务 -> 503；gpu-1 仍是大分区，不承接小规格
    r = client.post("/tasks/acquire", json={"size": "small", "task_id": "s3"})
    assert r.status_code == 503

    # 10) gpu-1 上的大任务不受影响，正常结束
    g1 = gpu_state(client, "gpu-1")
    assert g1["partitions"][0]["occupied_by"] == "job-b"
    assert client.post("/tasks/release", json={"task_id": "job-b"}).status_code == 200


def test_failed_report_disables_card_and_followup_report_recovers(client: TestClient) -> None:
    """改配失败且布局未知 -> 继续停用；本次批次补报确认后才恢复。"""
    client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "split"})

    # 驱动回报失败：布局未知，卡停用
    r = client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 1, "status": "failed"},
    )
    assert r.status_code == 200
    g0 = r.json()
    assert g0["state"] == "disabled"
    assert g0["layout"] == "large"          # 已确认布局保持旧值
    assert g0["partitions"] == []           # 实际布局未知，不暴露分区
    assert g0["generation"] == 1

    # 停用期间：不能发起新改配，也不接任务
    r = client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "split"})
    assert r.status_code == 409
    r = client.post("/tasks/acquire", json={"size": "large", "task_id": "x"})
    assert r.status_code == 200
    assert r.json()["gpu_id"] == "gpu-1"    # 另一张卡照常分配

    # 补报再失败：继续停用
    r = client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 1, "status": "failed"},
    )
    assert r.status_code == 200
    assert gpu_state(client, "gpu-0")["state"] == "disabled"

    # 本次批次补报 success（驱动确认实际就是 split）-> 恢复
    r = client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 1, "status": "success", "actual_layout": "split"},
    )
    assert r.status_code == 200
    g0 = r.json()
    assert g0["state"] == "active"
    assert g0["layout"] == "split"
    assert len(g0["partitions"]) == 2
    assert g0["pending_target"] is None


def test_stale_generation_reports_never_change_state(client: TestClient) -> None:
    """旧改配回报不能改变当前分配状态。"""
    # 批次 #1：成功改成 split
    client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "split"})
    client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 1, "status": "success", "actual_layout": "split"},
    )

    # 批次 #2：再发起改配回 large，进入 RECONFIGURING
    r = client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "large"})
    assert r.json()["generation"] == 2
    assert r.json()["state"] == "reconfiguring"

    # 旧批次 #1 的失败回报 -> 拒收，当前状态不变
    r = client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 1, "status": "failed"},
    )
    assert r.status_code == 409
    g0 = gpu_state(client, "gpu-0")
    assert g0["state"] == "reconfiguring"
    assert g0["generation"] == 2
    assert g0["layout"] == "split"

    # 旧批次 #1 的成功回报（即便带布局）-> 同样拒收
    r = client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 1, "status": "success", "actual_layout": "large"},
    )
    assert r.status_code == 409
    assert gpu_state(client, "gpu-0")["state"] == "reconfiguring"

    # 未来批次号也拒收
    r = client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 99, "status": "success", "actual_layout": "large"},
    )
    assert r.status_code == 409

    # 当前批次 #2 的正常回报生效
    r = client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 2, "status": "success", "actual_layout": "large"},
    )
    assert r.status_code == 200
    g0 = r.json()
    assert g0["state"] == "active"
    assert g0["layout"] == "large"
    assert len(g0["partitions"]) == 1

    # 批次结束后再重复回报 #2 -> 没有等待中的改配，拒收
    r = client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 2, "status": "success", "actual_layout": "large"},
    )
    assert r.status_code == 409


def test_unknown_gpu_and_task_return_404(client: TestClient) -> None:
    r = client.post("/reconfig", json={"gpu_id": "gpu-x", "target_layout": "split"})
    assert r.status_code == 404
    r = client.post("/tasks/release", json={"task_id": "nope"})
    assert r.status_code == 404


def test_duplicate_task_id_rejected(client: TestClient) -> None:
    assert client.post("/tasks/acquire", json={"size": "large", "task_id": "dup"}).status_code == 200
    r = client.post("/tasks/acquire", json={"size": "large", "task_id": "dup"})
    assert r.status_code == 409


def test_same_layout_reconfig_rejected(client: TestClient) -> None:
    r = client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "large"})
    assert r.status_code == 409


def test_server_generates_task_id_when_omitted(client: TestClient) -> None:
    r = client.post("/tasks/acquire", json={"size": "large"})
    assert r.status_code == 200
    assert r.json()["task_id"].startswith("task-")


def test_success_report_requires_actual_layout(client: TestClient) -> None:
    client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "split"})
    r = client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 1, "status": "success"},
    )
    assert r.status_code == 409
    assert gpu_state(client, "gpu-0")["state"] == "reconfiguring"


def test_draining_card_keeps_running_task_until_release(client: TestClient) -> None:
    client.post("/tasks/acquire", json={"size": "large", "task_id": "keep"})
    client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "split"})
    g0 = gpu_state(client, "gpu-0")
    assert g0["state"] == "draining"
    # 在跑任务照常保留
    assert g0["partitions"][0]["occupied_by"] == "keep"
    # 归还后才进入驱动环节
    client.post("/tasks/release", json={"task_id": "keep"})
    assert gpu_state(client, "gpu-0")["state"] == "reconfiguring"
