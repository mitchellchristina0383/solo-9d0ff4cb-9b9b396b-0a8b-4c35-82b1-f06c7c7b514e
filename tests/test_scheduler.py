"""端到端状态机测试（通过 FastAPI TestClient 打真实 HTTP 路由）。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, service


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


# ------------------------- 归还凭据 / 重跑防迟到归还 -------------------------

def _acquire(client: TestClient, task_id: str, size: str = "large") -> dict:
    r = client.post("/tasks/acquire", json={"size": size, "task_id": task_id})
    assert r.status_code == 200, r.text
    return r.json()


def test_every_acquire_returns_fresh_release_token(client: TestClient) -> None:
    first = _acquire(client, "job-a")
    assert first["release_token"]
    client.post("/tasks/release", json={"task_id": "job-a", "release_token": first["release_token"]})
    second = _acquire(client, "job-a")
    assert second["release_token"] and second["release_token"] != first["release_token"]


def test_first_acquire_since_reset_can_release_with_task_id_only(client: TestClient) -> None:
    """老客户端：启动/重置后首次成功领用仍可只凭 task_id 归还。"""
    _acquire(client, "legacy")
    r = client.post("/tasks/release", json={"task_id": "legacy"})
    assert r.status_code == 200, r.text
    assert gpu_state(client, "gpu-0")["partitions"][0]["occupied_by"] is None


def test_late_release_from_previous_round_cannot_clear_current_lease(client: TestClient) -> None:
    """迟到归还：上一轮凭据在重跑领用后晚到，必须被拒且不清本次占用。"""
    old = _acquire(client, "job-a")
    assert client.post(
        "/tasks/release", json={"task_id": "job-a", "release_token": old["release_token"]}
    ).status_code == 200
    new = _acquire(client, "job-a")
    assert new["release_token"] != old["release_token"]

    # 上一轮的归还请求晚到（旧凭据）
    r = client.post(
        "/tasks/release", json={"task_id": "job-a", "release_token": old["release_token"]}
    )
    assert r.status_code == 409
    g0 = gpu_state(client, "gpu-0")
    assert g0["partitions"][0]["occupied_by"] == "job-a"

    # 当前凭据可正常归还
    assert client.post(
        "/tasks/release", json={"task_id": "job-a", "release_token": new["release_token"]}
    ).status_code == 200
    assert gpu_state(client, "gpu-0")["partitions"][0]["occupied_by"] is None


def test_release_after_second_acquire_requires_token(client: TestClient) -> None:
    """同一 ID 再次领用后，只凭 task_id 的归还必须被拒。"""
    _acquire(client, "job-a")
    client.post("/tasks/release", json={"task_id": "job-a"})
    _acquire(client, "job-a")

    r = client.post("/tasks/release", json={"task_id": "job-a"})
    assert r.status_code == 409
    assert gpu_state(client, "gpu-0")["partitions"][0]["occupied_by"] == "job-a"


def test_failed_acquire_does_not_count_as_a_lease(client: TestClient) -> None:
    """失败的领用不算一次；首次成功领用后仍可只凭 task_id 归还。"""
    # 初始无小分区，small 领用失败
    assert client.post(
        "/tasks/acquire", json={"size": "small", "task_id": "job-a"}
    ).status_code == 503
    _acquire(client, "job-a")  # 首次成功领用
    r = client.post("/tasks/release", json={"task_id": "job-a"})
    assert r.status_code == 200, r.text


def test_late_release_during_draining_keeps_occupancy_and_wait(client: TestClient) -> None:
    """归还被拒时保留现有占用；正在排空的卡继续等待，不提前交给驱动。"""
    old = _acquire(client, "job-a")
    assert client.post(
        "/tasks/release", json={"task_id": "job-a", "release_token": old["release_token"]}
    ).status_code == 200
    new = _acquire(client, "job-a")

    # 值班员发起改配 -> 卡进入 draining，在跑任务保留
    assert client.post(
        "/reconfig", json={"gpu_id": "gpu-0", "target_layout": "split"}
    ).json()["state"] == "draining"

    # 迟到的旧凭据归还 -> 409，占用保留，仍为 draining
    r = client.post(
        "/tasks/release", json={"task_id": "job-a", "release_token": old["release_token"]}
    )
    assert r.status_code == 409
    g0 = gpu_state(client, "gpu-0")
    assert g0["state"] == "draining"
    assert g0["partitions"][0]["occupied_by"] == "job-a"

    # 正常凭据归还最后一个分区 -> 才交给驱动
    assert client.post(
        "/tasks/release", json={"task_id": "job-a", "release_token": new["release_token"]}
    ).status_code == 200
    assert gpu_state(client, "gpu-0")["state"] == "reconfiguring"


def test_rejected_release_does_not_advance_other_partitions(client: TestClient) -> None:
    """多分区排空：被拒的归还不能让卡提前改配，最后一个正常归还才推进。"""
    client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "split"})
    client.post(
        "/driver/report",
        json={"gpu_id": "gpu-0", "generation": 1, "status": "success", "actual_layout": "split"},
    )
    old = _acquire(client, "job-a", size="small")
    assert client.post(
        "/tasks/release", json={"task_id": "job-a", "release_token": old["release_token"]}
    ).status_code == 200
    tok_a = _acquire(client, "job-a", size="small")["release_token"]
    _acquire(client, "job-b", size="small")

    client.post("/reconfig", json={"gpu_id": "gpu-0", "target_layout": "large"})
    assert gpu_state(client, "gpu-0")["state"] == "draining"

    # job-a 用旧凭据归还 -> 拒，卡仍在 draining
    r = client.post(
        "/tasks/release", json={"task_id": "job-a", "release_token": old["release_token"]}
    )
    assert r.status_code == 409
    assert gpu_state(client, "gpu-0")["state"] == "draining"

    # job-b 正常归还：还有一个分区被占，仍 draining
    assert client.post("/tasks/release", json={"task_id": "job-b"}).status_code == 200
    assert gpu_state(client, "gpu-0")["state"] == "draining"

    # job-a 用当前凭据正常归还最后一个分区 -> 交给驱动
    assert client.post(
        "/tasks/release", json={"task_id": "job-a", "release_token": tok_a}
    ).status_code == 200
    assert gpu_state(client, "gpu-0")["state"] == "reconfiguring"


def test_task_record_count_does_not_grow_with_reacquire(client: TestClient) -> None:
    """同一 ID 反复领用，服务为该 ID 保留的记录数不增长。"""
    for _ in range(12):
        tok = _acquire(client, "job-a")["release_token"]
        assert client.post(
            "/tasks/release", json={"task_id": "job-a", "release_token": tok}
        ).status_code == 200
    records = [r for r in service._tasks.values() if r.task_id == "job-a"]
    assert len(records) == 1
    assert records[0].acquire_count == 12
    assert records[0].active is False


def test_release_unknown_task_still_404_even_with_token(client: TestClient) -> None:
    r = client.post(
        "/tasks/release", json={"task_id": "nope", "release_token": "rel_whatever"}
    )
    assert r.status_code == 404


def test_bogus_token_on_first_lease_is_rejected(client: TestClient) -> None:
    """首次领用虽可省略凭据，但一旦提交凭据就必须与当前领用匹配。"""
    _acquire(client, "job-a")
    r = client.post(
        "/tasks/release", json={"task_id": "job-a", "release_token": "rel_bogus"}
    )
    assert r.status_code == 409
    assert gpu_state(client, "gpu-0")["partitions"][0]["occupied_by"] == "job-a"
