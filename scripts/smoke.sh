#!/usr/bin/env bash
# 端到端冒烟：真实 uvicorn 进程 + curl，覆盖
# 1) 新申请与改配交错（停用卡停接、另一张照常、归还后交驱动、按确认布局恢复）
# 2) 改配失败停用 -> 同批次补报恢复
# 3) 旧批次回报拒收
# 4) 重跑复用 task_id 的归还凭据：迟到归还 / 老客户端 / 正常改配
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
CURL=(curl -sS --fail-with-body --no-progress-meter)

j() { python3 -c 'import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))' "$1"; }
post() { $CURL -X POST "$BASE$1" -H 'Content-Type: application/json' -d "$2"; }

echo "== 健康检查 & 初始状态 =="
$CURL "$BASE/health"; echo
$CURL "$BASE/gpus" | python3 -m json.tool
post /admin/reset '{}' >/dev/null

echo
echo "== 场景1：新申请与改配交错 =="
post /tasks/acquire '{"size":"large","task_id":"job-a"}' | python3 -m json.tool
post /reconfig '{"gpu_id":"gpu-0","target_layout":"split"}' | python3 -m json.tool
echo "-- gpu-0 停接，新大任务落到 gpu-1 --"
post /tasks/acquire '{"size":"large","task_id":"job-b"}' | python3 -m json.tool
echo "-- 再来一个大任务，期望 503 --"
post /tasks/acquire '{"size":"large","task_id":"job-c"}' || true
echo
echo "-- job-a 归还，gpu-0 自动进入 reconfiguring --"
post /tasks/release '{"task_id":"job-a"}' >/dev/null
$CURL "$BASE/gpus/gpu-0" | python3 -m json.tool
echo "-- 驱动 success 确认 split --"
post /driver/report '{"gpu_id":"gpu-0","generation":1,"status":"success","actual_layout":"split"}' | python3 -m json.tool
echo "-- 两个轻量任务领用两个小分区 --"
post /tasks/acquire '{"size":"small","task_id":"s1"}' | python3 -m json.tool
post /tasks/acquire '{"size":"small","task_id":"s2"}' | python3 -m json.tool
echo "-- 第三个小任务期望 503 --"
post /tasks/acquire '{"size":"small","task_id":"s3"}' || true

echo
echo "== 场景2：失败停用 + 本次补报 =="
post /tasks/release '{"task_id":"s1"}' >/dev/null
post /tasks/release '{"task_id":"s2"}' >/dev/null
post /tasks/release '{"task_id":"job-b"}' >/dev/null
post /reconfig '{"gpu_id":"gpu-0","target_layout":"large"}' | python3 -m json.tool
echo "-- 驱动 failed，期望 disabled --"
post /driver/report '{"gpu_id":"gpu-0","generation":2,"status":"failed"}' | python3 -m json.tool
echo "-- 停用期间大任务只能去 gpu-1 --"
post /tasks/acquire '{"size":"large","task_id":"job-d"}' | python3 -m json.tool
echo "-- 同批次补报 success，恢复 active --"
post /driver/report '{"gpu_id":"gpu-0","generation":2,"status":"success","actual_layout":"large"}' | python3 -m json.tool

echo
echo "== 场景3：旧批次回报拒收 =="
post /tasks/release '{"task_id":"job-d"}' >/dev/null
post /reconfig '{"gpu_id":"gpu-1","target_layout":"split"}' >/dev/null   # gpu-1 批次 #1
echo "-- 未来批次号 99 回报，期望 409 拒收 --"
post /driver/report '{"gpu_id":"gpu-1","generation":99,"status":"failed"}' || true
echo
echo "-- 当前批次 #1 成功回报 --"
post /driver/report '{"gpu_id":"gpu-1","generation":1,"status":"success","actual_layout":"split"}' >/dev/null
echo "-- 再发起批次 #2，然后拿旧批次 #1 回报，期望 409 且状态不变 --"
post /reconfig '{"gpu_id":"gpu-1","target_layout":"large"}' | j 'd["state"] + " gen=" + str(d["generation"])'
post /driver/report '{"gpu_id":"gpu-1","generation":1,"status":"failed"}' || true
echo
$CURL "$BASE/gpus/gpu-1" | j '"state=" + d["state"] + " gen=" + str(d["generation"]) + " layout=" + d["layout"]'
post /driver/report '{"gpu_id":"gpu-1","generation":2,"status":"success","actual_layout":"large"}' >/dev/null

echo
echo "== 场景4：重跑复用 task_id 的归还凭据 =="
post /admin/reset '{}' >/dev/null

echo "-- 老客户端：首次成功领用后只凭 task_id 归还，期望 200 --"
OLD=$(post /tasks/acquire '{"size":"large","task_id":"rerun"}' | j 'd["release_token"]')
post /tasks/release '{"task_id":"rerun"}' || true

echo "-- 同 ID 重跑再次领用，拿到新凭据（旧凭据应已不同）--"
NEW=$(post /tasks/acquire '{"size":"large","task_id":"rerun"}' | j 'd["release_token"]')
[ "$OLD" != "$NEW" ] && echo "新凭据 != 旧凭据 OK"

echo "-- 上一轮归还迟到（带旧凭据），期望 409 且本次占用保留 --"
post /tasks/release "{\"task_id\":\"rerun\",\"release_token\":\"$OLD\"}" || true
$CURL "$BASE/gpus/gpu-0" | j '"occupied_by=" + str(d["partitions"][0]["occupied_by"])'

echo "-- 再次领用后不带凭据归还，期望 409 --"
post /tasks/release '{"task_id":"rerun"}' || true

echo "-- 发起改配进入 draining，迟到旧凭据归还仍被拒，卡继续等待 --"
post /reconfig '{"gpu_id":"gpu-0","target_layout":"split"}' >/dev/null
post /tasks/release "{\"task_id\":\"rerun\",\"release_token\":\"$OLD\"}" || true
$CURL "$BASE/gpus/gpu-0" | j '"state=" + d["state"] + " occupied_by=" + str(d["partitions"][0]["occupied_by"])'

echo "-- 用当前凭据正常归还，最后一个分区释放后交给驱动 --"
post /tasks/release "{\"task_id\":\"rerun\",\"release_token\":\"$NEW\"}" >/dev/null
$CURL "$BASE/gpus/gpu-0" | j '"state=" + d["state"]'

echo "-- 驱动 success 确认 split，正常完成改配 --"
post /driver/report '{"gpu_id":"gpu-0","generation":1,"status":"success","actual_layout":"split"}' | j '"state=" + d["state"] + " layout=" + d["layout"]'

echo
echo "== 最终全景 =="
post /admin/reset '{}' >/dev/null
$CURL "$BASE/gpus" | python3 -m json.tool
echo
echo "SMOKE OK"
