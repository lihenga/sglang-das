# Spark / Mooncake 直通路径复核（2026-11-16）

本记录补充 [SWA 性能记录](spark_swa_tuning_20261116.md)。此前用 attention trace 和显存命中对照给出“计算/框架优先”的结论，证据不足以排除直通元数据、会话、搬运与写回竞争。本轮按启动脚本实际走到的路径复核，不继续修改 attention kernel。

## 环境与比较边界

- `.29 / zhy-4B`，8 张 BW DCU，单机 1P1D，P TP4 / D TP4。
- Spark `sglang-0518-mtp-sparkX25`：`78970d3bad84b9076ec2cb5d176dac29814f5cc3`；Mooncake `v0.3.14_dev_l4`：`f855c7e7cb26b2654a265b75d97d96bcecd0a445`。固定此前构建 wheel，未更换 native 二进制。
- 保留已有 SWA 裁剪和 `--chunked-prefill-size 32768 --max-prefill-tokens 32768`，各组统一使用；所有正式对照关闭原生 overlap scheduling。
- 64,000 输入，1 输出，48 个独立随机前缀，并发 32；PC95 / PC90 分别精确命中 60,800 / 57,600 tokens。后缀每轮不同，P/D L1 每轮清空。
- 每组检查全部响应和四个 P rank 的 source counters：device 命中 0、Mooncake memory read 0、DFS read > 0；读取/预取分配失败 0。开启预取组进一步要求 48 请求均为 `dfs_prefetched`，预取读取字节等于成功消费字节。
- 复用唯一纯算分母 26,608.94 input tokens/s，未再跑纯算。吞吐包含缓存输入。
- DFS、源码、临时文件、缓存和证据均位于 `/stortest/zhy_spcae/fixbug/spark-perf-tuning-20261116/`。
- Spark 是端到端性能环境；PR 的 `feature/optimize_main` 在独立源码中做单测。二者路径布局不同，不能将 Spark 性能当作目标分支真模型性能。

## 启动脚本实际路径

```text
router → P scheduler 收请求 / 匹配 L1
  → UnifiedCacheLinkerWrapper.match
      prefix hashes → MooncakeStore.batch_exists_v2 → batch_is_exist
      → TP/CP 可恢复前缀集合交集
  → 后台 prefetch 工作队列
      component keys → session start with sources / 引用计数
      → batch_get_session_prefetch
          DFS/ParaStor POSIX BatchRead → unified pinned host buffer
  → 四 rank READY / session lease revalidation / claim
      → 分配 device slots、插入/锁住缓存节点 → pending load
  → load thread 等 ready event
      Python pool layout / locations → native create_read_plan
      → borrowed sessions / adaptive source
      → page-wise build_all → chunked range reads
          host-prefetch cache 命中 → scatter batches / stream synchronize
      → 发布层 readiness → attention / TP collectives
  → Mooncake P→D KV transfer → D 返回 token
  → 后台 offload：GPU 事件 → batch_set_v2 → native put → 外部存储
```

这里的 DFS→host 预取、host→GPU scatter、P→D KV 传输是三个不同阶段；host 临时缓冲不是 SGLang host cache，也不把 DFS session 变为 Mooncake memory replica。

| 脚本配置 | 实际行为 |
|---|---|
| `SGLANG_MOONCAKE_READ_PLAN=1` | 走 native ReadPlan，地址展开在 C++，避免 Python 构造每个 range 的嵌套列表 |
| waiting-queue DFS prefetch，4 workers / 48 requests | 后台先读取整对象到 pinned host buffer，入批前跨 rank 检查 readiness |
| page-wise + adaptive source | 借用的 session 仍为 DFS，所以 host 预取完成后依旧走 page-wise；等待完整计划成功后才开放计算 |
| `MOONCAKE_READ_PLAN_PIPELINE=0` | 原配置没有 native layer pipeline；单独改成 1 仍受 effective page-wise 条件限制 |
| restore chunk 1 GiB | 命中 host 预取后仍按对象规划 chunk，每 chunk 提交/同步 scatter；正式 PC90 波次为 120 次内层调用 / 480 个 H2D kernel batches（四 rank 合计） |
| `MC_STORE_DFS_H2D_KERNEL=1` | 使用已有批量 scatter；已有目的内存区域缓存，避免逐 range 查询设备指针属性 |
| unified buffer / bounded requests | 预取、恢复与写回共享预算；本轮无分配失败，不代表无限并发可用 |
| session refresh age 5s / TTL 30s | 新鲜私有 session 可跳过重复刷新，陈旧/共享 session 保持原校验 |
| `ASYNC_LOOKUP` / `BATCH_EXISTS` / `LINEAR_SCAN` / `LAZY_PUT_METADATA` | 在本次固定 Spark 的 Python 源码中没有对应 `SGLANG_MOONCAKE_*` 环境变量读取点；不能据脚本赋值推断启用。当前真实调用是 `batch_is_exist` |

原有 native read-plan、整对象预取、统一 pinned buffer、批量 scatter、region cache、lease-age shortcut 都已生效或有对应调用证据；本 PR 不把这些已有优化算作新增成果。

## 按波次、逐 rank 的边界计时

先用临时 probe 在既有方法和 native API 边界记录 wall time 与 thread CPU time。窗口取响应 wall-clock 起止；仅纳入窗口内完整调用。方法存在嵌套，跨线程会重叠，不能把表中各行相加；`plan.wait` 与 `plan.run` 描述同一次加载，尤其不能重复计数。

有 probe 的波次 PC95 为 149,597、PC90 为 90,180 input tokens/s，与原组合接近；probe 仍可能扰动调度，下面不是无开销 profiler。

| 阶段，每个 P rank 的累计 wall time 范围 | PC95 波次（19.87s） | PC90 波次（33.42s） |
|---|---:|---:|
| `match`（含查找、hash、集合同步） | 0.921–0.939s | 1.001–1.022s |
| `_sync_restorable_prefix`（含 collective） | 0.134–0.165s | 0.162–0.202s |
| session start native API，48 次 | 0.541–0.572s | 0.525–0.615s |
| `prepare_load`（含 session start、锁等待与 key 准备） | 0.844–0.892s | 1.055–1.188s |
| native DFS prefetch，48 次、4 worker | 29.10–29.66s | 26.34–26.62s |
| `load_back`（含 revalidate、claim、device/tree 操作） | 1.316–1.326s | 1.609–1.620s |
| native session refresh | 0.730–0.762s，26 次 | 1.009–1.036s，36 次 |
| Python read-plan layouts / locations | 0.136–0.181s | 0.199–0.234s |
| native `create_read_plan` | 0.010–0.014s | 0.030–0.041s |
| native `plan.run`（含展开、读取/搬运、校验） | 1.028–1.583s | 1.149–1.632s |
| `batch_set_v2` 后台写回 | 2.107–2.523s | 4.003–4.966s |

预取累计时间超过波次 wall time 是因为四 worker 并行，不能解释成多等了 29 秒。PC90 每个 prefetch native 调用中位约 0.59–0.61s；排队、Python 准备和跨 rank 一致性不包含在这一个 API 时长里。

原最终两轮日志还确认：

- PC90 四 rank 总 DFS 预取/成功消费均为 **104,639,496,192 bytes**；PC95 为 **110,301,806,592 bytes**。恢复阶段 `dfs_read_bytes=0`、`fallback_dfs_h2d_bytes=0`。
- PC90 内层 scatter sync 累计 2.15–2.47s、submit 0.57–0.98s（四 rank 合计）；外层 range API 总计 4.51–4.64s。外层包含内层，不能相加。
- 约 319 万 / 337 万 range 在该布局下没有合并；是源/目的连续性约束，不说明已有 merge 开关没有执行。
- PC90 pointer-query 总耗时只有 8–14ms（四 rank 合计）；外层 `gc_us` 约 0.35–0.36s，计时包含获取 session mutex 的等待和 orphan-buffer 收集，不能把它全当 GC 执行，也不能据 range 数量认定指针查询是主瓶颈。
- P 日志报告的 KV transfer bytes/s 反推区间：PC95 平均约 100ms、最大 173ms；PC90 平均约 59ms、最大 112ms。它不含 bootstrap/排队，`forward_duration` 又一直计到 completion，不能当作纯 GPU 计算时间。
- DFS 是 ParaStor POSIX 路径，未清操作系统/存储缓存；这里测的是 DFS API/backend 读取，不能称为裸磁盘冷读带宽。
- backend 每请求 966 / 916 个对象被合并为约 146 / 138 个 read tasks，合并读取已有实际效果。192 次后台读取的 prepare 中位约 3.33ms，read 中位约 636 / 591ms；read 包含线程池任务执行/等待，不等于单个 `preadv` 系统调用时长，也不含 SGLang 的请求队列时间。

## 本轮改动和因果对照

新增两个 SGLang 直通元数据优化：

1. `MooncakeStore.batch_exists_v2` 的 trailing-window 判断从逐边界重扫窗口改为滚动统计缺页数，复杂度从 O(pages × window) 降到 O(pages)。保留所有稀疏可恢复停点、头部短窗口以及每页多物理对象的要求。
2. `_sync_restorable_prefix` 一次 indexed assignment 构造 CPU mask，避免对每个候选调用一次 tensor 标量赋值。仍执行同一次 MIN collective，保留稀疏集合交集、越界过滤和 device prefix 边界。

没有减少 Master 查询、放宽 session TTL，或用每 rank 最大值代替交集。没有将禁写回、layer pipeline 或诊断 probe 加入生产改动。

以下组保持同一服务进程、模型/kernel、32K 预算和输入分布，按 control → metadata → no-write → layer-pipeline → no-write → metadata → control 顺序运行；第二轮 PC 顺序反转。no-prefetch 需要另启服务，与其相邻的有 probe 波次也复现了接近 control 的结果。每个服务先跑完整 48 请求预热，不计入正式数据。

| 配置 | PC95 两轮 input tokens/s | PC90 两轮 input tokens/s | 相对 control 的合并吞吐 |
|---|---:|---:|---|
| 关闭队列预取，仍从 DFS 恢复 | 107,304 / 107,097 | 73,133 / 73,253 | −28.22% / −18.56% |
| control：原有直通路径 | 149,918 / 148,776 | 89,891 / 89,851 | 基准 |
| 仅上述两处 metadata 优化 | 150,060 / 150,461 | 89,966 / 89,395 | **+0.61% / −0.21%** |
| 仅禁后台写回，诊断组 | 153,135 / 154,917 | 93,620 / 93,149 | +3.13% / +3.91% |
| layer-wise + 真正启用 pipeline，仅一轮 | 137,967 | 86,273 | −7.62% / −4.00% |

合并吞吐按总 tokens / 总 elapsed 计算，未取每轮吞吐的算术平均。control 合并为 149,345 / 89,871，metadata 为 150,260 / 89,679。没有从这些小差值认定稳定端到端增益，尤其不能把局部函数加速写成整机加速。

开启预取相对关闭预取的合并吞吐约提升 **39.31% / 22.79%**。这是现有预取路径在本负载下的收益，不是本 PR 新增收益。禁写回改变缓存写入语义，仅用来估计竞争量级；不能作为生产优化方案。layer-pipeline 日志在四 rank 均确认 `enabled=1 groups=36`，本轮没有保留它。

去掉全部诊断 hook、将两处实际方法补丁应用到 Spark 原路径，再启动干净服务复测：**PC95 150,034 / PC90 90,066 input tokens/s**，相对唯一纯算为 **5.64× / 3.38×**。缓存命中与四 rank 来源校验均通过；3 条提示各生成 128 tokens，与原 Triton 冷恢复结果逐 token 一致。原 page-wise、预取和写回均已恢复，P/D/router health 均为 200。该单轮验收不改变“未证明稳定端到端收益”的结论，也不代表目标分支真模型性能验证。

### CPU 微基准与稀疏性

在同一容器、目标分支源码中，对实际函数使用零 RPC 的固定存在性结果测局部 CPU 开销。每项预热 5 次，11 组、每组 30 次调用，报告中位。mask 项不执行 collective。

| 函数 / 形状 | 原版 → 优化版 | 局部倍率 |
|---|---:|---:|
| trailing 判断，900 pages、8-page 窗口、稀疏窗口快照 | 1.485 → 0.921ms | 1.61× |
| trailing 判断，950 pages、8-page 窗口、稀疏窗口快照 | 1.551 → 0.909ms | 1.71× |
| trailing 判断，1000 pages、64-page 窗口、全命中 | 4.563 → 0.816ms | 5.59× |
| trailing 判断，8000 pages、512-page 窗口、全命中 | 236.128 → 6.821ms | 34.62× |
| mask，950 pages、4 候选 | 0.0422 → 0.0370ms | 1.14× |
| mask，950 pages、8 候选 | 0.0645 → 0.0386ms | 1.67× |
| mask，950 pages、15 候选 | 0.0989 → 0.0409ms | 2.42× |
| mask，950 pages、950 候选，全密集 | 4.727 → 0.277ms | 17.04× |

**实际日志里 PC95 只有 5–15 个可恢复停点，PC90 为 4–13 个，并非 900/950 个候选。** 大窗口/密集 micro 是扩展形状，不代表当前工作负载。实际稀疏 mask 只节省几十微秒，trailing 判断节省不到 1ms/请求，因而端到端几乎不变是合理结果。不能用 17×/35× 局部数字解释或冒充用户要求的整机 15×/9×。

### 单元测试

在 `.29 / zhy-4B` 的独立 `feature/optimize_main` 源码运行：

- 新增 2 个方法：遍历长度 0–6 的全部存在性掩码、4 种窗口、1/2 物理 component（1,016 个组合）；另测 5 个跨 rank 集合交集场景，含空集合、重复、越界、device frontier 和 rank maxima 不可直接合并。
- 原版与优化版均通过新增测试，独立暴力窗口参考保持不变；故意把移出窗口条件从 `>=` 改成 `>`，**184 个子场景失败**，恢复后通过。
- 定向 read-plan、waiting-queue/no-prefetch-needed、session-refresh、新增 lookup 回归：**64 项通过，1,040 个子场景通过**。
- 扩展加入 load-failure 全文件：原版/改动版均 **147 通过、相同 13 项失败**；失败集合逐项对比一致，主要为既有 fixture 缺少 `_waiting_queue_prefetch_enabled` 等状态。本 PR 未顺带修复这些无关 fixture；不能称全套回归全绿。
- 适用 pre-commit 检查通过。全仓 CI 注册检查仍因原有 `test_mamba_direct_linker.py`、`test_mooncake_dfs_health.py` 未注册失败；新增文件已注册 CPU CI。

原 SWA kernel 的 BW 数值/HND 测试仍见前一份记录，本轮不重复跑 kernel 测试。

```bash
PYTHONPATH=/path/to/checkout/python python3 -m pytest -q \
  test/registered/unit/mem_cache/test_mooncake_prefix_lookup.py \
  test/registered/unit/mem_cache/test_mooncake_direct_linker_read_plan.py \
  test/registered/unit/mem_cache/test_mooncake_waiting_queue_no_prefetch_needed.py \
  test/registered/unit/mem_cache/test_mooncake_session_refresh_age.py
```

## 瓶颈判断与下一步

**已验证：直通路径没有“没成本”，原有预取也不是无效。** 元数据、session refresh、host→GPU readiness 和写回都有可测成本。需要同时看 foreground critical path 与后台竞争，不能根据 GPU attention 占比或显存命中一组就排除 Mooncake。

**当前证据不足以支持某一项单独解释距 15×/9× 的差距。** 这两个目标需要波次分别缩短到约 7.70s / 12.83s，现有约 20.5s / 34s；几十毫秒的指针查询和 read-plan constructor 不具备这个量级的解释力。消除或重叠更大阶段仍要考虑正确性、显存容量和 batch 组成。

| 后续方向 | 证据与边界 | 验证要求 |
|---|---|---|
| SGLang ↔ Mooncake 会话批处理 | 每波 26/36 次刷新，约 0.7–1.0s/rank；Python session lock 内含 start RPC | 按同一入批批次合并 RPC；逐 key 保持租约、共享引用和失败回退一致，不能单纯调大刷新跳过年龄 |
| 写回批量化与节流 | 后台 put/metadata 可与前台争用 CPU、设备和带宽；累计时间不等于可回收时间 | 用禁写回诊断确定收益量级，再测有写入语义的 pacing/batching；禁写回本身不交付 |
| 已预取 native 恢复路径 | 已有 host bytes 仍进入对象分块、metadata 展开和逐 chunk 同步 | 只对完整有效 prefetch buffer 设计快路径；缺失、租约失效和 fallback 仍保留分块预算与完成后发布语义 |
| layer / batch 重叠 | page-wise 确实阻止该批 H2D 与其 attention 重叠，但 finer-grain API 也增加 CPU/同步开销 | 先验证真正进入 pipeline，再看全波吞吐；不能仅凭时间区间不相交就宣布有收益 |
| 逐 rank 计算、通信与调度 | 旧 trace 中 TP0 attention 较慢，其他 rank collective 较长；独立逐卡微基准无持续卡0劣化 | 全波同形状对齐，区分上游等待、真实通信、CPU 提交与 GPU 执行；不把 Python GPU 同步等待称为 Python 计算 |
| DFS backend 读/排队 | 预取读有实际耗时且能提高吞吐，当前恢复不重新读；worker 增多没收益并不能排除 session 锁或 backend 排队 | 低并发/冷缓存/竞争负载分别测 backend prepare/read、inflight bytes、队列和有效带宽；本轮不外推到这些场景 |

## 证据与复现

实验脚本：`run_direct_probe.py`、`run_no_prefetch.py`、`run_direct_ablations.py`；临时 probe 为 `hook/direct_path_probe.py`。单测与局部微基准为 `direct-path-audit/run_lookup_tests.py`、`micro_lookup.py`。driver / 服务日志、逐请求响应、四 rank Prometheus 差分、临时 source backups 和 timing JSONL 均保存在实验目录。

本地整理目录：`direct-path-audit-20261116/`；关键摘要为 `native-timing.json`、`boundaries.json`、`request-stages.json`。不会把原始大日志、模型文件或访问凭据提交进仓库。
