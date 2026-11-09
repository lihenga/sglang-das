# sglang-model → sglang-das 代码同步记录

## 同步范围（2026-11-06）

本批次以上一批的源上界 `d3db5682d1560c4d707feb4d17a92edac1b3c91f` 为游标，按拓扑顺序审查至新源上界 `d0146097effd7f95308bad5184081f42d0b5dd1f`（`sglang-model` 分支 `reference`/`sglang_glm_dev`）的7个候选提交，目标分支仍为 `main_glm53_flash_feature_cleanup`，同步前基线为 `eb437264d345f892c0e4cf2bca4f14d5509dc119`。

后续补迁移 ReplaySSM 延迟提交后，5 个源提交全部或部分迁移，2 个跳过。

## 本批次内容

- 融合 DeepEP normal dispatch/combine 与 INT8 MoE 核心的 ep_scatter/ep_gather，默认关闭，按部署条件门控。
- KDA Triton kernel 新增 aiter decode/verify 后端切换（`SGLANG_KDA_USE_AITER_DECODE`/`SGLANG_KDA_USE_AITER_VERIFY`）。
- `CommonKVManager._connect` 修复：监听 `EVENT_CLOSED`（不仅 `EVENT_DISCONNECTED`），首次连接失败的 ZMQ PUSH socket 不再被永久缓存复用，带有最小重连间隔。

## 本批次测试对比（2026-11-08 至 2026-11-09）

同步后代码为 `ee93d2a19`。D-only 性能测试使用 W8A8 INT8，输入 87,040 token，共享前缀 84,992 token，输出 1,024 token，并发 128，固定接受长度 4。每侧三轮，每轮 384/384 请求成功。精度使用 E/P/D、自然接受长度，AIME25 和 HumanEval 各运行一次，采样参数为 seed 42、temperature 1、top_p 0.95、输出上限 64,000 token。DeepEP 融合和本次新增的可选 KDA AITER 路由保持默认关闭。

| 指标 | 同步前 → 同步后 |
| --- | --- |
| 三轮平均输出吞吐（token/s） | 6243.91 → 6290.30（+0.74%） |
| 三轮平均 TPOT（ms） | 18.81 → 18.93（+0.64%） |
| AIME25 | 26/30 → 27/30 |
| HumanEval pass@1 | 164/164 → 163/164 |

两侧各 16 份 rank trace 均可解析。按各 rank 单次平均耗时取中位数，DeepEP dispatch 为 122.73 → 113.88 µs，HCU KDA 为 153.44 → 153.46 µs。该对比只有每侧一次 profile，DeepEP dispatch 的差值无法归因于默认关闭的融合。两项数据集各只跑一次，单题变化不足以确认稳定的精度变化。

## 运行行为变化

| 项目 | 本次行为 / 保留项 |
| --- | --- |
| `SGLANG_DEEPEP_DISPATCH_SCATTER_FUSED` | 默认关闭；开启后需 a2a=DeepEP、非TBO、非internode（≤8 ranks）、非MNNVL、`SGLANG_GROUPGEMM=1`、INT8 deepgemm MoE 路径，任一不满足则警告并回退到现有 ep_scatter。 |
| `SGLANG_DEEPEP_COMBINE_GATHER_FUSED` | 默认关闭；另需 topk ≤ 64（ROCm warp size）且 NVL buffer 的 fused-combine 行容量不小于 `chunked_prefill_size / attn_cp_size` 的每-rank批量，不满足则警告并回退到现有 ep_gather。 |
| `SGLANG_KDA_USE_AITER_DECODE` / `SGLANG_KDA_USE_AITER_VERIFY` | 默认关闭；开启后 `TritonKDAKernel.decode`/`target_verify` 走 aiter `sglang_fused_sigmoid_gating_delta_rule_update`，需安装 `aiter` 包。 |
| `SGLANG_DISAGGREGATION_ZMQ_SOCKET_RECONNECT_BACKOFF_MS` | 默认 1000ms；`CommonKVManager` 缓存的 PUSH socket 首次连接失败后，在这段时间内 `_connect` 抛 `zmq.Again`而不返回旧 socket，超时后重建。 |

## ReplaySSM 延迟提交对比（2026-11-09）

D-only 使用同一 `sglang-das` 版本，输入 87,040 token，输出 1,024 token，并发 128，固定接受长度 4。开关各测一轮，均为 384/384 请求成功。两组都设置 `SGLANG_KDA_USE_TILELANG_VERIFY=1`、`SGLANG_KDA_USE_AITER_VERIFY=0`，开启 ReplaySSM spec，使用 `--disable-radix-cache` 和 `--mamba-radix-cache-strategy no_buffer`。唯一切换项为 `SGLANG_KDA_REPLAYSSM_DEFER_COMMIT`，默认关闭。

| 指标 | 延迟提交关闭 → 开启 |
| --- | --- |
| 输出吞吐（token/s） | 6386.46 → 6458.44（+1.13%） |
| 平均 TPOT（ms） | 18.27 → 18.09（-0.99%） |

两组各 16 份 rank trace 均显示每 rank 340 次 TileLang verify，独立 commit 从 10 次降至 0 次，新增 10 次 pending 记录。按各算子 GPU 总时间的 rank 中位数相加，verify 与 commit 或 pending 合计 51.38 → 41.67 ms。DCU 上 FP32 和 FP16 checkpoint 对齐测试均通过。端到端结果来自单轮计时。

### 延迟提交精度对照

使用 E/P/D Router 和自然接受长度，沿用上面的 Decode 开关约束。每侧各跑一次，采样参数为 seed 42、temperature 1、top_p 0.95，输出上限 64,000 token。AIME25 为 26/30 → 25/30，HumanEval pass@1 为 162/164 → 163/164，顺序为关闭到开启。AIME25 两组各三题触及输出上限，其中两题相同；完整但错误的答案由一题增至两题。HumanEval 两组同一道词排序题使用了方向相反的选择逻辑，关闭组另有一题触及输出上限。 单轮每项相差一题，无法确认稳定的精度变化。

## 逐提交账本

| 源完整 SHA | 原主题 | 结论 | 目标提交 | 依据与适配说明 |
| --- | --- | --- | --- | --- |
| `7c5d4a7bf94c076a99078c2f4fd206c205889f45` | perf(moe): fuse ep_scatter into the DeepEP dispatch | 迁移 | 18798158c | 目标 `DeepEPMoE`/`DeepEPDispatcher`/`MaybeTboDeepEPDispatcher` 与源结构几乎一致，直接移植 `enable_dispatch_scatter`/`take_dispatch_scatter`/`fusion_unsupported_reason` 和 `_ep_moe_workspace_base`。 |
| `f6b08b48ea8b4f96216976d79fed6c0ecb98e4d9` | perf(moe): fuse ep_gather into the DeepEP combine | 迁移 | 18798158c | 与上一条合并迁移；`DeepEPBuffer.normal_nvl_bytes`/`combine_gather_capacity`、`set_combine_gather`、`_MAX_GATHER_TOPK` 等均直接移植。 |
| `f2b71915a43fac3202b70746ada30ed6ffb2436f` | Update sglang_fused_sigmoid_gating_delta_rule_update kernel. | 迁移 | 55c8f0732 | 开关只作用于目标 `TritonKDAKernel`，独立的 `HcuKDAKernel` 路径不受影响，按源逻辑添加 aiter 分支和日志；`beta_scale` 在目标 `kda_backend.py` 已是带默认值的 `**kwargs` 传递参数，无需调整调用方。 |
| `e03e6c397bcd883e446092815ed585f341349873` | fix: recover stale ZMQ sockets with bounded reconnect backoff | 部分迁移 | 79e11787a | 源的 lease 式 `ZMQSocketCache`（`disaggregation/common/utils.py`）在目标不存在；目标 `CommonKVManager` 用 `_socket_cache`/`_monitor_cache` 字典 + `_connect` 方法。移植源的核心修复思路（监听 `EVENT_CLOSED`、有界重连间隔）到该方法，新建 `_socket_unreachable_since` 字典替代源的 `_SocketEntry.unreachable_since`。源的容量公式、`_SocketEntry`/`_quarantine`/`retired` 生命周期、`_is_stale_locked` 等基于共享缓存池的机制不适用于目标每-manager 独立字典结构，全部省略。源测试的容量/陀离区场景不适用，改为新写 `test_common_kv_manager_connect.py` 针对目标 `_connect` 的场景。 |
| `e27af720aa50318de4b3e8b2329f522f77da9859` | fix(hicache): preserve HiMamba lock ownership during eviction | 跳过 | — | 源修复的 `HiMambaRadixCache`（`mamba_lock_ref`/`full_lock_ref` 双树、`mamba_lru_list`、`ongoing_write_through: dict[int, (node, params)]`）在目标不存在。目标已重构为组件化的 `UnifiedRadixCache`/`MambaComponent`（`unified_cache/components/mamba_component.py`），用 `component_data[ComponentType.MAMBA].lock_ref`/`skip_lock_node_ids` 机制表达等价语义，没有源那个需要 `mamba_lock_ref_acquired` 标记的 Full-only 预锁问题。`registry.py` 对 hybrid-ssm + hierarchical-cache 无条件走 `UnifiedRadixCache`，没有源的 `HiMambaRadixCache` 分派分支。未尝试将源的修复改写到组件架构上，因为无法验证等价性；若未来重新引入类似的 `HiMambaRadixCache` 双树架构应重新审查本条。 |
| `0d5f5bcbd20a9b399b0e33b1c4d4d057ef9c5678` | fix(hicache): harden locked leaf eviction lifecycle | 跳过 | — | 与上一条同因：本提交在源 `HiMambaRadixCache`/`MambaRadixCache` 双树上进一步强化锁生命周期（`req.lock_ref_params`、`DecodePreallocQueue` 释放路径、`StreamingSession` 会话恢复），依赖上一条的 `mamba_lock_ref_acquired`/`mamba_full_lock_protected` 机制。目标 `req.swa_uuid_for_lock`/`skip_lock_node_ids` 已是通用的跨组件锁归属表达方式，`DecodePreallocQueue._match_prefix_and_lock` 和 `MambaRadixCache`/`RadixCache`/`SWARadixCache`/`UnifiedRadixCache` 各自维护自己的 `inc_lock_ref`/`dec_lock_ref` 配对，没有源那种跳过的风险点。未找到可独立移植的残留问题；若上一条被重新引入应重新审查。 |
| `d0146097effd7f95308bad5184081f42d0b5dd1f` | perf(kda): fold ReplaySSM commit into the next TileLang verify | 适配迁移 | 55f564108 | 将 `sglang-model` 的延迟提交接入 `sglang-das` 的 HCU KDA 路径。按物理槽位记录 pending，下一轮 TileLang verify 折入状态，普通 Decode、Extend 和状态拷贝前物化。开启后 verify 无法走 TileLang 时显式报错。 |

本批次源上界 `d0146097effd7f95308bad5184081f42d0b5dd1f` 作为后续同步的新游标。

---

## 同步范围（2026-10-29）

本批次审查 sglang-model 自 2026-09-15 起、截至 `d3db5682d1560c4d707feb4d17a92edac1b3c91f` 的26个候选提交。目标分支为 `mx_int8/main_glm53_flash_feature_cleanup`，迁移前基线为 `975bec73b6e167e8512fcb1e743332eaf84f9301`。同步采用线性追加提交，保留目标分支已有功能与差异化实现。

## 本次提交映射

| 上次同步提交 | 本次目标提交 | 内容 |
| --- | --- | --- |
| 2cec40bb28 | 98c2c41aa5 | HiCache host Mamba槽位指标 |
| abc104c533 | 18d8d0b351 | fake-bootstrap radix显式开关，默认关闭 |
| 9fb5493552 | bf16476aa5 | compressed-tensors预量化输入能力和消费修复 |
| 072def91a7 | 2b6473f6c7 | HCU ReplaySSM fp16 checkpoint支持 |
| 5f9525641d | d234f1eff5 | 可选TileLang verify及安全修复，默认关闭 |
| 9ab9cc0cc9 | 8bca770525 | PREBUILT pinned/async H2D及allocator默认false |
| 36933d9a4e | 2b41790339 | uniform共享前缀round-robin，保留ordered/Zipf和warmup剔除 |
| d1b554a679 | d21a9fde8d | failed-prefill manager socket发送排空后清理 |
| de2b688e33 | 8487c9b82a | 目标接口适配及9项portable CPU测试 |
| 60f425ec3b | f91efef42a | Mamba descriptor异步H2D及2项GPU回归 |

新基线相对旧TX基线恢复了18个文件、约1300行功能，相关文件本次保留：GLM DSA decode_input/replay_metadata/quant_k_cache/gate_scale、packed input metadata、MoE topk、draft LM-head VP及其参数验证等。源b878794235的核心模块已由目标173dbf5270/33f11fd33a恢复；该条由旧账本“跳过”改为“已有覆盖/保留目标差异”。不重复覆盖整文件。源KDA旧backend和MTP metadata共享协议仍不同，不跨target/draft池别名共享对象。

## 测试与对比结果（2026-10-29）

对比基线为 `975bec73b6e167e8512fcb1e743332eaf84f9301`，同步后版本为 `eb1714844453cc23f8b43edd0cd0c820c2428a4d`。

### 功能验证

- 10个代码与测试提交已应用，无冲突；22个变更Python文件AST解析及差异空白检查通过。
- portable CPU测试9/9、量化接口测试3/3、TileLang GPU测试8/8、Mamba descriptor GPU测试2/2通过。量化接口测试使用mock GEMM，不代表完整量化数值验证。
- 两侧真实PD/HiCache的16并发请求验证通过；194个精度测试题目逐一匹配。

### 精度对比

采样参数一致：seed=42、temperature=1、top_p=0.95、max_tokens=64000；每侧各运行一次，不复用预测缓存。

| 数据集 | 同步前 | 同步后 | 变化 |
| --- | --- | --- | --- |
| AIME25 | 25/30（83.33%） | 27/30（90.00%） | +6.67个百分点 |
| HumanEval | 164/164（100%） | 164/164（100%） | 持平 |

AIME25同步前5个失败样本均达到64000输出token且未产生最终答案，部分存在反复验算；同步后其中3题转为正确、2题仍截断，另有1题由正确转为组合计数错误（答案95，标准答案81，错误地将块间选数视为独立并自动满足行约束）。HumanEval两侧均无失败样本。单次随机采样净增2题，不能据此认定同步带来了确定性的精度提升，也不能将失败归因于某条提交或量化误差。

### 性能对比

两侧固定模型、依赖、硬件、启动配置及同步前版本的压测客户端，保留runtime fast path和draft LM-head VP=8。公共descriptor兼容补丁在两侧一致应用，不计入同步收益；allocator采用各版本默认值，其默认行为变化包含在对比中。

工作负载为D-only fake-prefill：输入81920 token、共享前缀79872 token、输出1024 token、并发128。每侧3轮，每轮384个正式请求，预热剔除；六轮均384/384成功，每轮生成393216 token。

| 指标 | 同步前 | 同步后 | 变化 |
| --- | --- | --- | --- |
| 平均输出吞吐（token/s） | 5251.83 | 5210.71 | -0.78% |
| 平均TPOT（ms） | 13.55 | 13.57 | +0.15% |
| 平均TTFT（ms） | 934.32 | 1071.82 | +14.72% |
| P99 TTFT（ms） | 6029.64 | 8003.24 | +32.73% |
| P99 TPOT（ms） | 31.20 | 33.19 | +6.37% |
| P99 ITL（ms） | 62.13 | 57.30 | -7.77% |

表中指标均为三轮结果的算术均值；P99为三轮各自P99的均值，不是合并请求后的分位数。

| 轮次 | 同步前吞吐（token/s） | 同步后吞吐（token/s） | 同步前TPOT（ms） | 同步后TPOT（ms） |
| --- | --- | --- | --- | --- |
| 1 | 4948.22 | 5481.78 | 13.16 | 12.76 |
| 2 | 4703.74 | 5265.06 | 13.13 | 13.64 |
| 3 | 6103.52 | 4885.28 | 14.37 | 14.32 |

算子profile两侧各采集16个rank，全部解析成功。按rank、kernel、grid和block匹配，同调用次数的W4A8 GEMM与delta-rule耗时中位变化分别为-0.02%、+0.03%，基本持平；DeepEP dispatch为-12.79%，combine为+6.42%，同步API调用次数一致。

本轮未证实稳定性能收益：吞吐均值变化小于轮间波动，TPOT基本持平，TTFT均值及部分尾延迟变差。短profile和每侧三轮测试不足以将通信耗时变化归因于具体同步提交。

## 同步规范

1. 固定源上界、目标起点和分支后枚举候选。区分 author date 与 committer date，按源拓扑顺序分析依赖；跨仓库 SHA 不等不能当作未覆盖的证据。
2. 先找目标的同名/改名/拆分实现、调用者和测试，再判定迁移、部分迁移、已覆盖、跳过。文件路径不同不等于功能不存在；目标已主动删除的路径保持删除。
3. 适用补丁使用 `cherry-pick -x`；重构适配在提交正文记录完整 `Source-commits` 或源 SHA。依赖安全修复可以合并。追加提交保持线性，不穿插两仓库原历史。
4. 冲突按目标结构解决，保留目标独有模式、配置入口和数据格式；部分迁移逐项说明跳过内容。避免整文件采用源版本，尤其 GLM/DSA、encoder、runtime-context、KV allocator 和量化分派。
5. 每个源 SHA 在下表只有一条结论。跳过表示对应源路径未同步，不表示目标不存在同类问题；如果后续引入该功能，应重新审查相关跳过条目。
6. 区分静态/CPU/GPU/服务端验证。没有实测时不把源 commit message 中的性能或精度数字写成目标验证结论。

## 逐提交账本

以下证据路径均相对于目标 `python/sglang/srt/`，以 `kernels/` 开头的除外；测试位于 `test/registered/`。目标 SHA 指本同步分支中的提交，可配合最后的验证修正提交阅读。

| 源完整 SHA | 原主题 | 结论 | 目标提交 | 依据与适配说明 |
| --- | --- | --- | --- | --- |
| `7b10e8c737b86634fe1d07b31856bb26c9b362bc` | fix(nsa): avoid Q quantization JIT specialization by row count | 跳过 | — | 目标 GLM kpool 没有 `_quantize_q_rows`；其 Q/索引量化实现不同。保留 `layers/attention/glm5_next/kpool/kernels.py`，不恢复源 INT8 Q kernel 或已删除测试。 |
| `b878794235c9a29b3c7c604d4ce95b671cb54bda` | perf: 融合 NSA/MoE 输入准备算子并减少 H2D/D2D 搬运 | 已覆盖/保留差异 | 173dbf5270 / 33f11fd33a | GLM DSA decode_input/replay_metadata/gate_scale/quant_k_cache及packed input metadata已在目标基线恢复；保留目标KDA分派与池翻译。 |
| `76747d5adab2bdb077b2c6ebf26a4b97bed90f31` | fix(epd): release prefill RDMA buffer per-part to avoid pool exhaustion | 跳过 | — | 源旧 receiver 的“克隆每个 part 后立即释放 RDMA buffer”路径不存在；目标 `encoder/receiver.py:GlmWaitingRDMARequest` 将 GPU view 生命周期绑定到 mm_inputs。直接套用会提前释放仍被模型使用的存储。 |
| `b822c05fdf33ed6eb8f88d572f776f05b218af44` | fix(disaggregation): stop unsent encoder parts after abort | 已覆盖 | 目标现有实现 | 目标 `encoder/receiver.py` 的 abort 通知、`encoder/transfer_queue.py` 的 abort registry/队列清退及 `test_encoder_abort.py` 已处理未发送任务取消和运行任务排空；保留目标实现。 |
| `71a95a45af9b1240ed5c42094c8014ba85aa8356` | feat(observability): export host mamba pool used/total slot gauges | 迁移 | 98c2c41aa5 | 指标写入 `observability/metrics_collector.py`；采样位置改为 `managers/scheduler_components/metrics_reporter.py:_log_hicache_stats`，缺少 Mamba host pool 时清零。 |
| `ce99439c2e8d9e23003efd8dd96a120ae631b382` | fix(disagg): let stub-bootstrap benchmark requests enter the radix cache | 迁移 | 18d8d0b351 | 保留目标 rebootstrap 字段及 pinned-array 导入，在 `Req` 中增加默认关闭的 fake-bootstrap radix opt-in。 |
| `4b98a3bba25d622a2fe1ebfb035f07e1246bc69f` | MTP 复用 NSA DSA init_forward_metadata | 跳过 | — | 源 NSA/Mamba metadata 的共享路径不存在。目标 GLM 使用 DSA backend 与 in/out-graph metadata，Mamba 槽位有目标池翻译；不跨 target/draft 池直接共享源 metadata 对象。 |
| `f03b2ad3574e86a44174fd590abf303b961c3c85` | feat(kda): add env-gated TileLang target_verify kernel (odd-S capable) | 迁移 | d234f1eff5 | 与后续安全修复合并移植；目标入口为 `linear/kernels/kda_hcu.py:HcuKDAKernel.target_verify`，kernel 位于 `kernels/ops/attention/fla/hcu/`。 |
| `3d08bb73b8df5025f6bed3ec7731ce1aee755d66` | fix(kda): review fixes + stride/fp16 perf for the TileLang verify kernel | 迁移 | d234f1eff5 | 保留 padding/stride/fp16/fallback 修复和预编译；目标开关为 pool 的 `use_hcu_kda_replayssm`，capture API 适配见 `8487c9b82a`。 |
| `8fcefa0916bcb9ba1db61db5352ba94e13b8b86a` | perf(encoder): dedup the EPD decode-side multimodal preprocess | 跳过 | — | 目标没有源 content-key encoder cache/probe、`mm_content_key.py` 与对应批处理协议；目标 encoder 已拆为 runtime/preprocessor/server，保留当前协议。 |
| `2387a27403f2aff1834266b35bad0c8374f90af6` | fix(mm): port upstream canonicalize/rendezvous + content-hit guard + scheduler seed | 跳过 | — | 依赖上一条缺失的 content lookup，包括 URL canonicalize、rendezvous/content-hit grid guard；不引入孤立开关或旧 receiver/server。 |
| `2dc443654559859c9f547d27f33f654d90a82692` | feat: support w4a8 deepgemm. | 已覆盖/保留差异 | 目标 241207e321 | 目标已有 SlimQuant W4A8 DeepEP/DeepGEMM 与 HCU compressed-tensors W4A8；保留其 checkpoint 格式与 scheme 分派。源独有 `w4a8_int` compact-config 别名及旧 MoE scheme 不覆盖目标。 |
| `7b361c36b84d1c43aad4455adf2ff61d35ee3a88` | Supports both startup parameters '--enable-kda-replayssm-spec' and '--mamba-ssm-dtype float16'. | 迁移 | 2b6473f6c7 | 参数 hook 已允许低精度并告警；移除 HCU commit kernel 和 MambaPool 中剩余 fp32-only guard，保留 FP32 g/beta 和累加。 |
| `155d946de36506c568f8feb424779674a2990409` | fix w4a8 kernel fusion issue. | 迁移 | bf16476aa5 | 按具体 scheme 公布 prequantized-input 能力，消费已有 INT8/scale；保留目标 LM-head/KV scheme 与签名过滤。通用 factory 修正见 `8487c9b82a`。 |
| `a19c0ab7d760e16246150be1d78ddffc2e01496a` | perf(pd-decode): remove host syncs that stall kernel launch on batch-size change | 部分迁移 | 8bca770525 | 移植 PREBUILT 小张量 pinned/async H2D 与 KVALLOC 默认关闭，并适配微基准路径。目标 sampling 已异步；源 `new_seq_lens_cpu`/SeqLensCpuView filter/merge 流程已被 relay 管线替代，跳过该部分。 |
| `7e29940662c6078483c6287a354ad7633468c297` | fix(pd-decode): keep a full MTP seed for fake-prefill requests | 跳过 | — | 目标 Req/FutureMap 不使用源 `mtp_topk_indices_tensor`/`mtp_topk_indices_dim` PREBUILT seed 协议，而由 spec algorithm/relay 构造输入；不添加失效字段。 |
| `8c2af29cb973c8d0a06da48f895a95a91e2d880b` | bench: interleave shared-prefix requests and validate warmup | 部分迁移 | 2b41790339 | 默认 uniform 新数据 round-robin，warmup 取首批真实请求分波发送；保留目标 ordered/Zipf、warmup 剔除及至少一条正式请求的边界。源 auto_benchmark.py/lib 不存在，跳过其 output-length 工具改动。旧数据缓存保留原顺序。 |
| `b41bbf6680c6bf6f423c38d0d40ae1aecf1f8258` | fix(encoder): avoid decoding video pixels for metadata | 跳过 | — | 目标不存在源 metadata-only video decode endpoint/调用链；当前 `encoder/preprocessor.py` 走自己的媒体预处理与 shard 元数据流程。 |
| `b7c10b7f1aea5940eee79d29c85edc4925754694` | fix(encoder): bound CPU video decode buffers | 跳过 | — | 目标没有源 `_decode_indices_parallel`/`sample_video_sglang_sync` CPU bucket decoder，不能把源整段 GLM processor 覆盖回来。 |
| `011f5e65aa53ee8882161b04024de8dde50e547e` | fix(encoder): pin layout and tensor format for video metadata | 跳过 | — | 仅修复源 metadata-only meta/TCHW 输入，依赖 b41bbf6680 的缺失路径；目标没有该 meta 调用。 |
| `74c8e36b5e03e686c5bec431a16b972a7b40d533` | feat: add opt-in CP sharding and prefetch for MLA weights | 跳过 | — | 目标没有源 `zero3_attn_weight.py` / `distributed/attention_cp.py` 的 replicated-attention-weight sharding/prefetch 功能；保留目标 CP/GLM DSA 实现，不导入整套新权重生命周期。 |
| `2e249dcecd7835ee447fb538edc89c2fcf22de66` | perf: optimize CP attention weight prefetch with HIP IPC | 跳过 | — | 依赖上一条缺失的 ZeRO-3 attention-weight prefetch，HIP IPC 权重读取及其更新保护没有目标消费者。 |
| `edec1245bdac657da9c93db0b50485d872d7d6dc` | fix(disagg): close stale ZMQ sockets to a failed prefill on node failure | 部分迁移 | d21a9fde8d | 目标 common manager 已统一 Mooncake/NIXL node-failure 并清理 receiver sockets；补充 manager socket/monitor 清理及 send-lock 排空。源 lease-based `ZMQSocketCache` 不存在，保留目标字典缓存。 |
| `bf9db3629eccee8e1c2e41099c49f21ac89ac606` | fix(glm4v): require begin token before mm placeholders in EPD prompt rebuild | 已覆盖 | 目标现有实现 | `multimodal/processors/glm4v.py:build_input_ids_with_timestamps` 已要求完整 begin + modality token 锚点；保留无 begin-ID 时的兼容分支。 |
| `aa1732a1e998e5754faef40c2ffe5b3b150c69ba` | update deepgemm. | 已覆盖 | 目标 4c095c2130 | 目标已有 `m_grouped_i8_gemm_nt_masked` 与 `marlin_i8_masked_weight`，对应工具改名为 `hcu_deepgemm_w8a8_utils.py`。 |
| `d3db5682d1560c4d707feb4d17a92edac1b3c91f` | fix(epd): bound encoder host memory across request lifetimes | 跳过 | — | 针对源旧 encoder HTTP body admission、content-hit-before-reserve、cache storage accounting 与 `/send` 生命周期；目标没有这套 admission/cache 协议。保留目标 scheduler/runtime/transfer_queue，不声称已具备源全部 host-memory 限额保护。 |

