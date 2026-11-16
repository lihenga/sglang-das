# SWA extend attention 优化与 Spark 缓存性能记录

## 修改动机与范围

Triton `extend_attention_fwd` 的 SWA 路径从 KV tile 0 开始遍历，再通过窗口掩码跳过完全不可见的 tile。长前缀和较长 extend 下，这些循环仍支付索引、掩码和分支成本。

本改动在 prefix、extend 两段循环中，按 query tile 中最早的 query 计算第一个可能可见的 key，并向下对齐 `BLOCK_N`。保留边界 tile、原始掩码和归约顺序。无滑动窗口时下界仍为0。未修改 Mooncake、租约、预取准入或 attention 数学定义。

```text
原路径：tile 0 → 检查窗口 → 跳过无效 tile → 计算可见 tile
新路径：向下对齐最早可见 key → 保留边界 tile → 计算可见 tile
```

32K prefill 预算是实验启动参数，不是本 PR 对全局默认值的修改。实验计时 hook、FlashAttention 候选及环境兼容代码不进入本 PR。

## 性能证据的适用范围

以下端到端数据来自 **Spark 分支**，不是 `feature/optimize_main` 真模型性能验收：

- Spark `sglang-0518-mtp-sparkX25`：`78970d3bad84b9076ec2cb5d176dac29814f5cc3`。
- Mooncake `v0.3.14_dev_l4`：`f855c7e7cb26b2654a265b75d97d96bcecd0a445`。
- Spark-X2.5-4B BF16，8张 BW DCU；P TP4、D TP4，单机1P1D。
- 64,000输入tokens，48独立前缀，并发32，无发送速率上限，1输出token。
- PC95：60,800缓存+3,200后缀；PC90：57,600缓存+6,400后缀。64-token页对齐，每轮新后缀。
- P/D显存缓存每轮清空；4个P rank均验证精确命中、device命中0、memory读0、DFS读>0、预取读字节=成功消费字节、读取和分配失败0。
- 纯算关闭外部linker和radix cache，预热排除后只计时一轮：26,608.94 input tokens/s。吞吐包含缓存输入，不等于新增计算吞吐。

| 配置 | PC95 input tokens/s | PC90 input tokens/s |
|---|---:|---:|
| 原配置，两轮 | 146,985 / 148,040 | 86,663 / 86,764 |
| 原配置，后期相邻对照 | 118,309 | 86,279 |
| 仅 SWA 裁剪，两轮 | 119,163 / 145,365 | 88,482 / 87,796 |
| 仅32K预算，一轮 | 150,575 | 89,435 |
| SWA裁剪+32K预算，两轮 | 149,567 / 151,784 | 90,282 / 90,887 |
| 组合两轮总tokens/总耗时 | **150,667** | **90,583** |
| 相对唯一纯算基线 | **5.66倍** | **3.40倍** |

组合数据包含两个因素，不能全部归因于此 kernel patch。PC95原配置在不同时间段也有明显波动；共享 ParaStor 不是独占存储，不能把相邻组差值当成已证实的稳定27%收益。最初原配置与最终组合差距约2%（PC95）和4%–5%（PC90）。没有达到15倍/9倍目标。

单算子测试中，prefix512、query长度3200/65/6400、window511的样本约1.15→0.67ms；prefix60,800、query128/65样本约1.83→0.30ms。六组实际 BW GPU 对照逐元素相等；这不保证所有模型的端到端收益。组合的3条自然语言提示，各128生成tokens，相对原 Triton 恢复结果逐token一致，不替代模型质量评测。

128/512输出补测各一组；没有重跑相应纯算基线，故不计算其加速倍数：

| 输出 | PC | input tokens/s | output tokens/s | 平均流式TTFT(s) |
|---:|---:|---:|---:|---:|
| 128 | 95% | 137,524 | 275.05 | 10.059 |
| 128 | 90% | 84,760 | 169.52 | 16.586 |
| 512 | 95% | 87,763 | 702.10 | 7.081 |
| 512 | 90% | 72,309 | 578.47 | 14.054 |

## 后续瓶颈：分开看物理读取、搬运和计算

**已验证：** 原 Triton PC90 在48请求全部显存命中、DFS/L3读均0时只有99,769 input tokens/s（纯算3.75倍）；纯DFS约86.7K。因此，在现有计算和调度配置下，单独去除存储读取/搬运，仍不足以达到9倍。该诊断不是“所有IO优化的理论上限”。

已完成的DFS读取基本被队列等待部分覆盖。例如原配置PC90一轮：本地TP0预取提交→完成均值1.78s，完成→准入均值12.18s。前者包含准备、排队、同步及读取，不是物理磁盘耗时；后者也不是都可消除的框架开销，可能是等待计算资源。

Spark PC95、4个prefill step的 GPU trace：

| rank | attention累计ms | H2D scatter累计ms | all-gather累计ms |
|---|---:|---:|---:|
| TP0 | 3,084.65 | 771.54 | 278.24 |
| TP1 | 1,297.06 | 465.33 | 2,105.86 |
| TP2 | 1,300.70 | 462.07 | 2,102.39 |
| TP3 | 1,297.10 | 531.24 | 2,096.35 |

累计kernel时间可以跨stream重叠，不等于端到端占比。TP0 attention约63%累计kernel时间，但TP1–3约27%，不能把TP0单rank比例概括成全机结论。慢rank与其他rank的collective等待相伴，提示需区分实际通信和等待上游工作；尚未证明是硬件、拓扑或框架的某一项导致。相同attention形状逐卡独立运行，中位耗时77–78ms，无持续卡0偏慢的证据。

这份trace的H2D scatter与attention时间区间没有重叠，符合当前page-wise读完后才开放计算的结构。H2D属于Mooncake→设备的传输路径，不是物理DFS读。layer-wise对照未提升吞吐（PC95 111K、PC90 84K），故不能据此直接改成分层加载；细粒度调用、协调和正确性依赖也需计入。

后续优先级：

1. **SGLang attention计算和rank间负载/等待。** 新后缀仍需访问长历史KV；高PC不意味着成本按未命中tokens比例线性下降。采集同形状、完整预热、低扰动的逐rank kernel与collective区间，检查慢rank是否可重复，避免直接归因于网络。
2. **SGLang与Mooncake之间的H2D调度。** 检查预取缓冲到设备的scatter粒度、合并、跨batch/层重叠，同时保留租约与失败传播约束。分别量化scatter执行、等待、metadata，不能把它们都算成DFS读取慢。
3. **框架控制路径。** 原CPU采样大量停在GPU结果转CPU的同步位置，不能解释成Python本身占用62%。继续定位prepare/admission/collective和必要同步；用减少同步的对照检验，不按日志片段相加估算收益。
4. **物理DFS带宽/排队。** 保留每rank读取耗时与队列深度；现有读取成功且加worker无收益，优先级低于以上路径。共享存储竞争、低并发或冷启动场景仍可能受IO限制。

## 未采用的实验

AITER extend、worker4→8、layer-wise读取均未显示收益。较大attention tile也未优于原形状。HCU变长FlashAttention诊断达到196K/122K，但teacher-forcing logprob和生成一致性尚未验收，因此撤回；其性能不能作为本PR交付结果。没有为性能跳过租约或失败校验。

## 测试与复测

目标分支定向测试使用原实验环境的 `zhy-4B`，独立解包 `feature/optimize_main` 源码，原版和补丁版分别运行新增数值用例；不复用Spark模块冒充目标分支测试。新增用例使用独立torch数学参考，覆盖窗口边界、无窗口、因果/非因果、不同query长度、paged KV、自定义掩码及FP16/BF16。故意向上取整循环起点的mutation必须失败，防止测试只是在复述实现。

基于 `feature/optimize_main` 的 `61321c04df90161a79b09d59132a8937393f87f1`：原版和补丁版各通过新增2项测试（18个参数子用例）；向上取整mutation触发10个子用例失败；恢复补丁后，7项定向回归通过，包含compact grid、block配置、context attention和unified/regular对照。这些是实际BW卡上的kernel测试，不是目标分支DeepSeek真模型或端到端性能验收。

同一环境另以 `SGLANG_KV_LAYOUT_HCU_FA=1` 运行paged/custom-mask测试，6个FP16/BF16 HND布局子用例全部通过；常规page-major布局使用 `SGLANG_KV_LAYOUT_HCU_FA=0`。

Black、isort、ruff、AST及适用的pre-commit检查通过。全仓 `check_registered_tests.py` 在原基线和本改动上均因既有 `test_mamba_direct_linker.py`、`test_mooncake_dfs_health.py` 缺少CI注册而失败；本PR沿用已有kernel测试文件的CUDA/AMD/HCU注册，未修改这两个无关文件。

```bash
PYTHONPATH=/path/to/checkout/python python3 \
  test/registered/attention/test_triton_attention_kernels.py \
  TestTritonAttention.test_extend_attention_sliding_window_tile_boundaries \
  TestTritonAttention.test_extend_attention_sliding_window_paged_custom_mask -v
```

32K实验参数：`--chunked-prefill-size 32768 --max-prefill-tokens 32768`。这只是Spark实验起点；其他模型须重新核对显存和后缀分布，不能直接沿用本轮性能倍率。

原始响应、4rank指标、trace、启动配置、失败样本与候选补丁已保存在实验数据目录。这里记录证据口径和结论，不包含模型权重或运行日志大文件。
