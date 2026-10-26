# KVServe PD + Store/DFS 并发联调版

基于 `sglang-0518-mtp-sparkX25`，将 KVServe runtime API 1 接入 SGLang 的
Mooncake P/D 传输。Mooncake C++/RDMA 库和 KVServe 仓库无需修改。

## 范围

- 目标：Linux DCU/DTK 环境，Spark-X2.5-4B，P、D 同构 TP=2，PP/DP/CP=1。
- 仅支持分开的 NHD BF16 K/V 缓存，保留现有 Triton attention。
- 仅压缩全注意力缓存。星火配置为 36 层，其中原始层号
  `3,7,11,15,19,23,27,31,35` 为全注意力，27 层 SWA 沿用原始传输。
- 不再将请求并发硬限制为 1。示例默认 `--max-running-requests 12`，
  可用 `MAX_RUNNING_REQUESTS` 覆盖；它与传输槽数量是两个独立参数。
- 对齐原 vLLM 的多请求会话/传输槽排队机制，每个 rank 默认使用 2 个槽。
  不同请求可重叠传输/等待 ACK，同一请求的分块保持顺序。
  这不是对 vLLM 调度器的逐行复制，也不代表已获得相同吞吐。
- 先在跨节点 1P+1D 上测并发 1/2/4/12；完整 2P+3D 仍需目标环境验证。
- 保留原 P 端 Mooncake Store/DFS linker 的读写、页级加载和等待队列预取；
  Store 中仍保存原格式 BF16 KV，仅 P→D 的线缆数据经过 KVServe 压缩。
- 暂不支持推测解码、D 端前缀复用、异构 TP、
  page-major/unified-memory 布局、GPU sleep/恢复。D 的原有 host_pool
  备份配置可保留，但回退恢复仍需另行验证。

## 原理及失败行为

P 从真实缓存按物理页 gather 一小块全注意力 KV，转换到 runtime 的
`[chunk_pages, local_heads, page_size, K_dim + V_dim]`。压缩后，使用现有
Mooncake `batch_transfer_sync` 写入 D 的已注册 staging 缓冲区。
ZMQ 只传描述符和 ACK，不传 KV payload。

D 在调度线程的 receiver.poll 中解压和 scatter，完成后回复 ACK。
P 再按既有流程发送 SWA/辅助数据和最终 Success。双方校验 profile、
TP rank、页大小、层号和目标页顺序。关闭压缩保留原始传输路径。

P 使用本地 FIFO 槽队列；D 为每个请求/分块分配独立槽和 token，槽满后排队。
commit/status 立即返回 pending，不占住控制线程等待 GPU，避免一个请求的
解压阻塞其他请求的握手。只有调度线程完成解压和写回后才能确认完成。
不同槽的布局转换暂共用受锁和同步保护的 scratch；网络传输/ACK 等待不持该锁。
现有 4 个传输队列按请求分流，同一 D 的不同请求不再必然串在同一队列上。

取消在写入前、或原生传输已完成时，可安全归还槽；排队超时不消耗槽。
遇到无法证明完成的 RDMA 写、无法确认排空的取消/ACK 超时或解码异常，保留已注册
staging 内存并隔离该 rank 的压缩通路，需重启 P、D worker 后再测。
不会把超时槽交给下一请求，也不会悄悄退回未压缩或 CPU 编码。
正常取消/请求级拒绝不把整个 Mooncake peer 标成失联；请求 KV 页仍由
现有 source-drain/abort 流程管理，未能同步完成的源读取不能宣称排空。

控制端口由每个 D rank 动态绑定并通过既有 bootstrap 通告；P 必须能
访问该端口。仅在可信内网使用，不向公网开放控制/RDMA 接口。
控制协议已升级为版本 2，P、D 都必须更新，不支持与最初版本 1 混跑。

## Store/DFS 与压缩同时开启

已移除 KVServe 对外部 linker/HiCache 的额外启动禁令，保留 SGLang 原本的
缓存配置校验。按你的原部署，P 使用 Mooncake 直接 linker，D 保留 host_pool
回退备份；没有把两者改成另一套存储或关闭缓存复用。

Store/DFS 继续读取、保存原格式 BF16 KV，已有模型标签、哈希和物理池元数据
均不改。KVServe 只读取真实缓存，拷贝到独立工作区后压缩；不把原缓存原地量化，
也不把压缩包作为 Store 对象写回。因此可以复用未启用压缩时保存的 KV。
这不意味着 Store/DFS 的存储容量也会按 KVServe 压缩率减少。

外部加载可能先发布 radix 条目、稍后才写完数据。开启 KVServe + 外部 linker
时，batch result 处理会等待已发出的加载完成，再通过既有 TP MIN 共识处理
失败，之后才发布压缩传输。外部缓存/HiCache 场景仅延后可选的「提前发送前缀」
优化，仍复用命中的 KV，由正常 post-forward 路径发送；不是关闭前缀复用。
加载失败沿用原有 abort/source-drain 清理，不将未加载的数据作为成功 KV 发出。

示例 P 脚本在 off/raw/default 三种模式下都默认开启 Store/DFS：
复用原 master/client 服务，默认 master 是 `10.113.46.17:50156`，
metadata 地址为 `http://10.113.46.17:8095/metadata`，DFS 使用 POSIX，
路径为 `/PD/lb_space/smallmodel_storage17/`，模型标签仍为 `spark-x2.5-4b`。
可通过原 `MOONCAKE_*` 环境变量、`STORE_EXTRA_CONFIG`、`DFS_PREFETCH_WORKERS`
和 `DFS_PREFETCH_MAX_REQUESTS` 覆盖。必须确保节点上有相同的实际挂载和可用服务。
D 脚本默认保留 `--disaggregation-decode-retraction-backup host_pool`、
`--hicache-ratio 0.4`。原生产脚本和已有缓存文件不修改、不清空。

如果直接沿用原生产脚本，在两端添加本 checkout 的 `python/` 到 PYTHONPATH，
并在原启动命令中增加下面的选项，原 Store/DFS 环境变量和参数全部保留：

```bash
--kvserve-compression default \
--kvserve-reserved-mb 1024 \
--kvserve-rpc-timeout 120
```

仅为故障定位才可手动设置 `STORE_ENABLED=0`；正式联调保持默认 1。

## TP 要求与原 vLLM 分支一致

原 vLLM 分支的 Mooncake 压缩路径在 `_session_job_open_producer` 中要求
远端 TP、PP 分别等于本端，否则报 `Compressed Mooncake transport requires
matching TP/PP topology`。本适配同样不做异构 TP 的 KV-head 重分片。
这里 TP=2 是星火示例参数，不是硬编码只能取 2；普通未压缩路径的能力不能
直接推定为压缩路径也支持。KVServe runtime 本身负责按本地 TP 分片编解码，
两端拓扑匹配是 connector 的传输约束。

## 环境准备

如使用随版本提供的 Git bundle，在服务器已有的 SGLang 仓库中导入：

```bash
git fetch /path/to/kvserve-sglang-pd-store-dfs.bundle \
  sglang-kvserve-provider
git switch -c sglang-kvserve-provider FETCH_HEAD
```

bundle 包含首版、并发及 Store/DFS 兼容改动，目标仓库需已有基线提交 `565c1af1b`。
导入使用新分支名，不覆盖服务器已检出的首版分支；工作区有改动时先自行保存。

沿用已有 DCU + SGLang、Mooncake 和压缩算子环境。将此分支放入服务器，
确认实际加载的是该 checkout 的 Python 代码。联调脚本会自动把本仓库
的 `python/` 放到 PYTHONPATH 前面。

在同一 Python 环境安装本地 KVServe 包，保留已有 Torch/DTK 版本：

```bash
python -m pip install --no-deps -e /path/to/KVServe
```

这一步只注册 Python 包，不补齐底层算子依赖。当前公开 runtime 的
`default` 路径引用 humming 和 nvCOMP/ANS，使用已跑通环境中的兼容实现；
不要直接把 CUDA 版依赖清单覆盖安装到 DTK 环境。下面的预检会打印
实际 Torch/HIP 版本及 KVServe 加载路径，并直接验证算子可用性。
如果该环境不提供可用的编码后端，`default` 会报错；`raw` 模式不依赖 ANS。

## 三种模式

| 参数 | 行为 |
| --- | --- |
| 省略 `--kvserve-compression` | 原始 Mooncake P/D，不加载 KVServe |
| `--kvserve-compression raw` | 同一适配/传输/恢复流程，原样打包，不做有损量化或 ANS |
| `--kvserve-compression default` | runtime 默认按层策略：Hadamard、量化、ANS |
| `--kvserve-compression @/path/profile.json` | 明确指定 runtime API 1 的配置对象 |

SGLang 的 `default` 使用 runtime 默认算法参数，并将
`logical_blocks_per_chunk` 设为 64（4096 tokens，page_size=64）。
这与旧独立 `CompressedKVConnector` 的字符串 `default` 入口不同。
默认不需要星火 head score 文件；它是初始测试配置，不代表质量最优。

每 rank 默认从 KV-cache 预算扣除 1024 MiB，实际所需 staging/runtime
内存超过预算时启动失败。可用 `--kvserve-reserved-mb` 调整预算。
runtime 的 slot_count 默认是 2，本版实际使用全部槽。可在自定义配置中
设为 1/2/4；槽多会增加显存工作区，不等于能多接同样数量的请求。

自定义配置示例（P、D 内容必须一致，使用同一份文件）：

```json
{
  "compression": {
    "enable-compression-pipeline": true,
    "split_type": "layer",
    "pipeline": ["transformer", "quantizer", "codec"],
    "slot_count": 2,
    "logical_blocks_per_chunk": 64
  }
}
```

## 联调步骤

以下命令在本仓库根目录执行。脚本以已有两台机器为默认地址，提供环境变量覆盖。
使用同一模型、代码、KVServe 包、配置和可访问的 RDMA 网卡。

1. 在 P/D 的实际 DTK 环境分别做算子预检，不启动 HTTP 服务、不需要加载权重：

```bash
python examples/kvserve/check_runtime.py \
  --model-config /ai_data/models/Spark-X2.5-4B/config.json --mode raw
python examples/kvserve/check_runtime.py \
  --model-config /ai_data/models/Spark-X2.5-4B/config.json --mode default
```

预检使用真实 GPU/HIP 张量和实际 KVServe encode/decode，覆盖 63/64/65
tokens、非连续页和不同源/目标页。raw 必须逐元素一致；default 输出
KV 误差统计。此随机张量检查不等同于模型质量评测。

2. 先做不接入 KVServe 的基线。在两个节点各开一个终端：

```bash
# 10.113.46.17
MODE=off HIP_VISIBLE_DEVICES=0,1 bash examples/kvserve/launch_spark_pd.sh prefill
# 10.113.46.18
MODE=off HIP_VISIBLE_DEVICES=2,3 bash examples/kvserve/launch_spark_pd.sh decode
```

在路由节点的另一个终端启动路由并发请求：

```bash
bash examples/kvserve/launch_spark_pd.sh router
# 再开一个终端
python examples/kvserve/smoke_requests.py --output off.json
```

3. 停止上一步的测试进程，将 P/D 的 `MODE=off` 都替换为 `MODE=raw`，
重启 P、D 和测试路由，然后执行：

```bash
python examples/kvserve/smoke_requests.py --output raw.json --reference off.json
```

4. 两端都改为 `MODE=default`，再次重启测试进程，然后执行：

```bash
python examples/kvserve/smoke_requests.py --output compressed.json
```

请求覆盖 63/64/65、4096、16383/16384/16385 tokens。测试使用固定 token
输入及贪心采样检查链路，记录的是完整请求耗时，不能当作 TTFT。
default 的业务质量需另用 HotpotQA/你们的星火业务集评测。

5. 串行 raw 对照完成后，保留 P/D 默认的并发上限 12，依次增加实际请求
并发。每个请求使用不同 token ID，覆盖多请求数据隔离；repeat=2 提供 14 个
请求，足以让 12 路并发争用默认 2 个传输槽：

```bash
# 先在 off 模式采集同一组输入的并发基线
python examples/kvserve/smoke_requests.py \
  --concurrency 12 --repeat 2 --output off-c12.json
# 将两端重启为 raw，再运行
python examples/kvserve/smoke_requests.py \
  --concurrency 12 --repeat 2 --output raw-c12.json --reference off-c12.json
# 将两端重启为 default，再运行
python examples/kvserve/smoke_requests.py \
  --concurrency 12 --repeat 2 --output compressed-c12.json
```

先将 concurrency 改为 1、2、4，再到 12；每个设置分别重跑 off 基线。
默认 token-step=1，repeat 不复用相同输入；输入 token ID 必须在模型词表内。
输出严格对比用于发现异常；若批处理数值差异导致贪心结果分叉，还需结合
原始 KV roundtrip 和日志定位，不能仅凭生成文本差异认定传输错误。

6. 验证真实外部缓存复用，不能只看同一进程中的本地前缀命中：

```bash
# off 模式，保持默认 STORE_ENABLED=1，用固定输入预热并等待 offload 成功。
python examples/kvserve/smoke_requests.py \
  --lengths 4096 16385 --token-step 0 --repeat 2 --output store-warm.json
# 保留 Mooncake master/client 和 DFS 数据，仅重启模型 P/D、路由为 raw。
python examples/kvserve/smoke_requests.py \
  --lengths 4096 16385 --token-step 0 --repeat 2 \
  --require-storage-hit --reference store-warm.json --output store-raw.json
# 再重启模型 P/D、路由为 default，仍用同一 Store 标签、模型和输入。
python examples/kvserve/smoke_requests.py \
  --lengths 4096 16385 --token-step 0 --repeat 2 \
  --require-storage-hit --output store-compressed.json
```

脚本保存并打印 `cached_tokens_details`；要求至少一个请求报告 `storage > 0`，
直接 linker 加载现在也记入 storage，而不是误记成 host；该统计经现有 P/D 元数据传给 D。
否则报错，避免把 GPU/host 本地命中误当成外部缓存测试成功。若首次未落盘/外存，
先确认 write-through/offload 日志，不要删除生产缓存来制造测试条件。
Store 命中可能来自远端内存，`--require-storage-hit` 本身不证明发生过 DFS 磁盘读。
DFS-only 用例须在隔离测试命名空间/服务中准备对应条目，结合
`MC_STORE_DFS_READ_TRACE` 和已有 DFS 读日志确认；不要驱逐/清空生产 Store。
再覆盖并发 12、缓存未命中、混合命中和外部加载失败/取消的回归。

脚本可覆盖 `PREFILL_HOST`、`DECODE_HOST`、`MODEL_PATH`、`IB_DEVICES`、
`CHAT_TEMPLATE`、`CONTEXT_LENGTH`、`MAX_RUNNING_REQUESTS`、`KVSERVE_RESERVED_MB` 等变量。
通过 `DRY_RUN=1` 只打印命令。可选地加载服务器原有
`/guofy/packages/Spark_config/dcu_host_prepare.sh`，用 `DCU_HOST_PREPARE`
指定其他路径。本版示例已恢复原 Store/DFS 功能组，原生产脚本不改动。

## 日志与本地测试

启动时应看到 `KVServe bound`，包含全注意力层号、TP rank、工作区大小和
profile。每次发送记录 `kvserve_transfer`，包括 `raw_bytes`、`wire_bytes`
及 queue/encode/remote_wait/transfer/ack 耗时。`wire_bytes` 包含 payload、量化辅助数据和
raw tail，不含 ZMQ 控制报文。encode 包含 gather/布局转换；ack 时间包含
远端调度、解压及写回；槽位等待单独计入 queue/remote_wait，transfer 只计原生传输调用。
通过标准 serving benchmark 单独测 TTFT/TPOT/吞吐，不用随机张量压缩率
代替端到端结果。

CPU 测试只依赖 pytest、PyTorch、pyzmq，不要求安装整个 SGLang：

```bash
python -m pytest test/manual/kv_transfer/test_kvserve_compression.py \
  test/manual/kv_transfer/test_kvserve_concurrent_smoke.py -q
```

它验证真实张量的 NHD/HND 页映射、TP=2 head 切分、部分页保护，以及
真实 ZMQ 握手、12 请求共享不同槽数、FIFO 排队、ACK 顺序、取消和隔离；
RDMA 在测试中使用独立内存缓冲区模拟。
另检查 Store 已注册指针和原始字节不被压缩修改、加载完成/失败共识先于
发送，以及外部缓存命中探针不会因仅有本地命中而误通过。
DCU 算子、真实 RDMA、模型精度和端到端服务必须在目标服务器验证。

本机使用 Python 3.12 / CPU PyTorch 进行测试，并检查 Python 语法、
新增模块 lint、脚本语法及 dry-run。原分支的 `mooncake/conn.py` 在
PD_HIDDEN 路径已有两处 `dynamic_dst` 未定义的 F821，本次未修改该路径；
`scheduler.py` 另有原有的 `EXTERNAL_KV_LOAD_ERR_TYPE` 未使用导入 F401，未顺带清理。
`unified_radix_cache.py` 的 `UnifiedCacheLinker` 未使用导入 F401 也已在改动前存在。
本版禁用推测解码。没有在本机声称通过 DCU、真实 RDMA 或模型端到端验证。
