# Mooncake as L3 KV Cache

This document describes how to use Mooncake as the L3 KV cache for SGLang.

Related documentation:
* [Quick Start: SGLang HiCache with Mooncake Backend](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/hicache-quick-start.html)
* [Complete Guide: SGLang HiCache with Mooncake Backend](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/hicache-integration-v1.html)
* [Mooncake x SGLang HiCache System Design](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html)
* [HiCache System Design and Optimization](https://docs.sglang.io/advanced_features/hicache_design.html)
* [SGLang HiCache with Mooncake Backend Benchmark](https://kvcache-ai.github.io/Mooncake/performance/sglang-hicache-benchmark-results-v1.html)

## About Mooncake

Mooncake aims to enhance the inference efficiency of large language models (LLMs), especially in slow object storage environments, by constructing a multi-level caching pool on high-speed interconnected DRAM/SSD resources. Compared to traditional caching systems, Mooncake utilizes (GPUDirect) RDMA technology to transfer data directly in a zero-copy manner, while maximizing the use of multi-NIC resources on a single machine.

For more details about Mooncake, please refer to [Mooncake project](https://github.com/kvcache-ai/Mooncake) and [Mooncake documents](https://kvcache-ai.github.io/Mooncake/).

### Mooncake & SGLang HiCache

Mooncake serves as a high-performance L3 storage backend for SGLang HiCache, enabling distributed KV cache storage across multiple servers with RDMA-accelerated data transfer. This integration addresses the capacity limitations of traditional GPU-only or GPU+CPU caching by providing virtually unlimited cache storage through a distributed memory pool.

When a cache miss occurs in L1 and L2, HiCache automatically fetches the required KV cache from Mooncake's distributed memory pool. The system uses intelligent prefetching strategies to minimize latency, and utilize RDMA technology and zero-copy technique to ensure high-bandwidth, low-latency data transfer between SGLang instances and Mooncake storage nodes.

**Key Advantages:**

- **Scalable Capacity**: Aggregate memory across entire clusters into large distributed pools.
- **Cache Sharing**: KV caches can be shared by all SGLang instances in the cluster.
- **RDMA Acceleration**: Direct memory access eliminates CPU overhead and reduces latency.
- **Zero Copy**: Direct data transfer between L2 and Mooncake without intermediate copying, maximizing throughput.
- **Fault Tolerance**: Distributed architecture provides resilience against individual node failures.

This integration is particularly valuable for production deployments involving long-context models, multi-turn conversations, and high-throughput serving scenarios where traditional caching approaches become capacity-constrained.

## Install Mooncake

**Method 1: with pip**

```bash
pip install mooncake-transfer-engine
```

**Method 2: from source**

Clone Mooncake project:

```bash
git clone https://github.com/kvcache-ai/Mooncake --recursive
```

Install dependencies:

```bash
cd Mooncake
bash dependencies.sh
```

Build the project:

```bash
mkdir build
cd build
cmake ..
make -j
```

Install Mooncake:

```bash
sudo make install
```

For more details, please refer to [Mooncake official installation guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html).

## Deployment

**Mooncake** is a distributed system that efficiently aggregates memory resources across multiple servers. It can also be deployed on a single server for simpler setups.

When integrated with **SGLang**, the system conceptually consists of four key components: `the master service`, `metadata service` (Optional), `store service`  (Optional), and the `SGLang server`. Among them, the `master service` and `metadata service` are responsible for object and metadata maintenance. The `store service` manages a contiguous memory segment that contributes to the distributed KV cache, making its memory accessible to both local and remote `SGLang servers`. Data transfer occurs directly between the `store service` and `SGLang servers`, bypassing the `master service`.

### Single Server Deployment

**Launch Mooncake `metadata service` (Optional):**

```bash
python -m mooncake.http_metadata_server
```

This service is responsible for centralized metadata management including internal connection status and related metadata.

Deployment of the `metadata service` can be skipped in the following cases:
* Mooncake supports non-centralized metadata management via a P2P handshake mechanism to exchange metadata. When using this mode, deployment of the `metadata service` can be skipped.
* Mooncake also supports embedding `mededata service` into `master service`. In this case, only the `master service` needs to be started.

**Launch Mooncake `master service`:**

The `master service` orchestrates the logical storage space pool across the entire cluster, managing KV cache space allocation and eviction.

To start `mooncake_master`:

```bash
mooncake_master --eviction_high_watermark_ratio=0.95
```

To start `mooncake_master` with embedded `metadata service` (so that a separate `metadata service` deployment can be skipped):

```bash
mooncake_master --enable_http_metadata_server=true --http_metadata_server_port=8080 --eviction_high_watermark_ratio=0.95
```

**Understanding `eviction_high_watermark_ratio`:**

When a `PutStart` request fails due to insufficient memory, or when the eviction thread detects that space usage has reached the configured high watermark ratio, an eviction task is triggered to free up space by evicting a portion of objects.

Due to memory fragmentation, allocation failures may occur even when memory usage has not yet reached 100%. The actual threshold depends on the workload. This [benchmark document](https://kvcache-ai.github.io/Mooncake/performance/allocator-benchmark-result.html) provides memory allocation efficiency results under different scenarios. if excessive allocation failures are observed, consider lowering this parameter accordingly.

**Launch Mooncake `store service` (Optional):**

First, create and save a configuration file in JSON format. For example:

```json
{
    "local_hostname": "localhost",
    "metadata_server": "http://127.0.0.1:8080/metadata",
    "master_server_address": "127.0.0.1:50051",
    "protocol": "rdma",
    "device_name": "",
    "global_segment_size": "4gb",
    "local_buffer_size": 0
}
```

Note: If the `metadata service` is not deployed, set this field to:

```json
    "metadata_server": "P2PHANDSHAKE",
```

Then start the `store service`:

```bash
python -m mooncake.mooncake_store_service --config=[config_path] --port=8081
```

Mooncake `store service` configuration can also be provided via environment variables:

```bash
MOONCAKE_LOCAL_HOSTNAME="localhost" \
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER="127.0.0.1:50051" \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="" \
MOONCAKE_GLOBAL_SEGMENT_SIZE="4gb" \
MOONCAKE_LOCAL_BUFFER_SIZE=0 \
python -m mooncake.mooncake_store_service --port=8081
```

**Parameter Explanation:**

* `local_hostname`, `MOONCAKE_LOCAL_HOSTNAME`: The hostname of the `store service`.
* `metadata_server`, `MOONCAKE_TE_META_DATA_SERVER` : The network address of the `metadata service`. The default port is 8080. If the `metadata service` is not deployed, set this field to: `"metadata_server": "P2PHANDSHAKE"`.
* `master_server_address`, `MOONCAKE_MASTER`: The network address of the `master service`. The default port is 50051.
* `protocol`, `MOONCAKE_PROTOCOL`: The protocol used by Mooncake. Supported values are `"rdma"` or `"tcp"`. For optimal performance, `"rdma"` is recommended.
* `device_name`, `MOONCAKE_DEVICE`: The RDMA devices used by Mooncake. This field can usually be left empty, as Mooncake automatically discovers available NICs by default. This parameter is required only when the protocol is set to `"rdma"` **and** a specific set of NICs needs to be used. Example: `"device_name": "mlx5_0,mlx5_1"`. To list available devices, run `ibv_devices`. **Note:** If the environment variable `MC_MS_AUTO_DISC` is set to `1`, any `device_name` or `MOONCAKE_DEVICE` configuration will be overridden, and Mooncake will switch to auto-discovery mode.
  - For tensor parallel deployments where different ranks should use different devices, you can specify device configurations using JSON format:
    ```json
    {
    "device_name": "{0: \"ib0,ib1\", 1: \"ib2,ib3\", 2: \"ib4,ib5\"}"
    }
    ```
  - Or in environment variables:
    ```bash
    MOONCAKE_DEVICE="{\"0\": \"ib0,ib1\", \"1\": \"ib2,ib3\", \"2\": \"ib4,ib5\"}"
    ```
* `global_segment_size`, `MOONCAKE_GLOBAL_SEGMENT_SIZE`: The amount of memory contributed to the global memory pool. Accepts either bytes (integer) or a string with the `gb` suffix, e.g., `"4294967296"` or `"4gb"`. A larger value allows Mooncake to cache more KV tensors.
* `local_buffer_size`, `MOONCAKE_LOCAL_BUFFER_SIZE`: Local buffer is used to do request operations such as `Get` or `Put`. In this case, it is set to 0 because the instance functions solely as a storage server, contributing memory to the global pool without issuing any request operations.

**Important: Understanding Global Segment Size**

`global_segment_size` and `MOONCAKE_GLOBAL_SEGMENT_SIZE`: This parameter specifies the amount of memory each instance contributes to the distributed memory pool. The total memory available for KV cache storage across the cluster is the sum of the memory contributed by all instances.

Adjust this value according to system’s available memory and expected cache requirements.

Note: If `MOONCAKE_GLOBAL_SEGMENT_SIZE` is set to a non-zero value when starting the `SGLang server`, launching the `store service` can be skipped. In this case, the `SGLang server` also takes on the role of the `store service`, which simplifies deployment but couples the two components together. Users can choose the deployment approach that best fits their needs.

**Start the `SGLang server` with Mooncake enabled:**

There are three ways to configure Mooncake:

1. Via extra configuration passed through sglang parameters
2. Using JSON configuration files
3. Using environment variables

Mooncake loads configuration in the following priority order:

1. If Mooncake-specific options are provided in `--hicache-storage-backend-extra-config`, they are used first.
2. If not, Mooncake checks whether the environment variable `DEFAULT_MOONCAKE_CONFIG_PATH_ENV` is set, and loads the JSON config file from that path.
3. If neither of the above is provided, Mooncake falls back to environment variables.

**Using extra-config of sglang arguments to configure Mooncake**

```bash
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --model-path [model_path] \
    --hicache-storage-backend-extra-config '{"master_server_address": "127.0.0.1:50051", "local_hostname": "localhost", "metadata_server": "http://127.0.0.1:8080/metadata", "global_segment_size": "4gb", "protocol": "rdma", "device_name": "", "tenant_id": "tenant-a"}'
```

**Using JSON file to configure Mooncake**

SGLang server can load Mooncake config from `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH`.

```bash
export SGLANG_HICACHE_MOONCAKE_CONFIG_PATH=/sgl-workspace/sglang/benchmark/hicache/mooncake_config.json

echo '{
    "local_hostname": "localhost",
    "metadata_server": "http://127.0.0.1:8080/metadata",
    "master_server_address": "127.0.0.1:50051",
    "protocol": "rdma",
    "device_name": "",
    "global_segment_size": "4gb",
    "tenant_id": "tenant-a"
}' > ${SGLANG_HICACHE_MOONCAKE_CONFIG_PATH}

python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --model-path [model_path]
```

**Using env variables to configure Mooncake**

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER="127.0.0.1:50051" \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="" \
MOONCAKE_GLOBAL_SEGMENT_SIZE="4gb" \
MOONCAKE_TENANT_ID="tenant-a" \
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake\
    --model-path [model_path]
```

**Parameter Explanation:**

The Mooncake parameters used here are essentially the same as those configured for the `store service`.

In particular, for the `global segment size`, if at least one `store service` instance is running, this value can be set to `0`. In this case, the SGLang server will not contribute any memory to the system. Note that KV tensors stored in this contributed memory will be lost when the process exits; however, this will **not** cause any system errors.

**Important:** when `tp > 1`, each Tensor Parallel (TP) rank launches its own Mooncake backend instance and contributes `1/global_segment_size` memory. Therefore, the total memory consumption equals `global segment size`.

**Tenant Isolation (`tenant_id`):**

When `tenant_id` is set, SGLang forwards it to `MooncakeDistributedStore.setup(..., tenant_id=...)`. Producers and consumers that should share HiCache data must use the same `tenant_id`.

You can configure it through `tenant_id` in `--hicache-storage-backend-extra-config`, `tenant_id` in the JSON config file, or `MOONCAKE_TENANT_ID`.

> **Note:** strict isolation between tenants requires a Mooncake master started with `--enable_multi_tenants=true` and a tenant quota policy that explicitly registers each tenant. When strict multi-tenant mode is disabled, Mooncake ignores request tenant IDs for object placement and all objects use the `default` namespace. Non-default `tenant_id` also requires a Mooncake version that supports the `tenant_id` parameter in `MooncakeDistributedStore.setup()`. In `standalone_storage` mode, start the external `mooncake_client` with the matching `--tenant_id` because that process owns the real Mooncake client.

**SSD Offload (`enable_ssd_offload`):**

When `enable_ssd_offload` is set to `true`, SGLang will request that Mooncake enable SSD offloading for the KV cache. This allows Mooncake to spill overflow data from DRAM to local SSDs, effectively expanding the available L3 cache capacity.

If you need to explicitly control the SSD spill directory, set `ssd_offload_path` or the `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` environment variable. SGLang forwards this value to `MooncakeDistributedStore.setup(..., ssd_offload_path=...)`, while other SSD offload tuning parameters continue to be read directly by the Mooncake C++ library.

You can enable it in any of the three supported configuration methods:

- **Via `--hicache-storage-backend-extra-config`:**
  ```bash
  python -m sglang.launch_server \
      --enable-hierarchical-cache \
      --hicache-storage-backend mooncake \
      --model-path [model_path] \
      --hicache-storage-backend-extra-config '{"master_server_address": "127.0.0.1:50051", "enable_ssd_offload": true, "ssd_offload_path": "/mnt/mooncake-ssd"}'
  ```

- **Via JSON config file (`SGLANG_HICACHE_MOONCAKE_CONFIG_PATH`):**
  ```json
  {
      "master_server_address": "127.0.0.1:50051",
      "enable_ssd_offload": true,
      "ssd_offload_path": "/mnt/mooncake-ssd"
  }
  ```

- **Via environment variable:**
  ```bash
  MOONCAKE_MASTER="127.0.0.1:50051" \
  MOONCAKE_ENABLE_SSD_OFFLOAD=1 \
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="/mnt/mooncake-ssd" \
  python -m sglang.launch_server \
      --enable-hierarchical-cache \
      --hicache-storage-backend mooncake \
      --model-path [model_path]
  ```

> **Note:** `enable_ssd_offload` requires a Mooncake version that supports the `enable_ssd_offload` parameter in `MooncakeDistributedStore.setup()`. If the installed version does not support it, SGLang will automatically fall back to the old behavior and print a warning.

**Mooncake Group Semantics (`enable_group_semantics`):**

When `enable_group_semantics` is set to `true`, SGLang passes Mooncake `group_ids` for physical objects derived from the same logical HiCache page. This allows Mooncake to apply group-aware metadata routing, lease refresh, and eviction behavior to related KV objects such as MHA K/V pairs, split-head shards, MLA objects, and supported sidecar objects.

This option is disabled by default. It requires a Mooncake version that exposes `ReplicateConfig.group_ids`. If the installed Mooncake package does not support it, SGLang automatically falls back to the existing write path and prints a warning.

Example:

```bash
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --model-path [model_path] \
    --hicache-storage-backend-extra-config '{"master_server_address": "127.0.0.1:50051", "enable_group_semantics": true}'
```

**HiCache Related Parameters for SGLang Server**

For a comprehensive overview of HiCache-related parameters, please refer to [this document](https://docs.sglang.io/advanced_features/hicache_design.html#related-parameters).


Note that, for `--hicache-mem-layout {layer_first,page_first,page_first_direct}`,
the regular Mooncake backend path still uses `page_first` or `page_first_direct`.
When HiSparse provides an MLA host KV pool or DeepSeek V4 C4 side pool with
layer-first page metadata, Mooncake Store uses Mooncake's multi-buffer zero-copy
APIs (`batch_put_from_multi_buffers` / `batch_get_into_multi_buffers`) to store
each logical page across its per-layer buffers.

### Distributed Deployment

Distributed deployment of Mooncake is straightforward. Similar to the single-node setup, start one `metadata service` and one `master service` for this cluster. Then start a `store service` on each server.

Mooncake also supports high availability mode. This mode enhances fault tolerance by running the `master service` as a cluster of multiple master nodes coordinated through an `etcd` cluster. The master nodes use `etcd` to elect a leader, which is responsible for handling client requests. For more details about how to deploy in this mode, please refer to our [documents](https://kvcache-ai.github.io/Mooncake/).

### Deployment with Dummy Client (Experimental)

In addition to the standard deployment where SGLang acts as a full Mooncake node, you can use the **Dummy Client** mode. In this mode, SGLang connects to a local **Mooncake Store Service** (Real Client) via RPC/IPC. This decouples the SGLang process from the heavy RDMA and memory management, potentially improving stability and allowing the cache to persist even if the SGLang process restarts.

**Architecture:**
* **Mooncake Master**: Manages the cluster topology (same as standard).
* **Mooncake Store Service (Real Client)**: Manages the actual memory pool and RDMA connections. Must be running locally.
* **SGLang Server (Dummy Client)**: Connects to the local Store Service to access the cache.

#### 1. Launch Services (Master & Store)

First, start the `master service` and the `store service`. The `store service` acts as the Real Client.

**Start Master:**
```bash
mooncake_master --eviction_high_watermark_ratio=0.95
```

**Start Store Service (Real Client):** Crucially, the default port (50052) is used for internal RPC, which the Dummy Client will connect to.
```bash
mooncake_client --global_segment_size=4GB
```

**Parameter Explanation:**

- **`host`**: (string, default: "0.0.0.0"): The hostname of the client.

- **`port`**: (int, default: 50052): The port number the client service listens on.

- **`global_segment_size`**: (string, default: "4GB"): The size of the global segment to be allocated by the client.

- **`master_server_address`**: (string, default: "localhost:50051"): The address of the Master Service.

- **`metadata_server`**: (string, default: "http://localhost:8080/metadata"): The address of the metadata service.

- **`protocol`**: (string, default: "tcp"): The protocol used by the Transfer Engine.

- **`device_name`**: (string, default: ""): The device name used by the Transfer Engine.

- **`threads`**: (int, default: 1): The number of threads used by the client.

#### 2. Launch SGLang (Dummy Client)
Configure SGLang to connect to the Real Client using the client_server_address parameter.

**Using extra-config of sglang arguments to configure Mooncake**

```bash
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --model-path [model_path] \
    --hicache-storage-backend-extra-config '{"standalone_storage": true, "client_server_address": "127.0.0.1:50052"}'
```

**Using JSON file to configure Mooncake**

SGLang server can load Mooncake config from `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH`.

```bash
export SGLANG_HICACHE_MOONCAKE_CONFIG_PATH=/sgl-workspace/sglang/benchmark/hicache/mooncake_config.json

echo '{
    "standalone_storage": true,
    "client_server_address": "127.0.0.1:50052"
}' > ${SGLANG_HICACHE_MOONCAKE_CONFIG_PATH}

python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --model-path [model_path]
```

**Using env variables to configure Mooncake**

```bash
MOONCAKE_STANDALONE_STORAGE=1
MOONCAKE_CLIENT="127.0.0.1:50052"
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --model-path [model_path]
```

### Prefill/Decode Disaggregation

In **PD disaggregation**, the configurations for the `metadata service`, `mooncake master`, and the optional `store service` remain the same as described above. The difference is that SGLang introduces three distinct roles: `prefill worker`, `decode worker`, and `router`.

Among these, the `prefill worker` supports enabling **HiCache**. To run with PD disaggregation, start from the [PD configuration](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration-v1.html), and add the HiCache-related parameters (as previously described for the `SGLang server`) to the `prefill worker`.

In the example below, one `prefill worker`, one `decode worker`, and one `router` are launched. HiCache is enabled on the `prefill worker` to optimize prefill performance.

**Prefill worker**:

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER=127.0.0.1:50051 \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="mlx5_1" \
MOONCAKE_GLOBAL_SEGMENT_SIZE=4294967296 \
python -m sglang.launch_server \
    --model-path [model_path] \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-storage-prefetch-policy timeout \
    --hicache-storage-backend mooncake \
    --disaggregation-mode prefill \
    --disaggregation-ib-device "mlx5_1" \
    --base-gpu-id 0 \
    --port 30000
```

**Decode worker**:

```bash
python -m sglang.launch_server \
    --model-path [model_path] \
    --page-size 64 \
    --disaggregation-mode decode \
    --disaggregation-ib-device "mlx5_1" \
    --base-gpu-id 1 \
    --port 30001
```

**Router**:

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "http://127.0.0.1:30000" \
    --decode "http://127.0.0.1:30001" \
    --host 0.0.0.0 \
    --port 8000
```

## Troubleshooting

**RDMA Registration Failure:**

* In some environments, RDMA registration may require root privileges. In this case, try running the program as root.
* In certain environments (e.g., eRDMA), there is an upper limit on the total amount of RDMA memory that can be registered. Once this limit is exceeded, registration will fail. To resolve this, you can lower the value of `MOONCAKE_GLOBAL_SEGMENT_SIZE`, or reduce the host memory allocated to HiCache in the `SGLang server` (since this memory is fully registered with RDMA to enable zero-copy).

**HiCache CPU Memory Usage:**

When using HiCache, the default L2 host DRAM (CPU memory) size for KV cache is **2 times** the size of the L1 device memory (GPU memory) for KV cache.

If the model is small but the GPU memory is large — especially in multi-TP (tensor parallel) setups — this may cause the L1 KV cache to become very large, which in turn can consume excessive CPU DRAM.

In such cases, you should manually configure an appropriate L2 cache size based on your hardware. This can be done by setting `--hicache-ratio` or `--hicache-size`.

**More Information:**

Additional troubleshooting information can be found [here](https://kvcache-ai.github.io/Mooncake/troubleshooting/troubleshooting.html).

## Community, HYGON L3, and L4 package compatibility

The Store integration negotiates optional APIs instead of comparing wheel
version strings. The target baselines are community `v0.3.13.post1`, HYGON
`v0.3.13.post1_dev`, and the DAS `v0.3.13.post1_l4` branch. Install the package
variant appropriate for your Linux device and driver.

| Optional feature | Community / HYGON L3 | L4 with the corresponding APIs |
| --- | --- | --- |
| Basic Store and session range reads | Existing API path | Existing API path |
| Native strided ReadLayout | Selected automatically when present for layer-wise loads | Selected automatically when ReadPlan is not selected |
| Waiting-queue DFS prefetch | Disabled with a startup warning when requested | Negotiated across ranks; requires an available arena |
| `SGLANG_MOONCAKE_READ_PLAN=1` | Uses native ReadLayout when available, otherwise session ranges; address reuse disabled | Native ReadPlan |
| DFS health and access policy | Original package behavior; no SGLang health-based failover | Native probes and coordinated policy |
| Client HTTP metrics | Automatic port zero disables the optional endpoint with a warning; upgrade for automatic ports | Automatic ports enabled when declared by the package |

The same SGLang code and configuration can use either package. Optional API
selection first respects local capability declarations, then checks callable
methods for older packages without a query. Layer-wise loads automatically use
native ReadLayout when available, so Mooncake expands strided addresses in C++.
An existing ReadPlan request selects ReadPlan on a supporting L4 package and
falls back on L3; page-wise loads keep their aggregated read path. No package
version-string comparison or per-package SGLang patch is required.

An older SGLang caller can keep its existing setup, batch-put, session, and range
calls with the L4 wheel. It uses the L4 implementation behind those interfaces;
caller-driven features such as ReadPlan and waiting-queue prefetch still require
SGLang code that calls their APIs. Python API compatibility alone does not carry
over native hardware fixes: the installed L4 build must also retain the needed
L3 transport fixes.

Disabling prefetch omits the L4-only setup argument on packages that predate
prefetch arenas. Packages with a prefetch arena must support either the setup
gate or a successful startup-only `disable_dfs_prefetch()` call. The adapter
does not change process-wide environment variables. Unsupported non-default
tenants or SSD placement options fail explicitly rather than silently changing
where data is stored.

For standalone mode, the Python wheel and external `mooncake_client` must be
from matching packages. A local method's existence does not establish remote
RPC support. Mooncake Client and Master must have matching RPC protocols.
Fallbacks preserve the older package's feature limits: for example, a
community memory-only session path does not acquire L4 DFS-only reads.

Run dependency-free negotiation and startup tests from the repository root:

```bash
python3 test/manual/mooncake/test_package_compatibility.py
```

These tests isolate native dependencies; they do not replace model, GPU, or
cluster I/O tests for each package combination.

## Test Mooncake Store

This test is intended for developers to quickly verify that the MooncakeStore class interfaces are functioning correctly.

First, start the `metadata service` and `master service`. Then run the `test_mooncake_store.py`. 16MB global segments size is enough to run this test.

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER=127.0.0.1:50051 \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_GLOBAL_SEGMENT_SIZE=16777216 \
python3 [path of test_mooncake_store.py]
```

If all tests pass, the message "✅ All tests passed" will be printed at the end.

## 内嵌 client 指标统一采集

配套升级 SGLang 与 Mooncake 后，启动 SGLang 加 `--enable-metrics`，Prometheus 只需抓取 SGLang 业务端口的 `/metrics`，即可采集同机和跨机器 worker 内嵌 client 的指标。无需配置各 client 的端口，`MOONCAKE_CLIENT_METRICS_PORT_BASE` 已移除，即使环境中遗留该变量也不再用于端口分配。可选 client HTTP 监听使用端口 0，由系统从临时端口范围分配可用端口。

worker 每约 5 秒通过原生 `get_client_metrics()` 获取与 client HTTP 相同的完整快照，并主动上报 SGLang；抓取时读取最近快照，不会逐个连接 client HTTP。该链路不依赖 Master 的服务快照，不修改其协议。需使用包含 `get_client_metrics` 的 Mooncake 包；旧包会告警且无法提供本节功能。`MC_STORE_CLIENT_METRIC=0` 会关闭这些原始 client 指标。

导出的 client 指标保留原标签，并增加 `mooncake_client`（client 生命周期 ID）、`mooncake_host`、`mooncake_pid` 和 `mooncake_rank`，避免不同 worker 同名序列冲突。Python Prometheus exporter 会按标准将 counter 名称规范为 `_total` 后缀，例如原始 `mooncake_transfer_read_bytes` 在此入口为 `mooncake_transfer_read_bytes_total`；histogram 的 bucket/sum/count 保留。

- `sglang:mooncake_client_up`：30 秒内收到有效 client 指标快照为 1；指标关闭或断报为 0。首次上报前不会出现该 client。
- `sglang:mooncake_client_last_report_timestamp_seconds`：SGLang 最近接收该 client 快照的时间。断报 30 秒后停止导出它的原始指标，300 秒后移除其状态。

上报失败会在后续周期重试，乱序快照按 sequence 丢弃；SGLang 重启后可通过后续累计快照恢复。异常退出前未成功上报的数据不保证保留。跨机器 worker 使用 `dist_init_addr` 的主节点地址与 SGLang 业务端口，因此须能访问该入口。上报沿用现有 API key 和 TLS 配置；legacy gRPC 模式使用其 HTTP sidecar。多 tokenizer HTTP 进程通过现有 Prometheus 目录共享接收快照，无需跨机器共享文件系统。

本节只统一内嵌 client 指标；独立 daemon 及 Master 自身指标仍由各自入口导出。多 rank 的物理 I/O 可以按需汇总，同一逻辑请求产生的 token/请求计数应根据实际并行方式过滤，不能机械地将所有 rank 相加。

## 等待队列 DFS 预取效果监控

本节只针对 `--mooncake-enable-waiting-queue-dfs-prefetch`：forward 期间接收新请求，经后台队列提前读取 DFS；请求加载时再消费准备好的数据。

**SGLang 导出队列与任务指标，Mooncake master 导出实际读取和成功消费指标。** Prometheus 抓取两个服务入口即可，不需要配置或抓取每个 Mooncake 客户端端口。

### SGLang：队列与任务

启动 SGLang 时加 `--enable-metrics`，从业务端口 `/metrics` 抓取：

| 指标 | 口径 |
| --- | --- |
| `sglang:waiting_queue_prefetch_scheduler_events_total{event}` | forward_received、enqueued、submitted、skipped；forward 生成请求按子请求计数，健康检查及延迟处理消息不计 |
| `sglang:waiting_queue_prefetch_events_total{event}` | host 预取 submitted、请求 claimed、就绪未领取即取消 discarded_ready |
| `sglang:waiting_queue_prefetch_duration_seconds{result}` | host 排队、session 准备和 DFS 读取总耗时；结果为 dfs_prefetched、no_prefetch_needed、failed、cancelled |
| `sglang:waiting_queue_prefetch_ready_lead_seconds` | DFS 预取就绪到请求领取的时间余量 |

事件和任务数按 rank 记录。看请求数量时固定 TP0/PP0/CP0 并保留 DP 副本，避免重复计数；`no_prefetch_needed` 不是 L4 预取成功，`claimed` 也不保证最终 DMA 成功。提前量不能直接当成节省的 TTFT。

### Mooncake master：实际 L4 读取与成功消费

SGLang 已自动注册服务地址；配套更新 master 和 SGLang 使用的 Mooncake 包后，每约 5 秒通过现有后台 RPC 上报累计快照。该链路不依赖客户端 HTTP 或 `MC_STORE_CLIENT_METRIC`，也不增加逐请求 RPC。master 按 `service_endpoint`（SGLang 业务 IP:端口）汇总，支持 UUID/sequence 去重与断报恢复。

| master 指标 | 标签及口径 |
| --- | --- |
| `mooncake_master_service_l4_read_operations_total` | service_endpoint、mode=prefetch/on_demand、result=success/failure；实际 DFS BatchGet 批次 |
| `mooncake_master_service_l4_read_bytes_total` | 同上；成功读取的物理字节，部分失败批次的成功字节计入 failure |
| `mooncake_master_service_l4_read_duration_seconds` | 同上；实际读取耗时直方图，包含 bucket、sum、count |
| `mooncake_master_service_l4_restore_ranges_total` | service_endpoint、mode=prefetched/on_demand；成功消费的数据段数，不是 chunk 或请求数 |
| `mooncake_master_service_l4_restore_bytes_total` | 同上；完成 scatter 后的成功消费有效字节 |
| `mooncake_master_service_reporting_clients` | service_endpoint；30 秒内仍有有效上报的客户端数量，0 表示断报，历史 Counter 保留 |

客户端原有 `mooncake_l4_*` 指标保留排障使用；SGLang 开启 `--enable-metrics` 时仍可自动开启客户端 HTTP，但无需给 Prometheus 配置这些客户端端口。本次扩展了服务上报快照，应配套升级 master 和客户端 Mooncake 包。

### 抓取与看板

```yaml
scrape_configs:
  - job_name: sglang
    static_configs:
      - targets: ["<sglang-host>:<business-port>"]
  - job_name: mooncake-master
    static_configs:
      - targets: ["<master-host>:<master-metrics-port>"]
```

核心预取覆盖率，分子与分母均为成功消费字节；将服务地址替换为实际值：

```promql
100 * (
  sum(rate(mooncake_master_service_l4_restore_bytes_total{service_endpoint="<sglang-ip:port>",mode="prefetched"}[5m]))
  or 0 * sum(rate(mooncake_master_service_l4_restore_bytes_total{service_endpoint="<sglang-ip:port>"}[5m]))
) / sum(rate(mooncake_master_service_l4_restore_bytes_total{service_endpoint="<sglang-ip:port>"}[5m]))
```

没有 L4 消费时比例没有有效值，不能解释为 0% 命中。客户端停止上报时要结合 reporting_clients 判定，不能将历史窗口没有增量当成服务闲置。物理读取可能含整对象、对齐和重复读取，不与有效消费字节混用。

后台上报与 Prometheus 抓取存在延迟；窗口速率反映收到的上报，首次补报累计数据也可能产生脉冲。原有 `prefetched_tokens_total` 包含普通请求加载，不用来计算本机制覆盖率。压测时比较相同负载下开关预取的覆盖率、实时回源量、TTFT 与吞吐；SGLang 任务计数与 master 字节不直接相除。
