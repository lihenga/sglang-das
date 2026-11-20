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

For multi-node deployments that attach Mooncake at runtime via `PUT /hicache/storage-backend`, omit `local_hostname` from the attach payload and set `MOONCAKE_LOCAL_HOSTNAME` (or `LOCAL_HOSTNAME`) per node before launching SGLang. Each rank resolves `local_hostname` from its own process environment instead of a shared default.

When loading from a JSON config file, `local_hostname` follows the same per-process precedence: `MOONCAKE_LOCAL_HOSTNAME`, then `LOCAL_HOSTNAME`, then the value in the JSON file, then `"localhost"`.

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

> **Note:** `enable_ssd_offload` requires a Mooncake version that supports the `enable_ssd_offload` parameter in `MooncakeDistributedStore.setup()`. If the installed version does not support an explicitly requested SSD option, startup fails with a compatibility error; SGLang does not silently remove storage configuration.

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

## DeepSeek 4.1 direct Mooncake L3/L4

This port is based on HYGON `feat/20260920_deepseek-v4.1` at
`7888b28e77` and brings the Mooncake direct-cache behavior from
`lihenga/sglang-das` `feature/optimize_main` at `61321c04df`.
The 4.1 C1/C2 KV, indexer payload/scale, SWA, CP layer ownership and packed
DSpark draft mappings remain the target implementation.

### Enable the direct path

Add these options to the model's existing supported launch configuration:

```bash
--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend mooncake \
--schedule-policy fcfs \
--mooncake-enable-page-wise-load \
--mooncake-enable-waiting-queue-dfs-prefetch \
--mooncake-waiting-queue-dfs-prefetch-workers 1 \
--mooncake-waiting-queue-dfs-prefetch-max-requests 12
```

Do not combine direct linking with `--enable-hierarchical-cache` or
`--hicache-storage-backend`. DeepSeek 4.1 currently rejects PP > 1 and
encoder SWA bounded replay together with the external linker. This port does
not remove those model compatibility gates. The unified radix tree is already
the default; `SGLANG_ENABLE_UNIFIED_RADIX_TREE` is obsolete.

The added runtime options are opt-in. Keep the existing layer-wise path to
establish a baseline before enabling page-wise/ReadPlan and queued prefetch.
For a supporting embedded L4 client, the relevant settings are:

```bash
export SGLANG_MOONCAKE_READ_PLAN=1
export SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES=0
export SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE=1073741824
export MOONCAKE_READ_PLAN_ADAPTIVE_SOURCE=1

# Example per-embedded-client buffer budget; size for the actual model/load.
export MC_STORE_DFS_UNIFIED_BUFFER_SIZE_BYTES=12884901888
export MC_STORE_DFS_PREFETCH_CAPACITY_RATIO=0.5
```

ReadPlan page-wise reads pass the mode directly to native code. The ordinary
range path uses `--mooncake-page-wise-load-threshold` (default 10 expanded keys).
ReadLayout is selected when supported; otherwise session range reads remain
available. Address reuse is incompatible with page-wise mode.

Queued prefetch reads DFS objects into the native pinned session cache before
GPU admission. It requires FCFS, a non-PD or prefill role, PP=1, all-rank API
support and an available native arena. If any attention rank lacks these,
all ranks disable it before traffic. Distinct Gloo groups isolate background
prefetch and input reception from model-forward collectives. Unsupported input
such as multimodal/control/grammar work is deferred in receive order until the
forward returns; shared-memory materialization stays on the scheduler thread.

The background state uses attempt-scoped backend IDs, so an old submission ACK
or cleanup cannot cancel a newer request attempt with the same public RID.
Session references are shared by physical key; failures, cancellations and
shutdown retain buffers until the owning I/O has drained. The Python TreeCore
owns failed-chain detachment/reclamation; selecting Rust with this direct path
uses the existing compatibility fallback to Python (already HCU's default).

### L3/L4 package compatibility

Optional APIs are negotiated through declared capabilities and callable
methods, without parsing wheel version strings. Community, HYGON L3 and DAS L4
packages keep their baseline setup/session reads. Unsupported ReadPlan falls
back to ReadLayout or session ranges; unsupported queued DFS prefetch disables
that feature. Storage placement and tenant options are never silently dropped.
Both the older extension module and the newer `mooncake.store` facade are
supported for ReadLayout types. This matches the compatibility behavior from
the donor's community/HYGON L3/L4 integration PR, adapted to this branch's
resolved RuntimeContext config bags.

Direct-linker storage suffixes now include the pool group's layout tag. This
changes L3/L4 keys: existing untagged caches are cold after this upgrade, and
prefill workers running different key layouts cannot share those entries.

For standalone mode, use matching Python/client-daemon packages and matching
Client/Master RPC protocols. A callable Python method does not establish daemon
support; current L4 declares ReadPlan, ReadLayout and queued DFS prefetch for
embedded clients. No transport performance or driver compatibility follows
from successful Python API negotiation.

### DFS deployment and lease settings

Keep the working Mooncake master/client DFS configuration for the shared root,
bucket allocator/alignment/capacity, offload and replica placement. SGLang's
per-rank SSD subdirectories apply to local-disk backends; a distributed backend
retains its shared root and can start with L3 while that mount is unavailable.
The mounted root and actual DFS capacity/permissions still require deployment
validation.

`SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S=0` refreshes reused sessions
unconditionally. A positive age skips refresh only for a private, sufficiently
fresh session. Declare the real master TTL in seconds through
`SGLANG_MOONCAKE_LEASE_TTL_S`; for master `--default_kv_lease_ttl=30000`, use `30`.
Grouped or shared keys use conservative refresh behavior. Admission, queueing,
synchronization and actual reads still consume the lease; this is an RPC-saving
optimization, not a guarantee for arbitrary read duration.

The current `v0.3.14_dev_l4` native configuration uses the unified DFS buffer
and prefetch-capacity ratio. The old `MC_STORE_DFS_PREFETCH_ARENA_SIZE_BYTES`,
`MC_STORE_PINNED_RESTORE_ARENA_SIZE_BYTES`,
`MC_STORE_DFS_RESTORE_CHUNK_SIZE_BYTES`, and
`MC_STORE_DFS_WRITE_ARENA_COUNT` should not be used to calculate its capacity.
Read and write chunks use `MC_STORE_DFS_READ_CHUNK_SIZE_BYTES` and
`MC_STORE_DFS_WRITE_CHUNK_SIZE_BYTES`. Count embedded clients/processes when
budgeting memory; an external storage client has a separate budget.

DFS health has independent read/write probes and fail-closed policy before
warmup. Configure interval, failure/success thresholds and timeout with the
`--mooncake-dfs-health-*` options, or corresponding
`SGLANG_MOONCAKE_DFS_HEALTH_*` environment variables. Attention ranks apply a
common conservative policy across attention and PP ranks, including decode KV
offload. The defaults match `gh/dspark_opt` commit `6bcb41be33`: a 5-second probe
interval, 2 consecutive failures to disable, 2 consecutive successes to recover,
and a 5-second probe timeout. Explicit config, CLI settings and environment values
retain their existing precedence. Read and write probes keep independent health
states, but effective DFS reads require both states to be available.

Real data-I/O timeouts (`-1606` for writes, `-1607` for reads) immediately mark
the corresponding local capability unavailable and queue a restrictive native
policy update on the existing serialized worker. A write timeout disables both
reads and writes. BatchGet admission backpressure (`-1608`) does not disable DFS.
Recovery still requires the configured number of fresh successful probes;
results from probes started before a data timeout cannot count towards recovery.
A local timeout cannot reopen a capability disabled by rank consensus. Native
updates remain asynchronous, so the local state transition does not confirm a
native disable until the worker's call succeeds.

When DFS becomes unavailable, young prefetched sessions must revalidate and
borrowed ReadPlan sessions must reselect memory replicas before reading. A DFS
source remaining after reselection fails the load. Decode offload also applies
TP/CP/PP consensus, checks health-capability agreement before policy polling, and
retains bounded retries and rate-limited warnings for failed policy submissions.
Recovery does not promise backfill of every write made while DFS was unavailable.

Initial fail-closed setup must succeed synchronously before any client warms up.
In deferred mode, recovery uses scheduler-coordinated policies on a single
background worker; local data timeouts may only restrict permissions. Pending
updates coalesce to the latest target. Failed updates retry at the probe interval
without scheduler polling. Non-deferred probe updates remain synchronous. An
unconfirmed native reply stays dirty when the
target returns to the last confirmed policy. A hung call occupies the one worker
until it returns; it is neither cancelled nor replaced by concurrent RPCs.
Embedded RealClient policy updates are local atomic stores. Standalone
DummyClient updates synchronously call the external RealClient by RPC. Before
using deferred policy application with standalone storage, verify that the
installed Mooncake build releases the Python GIL in its policy binding;
otherwise its RPC can still stall the scheduler from the background worker.
The `dfs_health` capability only checks API availability and does not certify
GIL release. No minimum fixed wheel version is established by these checks.
An unconfirmed disable cannot guarantee that the native client stopped DFS
access; the last confirmed policy is not proof of its current permissions.
Storage/linker policy wrappers currently discard the manager's confirmation
result, and loads do not gate on pending policy application. Deferred application
alone therefore does not provide fail-closed load admission during that window.

Health-manager shutdown stops new probes and policy submissions and changes its
local read/write candidates to `UNKNOWN` (deny both). It waits at most one second
in total for its tracked workers. A pending native read, write or policy call
makes `close()` raise instead of reporting completion; synchronous policy callers
are included. The manager retains the client and worker handles. Repeated close
attempts cannot start replacements, and late successful probes cannot restore
availability after shutdown. After the native calls drain, retry close before
creating a replacement manager, whose initial native deny policy must succeed.
An indefinitely blocked native call requires retaining the old owner until
recovery or restarting the process. No native thread is cancelled or client freed
while an operation is using it.

The controller disables storage work and retains the backend and synchronization
groups if shutdown fails, including rollback after a partially failed attach.
`MooncakeStore.close()` retains its health manager when close raises. A retained
backend blocks all new attachments, including same-backend policy updates.
Explicit detach retries release it only after workers and native calls drain;
shutdown also retries retained backends with storage already disabled. No worker
is automatically restarted. A retained-owner detach failure raises through the
attachment entry point so the admin handler cannot report success merely because
storage was already disabled. Stopping health workers does not confirm that native
DFS access was disabled: existing or unconfirmed native permissions can remain in
effect. Refused shutdown is not a healthy attachment that can safely resume
storage traffic.

After detach, disabled ranks skip DFS-policy polling even if only some retain
their native owner. Before a new attach, attention ranks collectively check for
retained backends; any pending cleanup refuses attachment on every rank before
client creation. A locally completed detach therefore cannot let that rank
reattach ahead of a peer whose native call is still pending.

### Observability and validation

With `--enable-metrics`, each embedded client reports cumulative native metrics
to its SGLang service; HTTP and gRPC sidecar receivers expose them through
`/metrics`. Reports wait for receiver startup, retain authentication/TLS
settings and omit successful snapshot traffic from access logs. Queue events,
worker outcome/duration and ready-to-claim lead time are separate from actual
load completion. Direct `l4_*` counters cover successful external-store loads
or writes; the metric name alone does not prove every object came from DFS.

Snapshot ingestion is mounted only on the main HTTP service and the authenticated
gRPC HTTP sidecar, not on non-primary dummy health servers. It follows the
service's configured API-key policy; a service without an API key remains open.
The shared receiver retains at most 1024 client snapshots and 256 MiB of snapshot
files, including temporary files during replacement; quota exhaustion returns
HTTP 429. A replacement needs room for both the old and new file. Snapshots expire
after 300 seconds, with a background sweep every 30 seconds and another sweep
before publication, independently of Prometheus scraping.

Waiting-queue prefetch metrics do not establish end-to-end request tracing.
Background reads, session operations and drain/cleanup work are not guaranteed
to inherit the original request's trace context. Existing transfer spans remain
useful, but this port does not add complete tracing. The waiting-queue event
counter records `wait_prefetch` and `wait_session` admission observations (not
request counts or elapsed time); opt-in `DAS_PREFETCH_TRACE` logs correlate these
reasons with request IDs and bootstrap rooms.

CPU-only tests live under `test/registered/unit/{mem_cache,managers,disaggregation,observability}`.
`test/manual/mooncake/test_package_compatibility.py` exercises package negotiation
without native allocation. `test_native_read_plan.py` uses a private in-process
Master for CPU/TCP native ReadLayout/ReadPlan/session smoke.
`test_wheel_swap.py` checks multi-pool SGLang writes/reads against a matching
wheel; choose `MATRIX_DEVICES=cpu` for CPU checks. These tests do not substitute
for DeepSeek 4.1 output-consistency, HCU, RDMA, DFS fault, or performance runs.
The existing `test_unified_cache_linker_kl_dsv4.py` targets **DeepSeek-V4-Flash**
on CUDA; its injected-read-failure arm is not DeepSeek V4.1 or HCU acceptance.
Do not substitute that model or add an unvalidated HCU registration to claim
V4.1 coverage. A V4.1 service-level run requires the actual V4.1 checkpoint,
launch configuration and allocated accelerators. The Flash test requires an
observed linker abort followed
by a successful request, live health, nonzero remote-load exposure, and the
existing GSM8K accuracy threshold after warm/flush/re-read. This service-level
test needs the model and accelerator environment; CPU unit results do not
establish that it passes.

### Implementation batches and evidence

The functional changes are grouped as Store/API compatibility and diagnostics;
read/session plans plus queued prefetch and tree ownership; scheduler/PD
lifetime integration; then regression coverage and deployment notes. Existing
4.1 pool/model code remains authoritative. External linker deployments reject streaming sessions because their virtual session nodes do not expose external-load ownership for failure recovery. Other-model-specific GLM/Mamba
features are outside this DeepSeek 4.1 port. Existing PP > 1 and encoder bounded
replay restrictions remain in force.

Validation in the supplied `sglang0518_0920_yb` container uses an isolated task
directory and leaves its installed Mooncake package untouched. A matching
`v0.3.14_dev_l4` wheel was built with SHCA, HIP/HYGON and etcd via
`scripts/variant_build.sh --shca --hip --etcd`. The script enables fake HIP RPC
compatibility; this build does not establish HIP fabric RPC functionality.
Native CPU/TCP ReadLayout/ReadPlan/session checks and multi-pool SGLang
round-trips passed. The container has Python 3.10, which satisfies this branch's
Python >= 3.10 requirement. Component tests still do not establish a working full
serving installation with the model and native dependencies.

### Port review follow-ups

The port also validates direct lookup against the target Store signature and
registers a batch's request IDs atomically, so a later duplicate or state-creation
failure leaves no partial request registration. Drained PD aborts retain bounded
terminal room records to reject late bootstrap destinations; failure notices use
the target status-message API and preserve its generation-tagged drain ACKs.

The donor's untagged two-frame `ABORT_ACK` is not a replacement for the target
protocol. A drained prefill abort sends `KVPoll.Failed` using the target status
channel; decode's drain handshake retains `ABORT_ACK` with the room, prefill rank
and matching generation. Generation-less drain ACKs cannot authorize release.
For device destinations, decode can release after
`SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT` (default 30 seconds)
even without complete drain ACKs. That timeout is not proof that remote writes
stopped. Host-staged destinations remain held until drain consensus; the device
timeout fallback does not apply to them.

Terminal-room or notification capacity exhaustion retains a scheduler cleanup
owner for retry. Local transport maps are cleared independently of successful
notification handoff; Failed guards and target snapshots survive backpressure.
Notifications use a bounded worker with endpoint backoff and aggregated logs.
`sglang:mooncake_drained_abort_rejections_total` counts rejected handoffs.

Unknown allocator cleanup outcomes retain request resources instead of repeating
an unsafe free. `sglang:external_kv_quarantined_reqs` reports this quarantine;
4096 retained requests exhaust its capacity and stop the scheduler rather than
grow without bound. Recover these opaque allocator failures by restarting the
worker. This differs from native get-session cleanup, which retains bounded
credits and retries its idempotent end operation without preventing idle/flush.
Inflight user cancellation preserves a computed prefix when no external KV
failure has already made it unsafe to cache.

Forward ingress runs only when the synchronized batch contains extend work,
and is disabled when the input blocker or KV canary is active. Direct-linker
health policy shares the completion-count reduction. Cleanup presence and
inflight retirement still require rank consensus; a locally empty queue cannot
safely gate these operations. No throughput or TTFT A/B is established here.

The accompanying `sgl-model-gateway` changes add HTTP stream outcomes, SSE/JSON
error detection, paired PD cancellation with one shared deadline, and failure-log
rate limiting. HTTP PD requests are dispatched once without replay; regular HTTP,
external OpenAI and gRPC retries retain their existing behavior. The target
Responses API, bootstrap metadata and logprob merging remain supported. Build and
install the gateway wheel separately to use these changes; updating the Engine
package alone does not install them. See `sgl-model-gateway/bindings/python/README.md`
for the metric definitions and scope.
