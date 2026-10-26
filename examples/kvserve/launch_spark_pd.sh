#!/usr/bin/env bash
# Run one role in the foreground; choose MODE=off/raw/default on both nodes.
set -e
if [[ -f "${DCU_HOST_PREPARE:-/guofy/packages/Spark_config/dcu_host_prepare.sh}" ]]; then
    source "${DCU_HOST_PREPARE:-/guofy/packages/Spark_config/dcu_host_prepare.sh}"
fi
set -euo pipefail

role="${1:?Usage: bash launch_spark_pd.sh prefill|decode|router}"
script_path="${BASH_SOURCE[0]}"
script_directory="${script_path%/*}"
if [[ "${script_directory}" == "${script_path}" ]]; then
    script_directory=.
fi
project_root="$(cd -- "${script_directory}/../.." && pwd)"
export PYTHONPATH="${project_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_KV_LAYOUT_HCU_FA=0
export SGLANG_USE_HND_KVCACHE=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export SGLANG_DISAGG_STAGING_BUFFER=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0
export MC_MS_AUTO_DISC=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_PCI_RELAXED_ORDERING=0
export MC_TRANSFER_TIMEOUT="${MC_TRANSFER_TIMEOUT:-120}"
export MC_ALLOWED_IBV_DEVICES="${IB_DEVICES:-shca_0,shca_1,shca_2,shca_3}"
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="${MC_ALLOWED_IBV_DEVICES}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-shca}"
export NCCL_PLUGIN_P2P="${NCCL_PLUGIN_P2P:-ib}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1000

prefill_host="${PREFILL_HOST:-10.113.46.17}"
decode_host="${DECODE_HOST:-10.113.46.18}"
prefill_port="${PREFILL_PORT:-30000}"
decode_port="${DECODE_PORT:-30001}"
bootstrap_port="${BOOTSTRAP_PORT:-31820}"
mode="${MODE:-raw}"

if [[ "${role}" == router ]]; then
    command=(python -m sglang_router.launch_router
        --prefill "http://${prefill_host}:${prefill_port}" "${bootstrap_port}"
        --decode "http://${decode_host}:${decode_port}"
        --pd-disaggregation --mini-lb
        --host "${ROUTER_HOST:-${prefill_host}}" --port "${ROUTER_PORT:-30320}")
else
    case "${role}" in
        prefill)
            host="${prefill_host}"; port="${prefill_port}"
            export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
            ;;
        decode)
            host="${decode_host}"; port="${decode_port}"
            export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-2,3}"
            ;;
        *) echo "Unknown role: ${role}" >&2; exit 2 ;;
    esac
    command=(python -m sglang.launch_server
        --model-path "${MODEL_PATH:-/ai_data/models/Spark-X2.5-4B/}"
        --served-model-name spark-x2.5-4b --tp 2
        --dtype bfloat16 --kv-cache-dtype bf16
        --attention-backend triton --prefill-attention-backend triton
        --decode-attention-backend triton --page-size 64
        --chunked-prefill-size 16384 --context-length "${CONTEXT_LENGTH:-65536}"
        --max-running-requests "${MAX_RUNNING_REQUESTS:-12}" --mem-fraction-static 0.85
        --swa-full-tokens-ratio 0.15 --enable-metrics --enable-cache-report
        --disaggregation-mode "${role}"
        --disaggregation-transfer-backend mooncake
        --disaggregation-bootstrap-port "${bootstrap_port}"
        --disaggregation-ib-device "${MC_ALLOWED_IBV_DEVICES}"
        --host "${host}" --port "${port}")
    if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
        command+=(--chat-template "${CHAT_TEMPLATE}" --tool-call-parser spark
            --reasoning-parser qwen3 --enable-strict-thinking)
    fi
    if [[ "${mode}" != off ]]; then
        command+=(--kvserve-compression "${mode}"
            --kvserve-reserved-mb "${KVSERVE_RESERVED_MB:-1024}"
            --kvserve-rpc-timeout "${KVSERVE_RPC_TIMEOUT:-120}")
    fi
    if [[ "${role}" == prefill ]]; then
        case "${STORE_ENABLED:-1}" in
            0|1) ;;
            *) echo "STORE_ENABLED must be 0 or 1" >&2; exit 2 ;;
        esac
        if [[ "${STORE_ENABLED:-1}" == 1 ]]; then
            # Reuse the existing Store master/client services and raw BF16
            # namespace. Compression only changes the P->D wire payload.
            export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-${host}}"
            export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${host}}"
            export MOONCAKE_MASTER="${MOONCAKE_MASTER:-${prefill_host}:50156}"
            export MOONCAKE_TE_META_DATA_SERVER="${MOONCAKE_TE_META_DATA_SERVER:-http://${prefill_host}:8095/metadata}"
            export MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-${MC_ALLOWED_IBV_DEVICES}}"
            export MC_TE_FILTERS="${MC_TE_FILTERS:-${MC_ALLOWED_IBV_DEVICES}}"
            export MOONCAKE_ENABLE_DFS="${MOONCAKE_ENABLE_DFS:-1}"
            export MOONCAKE_DFS_FS_ADAPTER="${MOONCAKE_DFS_FS_ADAPTER:-posix}"
            export MOONCAKE_DFS_ROOT_DIR="${MOONCAKE_DFS_ROOT_DIR:-/PD/lb_space/smallmodel_storage17/}"
            export MOONCAKE_DFS_ALLOCATOR_TYPE="${MOONCAKE_DFS_ALLOCATOR_TYPE:-bucket}"
            export MOONCAKE_DFS_BUCKET_CAPACITY="${MOONCAKE_DFS_BUCKET_CAPACITY:-4294967296}"
            export MOONCAKE_DFS_MAX_BUCKET_COUNT="${MOONCAKE_DFS_MAX_BUCKET_COUNT:-5120}"
            export MOONCAKE_DFS_ALIGNMENT="${MOONCAKE_DFS_ALIGNMENT:-4096}"
            export MOONCAKE_DFS_SINGLE_TENANT="${MOONCAKE_DFS_SINGLE_TENANT:-True}"
            export MOONCAKE_DFS_BUCKET_META_LOG_THRESHOLD="${MOONCAKE_DFS_BUCKET_META_LOG_THRESHOLD:-4194304}"
            export MOONCAKE_DFS_EVICTION_ENABLED="${MOONCAKE_DFS_EVICTION_ENABLED:-True}"
            export MOONCAKE_ENABLE_SSD_OFFLOAD="${MOONCAKE_ENABLE_SSD_OFFLOAD:-1}"
            export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR="${MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR:-distributed_storage_backend}"
            export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="${MOONCAKE_OFFLOAD_FILE_STORAGE_PATH:-${MOONCAKE_DFS_ROOT_DIR}}"
            export MC_STORE_DFS_READ_TRACE="${MC_STORE_DFS_READ_TRACE:-1}"
            export MOONCAKE_DFS_BATCH_READ_THREADS="${MOONCAKE_DFS_BATCH_READ_THREADS:-32}"
            export MOONCAKE_DFS_FORCE_ONE_REPLICA="${MOONCAKE_DFS_FORCE_ONE_REPLICA:-True}"
            export MOONCAKE_DFS_BATCH_READ_MERGE_ENABLED="${MOONCAKE_DFS_BATCH_READ_MERGE_ENABLED:-1}"
            export MC_STORE_MEMCPY="${MC_STORE_MEMCPY:-1}"
            export MC_STORE_DFS_RESTORE_CHUNK_SIZE_BYTES="${MC_STORE_DFS_RESTORE_CHUNK_SIZE_BYTES:-1073741824}"
            export MC_STORE_PINNED_RESTORE_ARENA_SIZE_BYTES="${MC_STORE_PINNED_RESTORE_ARENA_SIZE_BYTES:-4294967296}"
            export MC_STORE_DFS_PREFETCH_ARENA_SIZE_BYTES="${MC_STORE_DFS_PREFETCH_ARENA_SIZE_BYTES:-6442450944}"
            export SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE="${SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE:-2147483648}"
            export SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS="${SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS:-400}"
            export SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S="${SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S:-5}"
            export MOONCAKE_CLIENT_METRICS_PORT_BASE="${MOONCAKE_CLIENT_METRICS_PORT_BASE:-9301}"
            export MOONCAKE_MASTER_METRICS_PORT="${MOONCAKE_MASTER_METRICS_PORT:-9014}"
            store_extra_config="${STORE_EXTRA_CONFIG:-}"
            if [[ -z "${store_extra_config}" ]]; then
                store_extra_config='{"extra_backend_tag": "spark-x2.5-4b"}'
            fi
            command+=(--enable-unified-cache-external-linker
                --unified-cache-external-linker-backend mooncake
                --hicache-storage-backend-extra-config "${store_extra_config}"
                --mooncake-enable-page-wise-load
                --mooncake-enable-waiting-queue-dfs-prefetch
                --mooncake-waiting-queue-dfs-prefetch-workers "${DFS_PREFETCH_WORKERS:-1}"
                --mooncake-waiting-queue-dfs-prefetch-max-requests "${DFS_PREFETCH_MAX_REQUESTS:-${MAX_RUNNING_REQUESTS:-12}}")
        fi
    elif [[ "${DECODE_RETRACTION_BACKUP:-host_pool}" == host_pool ]]; then
        command+=(--disaggregation-decode-retraction-backup host_pool
            --hicache-ratio "${HICACHE_RATIO:-0.4}")
    fi
fi

printf 'Starting role=%s mode=%s\n' "${role}" "${mode}"
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    exec "${command[@]}"
fi
