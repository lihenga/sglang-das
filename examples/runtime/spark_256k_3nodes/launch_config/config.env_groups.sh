#!/usr/bin/env bash
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export P_IP=13.1.12.21 D0_IP=13.1.12.12 D1_IP=13.1.12.13
export NET_IFACE=ib1 IB_DEVICES=shca_0,shca_1,shca_2,shca_3
export MODEL_PATH=/public/share/rogon_admin/models/300b_1024_ep3_main_w4a8_flexsmooth_ssz_dspark_w8a8_ssz_20261021
export DRAFT_MODEL_PATH="${MODEL_PATH}/draft"
export CHAT_TEMPLATE="${MODEL_PATH}/Spark-ChatTemplate_1024_infer.jinja"
export ROCSHMEM_TOPO_FILE_FORCE=/root/topo.config
export CONTEXT_LENGTH=262144 P_MAX_TOTAL_TOKENS=270336 D_MAX_TOTAL_TOKENS=270336
export P_MAX_RUNNING_REQUESTS="${EXPERIMENT_P_MAX_RUNNING_REQUESTS:-4}"
export D_MAX_RUNNING_REQUESTS="${EXPERIMENT_D_MAX_RUNNING_REQUESTS:-128}"
export RUN_ID="${RUN_ID:?Run through runner.py to record the process group}"
export RUN_TS="$RUN_ID" BOOTSTRAP_PORT=8998
export MOONCAKE_MASTER="${D0_IP}:50156"
export MOONCAKE_TE_META_DATA_SERVER="http://${D0_IP}:8095/metadata"
export MOONCAKE_MASTER_METRICS_PORT=9014
export MASTER_HOST="$D0_IP" MASTER_BIND_HOST="$D0_IP" METADATA_BIND_HOST="$D0_IP"
export MASTER_PORT=50156 METADATA_PORT=8095
export DIST_PORT=5118 DIST_INIT_ADDR="${D0_IP}:5118"
export STORE_PORT=50052 STORE_HTTP_PORT=9300
export STORE_GLOBAL_SEGMENT_SIZE="${STORE_GLOBAL_SEGMENT_SIZE:-192gb}"
export RDMA_DEVICES="$IB_DEVICES"
# Native 64-token objects are the default; 128/256 explicitly enable grouping.
# Keep the existing experiment override ahead of the public environment value.
export L3_STORAGE_PAGE="${EXPERIMENT_L3_STORAGE_PAGE:-${SGLANG_HICACHE_DIRECT_STORAGE_PAGE_SIZE:-${L3_STORAGE_PAGE:-64}}}"
case "$L3_STORAGE_PAGE" in 64|128|256) ;; *) echo "L3_STORAGE_PAGE must be 64, 128 or 256" >&2; exit 2;; esac
export SGLANG_HICACHE_DIRECT_STORAGE_PAGE_SIZE="$L3_STORAGE_PAGE"
export L3_CACHE_TAG="${L3_CACHE_TAG:-$RUN_ID}"
# Leave the explicit JSON option out so it cannot mask the environment.
export L3_EXTRA_CONFIG="{\"extra_backend_tag\":\"${L3_CACHE_TAG}\"}"
if [[ "${SOURCE_MODE:-installed}" == baseline ]]; then
  unset SGLANG_HICACHE_DIRECT_STORAGE_PAGE_SIZE
fi
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=10000 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=10000
