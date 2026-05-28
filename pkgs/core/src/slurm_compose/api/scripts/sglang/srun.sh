#!/usr/bin/env bash

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../include/common.sh"

if ! ray_health_check; then
    exit 1
fi

sglang_cmd=(
    sglang
    serve
    "${@}"
    --trust-remote-code
    --host 0.0.0.0
    --nnodes "${SLURM_NNODES}"
    --node-rank "${SLURM_PROCID}"
)

if [[ $SLURM_PROCID -eq 0 ]]; then
    SGLANG_PORT=${SGLANG_PORT:-$(get_free_port)}
    SGLANG_NCCL_PORT=${SGLANG_NCCL_PORT:-$(get_free_port)}

    ray_kv set --key "${SLURM_JOB_NAME}" --value "http://$(hostname --ip-address):${SGLANG_PORT}/v1"
    ray_kv set --key "${SLURM_JOB_NAME}/nccl" --value "$(hostname --ip-address):${SGLANG_NCCL_PORT}"
fi

echo "[INFO] Waiting for sglang address..."
SGLANG_ADDR=$(ray_kv get --key "${SLURM_JOB_NAME}")
SGLANG_PORT=$(echo "${SGLANG_ADDR}" | cut -d: -f3 | cut -d/ -f1)

echo "[INFO] Waiting for sglang head..."
SGLANG_NCCL_ADDR=$(ray_kv get --key "${SLURM_JOB_NAME}/nccl")

sglang_cmd+=(
    --port "${SGLANG_PORT}"
    --dist-init-addr "${SGLANG_NCCL_ADDR}"
)

if [[ $SLURM_PROCID -eq 0 ]]; then
    echo "[INFO] Running SGLang command: ${sglang_cmd[*]}"
    echo "[INFO] Starting SGLang server: IPAddress=$(hostname --ip-address) Port=${SGLANG_PORT}"
fi
"${sglang_cmd[@]}" &

if [[ $SLURM_PROCID -eq 0 ]]; then
    while [[ $(curl -o /dev/null -s -w "%{http_code}\n" "http://$(hostname --ip-address):${SGLANG_PORT}/v1/models") -ne 200 ]]; do
        sleep "${POLL_INTERVAL:-10s}"
    done
    ray_kv set --key "${SLURM_JOB_NAME}/ready" --value 1
fi

wait
