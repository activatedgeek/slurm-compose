#!/usr/bin/env bash

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../ray/srun.sh"

## TODO: a way to work without the env var.
if [[ -z "${RAY_RUNTIME_DIR}" ]]; then
    echo "[ERROR] Missing RAY_RUNTIME_DIR." >&2
    return 1
fi

if ! ray-health-check; then
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

    ray-kv set --key "${SLURM_JOB_NAME}" --value "http://$(hostname --ip-address):${SGLANG_PORT}/v1"
    ray-kv set --key "${SLURM_JOB_NAME}/nccl" --value "$(hostname --ip-address):${SGLANG_NCCL_PORT}"
fi

echo "[INFO] Waiting for sglang address..."
SGLANG_ADDR=$(retry_until -r -1 -- ray-kv get --key "${SLURM_JOB_NAME}")
SGLANG_PORT=$(echo "${SGLANG_ADDR}" | cut -d: -f3 | cut -d/ -f1)

echo "[INFO] Waiting for sglang head..."
SGLANG_NCCL_ADDR=$(retry_until -r -1 -- ray-kv get --key "${SLURM_JOB_NAME}/nccl")

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
    until [[ $(curl -o /dev/null -s -w "%{http_code}\n" "http://$(hostname --ip-address):${SGLANG_PORT}/v1/models") -eq 200 ]]; do
        sleep 10
    done
    ray-kv set --key "${SLURM_JOB_NAME}/ready" --value 1
fi

wait
