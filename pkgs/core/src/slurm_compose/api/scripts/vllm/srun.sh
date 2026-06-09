#!/usr/bin/env bash

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../include/common.sh"

if [[ -z "${VLLM_RUNTIME_DIR}" ]]; then
    echo "[ERROR] Missing VLLM_RUNTIME_DIR."
    exit 1
fi

if ! ray_health_check; then
    exit 1
fi

## Initialize a new ray cluster for vLLM.
if [[ $SLURM_NNODES -gt 1 ]]; then
    ## New vLLM images are not bundled with ray.
    if ! command -v ray > /dev/null 2>&1; then
        uv -q --no-progress pip install --system "ray[cgraph,default]==${RAY_VERSION:-2.54.1}"
    fi

    RAY_RUNTIME_DIR="${VLLM_RUNTIME_DIR}/ray" RAY_SKIP_UV_WRAP=1 RAY_SKIP_KV_INIT=1 \
    "$(dirname "${BASH_SOURCE[0]}")/../ray/srun.sh" &

    if ! RAY_RUNTIME_DIR="${VLLM_RUNTIME_DIR}/ray" RAY_SKIP_UV_WRAP=1 RAY_SKIP_KV_INIT=1 ray_health_check; then
        exit 1
    fi

    VLLM_RAY_ADDRESS="$(cat "${VLLM_RUNTIME_DIR}/ray/HEAD")"
fi

export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-"WARNING"}

if [[ $SLURM_PROCID -eq 0 ]]; then
    VLLM_PORT=${VLLM_PORT:-$(get_free_port)}

    vllm_cmd=(
        vllm
        serve
        "${@}"
        --trust-remote-code
        --host 0.0.0.0
        --port "${VLLM_PORT}"
    )
    if [[ $SLURM_NNODES -gt 1 ]]; then
        vllm_cmd+=(--distributed-executor-backend ray)
    fi

    echo "[INFO] Running vLLM command: ${vllm_cmd[*]}"
    echo "[INFO] Starting vLLM server: IPAddress=$(hostname --ip-address) Port=${VLLM_PORT}"

    RAY_ADDRESS="${VLLM_RAY_ADDRESS}" \
    "${vllm_cmd[@]}" &

    ray_kv set --key "${SLURM_JOB_NAME}" --value "http://$(hostname --ip-address):${VLLM_PORT}/v1"
    while [[ $(curl -o /dev/null -s -w "%{http_code}\n" "http://$(hostname --ip-address):${VLLM_PORT}/v1/models") -ne 200 ]]; do
        sleep "${POLL_INTERVAL:-10s}"
    done
    ray_kv set --key "${SLURM_JOB_NAME}/ready" --value 1
fi

wait
