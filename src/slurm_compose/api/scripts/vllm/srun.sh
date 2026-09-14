#!/usr/bin/env bash

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../ray/srun.sh"

RAY_VERSION=${RAY_VERSION:-2.54.1}


## When using vLLM images, ray may not be bundled.
function maybe_install_ray {
    if [[ ! -x "$(command -v ray)" ]]; then
        if ! install_uv > /dev/null; then
            echo "[ERROR] Failed installing uv." >&2
            return 1
        fi

        uv -q --no-progress \
            pip install --system "ray[cgraph,default]==${RAY_VERSION}"

        echo "[DEBUG] Ray ${RAY_VERSION} installed $(command -v ray)." >&2
    fi
}


function vllm-deploy {
    ## TODO: a way to work without the env var.
    if [[ -z "${VLLM_RUNTIME_DIR}" ]]; then
        echo "[ERROR] Missing VLLM_RUNTIME_DIR." >&2
        return 1
    fi

    ## Launch a new ray cluster for vllm multi-node deployment.
    if [[ $SLURM_NNODES -gt 1 ]]; then
        if ! maybe_install_ray > /dev/null; then
            echo "[ERROR] Failed installing ray." >&2
            return 1
        fi

        RAY_RUNTIME_DIR="${VLLM_RUNTIME_DIR}/ray" \
        RAY_SKIP_UV_WRAP=1 \
        RAY_SKIP_KV_INIT=1 \
        ray-start &

        if ! RAY_RUNTIME_DIR="${VLLM_RUNTIME_DIR}/ray" \
             RAY_SKIP_UV_WRAP=1 \
             RAY_SKIP_KV_INIT=1 \
             ray-health-check; then
            return 1
        fi
    fi

    ## Launch vllm serve on the head node.
    if [[ $SLURM_PROCID -eq 0 ]]; then
        local VLLM_PORT=${VLLM_PORT:-$(get_free_port)}

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

        echo "[DEBUG] Starting vLLM server: IPAddress=$(hostname --ip-address) Port=${VLLM_PORT}" >&2
        echo "[DEBUG] Running vLLM command: ${vllm_cmd[*]}" >&2

        VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-"WARNING"} \
        RAY_ADDRESS="$(cat "${VLLM_RUNTIME_DIR}/ray/HEAD")" \
        "${vllm_cmd[@]}" &

        ## Register with ray coordination cluster if available.
        if [[ -n "${RAY_RUNTIME_DIR}" ]]; then
            if ! ray-health-check; then
                return 1
            fi

            if ! retry_until -r -1 -- ray-kv set --key "${SLURM_JOB_NAME}" --value "http://$(hostname --ip-address):${VLLM_PORT}/v1"; then
                echo "[ERROR] Unable to set ray KV for key '${SLURM_JOB_NAME}'" >&2
                return 1
            fi

            until [[ $(curl -o /dev/null -s -w "%{http_code}\n" "http://$(hostname --ip-address):${VLLM_PORT}/v1/models") -eq 200 ]]; do
                sleep 10
            done

            if ! retry_until -r -1 -- ray-kv set --key "${SLURM_JOB_NAME}/ready" --value 1; then
                echo "[ERROR] Unable to set vLLM model ready in ray KV store." >&2
                return 1
            fi
        fi
    fi

    wait
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    vllm-deploy "${@}"
fi
