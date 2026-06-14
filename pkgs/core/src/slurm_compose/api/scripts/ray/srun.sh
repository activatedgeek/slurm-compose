#!/usr/bin/env bash

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../include/common.sh"

RAY_VERSION=${RAY_VERSION:-2.55.1}
PY_VERSION=${PY_VERSION:-3.12.13}


function wrapped_ray {
    if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
        if ! install_uv > /dev/null; then
            echo "[ERROR] Failed installing uv." >&2
            return 1
        fi

        uvx -q --no-progress \
            -p "${PY_VERSION}" -w "ray[cgraph,default]==${RAY_VERSION}" \
        ray "${@}"
    else
        ray "${@}"
    fi
}


function ray-py {
    if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
        if ! install_uv > /dev/null; then
            echo "[ERROR] Failed installing uv."
            return 1
        fi

        uv run -q --no-progress --no-project \
               -p "${PY_VERSION}" -w "ray[cgraph,default]==${RAY_VERSION}" \
        python3 "${@}"
    else
        python3 "${@}"
    fi
}


function ray-kv {
    local address actor_namespace="${RAY_ACTOR_NAMESPACE:-slurm-${SLURM_JOB_ID}}" actor_name="${RAY_ACTOR_NAME:-global_kv}"

    if [[ -n "${RAY_RUNTIME_DIR}" ]]; then
        if ! retry_until -r -1 -- test -f "${RAY_RUNTIME_DIR}/CLIENT" > /dev/null; then
            echo "[ERROR] Ray client file ${RAY_RUNTIME_DIR}/CLIENT not found." >&2
            return 1
        fi

        address="$(cat "${RAY_RUNTIME_DIR}/CLIENT")"
    fi

    local args=()
    while [[ $# -gt 0 ]]; do
        case "${1}" in
            --address) address="${2}"; shift 2 ;;
            --namespace) actor_namespace="${2}"; shift 2 ;;
            --name) actor_name="${2}"; shift 2 ;;
            *) args+=("${1}"); shift ;;
        esac
    done

    if [[ -z "${address}" ]]; then
        echo "[ERROR] Ray head address not set." >&2
        return 1
    fi

    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    ray-py "$(dirname "${BASH_SOURCE[0]}")/kv.py" \
        --address "${address}" \
        --namespace "${actor_namespace}" \
        --name "${actor_name}" \
        "${args[@]}"
}


# shellcheck disable=SC2120
function ray-health-check {
    local address

    if [[ -n "${RAY_RUNTIME_DIR}" ]]; then
        if ! retry_until -r -1 -- test -f "${RAY_RUNTIME_DIR}/HEAD"; then
            echo "[ERROR] Ray head file ${RAY_RUNTIME_DIR}/HEAD not found." >&2
            return 1
        fi

        address="$(cat "${RAY_RUNTIME_DIR}/HEAD")"
    fi

    while [[ $# -gt 0 ]]; do
        case "${1}" in
            --address) address="${2}"; shift 2 ;;
            *) shift ;;
        esac
    done

    if ! retry_until -r -1 -- wrapped_ray health-check \
                                --skip-version-check \
                                --address "${address}" > /dev/null; then
        echo "[ERROR] Ray health check failed / timed out." >&2
        return 1
    fi

    if [[ -z "${RAY_SKIP_KV_INIT}" ]]; then
        echo "[INFO] Initializing kv store at ray head ${address}..."
        if ! retry_until -r -1 -- ray-kv init > /dev/null; then
            echo "[ERROR] Unable to initialize ray KV store." >&2
            return 1
        fi
    fi

    echo "[INFO] Ray head ${address} ready!"
}


function ray-start {
    ## TODO: a way to work without the env var.
    if [[ -z "${RAY_RUNTIME_DIR}" ]]; then
        echo "[ERROR] Missing RAY_RUNTIME_DIR." >&2
        return 1
    fi

    ## https://docs.ray.io/en/latest/cluster/vms/user-guides/large-cluster-best-practices.html#system-configuration
    if [[ $(ulimit -Hn) == "unlimited" ]] || [[ $(ulimit -Hn) -lt 65535 ]]; then
        if ! ulimit -Sn 65535; then
            echo "[WARNING] Could not set ulimit (old value $(ulimit -Hn))" >&2
        fi
    fi

    ## All clients talking to this ray cluster must install using these exact versions.
    # shellcheck disable=SC2155
    local RAY_VERSION=$(wrapped_ray --version | cut -d' ' -f3) PY_VERSION=$(ray-py --version | cut -d' ' -f2)
    # shellcheck disable=SC2155
    local ray_tmpdir="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ray-${SLURM_JOB_ID}-XXXXXX")"

    if [[ $SLURM_PROCID -eq 0 ]]; then
        local RAY_PORT=${RAY_PORT:-$(get_free_port)} RAY_CLIENT_PORT=${RAY_CLIENT_PORT:-$(get_free_port)}
        local ray_head_pid

        ## Dump ray cluster information.
        mkdir -p "${RAY_RUNTIME_DIR}"
        echo "$(hostname --ip-address):${RAY_PORT}" > "${RAY_RUNTIME_DIR}/HEAD"
        echo "$(hostname --ip-address):${RAY_CLIENT_PORT}" > "${RAY_RUNTIME_DIR}/CLIENT"
        echo "${RAY_VERSION}" > "${RAY_RUNTIME_DIR}/VERSION"
        echo "${PY_VERSION}" > "${RAY_RUNTIME_DIR}/PY_VERSION"

        echo "[DEBUG] Ray info written to ${RAY_RUNTIME_DIR}." >&2

        TMPDIR="${ray_tmpdir}" \
        RAY_TMPDIR="${ray_tmpdir}" \
        wrapped_ray start --block \
            --head \
            --include-dashboard=false \
            --disable-usage-stats \
            --object-manager-port 0 \
            --node-manager-port 0 \
            --metrics-export-port 0 \
            --dashboard-agent-grpc-port 0 \
            --min-worker-port 64000 \
            --max-worker-port 64999 \
            --node-ip-address "$(hostname --ip-address)" \
            --ray-client-server-port "${RAY_CLIENT_PORT}" \
            --port "${RAY_PORT}" \
            --temp-dir "${ray_tmpdir}" \
            "${@}" & ray_head_pid=$!

        if ! ray-health-check; then
            return 1
        fi

        wait "${ray_head_pid}"
    else
        if ! ray-health-check; then
            return 1
        fi

        TMPDIR="${ray_tmpdir}" \
        RAY_TMPDIR="${ray_tmpdir}" \
        wrapped_ray start --block \
            --address "$(cat "${RAY_RUNTIME_DIR}/HEAD")" \
            "${@}"
    fi
}


if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    ray-start "${@}"
fi
