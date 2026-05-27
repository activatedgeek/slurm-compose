#!/usr/bin/env bash


function safe_echo {
    mask_vars=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--string)
                mask_str="${2}"
                shift 2
            ;;
            -m|--mask-variables)
                IFS=',' read -ra mask_vars <<< "${2}"
                shift 2
            ;;
            *)
                shift 1
            ;;
        esac
    done

    ## Also mask any other protected env vars.
    for env_row in $(env | grep -E 'TOKEN|API_KEY|PASSWORD'); do
        # shellcheck disable=SC2207
        mask_vars+=($(echo "${env_row}" | cut -d= -f1))
    done

    for mask_var in "${mask_vars[@]}"; do
        mask_str="$(echo "$mask_str" | sed -e "s|\\\$${mask_var}|*******|g" -e "s|\\\$\\\{${mask_var}\\\}|*******|g" -e "s|${!mask_var}|*******|g")"
    done

    echo "${mask_str}"
    unset mask_vars mask_var mask_str
}


function install_uv {
    if ! command -v uv > /dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/bin" sh
    fi
}


function get_free_port {
    install_uv >&2

    uv run -q --no-progress --no-project python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}


function wait_on_file {
    local f retries=12

    while [[ $# -gt 0 ]]; do
        case "${1}" in
            -f) f="${2}"; shift 2 ;;
            -r) retries="${2}"; shift 2 ;;
            *) shift ;;
        esac
    done

    ## Wait for a maximum of 4 hours.
    if [[ $retries -eq -1 ]]; then
        retries=1440
    fi

    until test -f "${f}"; do
        retries=$(( retries - 1 ))
        if [[ $retries -lt 0 ]]; then
            echo "[ERROR] Timout waiting for file ${f}." >&2
            return 1
        fi
        sleep 10
    done
}


function ray_health_check {
    if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
        install_uv >&2
    fi

    local address client_address version py_version retries=12

    if [[ -n "${RAY_RUNTIME_DIR}" ]]; then
        if ! wait_on_file -f "${RAY_RUNTIME_DIR}/HEAD" > /dev/null 2>&1; then
            echo "[ERROR] Ray head file ${RAY_RUNTIME_DIR}/HEAD not found." >&2
            return 1
        fi

        if ! wait_on_file -f "${RAY_RUNTIME_DIR}/CLIENT" > /dev/null 2>&1; then
            echo "[ERROR] Ray client file ${RAY_RUNTIME_DIR}/CLIENT not found." >&2
            return 1
        fi

        if ! wait_on_file -f "${RAY_RUNTIME_DIR}/VERSION" > /dev/null 2>&1; then
            echo "[ERROR] Ray version file ${RAY_RUNTIME_DIR}/VERSION not found." >&2
            return 1
        fi

        if ! wait_on_file -f "${RAY_RUNTIME_DIR}/PY_VERSION" > /dev/null 2>&1; then
            echo "[ERROR] Python version file ${RAY_RUNTIME_DIR}/PY_VERSION not found." >&2
            return 1
        fi

        address="$(cat "${RAY_RUNTIME_DIR}/HEAD")"
        client_address="$(cat "${RAY_RUNTIME_DIR}/CLIENT")"
        version="$(cat "${RAY_RUNTIME_DIR}/VERSION")"
        py_version="$(cat "${RAY_RUNTIME_DIR}/PY_VERSION")"
    fi

    while [[ $# -gt 0 ]]; do
        case "${1}" in
            -r) retries="${2}"; shift 2 ;;
            --address) address="${2}"; shift 2 ;;
            *) shift ;;
        esac
    done

    ## Wait for a maximum of 4 hours.
    if [[ $retries -eq -1 ]]; then
        retries=1440
    fi

    if [[ (-z "${address}") || (-z "${client_address}") || (-z "${version}") || (-z "${py_version}") ]]; then
        echo "[ERROR] Unable to construct ray address/version. Set RAY_RUNTIME_DIR." >&2
        return 1
    fi

    ray_cmd=(ray health-check --address "${address}")
    if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
        ray_cmd=(uvx -q --no-progress -p "${py_version}" -w "ray[cgraph,default]==${version}" "${ray_cmd[@]}")
    fi

    echo "[INFO] Waiting for ray head ${address} to be ready..." >&2
    until "${ray_cmd[@]}" > /dev/null 2>&1; do
        retries=$(( retries - 1 ))
        if [[ $retries -lt 0 ]]; then
            echo "[ERROR] Ray health check failed or timed out." >&2
            return 1
        fi
        sleep 10
    done

    echo "[INFO] Ray head ${address} ready!" >&2

    if [[ -z "${RAY_SKIP_KV_INIT}" ]]; then
        if ! ray_kv init; then
            echo "[ERROR] Unable to initialize ray KV store." >&2
            return 1
        fi
    fi
}


function ray_kv {
    if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
        install_uv >&2
    fi

    local address version py_version retries=12
    local actor_namespace="${RAY_ACTOR_NAMESPACE:-slurm-${SLURM_JOB_ID}}" actor_name="${RAY_ACTOR_NAME:-global_kv}"

    if [[ -n "${RAY_RUNTIME_DIR}" ]]; then
        if ! wait_on_file -f "${RAY_RUNTIME_DIR}/CLIENT" > /dev/null 2>&1; then
            echo "[ERROR] Ray client file ${RAY_RUNTIME_DIR}/CLIENT not found." >&2
            return 1
        fi

        if ! wait_on_file -f "${RAY_RUNTIME_DIR}/VERSION" > /dev/null 2>&1; then
            echo "[ERROR] Ray version file ${RAY_RUNTIME_DIR}/VERSION not found." >&2
            return 1
        fi

        if ! wait_on_file -f "${RAY_RUNTIME_DIR}/PY_VERSION" > /dev/null 2>&1; then
            echo "[ERROR] Python version file ${RAY_RUNTIME_DIR}/PY_VERSION not found." >&2
            return 1
        fi

        address="$(cat "${RAY_RUNTIME_DIR}/CLIENT")"
        version="$(cat "${RAY_RUNTIME_DIR}/VERSION")"
        py_version="$(cat "${RAY_RUNTIME_DIR}/PY_VERSION")"
    fi

    if [[ (-z "${address}") || (-z "${version}") || (-z "${py_version}") ]]; then
        echo "[ERROR] Unable to construct ray client address/version. Set RAY_RUNTIME_DIR." >&2
        return 1
    fi

    kv_cmd_args=()
    while [[ $# -gt 0 ]]; do
        case "${1}" in
            -r) retries="${2}"; shift 2 ;;
            *) kv_cmd_args+=("${1}"); shift ;;
        esac
    done

    ## Wait for a maximum of 4 hours.
    if [[ $retries -eq -1 ]]; then
        retries=1440
    fi

    kv_cmd=(
        python "$(dirname "${BASH_SOURCE[0]}")/../ray/kv.py"
        --address "${address}" --namespace "${actor_namespace}" --name "${actor_name}"
        "${kv_cmd_args[@]}"
    )
    if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
        kv_cmd=(uv run -q --no-progress --no-project -p "${py_version}" -w "ray[cgraph,default]==${version}" "${kv_cmd[@]}")
    fi

    until "${kv_cmd[@]}"; do
        retries=$(( retries - 1 ))
        if [[ $retries -lt 0 ]]; then
            echo "[ERROR] ray_kv command failed or timed out." >&2
            return 1
        fi
        sleep 10
    done
}
